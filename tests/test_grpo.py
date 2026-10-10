"""GRPO in trainer_core: records, sequences, advantages, the loss, and run_grpo_iter's loop, checkpoints and resume."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "trainer"))

import data as dc  # noqa: E402
import trainer_core as tcore  # noqa: E402


class _CharTokenizer:
  """One token per character (id = ord + 1), so trained tokens decode back to their text."""
  pad_token_id = 0
  eos_token_id = 0

  def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=False):
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in chat)

  def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
    return {"input_ids": [ord(c) + 1 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def _row(rid, response, *, reward=1.0, group="g", prompt="P"):
  return {"id": rid, "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": prompt}],
          "response": response, "group_id": group, "reward": reward}


def _records(rows):
  return [dc.Record.from_dict(r) for r in rows]


def _decode(tokens, mask):
  return "".join(chr(int(t) - 1) for t, m in zip(tokens, mask) if m)


def _trained_texts(data):
  return [_decode(t, w > 0) for t, w in zip(data["tokens"], data["weight"])]


def _write_rollouts(path, rows):
  path.write_text("".join(json.dumps(r) + "\n" for r in rows))
  return path


# ---------------------------------------------------------------------------
# records and sequences
# ---------------------------------------------------------------------------


def test_grpo_records_need_group_and_reward_and_unique_ids(tmp_path):
  ok = _write_rollouts(tmp_path / "ok.jsonl", [_row("a#s0", "A", reward=1.0), _row("a#s1", "B", reward=0.0)])
  assert [(r.id, r.group_id, r.reward) for r in tcore.load_grpo_records(str(ok))] == [("a#s0", "g", 1.0),
                                                                                     ("a#s1", "g", 0.0)]
  no_reward = dict(_row("a#s0", "A"))
  del no_reward["reward"]
  with pytest.raises(ValueError, match="no group_id or reward"):
    tcore.load_grpo_records(str(_write_rollouts(tmp_path / "x.jsonl", [no_reward])))
  with pytest.raises(ValueError, match="duplicate record id"):
    tcore.load_grpo_records(str(_write_rollouts(tmp_path / "d.jsonl", [_row("a#s0", "A"), _row("a#s0", "B")])))


def test_sequences_train_only_the_response_and_end_of_turn():
  recs = _records([_row("a#s0", "ANSWER-0", reward=1.0), _row("a#s1", "ANSWER-1", reward=0.0)])
  adv, _ = tcore.group_advantages({r.id: (r.group_id, r.reward) for r in recs})
  data, stats = tcore.build_grpo_sequences(recs, _CharTokenizer(), adv, max_seq_len=256, overlong="drop")
  assert _trained_texts(data) == ["ANSWER-0<|im_end|>", "ANSWER-1<|im_end|>"]
  assert data["adv"].tolist() == [1.0, -1.0]
  assert _decode(data["tokens"][0], data["pad_mask"][0]) == (
      "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\nANSWER-0<|im_end|>\n")
  assert stats["n_sequences"] == 2 and stats["tokens_trained"] == 2 * len("ANSWER-0<|im_end|>")
  assert data["tokens"].shape == (2, 256) and set(data) == {"tokens", "weight", "pad_mask", "adv"}


def test_sequences_skip_zero_variance_groups_and_handle_overlong():
  recs = _records([_row("a#s0", "A" * 300, reward=1.0, group="a"), _row("a#s1", "B", reward=0.0, group="a"),
                   _row("b#s0", "C", reward=1.0, group="b"), _row("b#s1", "D", reward=1.0, group="b")])
  adv, st = tcore.group_advantages({r.id: (r.group_id, r.reward) for r in recs})
  assert st["n_groups_zero_variance"] == 1 and set(adv) == {"a#s0", "a#s1"}
  data, stats = tcore.build_grpo_sequences(recs, _CharTokenizer(), adv, max_seq_len=256, overlong="drop")
  assert _trained_texts(data) == ["B<|im_end|>"]
  assert stats["n_records_skipped_zero_variance_group"] == 2 and stats["n_records_dropped_overlong"] == 1
  data, stats = tcore.build_grpo_sequences(recs, _CharTokenizer(), adv, max_seq_len=256, overlong="truncate")
  assert stats["n_records_truncated"] == 1 and data["pad_mask"][0].sum() == 256
  with pytest.raises(ValueError, match="no trainable sequences"):
    tcore.build_grpo_sequences(recs[2:], _CharTokenizer(), {}, max_seq_len=256, overlong="drop")


def test_grpo_args_accept_drop_partial():
  ap = argparse.ArgumentParser()
  tcore.add_grpo_iter_args(ap)
  a = ap.parse_args(["--rollouts", "r", "--iteration", "0", "--ref_hf_dir", "d", "--drop_partial"])
  assert a.drop_partial and ap.parse_args(["--rollouts", "r", "--iteration", "0", "--ref_hf_dir", "d"]).drop_partial is False


def test_check_minibatches_refuses_zero_steps():
  with pytest.raises(ValueError, match="no optimizer step"):
    tcore.check_minibatches(150, types.SimpleNamespace(drop_partial=True, mini_batch=256))
  tcore.check_minibatches(150, types.SimpleNamespace(drop_partial=False, mini_batch=256))
  tcore.check_minibatches(256, types.SimpleNamespace(drop_partial=True, mini_batch=256))


def test_group_advantages_std_and_mean_norm():
  eps = {"a1": ("g", 0.0), "a2": ("g", 0.1), "b1": ("h", 0.0), "b2": ("h", 1.0), "c1": ("k", 1.0), "c2": ("k", 1.0)}
  std_adv, _ = tcore.group_advantages(eps)
  mean_adv, st = tcore.group_advantages(eps, "mean")
  assert std_adv["a2"] == pytest.approx(1.0) and std_adv["b2"] == pytest.approx(1.0)  # 0 vs 0.1 scaled like 0 vs 1
  assert mean_adv["a2"] == pytest.approx(0.05) and mean_adv["b2"] == pytest.approx(0.5)
  assert "c1" not in mean_adv and st["n_groups_zero_variance"] == 1 and st["adv_norm"] == "mean"
  with pytest.raises(ValueError):
    tcore.group_advantages(eps, "rank")


def test_grpo_bucket_width_and_length_sorting():
  pm = np.zeros((2, 16), np.int32)
  pm[0, :5] = 1
  pm[1, :9] = 1
  assert tcore.bucket_width(pm, [], 16) == 16  # no buckets: full width
  assert tcore.bucket_width(pm, [4, 8, 12, 16], 16) == 12  # longest real row 9 -> smallest bucket >= 9
  assert tcore.bucket_width(pm[:1], [4, 8, 12, 16], 16) == 8
  assert tcore.bucket_width(pm, [4, 8], 16) == 16  # overflow falls back to the full width
  exact = np.zeros((1, 16), np.int32)
  exact[0, :8] = 1
  assert tcore.bucket_width(exact, [4, 8, 12, 16], 16) == 8  # a row exactly as long as a bucket fits it
  lengths = np.asarray([9, 2, 5, 2])
  assert tcore.length_sorted(np.asarray([0, 1, 2, 3]), lengths).tolist() == [1, 3, 2, 0]  # stable for ties
  assert sorted(tcore.length_sorted(np.asarray([3, 0, 2]), lengths).tolist()) == [0, 2, 3]  # same rows


def test_micro_row_scale_and_token_weight():
  assert tcore.micro_row_scale(2, 4).tolist() == [1.0, 1.0, 0.0, 0.0]
  assert tcore.micro_row_scale(2, 2).tolist() == [1.0, 1.0]
  shifted = np.asarray([[1, 1, 0], [1, 0, 0], [1, 1, 1]], np.float32)
  assert tcore.minibatch_token_weight(shifted, np.asarray([0, 1, 2])) == 6.0
  assert tcore.minibatch_token_weight(shifted, np.asarray([0, 2])) == 5.0


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------


def _token_stats(jnp):
  def token_stats(model, tokens, positions, pad_mask):
    return model + jnp.zeros(tokens.shape, jnp.float32), jnp.zeros(tokens.shape, jnp.float32)
  return token_stats


def _rows():
  rng = np.random.default_rng(0)
  logp = rng.normal(-1.0, 0.3, (2, 4)).astype(np.float32)
  old = (logp + rng.normal(0.0, 0.1, (2, 4))).astype(np.float32)
  ref = (logp + rng.normal(0.0, 0.1, (2, 4))).astype(np.float32)
  return logp, old, ref


def test_padding_row_contributes_nothing_to_the_loss():
  jax = pytest.importorskip("jax")
  jnp = jax.numpy
  logp, old, ref = _rows()
  adv = np.asarray([1.0, -2.0], np.float32)
  loss_fn = tcore.make_grpo_loss(_token_stats(jnp), 0.2, 0.1)
  t = jnp.zeros((2, 4), jnp.int32)
  w = jnp.ones((2, 4), jnp.float32)

  def loss(lp, a, scale, n):
    out, _ = loss_fn(jnp.asarray(lp), t[: len(lp)], w[: len(lp)], t[: len(lp)], t[: len(lp)], jnp.asarray(scale),
                     jnp.asarray(a), jnp.asarray(old[: len(lp)]), jnp.asarray(ref[: len(lp)]),
                     jnp.asarray(ref[: len(lp)]), jnp.float32(1.0 / n))
    return float(out)

  alone = loss(logp[:1], adv[:1], tcore.micro_row_scale(1, 1), 3)
  other = logp.copy()
  other[1] += 3.0
  assert loss(logp, adv, tcore.micro_row_scale(1, 2), 3) == alone
  assert loss(other, np.asarray([1.0, 5.0], np.float32), tcore.micro_row_scale(1, 2), 3) == alone
  assert loss(logp, adv, tcore.micro_row_scale(2, 2), 3) != alone


def test_grpo_loss_terms_by_hand():
  """PPO clip, dual clip, clamped k3 and beta on hand-picked values."""
  jax = pytest.importorskip("jax")
  jnp = jax.numpy
  eps, beta, clamp, dual = 0.2, 0.5, 1.0, 3.0
  logp = np.asarray([[0.0, np.log(2.0), np.log(0.1), 0.0]], np.float32)  # ratios 1, 2, 0.1 vs old = 0
  old = np.zeros((1, 4), np.float32)
  ref = np.asarray([[0.0, 0.0, 0.0, 0.0]], np.float32)
  w = np.asarray([[1.0, 1.0, 1.0, 1.0]], np.float32)  # shifted: positions 0..2 trained, 3 is the last position
  t = jnp.zeros((1, 4), jnp.int32)
  for a in (1.0, -1.0):
    loss, aux = tcore.make_grpo_loss(_token_stats(jnp), eps, beta, clamp, dual)(
        jnp.asarray(logp), t, jnp.asarray(w), t, t, jnp.ones((1,)), jnp.asarray([a], jnp.float32), jnp.asarray(old),
        jnp.asarray(ref), jnp.asarray(ref), jnp.float32(1.0))
    ratio = np.exp(logp[0, :3].astype(np.float64))
    pg = -np.minimum(ratio * a, np.clip(ratio, 1 - eps, 1 + eps) * a)
    if a < 0:
      pg = np.minimum(pg, -a * dual)
    d = ref[0, :3] - logp[0, :3].astype(np.float64)
    k3 = np.minimum(np.exp(d) - d - 1.0, clamp)
    assert float(loss) == pytest.approx(float(np.sum(pg + beta * k3)), rel=1e-5)
    assert float(aux["w"]) == 3.0 and float(aux["kl_clamped_tokens"]) == 1.0  # ratio 0.1: k3 = 10 - ln 10 - 1 > 1


# ---------------------------------------------------------------------------
# run_grpo_iter loop with the model, log-probs, optimizer and export stubbed
# ---------------------------------------------------------------------------


class _TinyLora:
  """A real nnx module with one base Param and two LoRAParams (what the LoRA checkpoints hold)."""

  def __new__(cls, value):
    from flax import nnx
    import jax.numpy as jnp

    class M(nnx.Module):
      def __init__(self, v):
        self.w = nnx.Param(jnp.ones((4, 4)))
        self.a = nnx.LoRAParam(jnp.full((4, 2), v))
        self.b = nnx.LoRAParam(jnp.full((2, 4), v))

    return M(value)


def _stub_grpo_iter(tmp_path, monkeypatch, extra_args, events=None, ckpts=None, real_ckpt=False, patch_data=None,
                    fail_saves=(), lora=None, rows=None):
  """Run run_grpo_iter's real data and mini-batch loop with the model, log-probs, optimizer and export
  stubbed; return the (row_scale, inv_ntok, tokens) of every micro step and the data it trained on.

  The Tunix checkpoint manager and restore_lora are stubbed too (ckpts: root -> latest step; saves of
  steps in fail_saves raise) unless real_ckpt. `events` (a list) receives the order of model loads,
  log-prob passes, managers, LoRA restores, checkpoint saves and the export. patch_data(data) may edit
  the sequences before training."""
  jax = pytest.importorskip("jax")
  pytest.importorskip("optax")
  pytest.importorskip("tunix")
  import flax
  import student_models as sm
  from tunix.sft import checkpoint_manager

  events = [] if events is None else events
  rows = rows or [_row("a#s0", "RESPONSE-A", reward=1.0, prompt="Task a"),
                  _row("a#s1", "B", reward=0.0, prompt="Task a")]
  rollouts = _write_rollouts(tmp_path / "rollouts.jsonl", rows)
  (tmp_path / "eval.json").write_text(json.dumps(["eval-task"]))
  ap = argparse.ArgumentParser()
  tcore.add_common_args(ap)
  tcore.add_grpo_iter_args(ap)
  args = ap.parse_args(["--hf_dir", str(tmp_path), "--out_dir", str(tmp_path / "out"), "--run_id", "r",
                        "--rollouts", str(rollouts), "--iteration", "0", "--ref_hf_dir", str(tmp_path),
                        "--max_seq_len", "1024", "--micro_batch", "2", "--mini_batch", "4",
                        "--allow_unaligned_rollouts", "--eval_tasks", str(tmp_path / "eval.json"), *extra_args])
  calls, seen = [], {}
  aux = {k: jax.numpy.float32(1.0) for k in ("w", "pg", "kl", "kl_raw", "kl_clamped_tokens", "kl_base", "clip",
                                             "ent", "ratio", "abs_log_ratio_max")}

  def micro_step(model, tokens, weight, pad_mask, row_scale, adv, old, ref, base, inv_ntok):
    calls.append((np.asarray(row_scale), float(inv_ntok), np.asarray(tokens)))
    return (jax.numpy.float32(0.0), aux), {"g": jax.numpy.ones(2)}

  if patch_data is not None:
    build = tcore.build_grpo_sequences

    def patched(*a, **k):
      data, stats = build(*a, **k)
      patch_data(data)
      return data, stats

    monkeypatch.setattr(tcore, "build_grpo_sequences", patched)
  ckpts = {} if ckpts is None else ckpts

  def logps(model, ts, data, mesh, micro, buckets):
    events.append(("logp", len(calls)))
    seen["data"] = data
    return np.zeros(data["tokens"].shape, np.float32)

  class Opt:
    def __init__(self, *a, **k):
      pass

    def update(self, *a):
      pass

  monkeypatch.setattr(sm, "apply_lora", lambda base, *a, **k: object() if lora is None else lora)
  monkeypatch.setattr(tcore, "compute_logps", logps)
  monkeypatch.setattr(tcore, "grpo_micro_step", lambda *a, **k: micro_step)

  class Manager:
    def __init__(self, root_directory, options):
      self.root = root_directory
      events.append(("manager", root_directory, options.max_to_keep, options.enable_async_checkpointing))

    def latest_step(self):
      return ckpts.get(self.root)

    def save(self, step, model, optimizer=None, save_only_lora_params=False, force=False):
      events.append(("save", step, len(calls), save_only_lora_params and optimizer is None))
      if step in fail_saves:
        raise OSError(f"transient write failure at step {step}")
      return True

    def close(self):
      events.append(("close",))

  def restore(lora, root):
    events.append(("restore", root, len(calls)))
    return ckpts[root]

  if not real_ckpt:
    monkeypatch.setattr(checkpoint_manager, "CheckpointManager", Manager)
    monkeypatch.setattr(tcore, "restore_lora", restore)
  monkeypatch.setattr(tcore, "export_merged", lambda *a, **k: events.append(("export",)) or "merged")
  monkeypatch.setattr(flax.nnx, "jit", lambda f: f)
  monkeypatch.setattr(flax.nnx, "Optimizer", Opt)
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("fsdp",))
  tcore.run_grpo_iter(args, load_policy=lambda d: events.append(("load", d)) or object(), mesh=mesh,
                      tokenizer=_CharTokenizer(), lora_targets="", forward=None, deltas_fn_factory=lambda lora: None)
  return calls, seen.get("data")


def _expected_inv_ntok(data):
  w = data["weight"]
  shifted = np.concatenate([w[:, 1:], np.zeros_like(w[:, :1])], axis=1)
  return 1.0 / float(shifted.sum())


def _metrics(tmp_path):
  return [json.loads(x) for x in open(tmp_path / "out" / "metrics" / "grpo_iter_metrics.jsonl")]


def _kinds(events, keep=("load", "logp", "manager", "restore", "save", "close", "export")):
  return [e[0] for e in events if e[0] in keep]


def test_grpo_loop_feeds_unit_row_scale_and_token_weight(tmp_path, monkeypatch):
  calls, data = _stub_grpo_iter(tmp_path, monkeypatch, [])
  assert len(calls) == 1 and calls[0][0].tolist() == [1.0, 1.0]
  assert calls[0][1] == pytest.approx(_expected_inv_ntok(data))
  assert sorted(_trained_texts(data)) == ["B<|im_end|>", "RESPONSE-A<|im_end|>"]


def test_grpo_loop_pads_a_short_micro_batch_with_zero_scale(tmp_path, monkeypatch):
  rows = [_row(f"a#s{k}", f"R{k}", reward=float(k % 2), prompt="Task a") for k in range(3)]
  calls, data = _stub_grpo_iter(tmp_path, monkeypatch, [], rows=rows)
  assert len(calls) == 2  # 3 sequences, micro 2: one full micro step and one padded
  assert sorted(np.concatenate([c[0] for c in calls]).tolist()) == [0.0, 1.0, 1.0, 1.0]
  assert all(c[1] == pytest.approx(_expected_inv_ntok(data)) for c in calls)


def test_grpo_iter_refuses_eval_ids_and_unaligned_record_counts(tmp_path, monkeypatch):
  rows = [_row("eval-task#s0", "A", reward=1.0), _row("eval-task#s1", "B", reward=0.0)]
  with pytest.raises(AssertionError, match="eval tasks"):
    _stub_grpo_iter(tmp_path, monkeypatch, [], rows=rows)
  ap = argparse.ArgumentParser()
  tcore.add_common_args(ap)
  tcore.add_grpo_iter_args(ap)
  base = ["--hf_dir", "h", "--out_dir", str(tmp_path / "o2"), "--run_id", "r", "--rollouts",
          str(tmp_path / "rollouts.jsonl"), "--iteration", "0", "--ref_hf_dir", "h", "--max_seq_len", "1024",
          "--eval_tasks", str(tmp_path / "eval.json")]
  _write_rollouts(tmp_path / "rollouts.jsonl", [_row("a#s0", "A", reward=1.0), _row("a#s1", "B", reward=0.0)])
  with pytest.raises(ValueError, match="not a multiple of 256"):
    tcore.run_grpo_iter(ap.parse_args(base), load_policy=None, mesh=None, tokenizer=_CharTokenizer(),
                        lora_targets="", forward=None, deltas_fn_factory=None)


def test_grpo_iter_checkpoints_lora_every_epoch_before_export(tmp_path, monkeypatch):
  events = []
  calls, _ = _stub_grpo_iter(tmp_path, monkeypatch, ["--ppo_epochs", "3", "--mini_batch", "2"], events)
  assert len(calls) == 3  # 2 sequences, mini 2 = micro 2: one optimizer step per epoch
  root = str(tmp_path / "out" / "ckpt")
  assert events[0] == ("manager", root, 2, False)  # opened (and checked) before any model load; synchronous saves
  assert _kinds(events) == ["manager", "load", "logp", "save", "save", "save", "close", "export"]
  # one LoRA-only save at the end of each epoch (step = optimizer steps so far), all before the export
  assert [e for e in events if e[0] == "save"] == [("save", 1, 1, True), ("save", 2, 2, True), ("save", 3, 3, True)]
  logged = [(m["step"], m["epoch"], m["root"]) for m in _metrics(tmp_path) if m["event"] == "lora_checkpoint"]
  assert logged == [(1, 0, root), (2, 1, root), (3, 2, root)]


def test_grpo_iter_checkpoint_root_follows_gcs_out(tmp_path, monkeypatch):
  events = []
  _stub_grpo_iter(tmp_path, monkeypatch, ["--gcs_out", "gs://b/run/grpo/iter_1/"], events)
  assert events[0] == ("manager", "gs://b/run/grpo/iter_1/ckpt", 2, False)


def test_grpo_iter_refuses_a_root_with_checkpoints_unless_resuming(tmp_path, monkeypatch):
  events = []
  with pytest.raises(ValueError, match="already holds LoRA checkpoint step 1.*--skip_epochs N.*merge"):
    _stub_grpo_iter(tmp_path, monkeypatch, ["--gcs_out", "gs://b/it1"], events, ckpts={"gs://b/it1/ckpt": 1})
  assert "load" not in _kinds(events)  # fails before any model load
  with pytest.raises(ValueError, match="already holds"):  # resuming from another root does not make it safe
    _stub_grpo_iter(tmp_path, monkeypatch, ["--gcs_out", "gs://b/it1", "--init_lora_ckpt", "gs://b/old",
                                            "--skip_epochs", "1", "--ppo_epochs", "2", "--mini_batch", "2"],
                    ckpts={"gs://b/it1/ckpt": 1, "gs://b/old": 1})


def test_grpo_iter_resume_restores_after_old_logps_and_skips_epochs(tmp_path, monkeypatch):
  events = []
  args = ["--ppo_epochs", "3", "--mini_batch", "2", "--init_lora_ckpt", "gs://b/ckpt", "--skip_epochs", "2"]
  calls, _ = _stub_grpo_iter(tmp_path, monkeypatch, args, events, ckpts={"gs://b/ckpt": 2})
  assert len(calls) == 1  # only the last epoch trains
  assert _kinds(events) == ["manager", "manager", "close", "load", "logp", "restore", "save", "close", "export"]
  assert ("restore", "gs://b/ckpt", 0) in events  # after the old log-probs, before any micro step
  assert [e for e in events if e[0] == "save"] == [("save", 3, 1, True)]  # continues from step 2
  steps = [m for m in _metrics(tmp_path) if m["event"] == "optimizer_step"]
  assert [(m["step"], m["epoch"]) for m in steps] == [(2, 2)]


def test_grpo_iter_resume_into_its_own_root(tmp_path, monkeypatch):
  events = []
  root = str(tmp_path / "out" / "ckpt")
  args = ["--ppo_epochs", "2", "--mini_batch", "2", "--init_lora_ckpt", root + "/", "--skip_epochs", "1"]
  calls, _ = _stub_grpo_iter(tmp_path, monkeypatch, args, events, ckpts={root: 1, root + "/": 1})
  assert len(calls) == 1 and [e for e in events if e[0] == "save"] == [("save", 2, 1, True)]


def test_grpo_iter_resume_refuses_a_checkpoint_from_another_epoch(tmp_path, monkeypatch):
  events = []
  args = ["--ppo_epochs", "3", "--mini_batch", "2", "--init_lora_ckpt", "gs://b/ckpt", "--skip_epochs", "1"]
  with pytest.raises(ValueError, match="is step 2, but 1 epoch"):
    _stub_grpo_iter(tmp_path, monkeypatch, args, events, ckpts={"gs://b/ckpt": 2})
  assert "load" not in _kinds(events)


def test_grpo_iter_resume_step_count_skips_dropped_partial_minibatches(tmp_path, monkeypatch):
  # 3 sequences, mini 2 with --drop_partial: each epoch takes 1 step and drops 1 partial.
  rows = [_row(f"a#s{k}", f"R{k}", reward=float(k % 2), prompt="Task a") for k in range(3)]
  args = ["--ppo_epochs", "2", "--mini_batch", "2", "--drop_partial", "--init_lora_ckpt", "gs://b/ckpt",
          "--skip_epochs", "1"]
  calls, _ = _stub_grpo_iter(tmp_path, monkeypatch, args, ckpts={"gs://b/ckpt": 1}, rows=rows)
  assert len(calls) == 1


def _zero_row(row):
  def patch(data):
    data["weight"][row] = 0.0
  return patch


def test_grpo_iter_resume_step_count_skips_minibatches_without_loss_weight(tmp_path, monkeypatch):
  # 2 sequences, mini 1, one without loss weight: each epoch takes 1 step.
  args = ["--ppo_epochs", "2", "--mini_batch", "1", "--micro_batch", "1",
          "--init_lora_ckpt", "gs://b/ckpt", "--skip_epochs", "1"]
  calls, _ = _stub_grpo_iter(tmp_path, monkeypatch, args, ckpts={"gs://b/ckpt": 1}, patch_data=_zero_row(1))
  assert len(calls) == 1


def test_grpo_iter_saves_nothing_when_no_step_ran(tmp_path, monkeypatch):
  events = []
  with pytest.raises(RuntimeError, match="no optimizer step ran"):
    _stub_grpo_iter(tmp_path, monkeypatch, ["--ppo_epochs", "2"], events, patch_data=lambda d: d["weight"].fill(0.0))
  assert "save" not in _kinds(events) and "export" not in _kinds(events)


def test_grpo_iter_export_only_resume_skips_the_log_prob_passes(tmp_path, monkeypatch):
  events = []
  args = ["--ppo_epochs", "2", "--mini_batch", "2", "--init_lora_ckpt", "gs://b/ckpt", "--skip_epochs", "2",
          "--ref_hf_dir", "/other/ref", "--diag_base_hf_dir", "/other/base"]
  calls, _ = _stub_grpo_iter(tmp_path, monkeypatch, args, events, ckpts={"gs://b/ckpt": 2})
  assert not calls and "logp" not in _kinds(events)
  assert [e for e in events if e[0] == "load"] == [("load", str(tmp_path))]  # pi_k only: no ref / base model
  assert _kinds(events)[-3:] == ["restore", "close", "export"]


def test_grpo_iter_separate_reference_is_loaded_before_the_policy(tmp_path, monkeypatch):
  events = []
  _stub_grpo_iter(tmp_path, monkeypatch, ["--ref_hf_dir", "/other/ref"], events)
  assert [e for e in events if e[0] in ("load", "logp")][:3] == [("load", "/other/ref"), ("logp", 0),
                                                               ("load", str(tmp_path))]


def test_grpo_iter_last_epoch_save_failure_still_exports(tmp_path, monkeypatch):
  events = []
  _stub_grpo_iter(tmp_path, monkeypatch, ["--ppo_epochs", "2", "--mini_batch", "2"], events, fail_saves={2})
  assert _kinds(events)[-2:] == ["close", "export"]
  failed = [m for m in _metrics(tmp_path) if m["event"] == "lora_checkpoint_failed"]
  assert [(m["step"], m["epoch"]) for m in failed] == [(2, 1)] and "transient" in failed[0]["error"]


def test_grpo_iter_earlier_or_multi_host_save_failure_is_fatal(tmp_path, monkeypatch):
  events = []
  with pytest.raises(OSError):
    _stub_grpo_iter(tmp_path, monkeypatch, ["--ppo_epochs", "2", "--mini_batch", "2"], events, fail_saves={1})
  assert "export" not in _kinds(events)
  import jax
  monkeypatch.setattr(jax, "process_count", lambda: 2)
  events = []
  (tmp_path / "mh").mkdir()
  with pytest.raises(OSError):
    _stub_grpo_iter(tmp_path / "mh", monkeypatch, ["--ppo_epochs", "2", "--mini_batch", "2"], events,
                    fail_saves={2})
  assert "export" not in _kinds(events)


def test_grpo_iter_skip_epochs_needs_a_checkpoint(tmp_path, monkeypatch):
  with pytest.raises(ValueError, match="init_lora_ckpt"):
    _stub_grpo_iter(tmp_path, monkeypatch, ["--ppo_epochs", "2", "--skip_epochs", "1"])
  with pytest.raises(ValueError, match="skip_epochs"):
    _stub_grpo_iter(tmp_path, monkeypatch, ["--ppo_epochs", "2", "--skip_epochs", "3", "--init_lora_ckpt", "x"])
  with pytest.raises(ValueError, match="pass --skip_epochs"):  # no silent re-training on a restored LoRA
    _stub_grpo_iter(tmp_path, monkeypatch, ["--ppo_epochs", "2", "--init_lora_ckpt", "x"], ckpts={"x": 1})


def test_grpo_iter_real_checkpoints_rerun_and_resume(tmp_path, monkeypatch):
  """Real orbax: a crashed run left step 1. Retraining into the root would hit StepAlreadyExists at
  its first save; the rerun is refused up front, and a resume continues to step 2."""
  pytest.importorskip("jax")
  root = str(tmp_path / "out" / "ckpt")
  crashed = _TinyLora(5.0)
  ck = tcore.lora_checkpointer(root)
  assert ck.save(1, crashed, save_only_lora_params=True, force=True)
  ck.close()
  again = tcore.lora_checkpointer(root)
  with pytest.raises(Exception, match="already exists"):  # what a plain rerun's first save would do
    again.save(1, crashed, save_only_lora_params=True, force=True)
  again.close()
  events = []
  with pytest.raises(ValueError, match="already holds LoRA checkpoint step 1"):
    _stub_grpo_iter(tmp_path, monkeypatch, ["--mini_batch", "2"], events, real_ckpt=True, lora=_TinyLora(0.0))
  assert "load" not in _kinds(events)
  resumed = _TinyLora(0.0)
  calls, _ = _stub_grpo_iter(tmp_path, monkeypatch, ["--ppo_epochs", "2", "--mini_batch", "2",
                                                     "--init_lora_ckpt", root, "--skip_epochs", "1"],
                             real_ckpt=True, lora=resumed)
  assert len(calls) == 1 and float(resumed.a.get_value()[0, 0]) == 5.0  # restored from the crashed run's LoRA
  assert tcore.latest_lora_step(root) == 2
  check = _TinyLora(0.0)
  assert tcore.restore_lora(check, root) == 2 and float(check.b.get_value()[0, 0]) == 5.0


def test_grpo_iter_releases_the_base_model_after_apply_lora(tmp_path, monkeypatch):
  """The policy's base model object is gone before the first log-prob pass (weights not held twice)."""
  import gc
  import weakref

  jax = pytest.importorskip("jax")
  pytest.importorskip("optax")
  pytest.importorskip("tunix")
  import student_models as sm

  rollouts = _write_rollouts(tmp_path / "rollouts.jsonl", [_row("e#s0", "A", reward=0.0), _row("e#s1", "B")])
  (tmp_path / "eval.json").write_text(json.dumps(["eval-task"]))
  ap = argparse.ArgumentParser()
  tcore.add_common_args(ap)
  tcore.add_grpo_iter_args(ap)
  args = ap.parse_args(["--hf_dir", str(tmp_path), "--out_dir", str(tmp_path / "out"), "--run_id", "r",
                        "--rollouts", str(rollouts), "--iteration", "0", "--ref_hf_dir", str(tmp_path),
                        "--max_seq_len", "1024", "--allow_unaligned_rollouts", "--eval_tasks", str(tmp_path / "eval.json")])

  class Base:
    pass

  refs = []

  def load_policy(_):
    b = Base()
    refs.append(weakref.ref(b))
    return b

  class Released(Exception):
    pass

  def first_logps(*_a, **_k):
    gc.collect()
    raise Released(refs[-1]() is None)

  monkeypatch.setattr(sm, "apply_lora", lambda base, *a, **k: object())
  monkeypatch.setattr(tcore, "compute_logps", first_logps)
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("fsdp",))
  with pytest.raises(Released) as e:
    tcore.run_grpo_iter(args, load_policy=load_policy, mesh=mesh, tokenizer=_CharTokenizer(),
                        lora_targets="", forward=None, deltas_fn_factory=None)
  assert len(refs) == 1 and e.value.args == (True,)
