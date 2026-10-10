"""Dense Qwen3 Tunix student: loader parity with transformers, attention + MLP LoRA export, CPU SFT steps."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TRAINER = REPO / "trainer"
sys.path.insert(0, str(TRAINER))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tunix")
pytest.importorskip("qwix")
from flax import nnx  # noqa: E402
from safetensors import safe_open  # noqa: E402

import data as dc  # noqa: E402
import student_models as sm  # noqa: E402
import trainer_core as tcore  # noqa: E402

# Logit parity bound. Both sides run float32 on CPU with the same weights; the only differences are
# reduction order (einsum vs torch matmul, RMSNorm and softmax accumulation), which leaves errors of
# order 1e-6 relative to logits of magnitude ~10. 1e-4 is two orders above that and far below the
# size of any real mismatch (a wrong RoPE base or a transposed projection moves logits by O(1)).
LOGIT_ATOL = 1e-4

HF_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _tiny_config(vocab: int = 512, tied: bool = True, rope_theta: float = 5_000_000.0):
  return transformers.Qwen3Config(
      vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
      num_key_value_heads=2, head_dim=16, rope_theta=rope_theta, tie_word_embeddings=tied,
      max_position_embeddings=1024, rms_norm_eps=1e-6, initializer_range=0.2)


def _save_tiny(path: pathlib.Path, *, vocab: int = 512, tied: bool = True, seed: int = 0):
  torch.manual_seed(seed)
  model = transformers.Qwen3ForCausalLM(_tiny_config(vocab, tied)).eval()
  model.save_pretrained(path)
  return model


def _mesh():
  return sm.make_mesh(1, 1)


def _hf_logits(model, tokens, pad):
  with torch.no_grad():
    return model(torch.tensor(tokens, dtype=torch.long), attention_mask=torch.tensor(pad)).logits.numpy()


def _tunix_logits(model, mesh, tokens, pad):
  with mesh:
    return np.asarray(sm.forward_logits(model, jnp.asarray(tokens), jnp.asarray(pad)))


def _batch(vocab, seed=0):
  tokens = np.random.default_rng(seed).integers(0, vocab, (2, 24)).astype(np.int32)
  pad = np.ones_like(tokens)
  pad[1, 17:] = 0  # right padding on the second row
  return tokens, pad


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tied", [True, False])
def test_tunix_dense_logits_match_transformers(tmp_path, tied):
  hf = _save_tiny(tmp_path, tied=tied)
  model = sm.load_qwen3(str(tmp_path), _mesh(), remat=False, flash=False, compute_dtype=jnp.float32)
  assert model.config.num_experts is None and model.config.hidden_dim == 128
  assert model.config.rope_theta == 5_000_000 and model.config.use_tied_embedding is tied
  tokens, pad = _batch(512)
  got, ref = _tunix_logits(model, _mesh(), tokens, pad), _hf_logits(hf, tokens, pad)
  real = pad.astype(bool)
  assert np.abs(ref[real]).max() > 1.0  # logits are not near zero, so the bound is meaningful
  assert np.abs(got - ref)[real].max() < LOGIT_ATOL


def test_wrong_rope_base_is_detected_by_the_parity_check(tmp_path):
  """Negative control: the same weights with Tunix's default RoPE base (1e6) miss the bound."""
  hf = _save_tiny(tmp_path)
  model = sm.load_qwen3(str(tmp_path), _mesh(), remat=False, flash=False, compute_dtype=jnp.float32)
  sm._bind_rope_theta(1_000_000)  # pylint: disable=protected-access
  try:
    tokens, pad = _batch(512)
    diff = np.abs(_tunix_logits(model, _mesh(), tokens, pad) - _hf_logits(hf, tokens, pad))[pad.astype(bool)]
  finally:
    sm._bind_rope_theta(5_000_000)  # pylint: disable=protected-access
  assert diff.max() > 100 * LOGIT_ATOL


@pytest.mark.parametrize("field,value", [
    ("model_type", "llama"), ("attention_bias", True), ("rope_scaling", {"type": "yarn", "factor": 4.0}),
    ("use_sliding_window", True), ("hidden_act", "gelu"), ("layer_types", ["full_attention", "sliding_attention"]),
    ("rope_parameters", {"rope_theta": 1e6, "rope_type": "yarn", "factor": 4.0}),
    ("quantization_config", {"quant_method": "fp8"}),
])
def test_dense_config_refuses_fields_tunix_cannot_express(tmp_path, field, value):
  _tiny_config().save_pretrained(tmp_path)
  cfg = json.loads((tmp_path / "config.json").read_text())
  cfg[field] = value
  (tmp_path / "config.json").write_text(json.dumps(cfg))
  with pytest.raises(ValueError):
    sm.qwen3_config_from_hf(str(tmp_path))


def test_dense_config_reads_the_real_qwen3_configs():
  """The released configs (transformers 4.x layout, top-level rope_theta) load with their own RoPE base."""
  hub = pytest.importorskip("huggingface_hub")
  for repo, theta, mlp in (("Qwen/Qwen3-1.7B", 1_000_000, 6144), ("Qwen/Qwen3-4B-Instruct-2507", 5_000_000, 9728)):
    try:
      d = hub.snapshot_download(repo, allow_patterns=["config.json"], local_files_only=True)
    except Exception:  # pylint: disable=broad-except
      pytest.skip(f"{repo} config.json is not in the local Hugging Face cache")
    cfg = sm.qwen3_config_from_hf(d)
    assert (cfg.rope_theta, cfg.hidden_dim, cfg.num_experts, cfg.use_tied_embedding) == (theta, mlp, None, True)


# ---------------------------------------------------------------------------
# LoRA on attention + MLP, merged export
# ---------------------------------------------------------------------------


def _randomize_lora(lora_model, seed=1):
  rng = np.random.default_rng(seed)
  for _, v in nnx.iter_graph(lora_model):
    if isinstance(v, nnx.LoRAParam):
      v.set_value(jnp.asarray(rng.normal(0, 0.3, v.get_value().shape), jnp.float32))


def _lora_params(lora_model):
  out = {}
  for path, v in nnx.iter_graph(lora_model):
    if isinstance(v, nnx.LoRAParam):
      out.setdefault(tuple(path[:-1]), {})[str(path[-1])[-1]] = np.asarray(v.get_value(), np.float64)
  return out


def _expected_hf_delta(module_path, a, b, rank, alpha, cfg):
  """W_hf += (alpha/rank) * B_hf @ A_hf, with A_hf [rank, in] and B_hf [out, rank] written out per module kind."""
  kind = module_path[-1]
  if kind in ("q_proj", "k_proj", "v_proj"):  # a [D, r], b [r, N, H]
    a_hf, b_hf = a.T, b.reshape(rank, -1).T
  elif kind == "o_proj":  # a [N, H, r], b [r, D]
    a_hf, b_hf = a.reshape(-1, rank).T, b.T
  else:  # nnx.Linear: a [in, r], b [r, out]
    a_hf, b_hf = a.T, b.T
  return (alpha / rank) * (b_hf @ a_hf)


def _hf_key(module_path):
  layer, part, kind = module_path[1], module_path[2], module_path[3]
  return f"model.layers.{layer}.{'self_attn' if part == 'attn' else 'mlp'}.{kind}.weight"


def _tensors(path):
  with safe_open(str(path), framework="numpy") as f:
    return {k: f.get_tensor(k) for k in f.keys()}


def test_lora_targets_all_seven_projections_and_export_merges_exactly_them(tmp_path):
  base_dir, out_dir = tmp_path / "base", tmp_path / "out"
  _save_tiny(base_dir)
  (base_dir / "tokenizer_config.json").write_text('{"chat_template": "x"}\n')
  rank, alpha = 4, 8.0
  model = sm.load_qwen3(str(base_dir), _mesh(), remat=False, flash=False, compute_dtype=jnp.float32)
  lora = sm.apply_lora(model, _mesh(), sm.LORA_TARGETS, rank, alpha)
  adapted = sorted({".".join(str(p) for p in k) for k in _lora_params(lora)})
  assert adapted == sorted(f"layers.{i}.{m}" for i in range(2) for m in
                           ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
  _randomize_lora(lora)
  params = _lora_params(lora)

  deltas = sm.lora_deltas_hf(lora, alpha, rank)
  manifest = tcore.merge_lora_into_hf_shards(str(base_dir), str(out_dir), deltas)
  expected_keys = sorted(f"model.layers.{i}.{t}.weight" for i in range(2) for t in HF_TARGETS)
  assert sorted(deltas) == expected_keys and manifest["modified_tensors"] == expected_keys

  base, merged = _tensors(base_dir / "model.safetensors"), _tensors(out_dir / "model.safetensors")
  assert sorted(base) == sorted(merged)
  cfg = model.config
  for key in base:
    if key in expected_keys:
      continue
    assert base[key].dtype == merged[key].dtype and base[key].tobytes() == merged[key].tobytes(), key
  for mpath, ab in params.items():
    key = _hf_key(mpath)
    want = base[key].astype(np.float64) + _expected_hf_delta(mpath, ab["a"], ab["b"], rank, alpha, cfg)
    assert merged[key].dtype == np.float32
    np.testing.assert_allclose(merged[key], want, rtol=0, atol=1e-5, err_msg=key)
    assert np.abs(merged[key] - base[key]).max() > 1e-2, key  # the delta is not trivially zero
  for p in base_dir.iterdir():
    if not p.name.endswith(".safetensors"):
      assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256((out_dir / p.name).read_bytes()).digest()


def test_merged_checkpoint_in_transformers_reproduces_the_lora_model(tmp_path):
  """Functional check of the export orientation: HF logits of the merged checkpoint = Tunix LoRA logits."""
  base_dir, out_dir = tmp_path / "base", tmp_path / "out"
  _save_tiny(base_dir)
  model = sm.load_qwen3(str(base_dir), _mesh(), remat=False, flash=False, compute_dtype=jnp.float32)
  lora = sm.apply_lora(model, _mesh(), sm.LORA_TARGETS, 4, 8.0)
  _randomize_lora(lora, seed=2)
  tokens, pad = _batch(512, seed=3)
  lora_logits = _tunix_logits(lora, _mesh(), tokens, pad)
  tcore.merge_lora_into_hf_shards(str(base_dir), str(out_dir), sm.lora_deltas_hf(lora, 8.0, 4))
  merged = transformers.Qwen3ForCausalLM.from_pretrained(out_dir, dtype=torch.float32).eval()
  base = transformers.Qwen3ForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval()
  real = pad.astype(bool)
  assert np.abs(_hf_logits(base, tokens, pad) - lora_logits)[real].max() > 0.1  # LoRA changed the model
  assert np.abs(_hf_logits(merged, tokens, pad) - lora_logits)[real].max() < 1e-3


def test_dense_delta_export_refuses_keys_outside_the_targets():
  assert re.fullmatch(sm.LORA_HF_KEY, "model.layers.3.mlp.down_proj.weight")
  assert not re.fullmatch(sm.LORA_HF_KEY, "model.layers.3.mlp.gate.weight")
  assert not re.fullmatch(sm.LORA_HF_KEY, "model.embed_tokens.weight")


# ---------------------------------------------------------------------------
# SFT steps on CPU through trainer_core.run_sft
# ---------------------------------------------------------------------------


def _qwen3_tokenizer():
  hub = pytest.importorskip("huggingface_hub")
  try:
    d = hub.snapshot_download("Qwen/Qwen3-1.7B", allow_patterns=["*.json", "*.txt", "*.jinja"], local_files_only=True)
  except Exception:  # pylint: disable=broad-except
    pytest.skip("Qwen/Qwen3-1.7B tokenizer is not in the local Hugging Face cache")
  return transformers.AutoTokenizer.from_pretrained(d)


def test_three_sft_steps_on_cpu_reduce_the_loss(tmp_path, monkeypatch):
  """run_sft on the tiny model with the real Qwen3 tokenizer and two single-turn examples.

  The TPU alignment unit (256) is lowered to 2 for this CPU-sized run, so rank, global batch,
  sequence length and loss chunk can be tiny; every other code path is the production one.
  """
  import train  # pylint: disable=g-import-not-at-top

  tok = _qwen3_tokenizer()
  base_dir = tmp_path / "base"
  _save_tiny(base_dir, vocab=151936)
  data = tmp_path / "sft.jsonl"
  examples = [("train-1", "How many shops?", "Count rows.\n```sql\nSELECT COUNT(*) FROM shop\n```"),
              ("train-2", "List names.", "Read the name column.\n```sql\nSELECT name FROM customer\n```")]
  with open(data, "w") as f:
    for qid, q, ans in examples:
      msgs = [{"role": "system", "content": "You write SQLite."}, {"role": "user", "content": q}]
      f.write(json.dumps({"id": f"{qid}#s0", "messages": msgs, "response": ans}) + "\n")
  eval_tasks = tmp_path / "eval.json"
  eval_tasks.write_text(json.dumps(["dev-1"]))
  monkeypatch.setattr(dc, "MXU", 2)
  ap = argparse.ArgumentParser()
  tcore.add_common_args(ap)
  tcore.add_sft_args(ap)
  out = tmp_path / "out"
  args = ap.parse_args([
      "--hf_dir", str(base_dir), "--out_dir", str(out), "--run_id", "cpu",
      "--data", str(data), "--eval_tasks", str(eval_tasks), "--rank", "4", "--alpha", "8",
      "--global_batch", "2", "--micro_batch", "2", "--max_seq_len", "64", "--loss_chunk", "64",
      "--epochs", "3", "--lr", "3e-2", "--warmup_frac", "0.0", "--save_every", "1000",
  ])
  mesh = _mesh()
  tcore.run_sft(args, mesh=mesh, tokenizer=tok, lora_targets=train.LORA_TARGETS,
                forward=train.forward_hidden, deltas_fn_factory=train.deltas_factory(args),
                load_base=lambda: sm.load_qwen3(str(base_dir), mesh, remat=False, flash=False,
                                                compute_dtype=jnp.float32))
  stats = json.loads((out / "sft_data_stats.json").read_text())
  assert stats["n_kept"] == 2 and stats["max_steps"] == 3 and stats["warmup_steps"] == 0
  assert stats["n_input"] == 2 and stats["tokens_trained"] > 0
  losses = [r["value"] for r in map(json.loads, (out / "metrics" / "trainer_metrics.jsonl").read_text().splitlines())
            if r["event"].endswith("loss") and "train" in r["event"]]
  assert len(losses) == 3, losses
  assert all(np.isfinite(losses)) and losses[0] > losses[1] > losses[2], losses


def test_warmup_frac_sets_the_warmup_from_the_step_count(tmp_path, monkeypatch):
  """--warmup_frac 0.05 of 117 steps -> 6 warm-up steps; without it --warmup_steps is used as before."""
  captured = {}

  class Stop(Exception):
    pass

  def fake_schedule(init, peak, warmup, decay, end_value):
    captured["warmup"], captured["decay"] = warmup, decay
    raise Stop

  optax = pytest.importorskip("optax")
  monkeypatch.setattr(optax, "warmup_cosine_decay_schedule", fake_schedule)
  monkeypatch.setattr(dc, "build_sft_arrays", lambda records, *a, **k: ([object()] * (39 * 256 + 10), {}))
  monkeypatch.setattr(tcore, "load_sft_records", lambda path: [])
  monkeypatch.setattr(dc, "sha256_file", lambda path: "0")
  (tmp_path / "eval.json").write_text("[]")
  ap = argparse.ArgumentParser()
  tcore.add_common_args(ap)
  tcore.add_sft_args(ap)
  base = ["--hf_dir", "x", "--out_dir", str(tmp_path / "o"), "--run_id", "r", "--data", "d.jsonl",
          "--eval_tasks", str(tmp_path / "eval.json"), "--max_seq_len", "4096"]
  for extra, want in ((["--warmup_frac", "0.05"], 6), ([], 2), (["--warmup_steps", "9"], 9)):
    with pytest.raises(Stop):
      tcore.run_sft(ap.parse_args(base + extra), mesh=None, tokenizer=[], lora_targets="",
                    forward=None, deltas_fn_factory=None, load_base=None)
    assert (captured["warmup"], captured["decay"]) == (want, 117)
  with pytest.raises(ValueError):
    tcore.run_sft(ap.parse_args(base + ["--warmup_frac", "1.5"]), mesh=None, tokenizer=[],
                  lora_targets="", forward=None, deltas_fn_factory=None, load_base=None)


def test_overlap_check_sees_through_the_sample_suffix(tmp_path):
  ev = tmp_path / "eval.json"
  ev.write_text(json.dumps(["dev-7"]))
  tcore.assert_disjoint({"train-7#s0", "train-8#s3"}, str(ev), "SFT data")
  with pytest.raises(AssertionError):
    tcore.assert_disjoint({"train-1#s0", "dev-7#s2"}, str(ev), "SFT data")
  with pytest.raises(AssertionError):
    tcore.assert_disjoint({"dev-7"}, str(ev), "SFT data")


def test_sft_refuses_records_that_carry_a_reward(tmp_path):
  data = tmp_path / "mixed.jsonl"
  msgs = [{"role": "user", "content": "q"}]
  data.write_text(json.dumps({"id": "train-1#s0", "messages": msgs, "response": "a"}) + "\n" +
                  json.dumps({"id": "train-1#s1", "messages": msgs, "response": "b", "group_id": "train-1",
                              "reward": 0.0}) + "\n")
  with pytest.raises(ValueError, match="carry a reward"):
    tcore.load_sft_records(str(data))


def test_probe_memory_compiles_the_sft_step_on_cpu(tmp_path):
  import train  # pylint: disable=g-import-not-at-top

  base_dir = tmp_path / "base"
  _save_tiny(base_dir)
  model = sm.load_qwen3(str(base_dir), _mesh(), remat="BLOCK", flash=False, compute_dtype=jnp.float32)
  args = argparse.Namespace(rank=4, alpha=8.0, loss_chunk=64, remats="BLOCK,DECODER", micro_batches="1,2",
                            seq_lens="64,128", out_dir=str(tmp_path / "out"))
  train.probe_memory(args, model, _mesh())
  rows = json.loads((tmp_path / "out" / "probe_memory.json").read_text())["rows"]
  assert len(rows) == 8 and all("error" not in r and r["total_gib"] >= 0 for r in rows), rows
