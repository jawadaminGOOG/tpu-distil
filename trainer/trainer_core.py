"""LoRA SFT, one-iteration GRPO and merge-export drivers for the student (Qwen3 or Gemma 4).

The entry point (train.py) supplies:
  load(hf_dir)         -> base model on the mesh
  lora_targets         -> qwix regex
  forward              -> loss-side forward (see make_ce_loss)
  deltas(lora_model)   -> HF-keyed LoRA deltas for merge_lora_into_hf_shards
Everything here logs the trainer's own values; nothing is rescaled or offset.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import re
import time
from typing import Any, Callable

import numpy as np

import data as dc

# Merged SFT export directory: <out_dir>/merged-<SFT_EXPORT_TAG> (and <gcs_out>/merged-<tag>).
SFT_EXPORT_TAG = "sft-bc"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_args(ap: argparse.ArgumentParser) -> None:
  ap.add_argument("--hf_dir", required=True, help="bf16 HF checkpoint of the student (local or gs://)")
  ap.add_argument("--out_dir", required=True, help="local run directory (checkpoints, metrics, merged model)")
  ap.add_argument("--gcs_out", default="", help="gs:// prefix; process 0 uploads --out_dir here at the end")
  ap.add_argument("--run_id", required=True)
  ap.add_argument("--fsdp", type=int, default=2)
  ap.add_argument("--tp", type=int, default=4)
  ap.add_argument("--rank", type=int, default=256)
  ap.add_argument("--alpha", type=float, default=512.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--eval_tasks", default="",
                  help="JSON list (or {\"tasks\": [...]}) of evaluation ids; training ids that match are refused. "
                       "Required by sft and grpo_iter")
  ap.add_argument("--init_lora_ckpt", default="",
                  help="restore LoRA params (latest step) from this Tunix ckpt root first; grpo_iter restores them "
                       "after computing the old log-probs from pi_k (see --skip_epochs)")
  ap.add_argument("--export_merged", action="store_true", help="write merged HF safetensors at the end")


def add_sft_args(ap: argparse.ArgumentParser) -> None:
  ap.add_argument("--data", required=True, help="SFT records JSONL (data.Record; local or gs://)")
  ap.add_argument("--max_seq_len", type=int, default=16384)
  ap.add_argument("--overlong", choices=["drop", "truncate"], default="drop")
  ap.add_argument("--global_batch", type=int, default=256)
  ap.add_argument("--micro_batch", type=int, default=8, help="global examples per forward/backward")
  ap.add_argument("--len_buckets", default="",
                  help="comma-separated padded lengths (multiples of the splash block), e.g. 2048,4096,8192,16384")
  ap.add_argument("--epochs", type=int, default=3)
  ap.add_argument("--lr", type=float, default=1e-4)
  ap.add_argument("--warmup_steps", type=int, default=2)
  ap.add_argument("--warmup_frac", type=float, default=None,
                  help="warm-up as a fraction of the optimizer steps (e.g. 0.05); replaces --warmup_steps when given")
  ap.add_argument("--weight_decay", type=float, default=0.0)
  ap.add_argument("--max_grad_norm", type=float, default=1.0)
  ap.add_argument("--save_every", type=int, default=4)
  ap.add_argument("--loss_chunk", type=int, default=1024, help="sequence chunk for the chunked LM head")


def add_grpo_iter_args(ap: argparse.ArgumentParser) -> None:
  ap.add_argument("--rollouts", required=True,
                  help="GRPO records JSONL of pi_k samples, each with group_id and reward (local or gs://)")
  ap.add_argument("--iteration", type=int, required=True, help="k: --hf_dir holds merged pi_k")
  ap.add_argument("--ref_hf_dir", required=True, help="KL reference (HF dir, local or gs://)")
  ap.add_argument("--diag_base_hf_dir", default="",
                  help="optional: original base; logs k3 KL to it as a diagnostic (never in the loss)")
  ap.add_argument("--max_seq_len", type=int, default=16384)
  ap.add_argument("--overlong", choices=["truncate", "drop"], default="truncate")
  ap.add_argument("--rollout_multiple", type=int, default=256, help="records per iteration must be a multiple")
  ap.add_argument("--allow_unaligned_rollouts", action="store_true")
  ap.add_argument("--mini_batch", type=int, default=256, help="global sequences per optimizer step")
  ap.add_argument("--drop_partial", action="store_true",
                  help="skip the last mini-batch of an epoch when it holds fewer than --mini_batch sequences")
  ap.add_argument("--micro_batch", type=int, default=8, help="global sequences per forward/backward")
  ap.add_argument("--ppo_epochs", type=int, default=1)
  ap.add_argument("--lr", type=float, default=5e-6)
  ap.add_argument("--weight_decay", type=float, default=0.0)
  ap.add_argument("--max_grad_norm", type=float, default=1.0)
  ap.add_argument("--epsilon", type=float, default=0.2)
  ap.add_argument("--beta", type=float, default=0.0, help="k3 KL coefficient in the loss (KL is always logged)")
  ap.add_argument("--kl_clamp", type=float, default=10.0, help="per-token cap on k3 (<= 0: no cap)")
  ap.add_argument("--dual_clip", type=float, default=3.0, help="dual-clip bound for negative advantages (<= 0: off)")
  ap.add_argument("--grpo_len_buckets", default="",
                  help="comma-separated padded widths; each micro-batch is cut to the smallest bucket that holds its "
                       "longest row (sequences are right-padded), so short samples are not computed at max_seq_len")
  ap.add_argument("--adv_norm", choices=["std", "mean"], default="std",
                  help="group advantage: (r - mean) / std, or r - mean (no std scaling)")
  ap.add_argument("--loss_chunk", type=int, default=1024)
  ap.add_argument("--skip_epochs", type=int, default=0,
                  help="resume: skip the first N PPO epochs, continuing from the LoRA in --init_lora_ckpt (written at "
                       "the end of epoch N by an earlier run; required with it). Old log-probs are still recomputed "
                       "from pi_k; the optimizer state (AdamW moments) restarts from zero. N = --ppo_epochs only "
                       "exports the checkpointed LoRA")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def eval_task_ids(path: str) -> set[str]:
  with open(path) as f:
    data = json.load(f)
  return set(data if isinstance(data, list) else data["tasks"])


def assert_disjoint(train_ids: set[str], eval_path: str, what: str) -> None:
  """Refuse training ids that are eval ids, also after removing a per-sample suffix ("<id>#s<k>")."""
  if not eval_path:
    raise ValueError(f"{what}: --eval_tasks is required (the evaluation ids training must not overlap)")
  eval_ids = eval_task_ids(eval_path)
  overlap = {t for t in train_ids if t in eval_ids or re.sub(r"#s[0-9]+$", "", t) in eval_ids}
  if overlap:
    raise AssertionError(f"{what}: {len(overlap)} training ids are eval tasks, e.g. {sorted(overlap)[:5]}")


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
  import jax  # pylint: disable=g-import-not-at-top

  if jax.process_index() == 0:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))


class JsonlMetricsBackend:
  """metrax LoggingBackend that appends each scalar exactly as the trainer logged it."""

  def __init__(self, path: str):
    self.path = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    self._f = open(path, "a")

  def log_scalar(self, event: str, value: Any, **kwargs: Any) -> None:
    # JAX monitoring events (e.g. compile timings during export) can arrive after the
    # trainer closes its loggers; append them rather than fail.
    if self._f.closed:
      self._f = open(self.path, "a")
    v = np.asarray(value)
    rec = {"event": event, "value": v.tolist(), "wall": time.time()}
    rec.update({k: _jsonable(x) for k, x in kwargs.items()})
    self._f.write(json.dumps(rec) + "\n")
    self._f.flush()

  def close(self) -> None:
    self._f.close()


def _jsonable(x: Any) -> Any:
  try:
    json.dumps(x)
    return x
  except TypeError:
    return np.asarray(x).tolist()


def metrics_options(out_dir: str, run_id: str, flush_every: int = 1):
  from tunix.sft import metrics_logger  # pylint: disable=g-import-not-at-top

  log_dir = os.path.join(out_dir, "metrics")
  return metrics_logger.MetricsLoggerOptions(
      log_dir=log_dir,
      run_name=run_id,
      flush_every_n_steps=flush_every,
      backend_kwargs={
          "custom_backend": [
              lambda: JsonlMetricsBackend(os.path.join(log_dir, "trainer_metrics.jsonl")),
              lambda: metrics_logger.TensorboardBackend(log_dir=log_dir, flush_every_n_steps=flush_every),
          ]
      },
  )


def restore_lora(lora_model: Any, ckpt_root: str) -> int:
  from tunix.sft import checkpoint_manager  # pylint: disable=g-import-not-at-top

  mgr = checkpoint_manager.CheckpointManager(root_directory=ckpt_root)
  step, _ = mgr.maybe_restore(lora_model, restore_only_lora_params=True)
  if not step:
    raise FileNotFoundError(f"no LoRA checkpoint under {ckpt_root}")
  print(f"[lora] restored step {step} from {ckpt_root}", flush=True)
  return step


# ---------------------------------------------------------------------------
# LoRA merge -> sharded HF safetensors (vLLM-loadable)
# ---------------------------------------------------------------------------


def merge_lora_into_hf_shards(
    hf_dir: str, out_dir: str, deltas: dict[str, np.ndarray], manifest_path: str | None = None,
    on_shard: Any = None,
) -> dict[str, Any]:
  """Write base + delta into a copy of a sharded HF checkpoint.

  `deltas` maps HF tensor names (e.g. "model.layers.3.self_attn.q_proj.weight")
  to float32 arrays already in HF layout ([out, in]) and already scaled by
  alpha / rank. Every delta key must match exactly one checkpoint tensor with
  the same shape; otherwise this raises before writing anything partial.
  Non-safetensors files (config, tokenizer, generation_config, chat template)
  are copied byte for byte so vLLM sees the same model definition.
  on_shard(path), if given, is called after each shard is written and hashed (e.g. to upload it
  and delete the local copy, so the export never holds more than one shard on disk).
  """
  import shutil  # pylint: disable=g-import-not-at-top
  from safetensors import safe_open  # pylint: disable=g-import-not-at-top
  from safetensors.numpy import save_file  # pylint: disable=g-import-not-at-top

  src = pathlib.Path(hf_dir)
  dst = pathlib.Path(out_dir)
  dst.mkdir(parents=True, exist_ok=True)
  index_path = src / "model.safetensors.index.json"
  if index_path.exists():
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
  else:
    single = src / "model.safetensors"
    with safe_open(str(single), framework="numpy") as f:
      weight_map = {k: single.name for k in f.keys()}

  missing = sorted(set(deltas) - set(weight_map))
  if missing:
    raise KeyError(f"{len(missing)} LoRA delta keys not in checkpoint, e.g. {missing[:3]}")

  by_shard: dict[str, list[str]] = {}
  for k, shard in weight_map.items():
    by_shard.setdefault(shard, []).append(k)

  used: set[str] = set()
  modified: list[str] = []
  shard_sha: dict[str, str] = {}
  for shard, keys in sorted(by_shard.items()):
    tensors: dict[str, np.ndarray] = {}
    # bf16 is not a numpy dtype; go through the flax framework (jax arrays on CPU).
    with safe_open(str(src / shard), framework="flax") as f:
      metadata = f.metadata()
      for k in keys:
        t = f.get_tensor(k)
        if k in deltas:
          d = deltas[k]
          if tuple(d.shape) != tuple(t.shape):
            raise ValueError(f"{k}: delta shape {d.shape} != base shape {t.shape}")
          merged = (np.asarray(t, dtype=np.float32) + d.astype(np.float32)).astype(t.dtype)
          tensors[k] = np.asarray(merged)
          used.add(k)
          modified.append(k)
        else:
          tensors[k] = np.asarray(t)
    save_file(tensors, str(dst / shard), metadata=metadata or {"format": "pt"})
    shard_sha[shard] = dc.sha256_file(dst / shard)
    del tensors
    if on_shard is not None:
      on_shard(dst / shard)

  unused = sorted(set(deltas) - used)
  if unused:
    raise AssertionError(f"{len(unused)} delta keys were not applied, e.g. {unused[:3]}")

  for p in src.iterdir():
    if p.is_file() and not p.name.endswith(".safetensors"):
      shutil.copy2(p, dst / p.name)

  manifest = {
      "base_dir": str(src),
      "out_dir": str(dst),
      "n_modified_tensors": len(modified),
      "modified_tensors": sorted(modified),
      "files": dict(sorted({**shard_sha, **{p.name: dc.sha256_file(p) for p in dst.iterdir()
                                              if p.is_file() and p.name not in shard_sha}}.items())),
  }
  if manifest_path:
    pathlib.Path(manifest_path).write_text(json.dumps(manifest, indent=2))
  return manifest


def export_merged(hf_dir: str, out_dir: str, deltas_fn: Callable[[], dict[str, np.ndarray]], tag: str,
                  stream_to: str | None = None) -> str:
  """All processes join the all-gather inside deltas_fn; process 0 writes the shards.

  stream_to (gs://…/merged-<tag>): upload each shard as soon as it is written and delete the
  local copy, so the export needs one shard of disk instead of a full checkpoint. The remaining
  small files follow with finalize_outputs.
  """
  import jax  # pylint: disable=g-import-not-at-top
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

  deltas = deltas_fn()
  dst = os.path.join(out_dir, f"merged-{tag}")
  if jax.process_index() == 0:
    on_shard = None
    if stream_to:
      import gcs_io  # pylint: disable=g-import-not-at-top

      def on_shard(path):
        gcs_io.upload_file(str(path), f"{stream_to.rstrip('/')}/{path.name}")
        path.unlink()

    manifest = merge_lora_into_hf_shards(hf_dir, dst, deltas, os.path.join(out_dir, f"merged-{tag}.manifest.json"),
                                         on_shard=on_shard)
    print(f"[merge] {manifest['n_modified_tensors']} tensors merged -> {dst}", flush=True)
  multihost_utils.sync_global_devices("merge_done")
  return dst


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


def make_ce_loss(forward: Callable[..., Any], chunk: int):
  """Token-weighted NLL on next-token targets, returned unreduced for global-batch averaging.

  forward(model, input_tokens, positions, pad_mask) returns either
    ("hidden", h[B, L, D], head_fn)  -> the LM head is applied per `chunk` positions under
                                        jax.checkpoint so full [B, L, V] logits never live
    ("logits", logits[B, L, V], None)
  The loss weight is the builder's per-token weight shifted to next-token targets; prompt,
  template and pad tokens are exactly 0 there (asserted at build time).
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from tunix.sft import utils as sft_utils  # pylint: disable=g-import-not-at-top

  def nll_from_logits(logits, targets, w):
    logits = logits.astype(jnp.float32)
    lse = jax.nn.logsumexp(logits, axis=-1)
    tgt = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return jnp.sum((lse - tgt) * w)

  def loss_fn(model, input_tokens, loss_weight, pad_mask, positions):
    targets = jnp.concatenate([input_tokens[:, 1:], jnp.zeros_like(input_tokens[:, :1])], axis=1)
    w = jnp.concatenate([loss_weight[:, 1:], jnp.zeros_like(loss_weight[:, :1])], axis=1).astype(jnp.float32)
    kind, out, head = forward(model, input_tokens, positions, pad_mask)
    if kind == "logits":
      total = nll_from_logits(out, targets, w)
    else:
      L = out.shape[1]
      if L % chunk:
        raise ValueError(f"sequence {L} not divisible by loss chunk {chunk}")
      step = jax.checkpoint(lambda h, t, ww: nll_from_logits(head(h), t, ww))
      total = jnp.zeros((), jnp.float32)
      for s in range(0, L, chunk):
        total = total + step(out[:, s : s + chunk], targets[:, s : s + chunk], w[:, s : s + chunk])
    denom = jnp.sum(w)
    return sft_utils.LossOutput(
        primary_loss=sft_utils.WeightedMetric(total, denom),
        aux_metrics={},
    )

  def gen_input(batch):
    return {
        "input_tokens": batch["input_tokens"],
        "loss_weight": batch["loss_weight"],
        "pad_mask": batch["pad_mask"],
        "positions": sft_utils.build_positions_from_mask(batch["pad_mask"]),
    }

  return loss_fn, gen_input


def load_sft_records(path: str) -> list[dc.Record]:
  """SFT records; a record holding a reward is a GRPO sample (correct or not) and is refused."""
  records = dc.load_records(path)
  graded = [r.id for r in records if r.reward is not None]
  if graded:
    raise ValueError(f"{path}: {len(graded)} records carry a reward (GRPO samples are not SFT data), e.g. {graded[:3]}")
  return records


def run_sft(args, *, base_model=None, mesh, tokenizer, lora_targets: str, forward, deltas_fn_factory, load_base=None):
  """LoRA SFT. Pass `load_base` (a zero-argument loader) instead of `base_model` to run the
  data and schedule checks before the model is loaded."""
  import jax  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top
  from tunix.sft import peft_trainer  # pylint: disable=g-import-not-at-top
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top

  import student_models as sm  # pylint: disable=g-import-not-at-top

  for name in ("rank", "max_seq_len", "global_batch", "loss_chunk"):
    dc.assert_mxu_multiple(name, int(getattr(args, name)))

  records = load_sft_records(args.data)
  assert_disjoint({r.id for r in records}, args.eval_tasks, "SFT data")
  examples, stats = dc.build_sft_arrays(records, tokenizer, max_seq_len=args.max_seq_len, overlong=args.overlong)
  steps_per_epoch = len(examples) // args.global_batch
  if steps_per_epoch == 0:
    raise ValueError(f"{len(examples)} kept examples < global_batch {args.global_batch}")
  max_steps = steps_per_epoch * args.epochs
  warmup_frac = getattr(args, "warmup_frac", None)
  if warmup_frac is not None and not 0.0 <= warmup_frac < 1.0:
    raise ValueError(f"--warmup_frac={warmup_frac} must be in [0, 1)")
  requested_warmup = args.warmup_steps if warmup_frac is None else int(round(warmup_frac * max_steps))
  # warmup_cosine_decay_schedule needs decay_steps (= max_steps) > warmup_steps.
  warmup_steps = min(requested_warmup, max_steps - 1)
  grad_accum = args.global_batch // args.micro_batch
  stats.update({
      "data_path": args.data,
      "data_sha256": dc.sha256_file(args.data),
      "hf_dir": args.hf_dir,
      "tokenizer_vocab": len(tokenizer),
      "max_seq_len": args.max_seq_len,
      "global_batch": args.global_batch,
      "micro_batch": args.micro_batch,
      "grad_accum": grad_accum,
      "optimizer_steps_per_epoch": steps_per_epoch,
      "epochs": args.epochs,
      "max_steps": max_steps,
      "warmup_steps": warmup_steps,
      "warmup_frac": warmup_frac,
      "lr": args.lr,
      "len_buckets": args.len_buckets,
      "examples_dropped_trailing_per_epoch": len(examples) - steps_per_epoch * args.global_batch,
      "processes": jax.process_count(),
  })
  write_json(os.path.join(args.out_dir, "sft_data_stats.json"), stats)
  print(f"[sft] {json.dumps({k: v for k, v in stats.items() if k != 'dropped_ids'})}", flush=True)
  schedule = (optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup_steps, max_steps, end_value=0.0)
              if max_steps > 1 else optax.constant_schedule(args.lr))

  if base_model is None:
    base_model = load_base()
  lora_model = sm.apply_lora(base_model, mesh, lora_targets, args.rank, args.alpha)
  # apply_lora re-places the weights into new arrays; the base model object still holds the old
  # ones. Drop it so the base weights are not resident twice during training.
  del base_model
  gc.collect()
  if args.init_lora_ckpt:
    restore_lora(lora_model, args.init_lora_ckpt)

  optimizer = optax.chain(
      optax.clip_by_global_norm(args.max_grad_norm),
      optax.inject_hyperparams(optax.adamw)(learning_rate=schedule, weight_decay=args.weight_decay),
  )
  cfg = peft_trainer.TrainingConfig(
      eval_every_n_steps=10**9,
      max_steps=max_steps,
      gradient_accumulation_steps=grad_accum,
      # Multi-host orbax needs a path every host sees: GCS when --gcs_out is set.
      checkpoint_root_directory=(f"{args.gcs_out.rstrip('/')}/ckpt" if args.gcs_out
                                 else os.path.join(args.out_dir, "ckpt")),
      checkpointing_options=ocp.CheckpointManagerOptions(save_interval_steps=args.save_every, max_to_keep=3),
      metrics_logging_options=metrics_options(args.out_dir, args.run_id),
      data_sharding_axis=("fsdp",),
  )
  loss_fn, gen_input = make_ce_loss(forward, args.loss_chunk)
  trainer = (
      peft_trainer.PeftTrainer(lora_model, optimizer, cfg)
      .with_gen_model_input_fn(gen_input)
      .with_loss_fn(loss_fn)
  )
  # Batches are placed on the mesh here (batch axis over "fsdp"), so Tunix's input sharding
  # passes them through: its own placement checks divisibility against the process-local
  # batch and silently replicates when that fails, which yields a wrong global batch.
  batches = ({k: _global(mesh, v) for k, v in b.items()} for b in dc.global_micro_batches(
      examples,
      global_batch=args.global_batch,
      micro_batch=args.micro_batch,
      epochs=args.epochs,
      seed=args.seed,
      buckets=[int(x) for x in args.len_buckets.split(",")] if args.len_buckets else None,
  ))
  t0 = time.time()
  with mesh:
    trainer.train(batches)
  write_json(os.path.join(args.out_dir, "sft_done.json"), {"wall_s": time.time() - t0, "max_steps": max_steps})
  if args.export_merged:
    export_merged(args.hf_dir, args.out_dir, deltas_fn_factory(lora_model), SFT_EXPORT_TAG,
                  stream_to=stream_target(args, SFT_EXPORT_TAG))
  return lora_model


# ---------------------------------------------------------------------------
# GRPO (one iteration per call)
#   pi_k served by vLLM -> G samples per training question, reward 1 if the SQL returns the gold
#   rows -> GRPO records (distil/build_grpo_rollouts.py) -> grpo_iter (this) -> merged pi_{k+1}
# ---------------------------------------------------------------------------


def group_advantages(samples: dict[str, tuple[str, float]], norm: str = "std") -> tuple[dict[str, float], dict[str, Any]]:
  """GRPO advantages per group g: A = (r - mean_g) / std_g (norm="std", population std) or A = r - mean_g
  (norm="mean", no std scaling, so a group's advantages keep the size of its reward differences).

  samples: record id -> (group_id, reward). Groups whose rewards all match
  carry no signal and are dropped (their records get no advantage).
  """
  if norm not in ("std", "mean"):
    raise ValueError(f"unknown advantage norm {norm}")
  groups: dict[str, list[str]] = {}
  for rid, (gid, _) in samples.items():
    groups.setdefault(gid, []).append(rid)
  adv: dict[str, float] = {}
  zero_var = []
  sizes = {}
  for gid, rids in groups.items():
    r = np.asarray([samples[i][1] for i in rids], np.float64)
    sizes[gid] = len(rids)
    std = float(r.std())
    if std == 0.0:
      zero_var.append(gid)
      continue
    mean = float(r.mean())
    for i, ri in zip(rids, r):
      adv[i] = float((ri - mean) / std) if norm == "std" else float(ri - mean)
  rewards = np.asarray([rw for _, rw in samples.values()], np.float64)
  a = np.asarray(list(adv.values()), np.float64)
  stats = {
      "adv_norm": norm,
      "n_samples": len(samples),
      "n_groups": len(groups),
      "n_groups_zero_variance": len(zero_var),
      "n_groups_used": len(groups) - len(zero_var),
      "zero_variance_group_ids": sorted(zero_var),
      "group_sizes": dict(sorted({str(k): sum(1 for v in sizes.values() if v == k) for k in set(sizes.values())}.items())),
      "reward_mean": float(rewards.mean()) if rewards.size else float("nan"),
      "reward_std": float(rewards.std()) if rewards.size else float("nan"),
      "n_samples_used": int(a.size),
      "adv_mean": float(a.mean()) if a.size else float("nan"),
      "adv_std": float(a.std()) if a.size else float("nan"),
      "adv_min": float(a.min()) if a.size else float("nan"),
      "adv_max": float(a.max()) if a.size else float("nan"),
  }
  return adv, stats


def make_token_stats(forward: Callable[..., Any], chunk: int):
  """fn(model, tokens, positions, pad_mask) -> (logp[B, L], entropy[B, L]) for next-token targets.

  Position t holds log pi(tokens[t+1] | tokens[<=t]); the last position is 0.
  Entropy is over the full vocabulary and carries no gradient (metric only).
  The LM head / log-softmax runs `chunk` positions at a time under jax.checkpoint.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  def from_logits(logits, targets):
    logits = logits.astype(jnp.float32)
    lse = jax.nn.logsumexp(logits, axis=-1)
    logp = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0] - lse
    sg = jax.lax.stop_gradient(logits)
    p = jax.nn.softmax(sg, axis=-1)
    ent = jax.lax.stop_gradient(lse) - jnp.sum(p * sg, axis=-1)
    return logp, ent

  def fn(model, tokens, positions, pad_mask):
    targets = jnp.concatenate([tokens[:, 1:], jnp.zeros_like(tokens[:, :1])], axis=1)
    kind, out, head = forward(model, tokens, positions, pad_mask)
    L = out.shape[1]
    if L % chunk:
      raise ValueError(f"sequence {L} not divisible by loss chunk {chunk}")
    if kind == "logits":
      step = jax.checkpoint(from_logits)
      parts = [step(out[:, s : s + chunk], targets[:, s : s + chunk]) for s in range(0, L, chunk)]
    else:
      step = jax.checkpoint(lambda h, t: from_logits(head(h), t))
      parts = [step(out[:, s : s + chunk], targets[:, s : s + chunk]) for s in range(0, L, chunk)]
    logp = jnp.concatenate([p[0] for p in parts], axis=1)
    ent = jnp.concatenate([p[1] for p in parts], axis=1)
    last = jnp.arange(L)[None, :] == L - 1
    return jnp.where(last, 0.0, logp), jnp.where(last, 0.0, ent)

  return fn


def make_grpo_loss(token_stats: Callable[..., Any], epsilon: float, beta: float, kl_clamp: float = 10.0,
                   dual_clip: float = 3.0):
  """Dual-clip PPO token loss + beta * clamped k3(pi || ref), summed over weighted tokens, scaled by inv_ntok.

  The weight is the builder's per-token weight shifted to next-token targets (prompt, template
  and pad positions are exactly 0) times a per-row scale: 1 for real rows and 0 for the rows
  that pad a micro-batch to its fixed size (micro_row_scale).
  k3 = exp(d) - d - 1 with d = ref_logp - logp grows exponentially in d, so one token whose
  log-probability falls far below the reference dominates the loss and its gradient; each token's
  k3 is clamped to kl_clamp (zero gradient beyond it). For negative advantages the policy loss is
  capped at -A * dual_clip, so a token with an extreme ratio cannot produce an unbounded gradient.
  kl_clamp <= 0 or dual_clip <= 0 disables the respective bound.
  Returns (scaled loss, aux sums); aux sums are divided by their token weight when logged.
  `base_logp` only feeds the diagnostic `kl_base` (no gradient).
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  def loss_fn(model, tokens, weight, pad_mask, positions, row_scale, adv, old_logp, ref_logp, base_logp, inv_ntok):
    w = jnp.concatenate([weight[:, 1:], jnp.zeros_like(weight[:, :1])], axis=1).astype(jnp.float32)
    w = w * row_scale[:, None]
    logp, ent = token_stats(model, tokens, positions, pad_mask)
    on = w > 0
    log_ratio = jnp.where(on, logp - old_logp, 0.0)
    ratio = jnp.exp(log_ratio)
    a = adv[:, None]
    pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * a)
    if dual_clip > 0:
      pg = jnp.where(a < 0, jnp.minimum(pg, -a * dual_clip), pg)
    d = jnp.where(on, ref_logp - logp, 0.0)
    k3_raw = jnp.exp(d) - d - 1.0
    k3 = jnp.minimum(k3_raw, kl_clamp) if kl_clamp > 0 else k3_raw
    db = jax.lax.stop_gradient(jnp.where(on, base_logp - logp, 0.0))
    k3_base = jnp.exp(db) - db - 1.0
    tok = jnp.where(on, pg + beta * k3, 0.0)
    clipped = ((a > 0) & (ratio > 1.0 + epsilon)) | ((a < 0) & (ratio < 1.0 - epsilon))
    aux = {
        "w": jnp.sum(w),
        "pg": jnp.sum(jnp.where(on, w * pg, 0.0)),
        "kl": jnp.sum(jnp.where(on, w * k3, 0.0)),
        "kl_raw": jnp.sum(jnp.where(on, w * k3_raw, 0.0)),
        "kl_clamped_tokens": jnp.sum(jnp.where(on & (k3_raw > k3), w, 0.0)),
        "kl_base": jnp.sum(jnp.where(on, w * k3_base, 0.0)),
        "clip": jnp.sum(jnp.where(on, w * clipped, 0.0)),
        "ent": jnp.sum(jnp.where(on, w * ent, 0.0)),
        "ratio": jnp.sum(jnp.where(on, w * ratio, 0.0)),
        "abs_log_ratio_max": jnp.max(jnp.abs(log_ratio)),
    }
    return jnp.sum(w * tok) * inv_ntok, aux

  return loss_fn


def load_grpo_records(path: str) -> list[dc.Record]:
  """GRPO records: each carries group_id and reward; ids are unique."""
  records = dc.load_records(path)
  seen: set[str] = set()
  for r in records:
    if r.group_id is None or r.reward is None:
      raise ValueError(f"{path}: record {r.id} has no group_id or reward (not a GRPO sample)")
    if r.id in seen:
      raise ValueError(f"{path}: duplicate record id {r.id}")
    seen.add(r.id)
  return records


def build_grpo_sequences(records, tokenizer, adv: dict[str, float], *, max_seq_len: int,
                         overlong: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  """Tokenize every record that has an advantage (prompt weight 0, response + end-of-turn weight 1).

  Records of zero-variance groups have no advantage and are skipped. A record longer than
  max_seq_len is dropped (overlong="drop") or truncated (overlong="truncate").
  """
  toks, wts, pads, advs, ids = [], [], [], [], []
  n_trunc = n_drop_overlong = n_zero = n_skipped_group = n_mismatch = 0
  units = []
  for rec in records:
    if rec.id not in adv:
      n_skipped_group += 1
      continue
    units.append(rec)
  for rec in units:
    try:
      r = dc.render_example(rec, tokenizer, max_seq_len=max_seq_len, overlong="drop")
    except dc.TemplateMismatch:
      n_mismatch += 1
      continue
    if r is None:
      if overlong == "drop":
        n_drop_overlong += 1
        continue
      r = dc.render_example(rec, tokenizer, max_seq_len=max_seq_len, overlong="truncate")
      n_trunc += 1
    if r.n_trained == 0:
      n_zero += 1
      continue
    toks.append(r.input_ids)
    wts.append(r.loss_weight)
    pads.append((r.seg != dc.SEG["pad"]).astype(np.int32))
    advs.append(adv[rec.id])
    ids.append(rec.id)
  if not toks:
    raise ValueError("no trainable sequences after advantage filtering")
  if n_mismatch > dc.MAX_TEMPLATE_MISMATCH_FRACTION * len(records):
    raise dc.TemplateMismatch(f"{n_mismatch} of {len(records)} records do not fit the chat template")
  data = {
      "tokens": np.stack(toks),
      "weight": np.stack(wts),
      "pad_mask": np.stack(pads),
      "adv": np.asarray(advs, np.float32),
  }
  stats = {
      "n_sequences": len(toks),
      "n_records_skipped_zero_variance_group": n_skipped_group,
      "n_records_truncated": n_trunc,
      "n_records_dropped_overlong": n_drop_overlong,
      "n_records_zero_trained_tokens": n_zero,
      "n_records_dropped_template_mismatch": n_mismatch,
      "tokens_trained": int((data["weight"] > 0).sum()),
      "tokens_real": int(data["pad_mask"].sum()),
  }
  return data, stats


def _global(mesh, arr: np.ndarray):
  import jax  # pylint: disable=g-import-not-at-top
  from jax.sharding import NamedSharding, PartitionSpec as P  # pylint: disable=g-import-not-at-top

  spec = P("fsdp", *([None] * (arr.ndim - 1)))
  sh = NamedSharding(mesh, spec)
  return jax.make_array_from_callback(arr.shape, sh, lambda idx: arr[idx])


def _gather(x) -> np.ndarray:
  import jax  # pylint: disable=g-import-not-at-top
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

  if jax.process_count() > 1:
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))
  return np.asarray(x)


def bucket_width(pad_mask_rows: np.ndarray, buckets: list[int], full: int) -> int:
  """Smallest bucket >= the longest real (right-padded) row in pad_mask_rows; `full` without buckets or overflow."""
  if not buckets:
    return full
  n = int(pad_mask_rows.sum(axis=1).max()) if pad_mask_rows.size else 0
  for b in sorted(buckets):
    if b >= n and b <= full:
      return b
  return full


def length_sorted(rows: np.ndarray, lengths: np.ndarray) -> np.ndarray:
  """rows reordered by real length (stable), so consecutive micro-batches have similar widths."""
  return rows[np.argsort(lengths[rows], kind="stable")]


def micro_row_scale(n_real: int, micro: int) -> np.ndarray:
  """Per-row loss scale of one micro-batch: 1 for each real row, then 0 for the padding rows."""
  return np.concatenate([np.ones(n_real, np.float32), np.zeros(micro - n_real, np.float32)])


def minibatch_token_weight(shifted_w: np.ndarray, rows: np.ndarray) -> float:
  """Total next-token loss weight of a mini-batch; the loss divides by it."""
  return float(shifted_w[rows].sum())


def compute_logps(model, token_stats_jit, data: dict[str, np.ndarray], mesh, micro: int,
                  buckets: list[int] | None = None) -> np.ndarray:
  from tunix.sft import utils as sft_utils  # pylint: disable=g-import-not-at-top

  n, width = data["tokens"].shape
  out = np.zeros(data["tokens"].shape, np.float32)
  order = length_sorted(np.arange(n), data["pad_mask"].sum(axis=1)) if buckets else np.arange(n)
  for s in range(0, n, micro):
    idx = order[s : s + micro]
    real = len(idx)
    full = np.concatenate([idx, np.repeat(idx[:1], micro - real)]) if real < micro else idx
    L = bucket_width(data["pad_mask"][full], buckets or [], width)
    t = _global(mesh, data["tokens"][full][:, :L])
    pm = _global(mesh, data["pad_mask"][full][:, :L])
    pos = sft_utils.build_positions_from_mask(pm)
    logp, _ = token_stats_jit(model, t, pos, pm)
    out[idx, :L] = _gather(logp)[:real]
  return out


def check_minibatches(n_sequences: int, args) -> None:
  """Refuse an iteration that could not take a single optimizer step."""
  if args.drop_partial and n_sequences < args.mini_batch:
    raise ValueError(f"{n_sequences} trainable sequences < --mini_batch {args.mini_batch} with --drop_partial: "
                     "no optimizer step would run")


def epoch_minibatches(args, epoch: int, n: int, lengths: np.ndarray, buckets: list[int]):
  """(rows, dropped_partial) for each mini-batch of PPO epoch `epoch`, in training order.

  The order depends only on (seed, iteration, epoch), so a resumed run replays the same epochs.
  """
  order = np.random.default_rng(args.seed + 1000 * args.iteration + epoch).permutation(n)
  for mb_start in range(0, n, args.mini_batch):
    rows = order[mb_start : mb_start + args.mini_batch]
    if buckets:
      rows = length_sorted(rows, lengths)  # same rows, so the summed gradient is unchanged
    yield rows, bool(args.drop_partial and len(rows) < args.mini_batch)


def optimizer_steps_in_epochs(args, epochs: int, n: int, lengths: np.ndarray, buckets: list[int],
                              shifted_w: np.ndarray) -> int:
  """Optimizer steps the first `epochs` PPO epochs take: mini-batches that are neither a dropped
  partial one nor without loss weight (the training loop skips both)."""
  return sum(1 for e in range(epochs) for rows, partial in epoch_minibatches(args, e, n, lengths, buckets)
             if not partial and minibatch_token_weight(shifted_w, rows) != 0.0)


def _same_root(a: str, b: str) -> bool:
  norm = lambda r: r.rstrip("/") if "://" in r else os.path.abspath(r)  # noqa: E731
  return norm(a) == norm(b)


def latest_lora_step(root: str) -> int | None:
  """Latest step of the LoRA checkpoints under root (None when there is none)."""
  mgr = lora_checkpointer(root)
  try:
    return mgr.latest_step()
  finally:
    mgr.close()


def lora_ckpt_root(args) -> str:
  """LoRA checkpoint root: <gcs_out>/ckpt (orbax writes gs:// directly; every host sees it), else <out_dir>/ckpt."""
  return f"{args.gcs_out.rstrip('/')}/ckpt" if args.gcs_out else os.path.join(args.out_dir, "ckpt")


def lora_checkpointer(root: str, max_to_keep: int = 2):
  """Tunix CheckpointManager that saves synchronously (the save is on disk when save() returns).

  Its LoRA-only checkpoints are the format restore_lora reads.
  """
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top
  from tunix.sft import checkpoint_manager  # pylint: disable=g-import-not-at-top

  return checkpoint_manager.CheckpointManager(
      root_directory=root,
      options=ocp.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=max_to_keep,
                                           enable_async_checkpointing=False))


def grpo_micro_step(token_stats: Callable[..., Any], epsilon: float, beta: float, kl_clamp: float = 10.0,
                    dual_clip: float = 3.0):
  """The GRPO micro-batch step (un-jitted): ((loss, aux), LoRA grads) for one micro-batch.

  fn(model, tokens, weight, pad_mask, row_scale, adv, old_logp, ref_logp, base_logp, inv_ntok);
  positions come from pad_mask.
  """
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from tunix.sft import utils as sft_utils  # pylint: disable=g-import-not-at-top

  loss_fn = make_grpo_loss(token_stats, epsilon, beta, kl_clamp, dual_clip)
  grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam), has_aux=True)

  def micro_step(model, tokens, weight, pad_mask, row_scale, adv_r, old, ref, base_lp, inv_ntok):
    positions = sft_utils.build_positions_from_mask(pad_mask)
    return grad_fn(model, tokens, weight, pad_mask, positions, row_scale, adv_r, old, ref, base_lp, inv_ntok)

  return micro_step


def run_grpo_iter(args, *, load_policy: Callable[[str], Any], mesh, tokenizer, lora_targets: str,
                  forward, deltas_fn_factory):
  """One GRPO iteration on pi_k samples; writes merged pi_{k+1}.

  load_policy(hf_dir) -> base model. Old log-probs are recomputed here from pi_k (the
  LoRA-wrapped model before any update: LoRA B = 0), never taken from the sampler.
  The LoRA params are checkpointed (lora_ckpt_root, step = optimizer steps so far) at the end of
  every PPO epoch, so the last one precedes the merged export. A root that already holds a
  checkpoint is refused at startup unless this run resumes from it. A resumed run (--init_lora_ckpt,
  --skip_epochs N) computes the old log-probs from the fresh LoRA first, then restores the
  checkpoint and trains epochs N.. with a fresh optimizer state; with N = --ppo_epochs it skips the
  log-prob passes and only exports. A failed save after the last epoch is logged and the export
  still runs (single host only: on several hosts one process must not continue alone).
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top

  import student_models as sm  # pylint: disable=g-import-not-at-top

  for name in ("rank", "max_seq_len", "loss_chunk"):
    dc.assert_mxu_multiple(name, int(getattr(args, name)))
  if args.mini_batch % args.micro_batch:
    raise ValueError("mini_batch must be a multiple of micro_batch")
  skip_epochs = int(getattr(args, "skip_epochs", 0))
  init_ckpt = getattr(args, "init_lora_ckpt", "")
  if not 0 <= skip_epochs <= args.ppo_epochs:
    raise ValueError(f"--skip_epochs {skip_epochs} must be in [0, --ppo_epochs {args.ppo_epochs}]")
  if skip_epochs and not init_ckpt:
    raise ValueError("--skip_epochs needs --init_lora_ckpt (the LoRA written at the end of the skipped epochs)")
  if init_ckpt and not skip_epochs:
    raise ValueError("--init_lora_ckpt in grpo_iter resumes an interrupted iteration: pass --skip_epochs N "
                     "(the PPO epochs that checkpoint finished)")
  export_only = bool(skip_epochs) and skip_epochs == args.ppo_epochs
  # Refuse a root that already holds this iteration's checkpoints before any model is loaded:
  # retraining into it would collide with the existing steps at the first save.
  ckpt_root = lora_ckpt_root(args)
  ckpt = lora_checkpointer(ckpt_root)
  existing = ckpt.latest_step()
  same_root = bool(init_ckpt) and _same_root(init_ckpt, ckpt_root)
  if existing is not None and not same_root:
    ckpt.close()
    raise ValueError(
        f"{ckpt_root} already holds LoRA checkpoint step {existing} from an earlier run of this iteration. "
        f"Resume it (--init_lora_ckpt {ckpt_root} --skip_epochs N, N = the PPO epochs it finished), export it "
        f"(merge --hf_dir <pi_k> --ckpt {ckpt_root} --tag iter_{args.iteration + 1}), or write to a new --gcs_out")
  resume_step = existing if same_root else (latest_lora_step(init_ckpt) if init_ckpt else None)
  if init_ckpt and resume_step is None:
    ckpt.close()
    raise FileNotFoundError(f"no LoRA checkpoint under {init_ckpt}")
  t_start = time.time()
  metrics_path = os.path.join(args.out_dir, "metrics", "grpo_iter_metrics.jsonl")

  def log(rec: dict[str, Any]) -> None:
    if jax.process_index() == 0:
      pathlib.Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
      with open(metrics_path, "a") as f:
        f.write(json.dumps({"iteration": args.iteration, "wall": time.time(), **rec}) + "\n")
      print(f"[grpo_iter] {json.dumps(rec)}", flush=True)

  # 1. samples -> advantages -> sequences
  rollouts = args.rollouts
  records = load_grpo_records(rollouts)
  assert_disjoint({r.id for r in records}, args.eval_tasks, "GRPO rollouts")
  samples = {r.id: (r.group_id, float(r.reward)) for r in records}
  if len(samples) % args.rollout_multiple and not args.allow_unaligned_rollouts:
    raise ValueError(f"{len(samples)} records is not a multiple of {args.rollout_multiple} "
                     "(pass --allow_unaligned_rollouts to override)")
  adv, adv_stats = group_advantages(samples, args.adv_norm)
  data, seq_stats = build_grpo_sequences(records, tokenizer, adv, max_seq_len=args.max_seq_len,
                                         overlong=args.overlong)
  iter_stats = {"event": "iteration_data", "rollouts": args.rollouts,
                "rollouts_sha256": dc.sha256_file(rollouts), "policy_hf_dir": args.hf_dir,
                "ref_hf_dir": args.ref_hf_dir, **adv_stats, **seq_stats}
  log({k: v for k, v in iter_stats.items() if k != "zero_variance_group_ids"})
  write_json(os.path.join(args.out_dir, "grpo_iter_data_stats.json"), iter_stats)

  n = data["tokens"].shape[0]
  check_minibatches(n, args)
  buckets = [int(x) for x in args.grpo_len_buckets.split(",")] if getattr(args, "grpo_len_buckets", "") else []
  lengths = data["pad_mask"].sum(axis=1)
  ts = make_token_stats(forward, args.loss_chunk)
  ts_jit = nnx.jit(ts)
  shifted_w = np.concatenate([data["weight"][:, 1:], np.zeros_like(data["weight"][:, :1])], axis=1)
  if skip_epochs:
    done = optimizer_steps_in_epochs(args, skip_epochs, n, lengths, buckets, shifted_w)
    if resume_step != done:
      ckpt.close()
      raise ValueError(f"the latest checkpoint in {init_ckpt} is step {resume_step}, but {skip_epochs} epoch(s) take "
                       f"{done} optimizer steps; pass the --skip_epochs that matches it")

  # 2. reference log-probs (separate model unless ref == pi_k), freed before loading pi_k
  policy_dir, ref_dir = args.hf_dir, args.ref_hf_dir  # local (resolve_inputs)
  same_ref = os.path.realpath(ref_dir) == os.path.realpath(policy_dir)
  ref_logp = base_logp = None
  base_dir = args.diag_base_hf_dir
  with mesh:
    if not same_ref and not export_only:
      ref_model = load_policy(ref_dir)
      ref_logp = compute_logps(ref_model, ts_jit, data, mesh, args.micro_batch, buckets)
      del ref_model
      gc.collect()
    if base_dir and os.path.realpath(base_dir) == os.path.realpath(ref_dir) and ref_logp is not None:
      base_logp = ref_logp
    elif base_dir and os.path.realpath(base_dir) != os.path.realpath(policy_dir) and not export_only:
      base_model = load_policy(base_dir)
      base_logp = compute_logps(base_model, ts_jit, data, mesh, args.micro_batch, buckets)
      del base_model
      gc.collect()

    # 3. pi_k + fresh LoRA; old log-probs from it before any update
    base = load_policy(policy_dir)
    lora = sm.apply_lora(base, mesh, lora_targets, args.rank, args.alpha, seed=args.seed + args.iteration)
    # apply_lora re-places the weights into new arrays; the base model object still holds the old
    # ones. Drop it so the base weights are not resident twice during training.
    del base
    gc.collect()
    # With no epoch left to train, the log-probs are unused: restore and export only.
    old_logp = None if export_only else compute_logps(lora, ts_jit, data, mesh, args.micro_batch, buckets)
    if same_ref and not export_only:
      ref_logp = old_logp.copy()
    if base_dir and base_logp is None and not export_only:  # base == pi_k (or == ref == pi_k)
      base_logp = old_logp.copy()
    diag = base_logp is not None
    if not diag:
      base_logp = ref_logp  # placeholder input; kl_base is not logged
    # Resume only after the old log-probs: they must come from pi_k, not from the checkpointed LoRA.
    step = restore_lora(lora, init_ckpt) if init_ckpt else 0
    if init_ckpt:
      log({"event": "lora_restored", "root": init_ckpt, "step": step, "skip_epochs": skip_epochs,
           "export_only": export_only})
      if step != resume_step:
        raise RuntimeError(f"restored step {step} from {init_ckpt}, expected {resume_step}")
    saved_step = step

    tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
                     optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay))
    optimizer = nnx.Optimizer(lora, tx, wrt=nnx.LoRAParam)
    micro_step = nnx.jit(grpo_micro_step(ts, args.epsilon, args.beta, args.kl_clamp, args.dual_clip))

    @nnx.jit
    def apply(opt, model, grads):
      gnorm = optax.global_norm(grads)
      opt.update(model, grads)
      return gnorm

    for epoch in range(skip_epochs, args.ppo_epochs):
      for rows, partial in epoch_minibatches(args, epoch, n, lengths, buckets):
        if partial:
          log({"event": "partial_minibatch_dropped", "epoch": epoch, "sequences": int(len(rows))})
          continue
        ntok = minibatch_token_weight(shifted_w, rows)
        if ntok == 0.0:
          continue
        grads = None
        sums = {k: 0.0 for k in ("w", "pg", "kl", "kl_raw", "kl_clamped_tokens", "kl_base", "clip", "ent", "ratio")}
        lr_max = 0.0
        for s in range(0, len(rows), args.micro_batch):
          idx = rows[s : s + args.micro_batch]
          real = len(idx)
          full = np.concatenate([idx, np.repeat(idx[:1], args.micro_batch - real)]) if real < args.micro_batch else idx
          scale = micro_row_scale(real, args.micro_batch)
          L = bucket_width(data["pad_mask"][full], buckets, data["tokens"].shape[1])
          g = lambda a: _global(mesh, a)  # noqa: E731
          c = lambda a: g(a[full][:, :L])  # noqa: E731
          (_, aux), gr = micro_step(lora, c(data["tokens"]), c(data["weight"]), c(data["pad_mask"]),
                                    g(scale), g(data["adv"][full]), c(old_logp), c(ref_logp), c(base_logp),
                                    jnp.float32(1.0 / ntok))
          grads = gr if grads is None else jax.tree.map(jnp.add, grads, gr)
          for k in sums:
            sums[k] += float(aux[k])
          lr_max = max(lr_max, float(aux["abs_log_ratio_max"]))
        gnorm = float(apply(optimizer, lora, grads))
        w = sums["w"]
        pg, kl = sums["pg"] / w, sums["kl"] / w
        rec = {"event": "optimizer_step", "step": step, "epoch": epoch, "sequences": int(len(rows)),
               "tokens": w, "loss": pg + args.beta * kl, "pg_loss": pg, "kl": kl, "kl_raw": sums["kl_raw"] / w,
               "kl_clamped_frac": sums["kl_clamped_tokens"] / w, "clip_frac": sums["clip"] / w,
               "entropy": sums["ent"] / w, "ratio_mean": sums["ratio"] / w, "abs_log_ratio_max": lr_max,
               "grad_norm": gnorm, "lr": args.lr, "beta": args.beta}
        if diag:
          rec["kl_base_diag"] = sums["kl_base"] / w
        log(rec)
        step += 1
      if step > saved_step:
        try:
          ckpt.save(step, lora, save_only_lora_params=True, force=True)
        except Exception as e:  # pylint: disable=broad-except
          if epoch != args.ppo_epochs - 1 or jax.process_count() > 1:
            raise
          log({"event": "lora_checkpoint_failed", "step": step, "epoch": epoch, "root": ckpt_root,
               "error": f"{type(e).__name__}: {str(e)[:300]}"})
        else:
          log({"event": "lora_checkpoint", "step": step, "epoch": epoch, "root": ckpt_root})
        saved_step = step
    ckpt.close()
    if step == 0:
      raise RuntimeError("no optimizer step ran; refusing to export an unchanged policy as the next iteration")

    # 4. merge pi_k + delta -> pi_{k+1}
    tag = f"iter_{args.iteration + 1}"
    merged = export_merged(policy_dir, args.out_dir, deltas_fn_factory(lora), tag, stream_to=stream_target(args, tag))
  done = {"event": "iteration_done", "optimizer_steps": step, "wall_s": time.time() - t_start, "merged": merged}
  log(done)
  write_json(os.path.join(args.out_dir, "grpo_iter_done.json"), done)
  return lora


def resolve_inputs(args) -> None:
  """Download gs:// inputs to this host's local disk (idempotent)."""
  import gcs_io  # pylint: disable=g-import-not-at-top

  for name in ("hf_dir", "data", "ref_hf_dir", "diag_base_hf_dir", "rollouts"):
    v = getattr(args, name, None)
    if v:
      setattr(args, name, gcs_io.ensure_local(v))


def stream_target(args, tag: str) -> str | None:
  """gs:// destination for a streamed export (EXPORT_STREAM=1 and --gcs_out set), else None."""
  if os.environ.get("EXPORT_STREAM") == "1" and getattr(args, "gcs_out", ""):
    return f"{args.gcs_out.rstrip('/')}/merged-{tag}"
  return None


def finalize_outputs(args) -> None:
  import jax  # pylint: disable=g-import-not-at-top
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

  import gcs_io  # pylint: disable=g-import-not-at-top

  if args.gcs_out and jax.process_index() == 0:
    gcs_io.upload_dir(args.out_dir, args.gcs_out.rstrip("/"))
  multihost_utils.sync_global_devices("upload_done")
