#!/usr/bin/env python3
"""Student on Tunix — dense Qwen3 (model_type "qwen3", e.g. Qwen3-1.7B) or the Gemma 4 text decoder
(model_type "gemma4", e.g. Gemma 4 E2B-it), chosen from --hf_dir's config.json: LoRA SFT, one-iteration
GRPO, merge-export and an HBM probe.

  sft   --data D                         LoRA SFT on single-turn records (data.Record); loss on the
                                         response and its end-of-turn token only
  grpo_iter --rollouts R --iteration k   one GRPO update on pi_k's graded samples -> merged pi_{k+1}
            --ref_hf_dir REF             (old log-probs recomputed from pi_k; k3 KL to REF)
  merge --ckpt <root> --tag T            LoRA ckpt (an SFT checkpoint, or a grpo_iter epoch checkpoint
                                         on --hf_dir pi_k) -> merged-T sharded HF safetensors for vLLM
  probe_memory                           AOT-compile the SFT step per (remat, micro batch, length); HBM use

LoRA: rank 256, alpha 512 (launcher defaults) on attention q/k/v/o and MLP gate/up/down (Gemma 4: the
fused k/v projection, only in layers that compute their own K/V). Each GRPO iteration trains a fresh
LoRA on top of merged pi_k.

--hf_dir / --data / --rollouts / --ref_hf_dir accept gs:// (downloaded per host);
--gcs_out uploads --out_dir from process 0 at the end.

Launch on each host of the v6e slice (jax.distributed picks up TPU_WORKER_*), normally through
train_tpu_vm.sh:
  python3 trainer/train.py sft --hf_dir /mnt/data/models/Qwen3-1.7B --out_dir /mnt/data/train/kd \\
      --run_id kd --data sft.jsonl --eval_tasks dev_ids.json --fsdp 8 --tp 1 --export_merged \\
      --gcs_out gs://<bucket>/Qwen3-1.7B/kd
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import student_models as sm  # noqa: E402  pylint: disable=g-import-not-at-top
import trainer_core as tc  # noqa: E402  pylint: disable=g-import-not-at-top

LORA_TARGETS = sm.LORA_TARGETS  # Qwen3; a Gemma 4 run takes sm.lora_targets(hf_dir)

def forward_hidden(model, input_tokens, positions, pad_mask):
  """Hidden states (no LM head) so the loss applies the head chunk by chunk.

  Splash attention is causal with segment ids; the XLA attention path needs an explicit
  causal mask, which is built here when splash is off.
  """
  from tunix.sft import utils as sft_utils  # pylint: disable=g-import-not-at-top

  attn = None if model.config.use_flash_attention else sft_utils.make_causal_attn_mask(pad_mask)
  h, _ = model(input_tokens, positions, None, attn, segment_ids=pad_mask.astype("int32"), skip_lm_head=True)
  return "hidden", h, lambda x: sm.lm_head_f32(model, x)


def deltas_factory(args):
  family = sm.model_family(args.hf_dir)
  return lambda lora_model: (lambda: sm.lora_deltas_hf(lora_model, args.alpha, args.rank, family))


def probe_memory(args, base_model, mesh, lora_targets: str = sm.LORA_TARGETS) -> None:
  """AOT-compile the LoRA SFT train step for each (remat, micro_batch, seq_len); print HBM use.

  The step is the one PeftTrainer runs: value_and_grad of the chunked-head CE loss with
  respect to LoRA params only, inputs sharded on "fsdp". Nothing is executed.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from jax.sharding import NamedSharding, PartitionSpec as P  # pylint: disable=g-import-not-at-top
  lora = sm.apply_lora(base_model, mesh, lora_targets, args.rank, args.alpha)
  loss_fn, _ = tc.make_ce_loss(forward_hidden, args.loss_chunk)

  def step(model, tokens, weight, pad_mask, positions):
    def f(m):
      out = loss_fn(m, tokens, weight, pad_mask, positions)
      return out.primary_loss.unreduced_sum / jnp.maximum(out.primary_loss.denominator, 1.0)
    return nnx.value_and_grad(f, argnums=nnx.DiffState(0, nnx.LoRAParam))(model)

  cfgs = {id(c): c for _, v in nnx.iter_graph(lora) if (c := getattr(v, "config", None)) is not None
          and hasattr(c, "remat_config")}
  rows = []
  for remat in args.remats.split(","):
    for c in cfgs.values():
      c.remat_config = getattr(type(c.remat_config), remat)
    for mb in (int(x) for x in args.micro_batches.split(",")):
      for L in (int(x) for x in args.seq_lens.split(",")):
        sh = NamedSharding(mesh, P("fsdp", None))
        spec = lambda dt: jax.ShapeDtypeStruct((mb, L), dt, sharding=sh)  # noqa: E731
        row = {"remat": remat, "micro_batch": mb, "seq_len": L}
        try:
          with mesh:
            compiled = nnx.jit(step).lower(lora, spec(jnp.int32), spec(jnp.float32), spec(jnp.int32),
                                          spec(jnp.int32)).compile()
          ma = compiled.memory_analysis()
          row.update({k: round(getattr(ma, k) / 2**30, 3) for k in (
              "temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes", "generated_code_size_in_bytes")})
          row["total_gib"] = round(row["temp_size_in_bytes"] + row["argument_size_in_bytes"] + row["output_size_in_bytes"], 3)
        except Exception as e:  # pylint: disable=broad-except
          row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rows.append(row)
        print("[probe] " + json.dumps(row), flush=True)
  if jax.process_index() == 0:
    stats = jax.local_devices()[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    tc.write_json(os.path.join(args.out_dir, "probe_memory.json"),
                  {"device_bytes_limit_gib": round(limit / 2**30, 3) if limit else None, "rows": rows})


def build_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  for name, extra in (("sft", tc.add_sft_args), ("grpo_iter", tc.add_grpo_iter_args), ("merge", None),
                      ("probe_memory", None)):
    p = sub.add_parser(name)
    tc.add_common_args(p)
    p.add_argument("--flash", default="on", choices=["on", "off"], help="splash attention (TPU); off for CPU runs")
    p.add_argument("--remat", default="BLOCK", choices=["NONE", "BLOCK", "DECODER"],
                   help="Tunix remat: BLOCK = attention block and MLP separately; DECODER = whole layer")
    if extra:
      extra(p)
    if name == "probe_memory":
      p.add_argument("--micro_batches", default="4,8")
      p.add_argument("--seq_lens", default="8192,16384")
      p.add_argument("--remats", default="BLOCK,DECODER")
      p.add_argument("--loss_chunk", type=int, default=1024)
    if name == "merge":
      p.add_argument("--ckpt", required=True, help="Tunix checkpoint root holding LoRA params")
      p.add_argument("--tag", default="export")
  return ap


def main() -> int:
  args = build_parser().parse_args()

  sm.maybe_init_distributed()
  import jax  # pylint: disable=g-import-not-at-top
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  os.makedirs(args.out_dir, exist_ok=True)
  tc.resolve_inputs(args)
  tok = AutoTokenizer.from_pretrained(args.hf_dir)
  mesh = sm.make_mesh(args.fsdp, args.tp)
  family = sm.model_family(args.hf_dir)
  lora_targets = sm.lora_targets(args.hf_dir)
  print(f"[{family}] devices={jax.device_count()} processes={jax.process_count()} mesh={dict(mesh.shape)}", flush=True)

  def load(hf_dir):
    # Flash (splash) attention with segment ids for training; the causal kernel masks padding.
    if sm.model_family(hf_dir) != family:
      raise ValueError(f"{hf_dir} is not a {family} checkpoint")
    return sm.load_student(hf_dir, mesh, remat=args.remat, flash=args.flash == "on")

  if args.cmd == "sft":
    tc.run_sft(args, load_base=lambda: load(args.hf_dir), mesh=mesh, tokenizer=tok, lora_targets=lora_targets,
               forward=forward_hidden, deltas_fn_factory=deltas_factory(args))
  elif args.cmd == "grpo_iter":
    tc.run_grpo_iter(args, load_policy=load, mesh=mesh, tokenizer=tok, lora_targets=lora_targets,
                     forward=forward_hidden, deltas_fn_factory=deltas_factory(args))
  elif args.cmd == "probe_memory":
    probe_memory(args, load(args.hf_dir), mesh, lora_targets)
  else:
    lora = sm.apply_lora(load(args.hf_dir), mesh, lora_targets, args.rank, args.alpha)
    tc.restore_lora(lora, args.ckpt)
    # EXPORT_STREAM=1 with --gcs_out uploads each shard as it is written (no full local copy).
    tc.export_merged(args.hf_dir, args.out_dir, deltas_factory(args)(lora), args.tag,
                     stream_to=tc.stream_target(args, args.tag))
  tc.finalize_outputs(args)
  return 0


if __name__ == "__main__":
  sys.exit(main())
