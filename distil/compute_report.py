#!/usr/bin/env python3
"""Compute, data and cost accounting for a distillation run, regenerated from raw logs.

Input is a JSON manifest that points at the raw files (it holds paths and hardware facts, never
results):
  {
    "model_config": "config.json",                 # Hugging Face config of the student
    "lora_rank": 256,
    "chip": {"name": "TPU v6e", "peak_bf16_tflops": 918, "source": "<where the peak comes from>"},
    "prices_dir": "prices/", "tpu_sku": "C4E3-1652-BF17", "gpu_sku": "23B6-727C-4CFC", "gpu_sku_alt": "...",
    "teacher_gpus": 8,
    "sft":  [{"name", "role": "recipe"|"overhead", "stats": sft_data_stats.json, "done": sft_done.json, "chips": 8}],
    "grpo": [{"name", "role", "stats": grpo_iter_data_stats.json, "done": grpo_iter_done.json, "chips": 8}],
    "generation": [{"name", "role", "device": "tpu"|"teacher", "chips": n, "gen": gen.jsonl, "log": gen.log}],
    "benchmarks": [{"name", "role", "device", "chips", "bench": bench_serving.py output JSON}],
    "teacher_uptime_logs": ["keepwarm.log"],
    "data": [{"name", "manifest": build manifest JSON}]
  }

FLOP model (per token, bf16 matmuls; norms, softmax and element-wise ops ignored):
  forward  = 2 x (decoder matmul params + LM-head params) + attention 4 x L x (heads x head_dim) x s / 2
             (s = mean sequence length of the run; causal attention averages s/2 keys per query),
  adapters = 2 x LoRA params (rank r on q,k,v,o,gate,up,down of every layer).
  LoRA SFT/GRPO update, model FLOPs    = 2 x forward (forward + activation gradients; frozen weights get
                                          no weight gradients) + 3 x adapter forward,
  LoRA SFT/GRPO update, hardware FLOPs = model FLOPs + 1 x forward + 1 x adapter forward (decoder
                                          rematerialisation recomputes the forward pass),
  GRPO log-probability pass            = 1 x forward per pass (old policy; plus the KL reference when it
                                          differs from the policy),
  student generation                   = 1 x forward per prompt and completion token (no prefix-cache credit).
Accelerator time = logged wall time x accelerators in use (compile and data loading included; replica
start-up and idle serving excluded). Teacher endpoint up-time comes from a keep-warm log that pings it every
5 minutes: each run of consecutive successful pings counts as up for its span plus one interval.

Usage: compute_report.py --manifest manifest.json --out compute.json --md COMPUTE.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
from pathlib import Path

KEEPWARM_INTERVAL_S = 300


def load(p):
  return json.loads(Path(p).read_text())


def model_dims(cfg: dict, rank: int) -> dict:
  h, L, ffn, v = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"], cfg["vocab_size"]
  q_out = cfg["num_attention_heads"] * cfg["head_dim"]
  kv_out = cfg["num_key_value_heads"] * cfg["head_dim"]
  shapes = {"q": (h, q_out), "k": (h, kv_out), "v": (h, kv_out), "o": (q_out, h),
            "gate": (h, ffn), "up": (h, ffn), "down": (ffn, h)}
  decoder = L * sum(i * o for i, o in shapes.values())
  lora = L * sum(rank * (i + o) for i, o in shapes.values())
  return {"decoder_matmul_params": decoder, "lm_head_params": h * v, "embedding_tied": bool(cfg.get("tie_word_embeddings")),
          "lora_params": lora, "layers": L, "attn_width": q_out}


def forward_flops(d: dict, seq_len: float) -> float:
  return 2 * (d["decoder_matmul_params"] + d["lm_head_params"]) + 4 * d["layers"] * d["attn_width"] * seq_len / 2


def update_flops(d: dict, seq_len: float) -> tuple[float, float]:
  fwd, ad = forward_flops(d, seq_len), 2 * d["lora_params"]
  model = 2 * fwd + 3 * ad
  return model, model + fwd + ad


def log_elapsed(log: Path) -> float:
  last = None
  for line in open(log):
    m = re.search(r"elapsed=(\d+)s", line)
    if m:
      last = float(m.group(1))
  if last is None:
    raise SystemExit(f"no elapsed= line in {log}")
  return last


def gen_tokens(gen: Path) -> dict:
  n = prompt = completion = 0
  for line in open(gen):
    r = json.loads(line)
    if not r.get("choices"):
      continue
    u = r.get("usage") or {}
    n += len(r["choices"])
    prompt += u.get("prompt_tokens", 0) or 0
    completion += u.get("completion_tokens", 0) or 0
  return {"requests_samples": n, "prompt_tokens": prompt, "completion_tokens": completion}


def uptime_s(logs: list[str]) -> tuple[float, list[list[str]]]:
  """Seconds the endpoint was up: runs of successful pings with no gap over two intervals, each plus one interval."""
  ts = []
  for p in logs:
    for line in open(p):
      parts = line.split()
      if len(parts) >= 2 and parts[1].isdigit():
        ts.append((dt.datetime.fromisoformat(parts[0].replace("Z", "+00:00")), parts[1] == "200"))
  ts.sort()
  windows, cur = [], None
  for t, ok in ts:
    if ok and cur is not None and (t - cur[1]).total_seconds() <= 2 * KEEPWARM_INTERVAL_S:
      cur[1] = t
      continue
    if cur is not None:
      windows.append(cur)
    cur = [t, t] if ok else None
  if cur is not None:
    windows.append(cur)
  total = sum((b - a).total_seconds() + KEEPWARM_INTERVAL_S for a, b in windows)
  return total, [[a.isoformat(), b.isoformat()] for a, b in windows]


def sku_price(prices_dir: Path, sku: str) -> dict:
  for f in sorted(glob.glob(str(prices_dir / "*.json"))):
    for s in json.load(open(f)).get("skus", []):
      if s["skuId"] == sku:
        pi = s["pricingInfo"][0]
        r = pi["pricingExpression"]["tieredRates"][-1]["unitPrice"]
        return {"sku": sku, "description": s["description"], "usd_per_hour": float(r["units"]) + r["nanos"] / 1e9,
                "effective_time": pi.get("effectiveTime")}
  raise SystemExit(f"SKU {sku} not found in {prices_dir}")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--md", type=Path, required=True)
  args = ap.parse_args()
  m = load(args.manifest)
  base = args.manifest.parent
  P = lambda p: (base / p) if not Path(p).is_absolute() else Path(p)  # noqa: E731
  d = model_dims(load(P(m["model_config"])), m["lora_rank"])
  peak = m["chip"]["peak_bf16_tflops"] * 1e12
  tpu_price = sku_price(P(m["prices_dir"]), m["tpu_sku"])
  gpu_price = sku_price(P(m["prices_dir"]), m["gpu_sku"])
  gpu_alt = sku_price(P(m["prices_dir"]), m["gpu_sku_alt"]) if m.get("gpu_sku_alt") else None

  rows = []
  for kind in ("sft", "grpo"):
    for r in m.get(kind, []):
      st, done = load(P(r["stats"])), load(P(r["done"]))
      tokens = st["tokens_real"]
      seqs = st.get("n_kept") or st.get("n_sequences")
      epochs = st.get("epochs", 1) if kind == "sft" else 1
      s = tokens / seqs
      upd_model, upd_hw = update_flops(d, s)
      tok_upd = tokens * epochs
      model_f, hw_f = tok_upd * upd_model, tok_upd * upd_hw
      if kind == "grpo":
        passes = 1 + (0 if st.get("ref_hf_dir") == st.get("policy_hf_dir") else 1)
        lp = passes * tokens * forward_flops(d, s)
        model_f += lp; hw_f += lp
      chip_s = done["wall_s"] * r["chips"]
      rows.append({"name": r["name"], "kind": kind, "role": r["role"], "tokens_processed": tok_upd,
                   "tokens_trained": st.get("tokens_trained", 0) * epochs, "mean_seq_len": round(s, 1),
                   "model_flops": model_f, "hardware_flops": hw_f, "wall_s": done["wall_s"], "chips": r["chips"],
                   "accel_hours": chip_s / 3600, "device": "tpu",
                   "mfu_model": model_f / (chip_s * peak), "hfu": hw_f / (chip_s * peak)})
  for g in m.get("generation", []):
    tok = gen_tokens(P(g["gen"]))
    el = log_elapsed(P(g["log"]))
    row = {"name": g["name"], "kind": "generation", "role": g["role"], "device": g["device"], "wall_s": el,
           "chips": g["chips"], "accel_hours": el * g["chips"] / 3600, **tok}
    if g["device"] == "tpu":
      seq = (tok["prompt_tokens"] + tok["completion_tokens"]) / max(tok["requests_samples"], 1)
      row["model_flops"] = row["hardware_flops"] = (tok["prompt_tokens"] + tok["completion_tokens"]) * forward_flops(d, seq)
    rows.append(row)

  for b in m.get("benchmarks", []):
    bj = load(P(b["bench"]))
    el = sum(lv["wall_s"] for lv in bj["levels"])
    tok = {"requests_samples": sum(lv["ok"] for lv in bj["levels"]),
           "prompt_tokens": sum(sum(r.get("prompt_tokens", 0) for r in lv["records"]) for lv in bj["levels"]),
           "completion_tokens": sum(sum(r.get("completion_tokens", 0) for r in lv["records"]) for lv in bj["levels"])}
    rows.append({"name": b["name"], "kind": "benchmark", "role": b["role"], "device": b["device"], "wall_s": el,
                 "chips": b["chips"], "accel_hours": el * b["chips"] / 3600, **tok})

  up_s, windows = uptime_s([str(P(p)) for p in m.get("teacher_uptime_logs", [])])
  data = []
  for x in m.get("data", []):
    man = load(P(x["manifest"]))
    data.append({"name": x["name"], **{k: v for k, v in man.get("counts", {}).items()},
                 **({"questions_covered": man["questions_covered"]} if "questions_covered" in man else {})})

  def summ(role):
    rs = [r for r in rows if r["role"] == role]
    tpu_h = sum(r["accel_hours"] for r in rs if r["device"] == "tpu")
    gpu_h = sum(r["accel_hours"] for r in rs if r["device"] == "teacher")
    return {"tpu_chip_hours": tpu_h, "teacher_gpu_hours_active": gpu_h,
            "training_model_flops": sum(r.get("model_flops", 0) for r in rs if r["kind"] in ("sft", "grpo")),
            "training_hardware_flops": sum(r.get("hardware_flops", 0) for r in rs if r["kind"] in ("sft", "grpo")),
            "student_generation_flops": sum(r.get("model_flops", 0) for r in rs if r["kind"] == "generation"),
            "cost_usd_tpu": tpu_h * tpu_price["usd_per_hour"],
            "cost_usd_teacher_active": gpu_h * gpu_price["usd_per_hour"],
            "cost_usd_teacher_active_alt": gpu_h * gpu_alt["usd_per_hour"] if gpu_alt else None}
  recipe, overhead = summ("recipe"), summ("overhead")
  teacher_up_h = up_s / 3600 * m["teacher_gpus"]
  out = {"model": d, "chip": m["chip"], "prices": {"tpu": tpu_price, "gpu": gpu_price, "gpu_alt": gpu_alt},
         "runs": rows, "recipe": recipe, "overhead": overhead,
         "teacher_endpoint": {"uptime_hours": up_s / 3600, "gpu_hours": teacher_up_h, "windows": windows,
                              "cost_usd": teacher_up_h * gpu_price["usd_per_hour"],
                              "cost_usd_alt": teacher_up_h * gpu_alt["usd_per_hour"] if gpu_alt else None},
         "data": data}
  args.out.write_text(json.dumps(out, indent=1, default=str))

  f = lambda x: f"{x:.3g}"  # noqa: E731
  md = ["| Run | Role | Tokens processed | Model FLOPs | Wall (s) | Accelerators | Accelerator-hours | MFU |",
        "|---|---|---|---|---|---|---|---|"]
  for r in rows:
    mfu = f"{100 * r['mfu_model']:.1f}%" if "mfu_model" in r else "—"
    md.append(f"| {r['name']} | {r['role']} | {r.get('tokens_processed', r.get('prompt_tokens', 0) + r.get('completion_tokens', 0)):,} | "
              f"{f(r['model_flops']) if 'model_flops' in r else '—'} | {r['wall_s']:.0f} | {r['chips']} {'v6e chip' if r['device'] == 'tpu' else 'H200'} | "
              f"{r['accel_hours']:.2f} | {mfu} |")
  for label, s in (("Recipe", recipe), ("Experiment overhead", overhead)):
    md += ["", f"**{label}:** training {f(s['training_model_flops'])} model FLOPs ({f(s['training_hardware_flops'])} with "
           f"rematerialisation); student generation {f(s['student_generation_flops'])} FLOPs; {s['tpu_chip_hours']:.2f} TPU v6e "
           f"chip-hours (${s['cost_usd_tpu']:.2f}); {s['teacher_gpu_hours_active']:.2f} active teacher H200-hours "
           f"(${s['cost_usd_teacher_active']:.2f}"
           + (f"; ${s['cost_usd_teacher_active_alt']:.2f} at the alternative rate" if s['cost_usd_teacher_active_alt'] else "") + ")."]
  te = out["teacher_endpoint"]
  md += ["", f"**Teacher endpoint up-time (billed whether busy or idle):** {te['uptime_hours']:.2f} h × {m['teacher_gpus']} GPUs = "
         f"{te['gpu_hours']:.1f} GPU-hours (${te['cost_usd']:.2f}" + (f"; ${te['cost_usd_alt']:.2f} alt" if te['cost_usd_alt'] else "") + ")."]
  args.md.write_text("\n".join(md) + "\n")
  print("\n".join(md))


if __name__ == "__main__":
  main()
