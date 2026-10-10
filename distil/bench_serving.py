#!/usr/bin/env python3
"""Serving throughput and latency of one OpenAI-compatible endpoint over a concurrency ladder.

The workload is a fixed, seeded sample of prompts from a prompts JSONL (the real task prompts, so
prompt and output lengths are the task's own). For each concurrency level every request in the
sample is sent with that many in flight; the level's wall time, request rate, output-token rate and
latency percentiles are recorded, along with each request's token counts. A short warm-up runs
first and is not measured.

Usage:
  bench_serving.py --prompts dev_prompts.jsonl --endpoint http://host:8104/v1 --model kd \
    --concurrency 1,8,32,64,128 --n 512 --max-tokens 1024 --out bench_kd.json \
    [--chat-template-kwargs '{"enable_thinking": false}'] [--n-low 64 --low-below 8]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher_client  # noqa: E402  pylint: disable=wrong-import-position


def percentile(xs: list[float], q: float) -> float:
  s = sorted(xs)
  return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def run_level(rows: list[dict], params: dict, endpoint: str, concurrency: int, timeout: float) -> dict:
  def one(row: dict) -> dict:
    t = time.time()
    try:
      resp = teacher_client.call(endpoint, dict(params, messages=row["messages"]), timeout=timeout,
                                 max_attempts=3)
      u = resp.get("usage") or {}
      return {"id": row["id"], "latency_s": time.time() - t, "prompt_tokens": u.get("prompt_tokens", 0),
              "completion_tokens": u.get("completion_tokens", 0),
              "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason")}
    except Exception as e:  # pylint: disable=broad-except
      return {"id": row["id"], "latency_s": time.time() - t, "error": f"{type(e).__name__}: {e}"[:300]}

  t0 = time.time()
  with ThreadPoolExecutor(max_workers=concurrency) as ex:
    recs = list(ex.map(one, rows))
  wall = time.time() - t0
  ok = [r for r in recs if "error" not in r]
  lat = [r["latency_s"] for r in ok] or [float("nan")]
  out_tok = sum(r["completion_tokens"] for r in ok)
  return {
      "concurrency": concurrency, "requests": len(recs), "ok": len(ok), "errors": len(recs) - len(ok),
      "wall_s": round(wall, 3), "req_per_s": round(len(ok) / wall, 4),
      "output_tok_per_s": round(out_tok / wall, 1),
      "prompt_tok_per_s": round(sum(r["prompt_tokens"] for r in ok) / wall, 1),
      "mean_completion_tokens": round(out_tok / max(len(ok), 1), 1),
      "latency_p50_s": round(percentile(lat, 0.5), 3), "latency_p90_s": round(percentile(lat, 0.9), 3),
      "latency_p99_s": round(percentile(lat, 0.99), 3),
      "finish_length": sum(1 for r in ok if r.get("finish_reason") == "length"),
      "records": recs,
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--prompts", type=Path, required=True)
  ap.add_argument("--endpoint", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--concurrency", default="1,8,32,64,128")
  ap.add_argument("--n", type=int, default=512, help="requests per level")
  ap.add_argument("--n-low", type=int, default=64, help="requests per level below --low-below")
  ap.add_argument("--low-below", type=int, default=8)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--max-tokens", type=int, default=1024)
  ap.add_argument("--temperature", type=float, default=0.0)
  ap.add_argument("--chat-template-kwargs", default=None)
  ap.add_argument("--warmup", type=int, default=8)
  ap.add_argument("--timeout", type=float, default=1800)
  ap.add_argument("--hardware", default="", help="free-text description recorded in the output")
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()

  rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
  sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
  params = {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens}
  if args.chat_template_kwargs:
    params["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)

  started = time.time()
  run_level(sample[:args.warmup], params, args.endpoint, args.warmup, args.timeout)
  levels = []
  for c in [int(x) for x in args.concurrency.split(",")]:
    use = sample[:args.n_low] if c < args.low_below else sample
    lv = run_level(use, params, args.endpoint, c, args.timeout)
    print(json.dumps({k: v for k, v in lv.items() if k != "records"}), flush=True)
    levels.append(lv)
  out = {"endpoint": args.endpoint, "model": args.model, "hardware": args.hardware, "params": params,
         "prompts": str(args.prompts), "seed": args.seed, "n": len(sample), "started": started,
         "finished": time.time(),
         "levels": levels}
  args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
