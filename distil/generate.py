#!/usr/bin/env python3
"""Run chat completions for a prompts JSONL against one or more OpenAI-compatible endpoints.

Each output row holds the prompt id, the endpoint, the exact request parameters, every returned
choice (content, reasoning_content, finish_reason), token usage and wall-clock latency. Output is
appended; ids already present with a response are skipped, so an interrupted run resumes.

Usage:
  generate.py --prompts dev_prompts.jsonl --out gen.jsonl --model qwen3-1.7b \
    --endpoint http://10.0.0.1:8100/v1 --endpoint http://10.0.0.1:8101/v1 \
    --temperature 0 --max-tokens 2048 --n 1 --concurrency 64 \
    --chat-template-kwargs '{"enable_thinking": false}'
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher_client  # noqa: E402  pylint: disable=wrong-import-position


def done_ids(path: Path) -> set[str]:
  if not path.exists():
    return set()
  ids = set()
  with open(path) as f:
    for line in f:
      try:
        r = json.loads(line)
      except json.JSONDecodeError:
        continue
      if r.get("choices"):
        ids.add(r["id"])
  return ids


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--prompts", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--endpoint", action="append", required=True)
  ap.add_argument("--temperature", type=float, default=0.0)
  ap.add_argument("--top-p", type=float, default=None)
  ap.add_argument("--max-tokens", type=int, default=2048)
  ap.add_argument("--n", type=int, default=1)
  ap.add_argument("--seed", type=int, default=None)
  ap.add_argument("--concurrency", type=int, default=32)
  ap.add_argument("--chat-template-kwargs", default=None, help="JSON object passed through")
  ap.add_argument("--extra-body", default=None, help="JSON object merged into the request body")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--ids", type=Path, default=None, help="file with one prompt id per line to run")
  ap.add_argument("--timeout", type=float, default=1800)
  args = ap.parse_args()

  rows = list(read_jsonl(args.prompts))
  if args.ids:
    keep = {l.strip() for l in open(args.ids) if l.strip()}
    rows = [r for r in rows if r["id"] in keep]
  if args.limit:
    rows = rows[:args.limit]
  skip = done_ids(args.out)
  todo = [r for r in rows if r["id"] not in skip]
  print(f"prompts={len(rows)} done={len(rows) - len(todo)} todo={len(todo)}", flush=True)

  params = {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens,
            "n": args.n}
  if args.top_p is not None:
    params["top_p"] = args.top_p
  if args.seed is not None:
    params["seed"] = args.seed
  if args.chat_template_kwargs:
    params["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
  if args.extra_body:
    params.update(json.loads(args.extra_body))

  endpoints = itertools.cycle(args.endpoint)
  lock = threading.Lock()
  out = open(args.out, "a")
  stats = {"ok": 0, "err": 0, "t0": time.time(), "completion_tokens": 0}

  def one(row: dict, endpoint: str) -> dict:
    body = dict(params, messages=row["messages"])
    t = time.time()
    rec = {"id": row["id"], "endpoint": endpoint, "params": params, "started_at": t}
    try:
      resp = teacher_client.call(endpoint, body, timeout=args.timeout)
      rec["choices"] = [{"content": c.get("message", {}).get("content"),
                         "reasoning_content": c.get("message", {}).get("reasoning_content"),
                         "finish_reason": c.get("finish_reason")} for c in resp.get("choices", [])]
      rec["usage"] = resp.get("usage")
      rec["served_model"] = resp.get("model")
    except Exception as e:  # pylint: disable=broad-except
      rec["error"] = f"{type(e).__name__}: {e}"[:1000]
    rec["latency_s"] = round(time.time() - t, 3)
    return rec

  with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
    futs = [ex.submit(one, r, next(endpoints)) for r in todo]
    for i, f in enumerate(as_completed(futs), 1):
      rec = f.result()
      with lock:
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if "error" in rec:
          stats["err"] += 1
        else:
          stats["ok"] += 1
          stats["completion_tokens"] += (rec.get("usage") or {}).get("completion_tokens", 0) or 0
      if i % 100 == 0 or i == len(todo):
        el = time.time() - stats["t0"]
        print(f"{i}/{len(todo)} ok={stats['ok']} err={stats['err']} "
              f"elapsed={el:.0f}s out_tok/s={stats['completion_tokens'] / max(el, 1e-9):.0f}",
              flush=True)
  out.close()


def read_jsonl(path: Path):
  with open(path) as f:
    for line in f:
      if line.strip():
        yield json.loads(line)


if __name__ == "__main__":
  main()
