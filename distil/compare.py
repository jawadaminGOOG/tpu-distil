#!/usr/bin/env python3
"""Paired comparison of two graded runs on the same questions.

Per question, each arm's score is its sample-0 correctness (or, with --mean-samples, its mean
correctness over all samples). The difference B − A is reported in EX points with a paired
bootstrap 95% CI over questions (10,000 resamples, fixed seed) and the win/loss counts.

Usage: compare.py --a base.graded.jsonl --b kd.graded.jsonl [--label-a base --label-b kd]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def per_question(path: Path, mean_samples: bool) -> dict[str, float]:
  by: dict[str, list[tuple[int, bool]]] = defaultdict(list)
  with open(path) as f:
    for line in f:
      if line.strip():
        r = json.loads(line)
        by[r["id"]].append((r["sample"], bool(r["correct"])))
  if mean_samples:
    return {k: sum(c for _, c in v) / len(v) for k, v in by.items()}
  return {k: float(min(v)[1]) for k, v in by.items()}


def paired_bootstrap(diffs: list[float], resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
  rng = random.Random(seed)
  n = len(diffs)
  means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
  return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def compare(a: dict[str, float], b: dict[str, float], resamples: int = 10_000) -> dict:
  ids = sorted(set(a) & set(b))
  if set(a) != set(b):
    raise SystemExit(f"question sets differ: {len(set(a) - set(b))} only in A, "
                     f"{len(set(b) - set(a))} only in B")
  diffs = [b[i] - a[i] for i in ids]
  lo, hi = paired_bootstrap(diffs, resamples)
  return {
      "n": len(ids),
      "a": round(100 * sum(a[i] for i in ids) / len(ids), 2),
      "b": round(100 * sum(b[i] for i in ids) / len(ids), 2),
      "delta": round(100 * sum(diffs) / len(ids), 2),
      "ci95": [round(100 * lo, 2), round(100 * hi, 2)],
      "b_better": sum(d > 0 for d in diffs), "a_better": sum(d < 0 for d in diffs),
      "resamples": resamples, "seed": 0,
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--a", type=Path, required=True)
  ap.add_argument("--b", type=Path, required=True)
  ap.add_argument("--label-a", default="A")
  ap.add_argument("--label-b", default="B")
  ap.add_argument("--mean-samples", action="store_true")
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()
  res = compare(per_question(args.a, args.mean_samples), per_question(args.b, args.mean_samples))
  res.update({"label_a": args.label_a, "label_b": args.label_b, "file_a": str(args.a),
              "file_b": str(args.b), "mean_samples": args.mean_samples})
  if args.out:
    args.out.write_text(json.dumps(res, indent=1))
  print(json.dumps(res))


if __name__ == "__main__":
  main()
