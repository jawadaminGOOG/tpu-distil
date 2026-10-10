#!/usr/bin/env python3
"""pass@1, pass@k and the share of mixed-outcome questions from a graded file with k samples per question.

pass@1 is the mean per-sample accuracy; pass@k is the share of questions with at least one correct
sample; mixed is the share of questions with some but not all samples correct (the questions that
give a group-relative policy gradient a non-zero signal).

Usage: passk.py graded.jsonl [graded2.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def passk(path: str) -> dict:
  by: dict[str, list[bool]] = defaultdict(list)
  with open(path) as f:
    for line in f:
      if line.strip():
        r = json.loads(line)
        by[r["id"]].append(bool(r["correct"]))
  ks = {len(v) for v in by.values()}
  n = len(by)
  return {
      "file": path, "questions": n, "samples_per_question": sorted(ks),
      "pass_at_1": round(100 * sum(sum(v) / len(v) for v in by.values()) / n, 2),
      "pass_at_k": round(100 * sum(any(v) for v in by.values()) / n, 2),
      "mixed_pct": round(100 * sum(0 < sum(v) < len(v) for v in by.values()) / n, 2),
      "all_correct_pct": round(100 * sum(all(v) for v in by.values()) / n, 2),
      "all_wrong_pct": round(100 * sum(not any(v) for v in by.values()) / n, 2),
  }


if __name__ == "__main__":
  for p in sys.argv[1:]:
    print(json.dumps(passk(p)))
