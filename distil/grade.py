#!/usr/bin/env python3
"""Grade BIRD generations by execution accuracy, or run the evaluator's own controls.

  grade.py --bird-root R --split dev --gen gen.jsonl --out graded.jsonl
      One row per (id, sample): extracted SQL, correct, error. A summary JSON is written to
      <out>.summary.json (EX over sample 0, mean EX over all samples, by difficulty).
  grade.py --bird-root R --split dev --control gold|shuffled --out control.jsonl
      gold: each gold query graded against itself (must be 100% on executable gold).
      shuffled: each question graded with another question's gold query on the same database
      (a negative control; must be near 0%).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bird

_ROOT: Path
_SPLIT: str
_TIMEOUT: float


def _init(root: Path, split: str, timeout: float) -> None:
  global _ROOT, _SPLIT, _TIMEOUT  # pylint: disable=global-statement
  _ROOT, _SPLIT, _TIMEOUT = root, split, timeout


def _grade_group(task: tuple[str, str, str, list[tuple[int, str | None]]]) -> list[dict]:
  """Grade every sample of one question; the gold rows are computed once."""
  qid, db_id, gold_sql, samples = task
  path = bird.db_path(_ROOT, _SPLIT, db_id)
  try:
    gold = bird.run_query(path, gold_sql, _TIMEOUT)
    gold_err = ""
  except bird.QueryError as e:
    gold, gold_err = None, str(e)[:300]
  out = []
  for idx, sql in samples:
    if gold is None:
      out.append({"id": qid, "db_id": db_id, "sample": idx, "pred_sql": sql, "correct": False,
                  "error": "gold_failed", "gold_error": gold_err})
      continue
    ok, err = bird.execution_match(path, sql, gold, _TIMEOUT)
    out.append({"id": qid, "db_id": db_id, "sample": idx, "pred_sql": sql, "correct": ok,
                "error": err})
  return out


def summarize(rows: list[dict], items: dict[str, bird.Item]) -> dict:
  by_id: dict[str, list[dict]] = defaultdict(list)
  for r in rows:
    by_id[r["id"]].append(r)
  s0 = [next(x for x in v if x["sample"] == min(y["sample"] for y in v)) for v in by_id.values()]
  mean_all = sum(sum(x["correct"] for x in v) / len(v) for v in by_id.values()) / max(len(by_id), 1)
  diff: dict[str, list[bool]] = defaultdict(list)
  for r in s0:
    diff[items[r["id"]].difficulty or "all"].append(r["correct"])
  errs: dict[str, int] = defaultdict(int)
  for r in s0:
    if not r["correct"]:
      key = r["error"].split(":")[0] if r["error"] else "wrong_result"
      errs[key] += 1
  return {
      "questions": len(by_id), "samples": len(rows),
      "ex_sample0": round(100 * sum(r["correct"] for r in s0) / max(len(s0), 1), 2),
      "ex_mean_all_samples": round(100 * mean_all, 2),
      "by_difficulty": {k: {"n": len(v), "ex": round(100 * sum(v) / len(v), 2)}
                        for k, v in sorted(diff.items())},
      "gold_failed": sorted(r["id"] for r in s0 if r["error"] == "gold_failed"),
      "failure_kinds_sample0": dict(sorted(errs.items(), key=lambda kv: -kv[1])),
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--bird-root", type=Path, required=True)
  ap.add_argument("--split", choices=["dev", "train"], required=True)
  ap.add_argument("--gen", type=Path)
  ap.add_argument("--control", choices=["gold", "shuffled"])
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--timeout", type=float, default=30.0)
  ap.add_argument("--workers", type=int, default=48)
  args = ap.parse_args()
  if bool(args.gen) == bool(args.control):
    ap.error("give exactly one of --gen or --control")

  items = {it.id: it for it in bird.load_items(args.bird_root, args.split)}
  tasks = []
  if args.control == "gold":
    tasks = [(i, it.db_id, it.gold_sql, [(0, it.gold_sql)]) for i, it in items.items()]
  elif args.control == "shuffled":
    rng = random.Random(0)
    per_db: dict[str, list[bird.Item]] = defaultdict(list)
    for it in items.values():
      per_db[it.db_id].append(it)
    for its in per_db.values():
      for it in its:
        others = [o for o in its if o.gold_sql.strip() != it.gold_sql.strip()]
        if others:
          tasks.append((it.id, it.db_id, it.gold_sql, [(0, rng.choice(others).gold_sql)]))
  else:
    samples: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    for r in bird.iter_jsonl(args.gen):
      if r["id"] not in items or not r.get("choices"):
        continue
      for k, c in enumerate(r["choices"]):
        samples[r["id"]].append((k, bird.extract_sql(c.get("content"))))
    missing = sorted(set(items) - set(samples)) if args.split == "dev" else []
    if missing:
      print(f"WARNING: {len(missing)} dev questions have no generation (graded as incorrect)")
      for m in missing:
        samples[m].append((0, None))
    tasks = [(i, items[i].db_id, items[i].gold_sql, s) for i, s in samples.items()]

  with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                           initargs=(args.bird_root, args.split, args.timeout)) as ex:
    rows = [r for group in ex.map(_grade_group, tasks, chunksize=4) for r in group]
  rows.sort(key=lambda r: (r["id"], r["sample"]))
  with open(args.out, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  summary = summarize(rows, items)
  summary.update({"source": str(args.gen or f"control:{args.control}"), "split": args.split,
                  "timeout_s": args.timeout})
  Path(str(args.out) + ".summary.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps({k: summary[k] for k in ("questions", "samples", "ex_sample0",
                                            "ex_mean_all_samples")}))


if __name__ == "__main__":
  main()
