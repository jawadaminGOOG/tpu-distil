#!/usr/bin/env python3
"""Grade generations for a task, or run the evaluator's own controls.

  grade.py [--task bird] --root R --split dev --gen gen.jsonl --out graded.jsonl
      One row per (id, sample): the extracted answer (BIRD: pred_sql; FinQA: pred_answer), correct,
      error. A summary JSON is written to <out>.summary.json (accuracy over sample 0, mean accuracy
      over all samples, by difficulty when the task has one). On the task's full-coverage splits
      (BIRD: dev; FinQA: dev and test) a question without a generation is graded incorrect.
  grade.py [--task bird] --root R --split dev --control gold|shuffled --out control.jsonl
      gold: each gold answer graded against itself (must be 100% on gradable gold).
      shuffled: each question graded with another question's gold answer (a negative control; must
      be low). BIRD draws the other query from the same database; FinQA from the whole split.

BIRD grades by execution accuracy in a process pool (--workers, --timeout); FinQA by numeric
tolerance (see tasks/finqa.py). `--root` is also accepted as `--bird-root`. The summary keeps the
key names ex_sample0 / ex_mean_all_samples for every task: the share of correct answers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bird  # noqa: E402  pylint: disable=wrong-import-position
from tasks import NAMES, get_task  # noqa: E402  pylint: disable=wrong-import-position


def summarize(rows: list[dict], items: dict) -> dict:
  by_id: dict[str, list[dict]] = defaultdict(list)
  for r in rows:
    by_id[r["id"]].append(r)
  s0 = [next(x for x in v if x["sample"] == min(y["sample"] for y in v)) for v in by_id.values()]
  mean_all = sum(sum(x["correct"] for x in v) / len(v) for v in by_id.values()) / max(len(by_id), 1)
  diff: dict[str, list[bool]] = defaultdict(list)
  for r in s0:
    diff[getattr(items[r["id"]], "difficulty", "") or "all"].append(r["correct"])
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


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", choices=NAMES, default="bird")
  ap.add_argument("--root", "--bird-root", dest="root", type=Path, required=True, help="the task's data directory")
  ap.add_argument("--split", required=True)
  ap.add_argument("--gen", type=Path)
  ap.add_argument("--control", choices=["gold", "shuffled"])
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--timeout", type=float, default=30.0)
  ap.add_argument("--workers", type=int, default=48)
  args = ap.parse_args(argv)
  if bool(args.gen) == bool(args.control):
    ap.error("give exactly one of --gen or --control")
  task = get_task(args.task)
  if args.split not in task.SPLITS:
    ap.error(f"--split must be one of {', '.join(task.SPLITS)} for task {task.NAME}")

  items = {it.id: it for it in task.load_items(args.root, args.split)}
  if args.control:
    samples = task.control_samples(items, args.control)
  else:
    samples: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    for r in bird.iter_jsonl(args.gen):
      if r["id"] not in items or not r.get("choices"):
        continue
      for k, c in enumerate(r["choices"]):
        samples[r["id"]].append((k, task.extract_answer(c.get("content"))))
    missing = sorted(set(items) - set(samples)) if args.split in task.FULL_COVERAGE_SPLITS else []
    if missing:
      print(f"WARNING: {len(missing)} {args.split} questions have no generation (graded as incorrect)")
      for m in missing:
        samples[m].append((0, None))

  rows = task.grade(args.root, args.split, items, samples, workers=args.workers, timeout=args.timeout)
  rows.sort(key=lambda r: (r["id"], r["sample"]))
  with open(args.out, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  summary = summarize(rows, items)
  summary.update({"task": task.NAME, "source": str(args.gen or f"control:{args.control}"), "split": args.split,
                  "timeout_s": args.timeout})
  Path(str(args.out) + ".summary.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps({k: summary[k] for k in ("questions", "samples", "ex_sample0",
                                            "ex_mean_all_samples")}))


if __name__ == "__main__":
  main()
