#!/usr/bin/env python3
"""Write chat prompts for one split of a task as JSONL: one row per question with id, group and messages.

Usage:
  prompts.py [--task bird] --root ~/kd/bird --split dev --out dev_prompts.jsonl [--exclude-groups a,b]
  prompts.py --task finqa --root ~/kd/finqa --split train --out train_prompts.jsonl --exclude-groups-from test,dev \
      --exclude-questions-from test,dev

`--root` is also accepted as `--bird-root`; `--exclude-groups` / `--only-groups` as `--exclude-dbs` /
`--only-dbs`. The group is the task's question group (BIRD: the database, row key "db_id"; FinQA: the
report page, row key "report"). `--exclude-groups-from` leaves out every question whose group also
appears in the named splits (for example training questions on an evaluation report page).
`--exclude-questions-from` leaves out every question that equals a question of the named splits after
case/space normalisation, or shares a word 13-gram with one of their questions (or, for BIRD, with
their evidence or its own evidence); it uses the same code as leak_audit.py (overlap.py), so the
audit finds no such overlap in the written split.
A manifest beside the output records the task, the prompt-template SHA-256, the row count, the groups
and what was excluded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlap import N, OverlapIndex  # noqa: E402  pylint: disable=wrong-import-position
from tasks import NAMES, get_task  # noqa: E402  pylint: disable=wrong-import-position


def _csv(s: str) -> list[str]:
  return [x for x in s.split(",") if x]


def _extra_texts(item) -> list[str]:
  """Texts besides the question that leak_audit.py checks for n-gram overlap (BIRD evidence)."""
  evidence = getattr(item, "evidence", "")
  return [evidence] if evidence else []


def _check_splits(ap, task, flag: str, splits: list[str], own: str) -> None:
  for s in splits:
    if s not in task.SPLITS:
      ap.error(f"{flag}: unknown split {s!r} for task {task.NAME}")
    if s == own:
      ap.error(f"{flag} must not name the split being written")


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  ap.add_argument("--task", choices=NAMES, default="bird")
  ap.add_argument("--root", "--bird-root", dest="root", type=Path, required=True, help="the task's data directory")
  ap.add_argument("--split", required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--exclude-groups", "--exclude-dbs", dest="exclude_groups", default="",
                  help="comma-separated groups to leave out")
  ap.add_argument("--only-groups", "--only-dbs", dest="only_groups", default="",
                  help="comma-separated groups to keep (others dropped)")
  ap.add_argument("--exclude-groups-from", default="",
                  help="comma-separated splits; questions whose group appears in any of them are left out")
  ap.add_argument("--exclude-questions-from", default="",
                  help=f"comma-separated splits; questions that equal one of theirs or share a word {N}-gram "
                       "with it are left out")
  args = ap.parse_args(argv)
  task = get_task(args.task)
  if args.split not in task.SPLITS:
    ap.error(f"--split must be one of {', '.join(task.SPLITS)} for task {task.NAME}")
  from_splits = _csv(args.exclude_groups_from)
  _check_splits(ap, task, "--exclude-groups-from", from_splits, args.split)
  q_splits = _csv(args.exclude_questions_from)
  _check_splits(ap, task, "--exclude-questions-from", q_splits, args.split)

  exclude = set(_csv(args.exclude_groups))
  only = set(_csv(args.only_groups))
  foreign: set[str] = set()
  for s in from_splits:
    foreign |= {it.group for it in task.load_items(args.root, s)}
  all_items = task.load_items(args.root, args.split)
  items = [it for it in all_items
           if it.group not in exclude and it.group not in foreign and (not only or it.group in only)]
  dropped_exact: list[str] = []
  dropped_ngram: list[str] = []
  if q_splits:
    refs = [it for s in q_splits for it in task.load_items(args.root, s)]
    index = OverlapIndex([it.question for it in refs], [t for it in refs for t in _extra_texts(it)])
    kept = []
    for it in items:
      if index.matches_exactly(it.question):
        dropped_exact.append(it.id)
      elif index.shares_ngram(it.question, *_extra_texts(it)):
        dropped_ngram.append(it.id)
      else:
        kept.append(it)
    items = kept
  rows, extra = task.prompt_rows(args.root, args.split, items)
  with open(args.out, "w") as f:
    for row in rows:
      f.write(json.dumps(row) + "\n")
  manifest = {"task": task.NAME, "split": args.split, "rows": len(rows), **extra,
              "prompt_template_sha256": task.prompt_template_sha256(),
              "excluded": {"groups": sorted(exclude), "only_groups": sorted(only), "groups_from_splits": from_splits,
                           "groups_shared_with_those_splits": len(foreign & {it.group for it in all_items}),
                           "rows_left_out": len(all_items) - len(items),
                           "questions_from_splits": q_splits, "ngram_n": N,
                           "question_overlap_dropped": len(dropped_exact) + len(dropped_ngram),
                           "question_overlap_exact_ids": dropped_exact,
                           "question_overlap_ngram_only_ids": dropped_ngram}}
  Path(str(args.out) + ".manifest.json").write_text(json.dumps(manifest, indent=1))
  print(json.dumps({k: manifest[k] for k in ("task", "split", "rows", "prompt_template_sha256")}))


if __name__ == "__main__":
  main()
