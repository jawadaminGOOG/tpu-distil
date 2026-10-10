#!/usr/bin/env python3
"""Audit SFT files against the BIRD evaluation split and re-verify every kept SQL by execution.

For each SFT JSONL (trainer records written by build_sft.py) it reports:
  - rows whose database is an evaluation database;
  - rows whose question text equals an evaluation question (case/space-normalised);
  - rows sharing any word 13-gram with an evaluation question or evidence text;
  - rows whose SQL, re-extracted from the trained response and re-executed on the training
    database, does not return the gold rows (independent of grade.py's verdict);
  - rows whose training id is not a BIRD train question.
Exit status 1 if any count is non-zero. Writes a JSON report.

Usage: leak_audit.py --bird-root R --sft a.jsonl [--sft b.jsonl ...] --out audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bird

N = 13
_ROOT: Path


def _norm(text: str) -> str:
  return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
  w = _norm(text).split()
  return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _question_of(prompt: str) -> str:
  m = re.search(r"\nQuestion: (.*?)\n\n", prompt, re.DOTALL)
  return m.group(1).strip() if m else ""


def _evidence_of(prompt: str) -> str:
  m = re.search(r"\nExternal knowledge: (.*?)\n\nQuestion: ", prompt, re.DOTALL)
  return m.group(1).strip() if m else ""


def _init(root: Path) -> None:
  global _ROOT  # pylint: disable=global-statement
  _ROOT = root


def _reverify(task: tuple[str, str, str, str]) -> tuple[str, bool, str]:
  rid, db_id, gold_sql, response = task
  path = bird.db_path(_ROOT, "train", db_id)
  if not path.exists():
    return rid, False, f"no training database {db_id}"
  try:
    gold = bird.run_query(path, gold_sql)
  except bird.QueryError as e:
    return rid, False, f"gold_failed: {e}"
  ok, err = bird.execution_match(path, bird.extract_sql(response), gold)
  return rid, ok, err


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--bird-root", type=Path, required=True)
  ap.add_argument("--sft", type=Path, action="append", required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--workers", type=int, default=48)
  args = ap.parse_args()

  dev = bird.load_items(args.bird_root, "dev")
  dev_dbs = {it.db_id for it in dev}
  dev_q = {_norm(it.question) for it in dev}
  dev_grams: set[tuple[str, ...]] = set()
  for it in dev:
    dev_grams |= ngrams(it.question) | ngrams(it.evidence)
  train = {it.id: it for it in bird.load_items(args.bird_root, "train")}

  report = {"ngram_n": N, "dev_questions": len(dev), "dev_dbs": sorted(dev_dbs), "files": {}}
  failed = False
  for path in args.sft:
    rows = list(bird.iter_jsonl(path))
    bad_id, bad_db, exact, gram, tasks = [], [], [], [], []
    for r in rows:
      rid = r["id"]
      qid = rid.rsplit("#s", 1)[0]
      db = r.get("db_id", "")
      prompt = r["messages"][-1]["content"]
      q = _question_of(prompt)
      if qid not in train or train[qid].db_id != db or _norm(train[qid].question) != _norm(q):
        bad_id.append(rid)
      if db in dev_dbs:
        bad_db.append(rid)
      if _norm(q) in dev_q:
        exact.append(rid)
      if (ngrams(q) | ngrams(_evidence_of(prompt))) & dev_grams:
        gram.append(rid)
      if qid in train:
        tasks.append((rid, db, train[qid].gold_sql, r["response"]))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.bird_root,)) as ex:
      results = list(ex.map(_reverify, tasks, chunksize=16))
    wrong = [(t, e) for t, ok, e in results if not ok]
    rep = {"rows": len(rows), "not_a_train_question": len(bad_id), "eval_db": len(bad_db),
           "exact_question_match": len(exact), "ngram13_overlap": len(gram),
           "reverify_failed": len(wrong), "examples": {"not_a_train_question": bad_id[:5], "eval_db": bad_db[:5],
                                                       "exact": exact[:5], "ngram13": gram[:5],
                                                       "reverify_failed": wrong[:5]}}
    report["files"][str(path)] = rep
    failed |= any(rep[k] for k in ("not_a_train_question", "eval_db", "exact_question_match", "ngram13_overlap",
                                   "reverify_failed"))
    print(json.dumps({"file": str(path), **{k: v for k, v in rep.items() if k != "examples"}}))
  report["passed"] = not failed
  args.out.write_text(json.dumps(report, indent=1))
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
