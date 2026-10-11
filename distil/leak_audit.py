#!/usr/bin/env python3
"""Audit SFT files against a task's evaluation split and re-verify every kept answer.

BIRD (default, `--task bird`; evaluation split dev). For each SFT JSONL (trainer records written by
build_sft.py) it reports:
  - rows whose database is an evaluation database;
  - rows whose question text equals an evaluation question (case/space-normalised);
  - rows sharing any word 13-gram with an evaluation question or evidence text;
  - rows whose SQL, re-extracted from the trained response and re-executed on the training
    database, does not return the gold rows (independent of grade.py's verdict);
  - rows whose training id is not a BIRD train question.

FinQA (`--task finqa`; evaluation split test, more with --eval-splits test,dev). It reports:
  - rows whose training id is not a FinQA train question with the same report page and question;
  - rows whose report page (the row's own, or its train question's) appears in an evaluation split;
  - rows whose question text equals an evaluation question (case/space-normalised);
  - rows whose question shares any word 13-gram with an evaluation question;
  - rows whose answer, re-extracted from the trained response and re-scored against the train
    question's gold answer, is not correct (independent of grade.py's verdict).

Exit status 1 if any count is non-zero. Writes a JSON report.

Usage: leak_audit.py [--task bird|finqa] --root R --sft a.jsonl [--sft b.jsonl ...] --out audit.json
       (`--root` is also accepted as `--bird-root`)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bird  # noqa: E402  pylint: disable=wrong-import-position
from tasks import NAMES, finqa  # noqa: E402  pylint: disable=wrong-import-position
from overlap import N, ngrams  # noqa: E402  pylint: disable=wrong-import-position
from overlap import normalize_text as _norm  # noqa: E402  pylint: disable=wrong-import-position

AUDIT_KEYS = {
    "bird": ("not_a_train_question", "eval_db", "exact_question_match", "ngram13_overlap", "reverify_failed"),
    "finqa": ("not_a_train_question", "eval_report", "exact_question_match", "ngram13_overlap", "rescore_failed"),
}
_ROOT: Path


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


def audit_finqa(root: Path, sft_paths: list[Path], eval_splits: list[str]) -> tuple[dict, bool]:
  """The FinQA audit (see the module docstring): the report and whether every file is clean."""
  evals = [it for s in eval_splits for it in finqa.load_items(root, s)]
  eval_reports = {it.report for it in evals}
  eval_q = {_norm(it.question) for it in evals}
  eval_grams: set[tuple[str, ...]] = set()
  for it in evals:
    eval_grams |= ngrams(it.question)
  train = {it.id: it for it in finqa.load_items(root, "train")}
  report = {"task": "finqa", "ngram_n": N, "eval_splits": eval_splits, "eval_questions": len(evals),
            "eval_reports": len(eval_reports), "files": {}}
  failed = False
  for path in sft_paths:
    rows = list(bird.iter_jsonl(path))
    bad_id, bad_report, exact, gram, wrong = [], [], [], [], []
    for r in rows:
      rid = r["id"]
      qid = rid.rsplit("#s", 1)[0]
      rep_name = r.get("report", "")
      q = finqa.question_of(r["messages"][-1]["content"])
      it = train.get(qid)
      if it is None or it.report != rep_name or _norm(it.question) != _norm(q):
        bad_id.append(rid)
      if rep_name in eval_reports or (it is not None and it.report in eval_reports):
        bad_report.append(rid)
      if _norm(q) in eval_q:
        exact.append(rid)
      if ngrams(q) & eval_grams:
        gram.append(rid)
      if it is None:
        wrong.append((rid, "not a train question"))
      else:
        pred = finqa.extract_answer(r["response"])
        ok, err = finqa.score(pred, it.gold)
        if not ok:
          wrong.append((rid, err or f"wrong_answer: {pred} vs gold {it.gold}"))
    rep = {"rows": len(rows), "not_a_train_question": len(bad_id), "eval_report": len(bad_report),
           "exact_question_match": len(exact), "ngram13_overlap": len(gram), "rescore_failed": len(wrong),
           "examples": {"not_a_train_question": bad_id[:5], "eval_report": bad_report[:5], "exact": exact[:5],
                        "ngram13": gram[:5], "rescore_failed": wrong[:5]}}
    report["files"][str(path)] = rep
    failed |= any(rep[k] for k in AUDIT_KEYS["finqa"])
    print(json.dumps({"file": str(path), **{k: v for k, v in rep.items() if k != "examples"}}))
  return report, not failed


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", choices=NAMES, default="bird")
  ap.add_argument("--root", "--bird-root", dest="root", type=Path, required=True, help="the task's data directory")
  ap.add_argument("--sft", type=Path, action="append", required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--workers", type=int, default=48)
  ap.add_argument("--eval-splits", default="", help="FinQA: comma-separated evaluation splits (default: test)")
  args = ap.parse_args()

  if args.task == "finqa":
    splits = [s for s in args.eval_splits.split(",") if s] or [finqa.EVAL_SPLIT]
    bad = [s for s in splits if s not in finqa.SPLITS or s == "train"]
    if bad:
      ap.error(f"--eval-splits: {bad} are not FinQA evaluation splits")
    report, passed = audit_finqa(args.root, args.sft, splits)
    report["passed"] = passed
    args.out.write_text(json.dumps(report, indent=1))
    return 0 if passed else 1
  if args.eval_splits:
    ap.error("--eval-splits applies to FinQA only (the BIRD evaluation split is dev)")

  dev = bird.load_items(args.root, "dev")
  dev_dbs = {it.db_id for it in dev}
  dev_q = {_norm(it.question) for it in dev}
  dev_grams: set[tuple[str, ...]] = set()
  for it in dev:
    dev_grams |= ngrams(it.question) | ngrams(it.evidence)
  train = {it.id: it for it in bird.load_items(args.root, "train")}

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
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.root,)) as ex:
      results = list(ex.map(_reverify, tasks, chunksize=16))
    wrong = [(t, e) for t, ok, e in results if not ok]
    rep = {"rows": len(rows), "not_a_train_question": len(bad_id), "eval_db": len(bad_db),
           "exact_question_match": len(exact), "ngram13_overlap": len(gram),
           "reverify_failed": len(wrong), "examples": {"not_a_train_question": bad_id[:5], "eval_db": bad_db[:5],
                                                       "exact": exact[:5], "ngram13": gram[:5],
                                                       "reverify_failed": wrong[:5]}}
    report["files"][str(path)] = rep
    failed |= any(rep[k] for k in AUDIT_KEYS["bird"])
    print(json.dumps({"file": str(path), **{k: v for k, v in rep.items() if k != "examples"}}))
  report["passed"] = not failed
  args.out.write_text(json.dumps(report, indent=1))
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
