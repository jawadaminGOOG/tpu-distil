#!/usr/bin/env python3
"""Build teacher correction prompts from a student's failed attempts on training questions (BIRD or MATH).

For every question with at least one incorrect graded sample, the first incorrect sample (lowest
sample index) is shown to the teacher together with the original prompt and the execution outcome
of the student's query on the training database (the error text, or the row count and up to
`SHOW_ROWS` rows). The teacher is told the result is wrong and asked to rewrite the answer with
minimal changes in the same format. The gold SQL and the gold rows are never part of the prompt.
Questions whose gold query fails to execute are skipped.

MATH (`--task math`): the feedback is a sentence about the student's own answer only: it was cut off
at the length limit before giving a final answer, it gives no final answer in \\boxed{}, or it gives
the final answer X, which is wrong. The gold answer is never part of the prompt (asserted: a named
answer must differ from gold after normalisation).

Output rows keep the question id (`train-<n>`), so the teacher's generations can be graded with
grade.py and turned into training rows with build_sft.py against the original prompts file: the
training example is the original single-turn prompt followed by the corrected response.

Usage: correct.py [--task bird] --bird-root R --prompts train_prompts.jsonl --gen gen_self.jsonl \
         --graded graded_self.jsonl --out correction_prompts.jsonl [--workers 48]
       correct.py --task math --root R --prompts train_prompts.jsonl --gen gen_self.jsonl \
         --graded graded_self.jsonl --out correction_prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bird  # noqa: E402
import build_sft  # noqa: E402
from tasks import get_task  # noqa: E402

SHOW_ROWS = 5
CELL_CHARS = 60

CORRECTION_TEMPLATE = (
    "{original}\n\n---\n"
    "A student wrote the following answer:\n<<<\n{student}\n>>>\n\n"
    "When run on the database, the student's query {feedback}\n\nThis result is wrong.\n\n"
    "Rewrite the answer so that the query is correct. Keep the student's reasoning and SQL wherever they "
    "are right, and change only what is wrong. Use the same format: brief reasoning, then the final "
    "SQLite query in a single ```sql code block. Output only the rewritten answer."
)

MATH_CORRECTION_TEMPLATE = (
    "{original}\n\n---\n"
    "A student wrote the following answer:\n<<<\n{student}\n>>>\n\n"
    "The student's answer {feedback}\n\nThis answer is wrong.\n\n"
    "Rewrite the answer so that it is correct. Keep the student's reasoning wherever it is right, and change "
    "only what is wrong. Use the same format: brief step-by-step reasoning, then the final answer as "
    "`\\boxed{{<answer>}}` on the last line. Output only the rewritten answer."
)

_ROOT: Path


def _init(root: Path) -> None:
  global _ROOT  # pylint: disable=global-statement
  _ROOT = root


def _cell(v) -> str:
  s = "NULL" if v is None else str(v).replace("\n", " ")
  return s if len(s) <= CELL_CHARS else s[:CELL_CHARS] + "..."


def feedback(path: Path, sql: str | None, timeout_s: float = 30.0) -> str:
  """A sentence describing what the student's query did, from executing it (never the gold)."""
  if not sql:
    return "could not be found in the answer (no SQL query was given)"
  conn = bird.connect_ro(path)
  try:
    import time  # pylint: disable=import-outside-toplevel
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(bird.MAX_ROWS + 1)
  except Exception as e:  # pylint: disable=broad-except
    msg = str(e)
    if "interrupted" in msg:
      return f"did not finish within {int(timeout_s)} seconds"
    return f"failed with the error: {type(e).__name__}: {msg[:300]}"
  finally:
    conn.close()
  if len(rows) > bird.MAX_ROWS:
    return f"returned more than {bird.MAX_ROWS} rows"
  if not rows:
    return "returned no rows"
  shown = "\n".join(" | ".join(_cell(v) for v in r) for r in rows[:SHOW_ROWS])
  more = f" (first {SHOW_ROWS} shown)" if len(rows) > SHOW_ROWS else ""
  return f"returned {len(rows)} row(s){more}:\n{' | '.join(cols)}\n{shown}"


def _build(task: tuple[str, str, list[dict], str, int]) -> dict:
  qid, db_id, messages, student, k = task
  fb = feedback(bird.db_path(_ROOT, "train", db_id), bird.extract_sql(student))
  user = CORRECTION_TEMPLATE.format(original=messages[-1]["content"], student=student, feedback=fb)
  return {"id": qid, "db_id": db_id, "split": "train", "source_sample": k, "student_response": student,
          "messages": messages[:-1] + [{"role": "user", "content": user}]}


def gold_leaks(row: dict, gold_sql: str) -> bool:
  """True if the gold SQL appears in the prompt outside the student's own quoted answer."""
  g = gold_sql.strip()
  outside = row["messages"][-1]["content"].replace(row["student_response"], "")
  return bool(g) and (g in outside or any(g in m["content"] for m in row["messages"][:-1]))


def math_feedback(content: str, finish_reason: str | None, pred: str | None, gold: str) -> str:
  """What was wrong with a MATH answer, from the student's own answer only (never the gold)."""
  m = get_task("math")
  if finish_reason == "length":
    return "was cut off at the length limit before giving a final answer."
  if pred is None or not str(pred).strip():
    return "does not give a final answer in \\boxed{}."
  if m.normalize_answer(pred) == m.normalize_answer(gold):
    raise ValueError("a feedback sentence would name the gold answer")
  return f"gives the final answer {pred}, which is wrong."


def main_math(args) -> None:
  m = get_task("math")
  prompts = {r["id"]: r for r in bird.iter_jsonl(args.prompts)}
  gens = build_sft.load_generations(args.gen)
  gold = {it.id: it.gold for it in m.load_items(args.root, "train")}
  failed: dict[str, dict[int, dict]] = defaultdict(dict)
  for r in bird.iter_jsonl(args.graded):
    if not r["correct"]:
      failed[r["id"]][int(r["sample"])] = r
  rows, kinds = [], defaultdict(int)
  for qid in sorted(failed):
    if qid not in prompts or qid not in gens:
      continue
    k = min(failed[qid])
    choice = gens[qid]["choices"][k]
    student = build_sft.normalize_response(choice.get("content") or "")
    fb = math_feedback(student, choice.get("finish_reason"), failed[qid][k].get(m.PRED_KEY), gold[qid])
    kinds["truncated" if fb.startswith("was cut off") else "no_answer" if fb.startswith("does not") else "wrong_answer"] += 1
    p = prompts[qid]
    user = MATH_CORRECTION_TEMPLATE.format(original=p["messages"][-1]["content"], student=student, feedback=fb)
    rows.append({"id": qid, m.GROUP_KEY: p[m.GROUP_KEY], "split": "train", "source_sample": k,
                 "student_response": student, "messages": p["messages"][:-1] + [{"role": "user", "content": user}]})
  with open(args.out, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  print(json.dumps({"questions_with_failure": len(failed), "prompts": len(rows), "feedback_kinds": dict(kinds)}))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", choices=("bird", "math"), default="bird")
  ap.add_argument("--bird-root", "--root", dest="bird_root", type=Path, required=True)
  ap.add_argument("--prompts", type=Path, required=True)
  ap.add_argument("--gen", type=Path, required=True)
  ap.add_argument("--graded", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--workers", type=int, default=48)
  args = ap.parse_args()
  if args.task == "math":
    args.root = args.bird_root
    main_math(args)
    return

  prompts = {r["id"]: r for r in bird.iter_jsonl(args.prompts)}
  gens = build_sft.load_generations(args.gen)
  gold = {it.id: it.gold_sql for it in bird.load_items(args.bird_root, "train")}
  failed: dict[str, list[int]] = defaultdict(list)
  skipped_gold = 0
  for r in bird.iter_jsonl(args.graded):
    if r["error"] == "gold_failed":
      skipped_gold += 1
    elif not r["correct"]:
      failed[r["id"]].append(int(r["sample"]))
  tasks = []
  for qid in sorted(failed):
    if qid not in prompts or qid not in gens:
      continue
    k = min(failed[qid])
    student = build_sft.normalize_response(gens[qid]["choices"][k].get("content") or "")
    tasks.append((qid, prompts[qid]["db_id"], prompts[qid]["messages"], student, k))
  with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.bird_root,)) as ex:
    rows = list(ex.map(_build, tasks, chunksize=8))
  leaked = [r["id"] for r in rows if gold_leaks(r, gold[r["id"]])]
  if leaked:
    raise SystemExit(f"{len(leaked)} correction prompts contain the gold SQL, e.g. {leaked[:3]}")
  with open(args.out, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  print(json.dumps({"questions_with_failure": len(failed), "prompts": len(rows),
                    "graded_rows_with_failed_gold": skipped_gold}))


if __name__ == "__main__":
  main()
