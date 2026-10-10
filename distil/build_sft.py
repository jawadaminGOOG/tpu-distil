#!/usr/bin/env python3
"""Turn graded generations into the SFT trainer's record JSONL (single-turn, correct samples only).

Usage:
  build_sft.py --prompts train_prompts.jsonl --gen gen.jsonl --graded graded.jsonl --out sft.jsonl \\
      [--max-per-question 4] [--n-target 10000 --seed 0] [--exclude-dbs a,b]

Inputs: prompts.py rows {id, db_id, messages}, generate.py rows {id, choices: [{content, ...}]}, and
grade.py rows {id, sample, pred_sql, correct, error} (sample = index into that id's choices).
Only rows with correct == true are ever written. Per question, at most --max-per-question samples
are kept, each with a distinct SQL after whitespace/case normalisation (lowest sample index first).
--n-target draws exactly N of the remaining rows with random.Random(--seed); fewer available rows
is an error. Before training, an empty reasoning block at the start of a response ("</think>" or
"<think></think>", as a non-thinking endpoint may return it) and leading whitespace are removed: the
Qwen3 chat templates cannot reproduce leading whitespace after the serving prompt. Two kinds of
correct response are dropped and counted: responses cut off by the length limit (finish_reason
"length") and responses that still hold thinking markup (<think>, </think>), since the student
trains and serves in non-thinking mode.

Only training questions are accepted: every prompt id must start with "train-", and a prompt on a
database named in --forbid-dbs (the evaluation databases) is an error, not a silent drop.

Each output row (trainer/data.py Record): id "<id>#s<sample>", messages = the full prompt chat
(system + user), response = the response after the normalisation above, db_id, sample, and
served_model when the generation recorded it. A manifest (<out>.manifest.json) records the counts,
the questions covered, per-database counts and the SHA-256 of every input and of the output.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bird  # noqa: E402  pylint: disable=g-import-not-at-top

THINK_MARKUP = ("<think>", "</think>")
_EMPTY_THINK = re.compile(r"\A\s*(?:<think>\s*)?</think>")


def normalize_response(content: str) -> str:
  """The response without an empty leading reasoning block and without leading whitespace."""
  return _EMPTY_THINK.sub("", content, count=1).lstrip()


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def read_jsonl(path: Path):
  with open(path) as f:
    for line_no, line in enumerate(f, 1):
      line = line.strip()
      if line:
        try:
          yield json.loads(line)
        except json.JSONDecodeError as e:
          raise ValueError(f"{path}:{line_no}: not JSON ({e})") from e


def normalize_sql(sql: str) -> str:
  """Whitespace- and case-insensitive form of a query, used to count distinct SQL per question."""
  s = " ".join(sql.split()).lower()
  while s.endswith(";"):
    s = s[:-1].rstrip()
  return s


def load_prompts(path: Path, forbid_dbs: set[str]) -> dict[str, dict]:
  out = {}
  for r in read_jsonl(path):
    if not str(r["id"]).startswith("train-"):
      raise ValueError(f"{path}: prompt {r['id']} is not a training question (id must start with 'train-')")
    if r.get("db_id") in forbid_dbs:
      raise ValueError(f"{path}: prompt {r['id']} is on evaluation database {r['db_id']}")
    if r["id"] in out:
      raise ValueError(f"{path}: duplicate prompt id {r['id']}")
    msgs = r["messages"]
    if not msgs or msgs[-1].get("role") != "user":
      raise ValueError(f"{path}: prompt {r['id']} does not end with a user message")
    if r.get("split") == "dev":
      raise ValueError(f"{path}: prompt {r['id']} is from BIRD dev, which is never trained on")
    out[r["id"]] = r
  return out


def load_generations(path: Path) -> dict[str, dict]:
  """id -> generation row; rows without choices (failed requests) are ignored."""
  out = {}
  for r in read_jsonl(path):
    if not r.get("choices"):
      continue
    if r["id"] in out:
      raise ValueError(f"{path}: id {r['id']} has more than one generation row with choices")
    out[r["id"]] = r
  return out


def build(prompts: dict[str, dict], gens: dict[str, dict], graded: list[dict], *, max_per_question: int | None,
          n_target: int | None, seed: int, exclude_dbs: set[str]) -> tuple[list[dict], dict]:
  counts = Counter()
  counts["graded_rows"] = len(graded)
  correct = [r for r in graded if r.get("correct") is True]
  counts["correct_rows"] = len(correct)
  seen: set[tuple[str, int]] = set()
  by_q: dict[str, list[tuple[int, dict, dict]]] = {}
  for r in correct:
    qid, k = r["id"], int(r["sample"])
    if (qid, k) in seen:
      raise ValueError(f"graded rows repeat ({qid}, sample {k})")
    seen.add((qid, k))
    if qid not in prompts:
      raise ValueError(f"graded id {qid} is not in --prompts")
    if qid not in gens:
      raise ValueError(f"graded id {qid} has no generation in --gen")
    if prompts[qid]["db_id"] in exclude_dbs:
      counts["dropped_excluded_db"] += 1
      continue
    choices = gens[qid]["choices"]
    if not 0 <= k < len(choices):
      raise ValueError(f"graded ({qid}, sample {k}) is outside the {len(choices)} generated choices")
    choice = choices[k]
    content = choice.get("content")
    if bird.extract_sql(content) != r["pred_sql"]:
      raise ValueError(f"graded ({qid}, sample {k}): pred_sql is not the SQL of the generated response")
    if choice.get("finish_reason") == "length":
      counts["dropped_truncated"] += 1
      continue
    text = normalize_response(content)
    if text != content:
      counts["normalized_prefix"] += 1
    if any(t in text for t in THINK_MARKUP):
      counts["dropped_think_markup"] += 1
      continue
    by_q.setdefault(qid, []).append((k, r, dict(choice, content=text)))

  candidates = []
  for qid in sorted(by_q):
    kept_sql: set[str] = set()
    for k, r, choice in sorted(by_q[qid], key=lambda x: x[0]):
      norm = normalize_sql(r["pred_sql"])
      if norm in kept_sql:
        counts["dropped_duplicate_sql"] += 1
        continue
      if max_per_question is not None and len(kept_sql) >= max_per_question:
        counts["dropped_over_cap"] += 1
        continue
      kept_sql.add(norm)
      candidates.append((qid, k, choice))
  counts["candidates"] = len(candidates)

  if n_target is not None:
    if len(candidates) < n_target:
      raise ValueError(f"only {len(candidates)} rows available, fewer than --n-target {n_target}")
    candidates = sorted(random.Random(seed).sample(candidates, n_target), key=lambda c: (c[0], c[1]))

  rows = []
  for qid, k, choice in candidates:
    p = prompts[qid]
    row = {
        "id": f"{qid}#s{k}",
        "messages": p["messages"],
        "response": choice["content"],
        "db_id": p["db_id"],
        "sample": k,
    }
    if gens[qid].get("served_model"):
      row["served_model"] = gens[qid]["served_model"]
    rows.append(row)
  counts["written"] = len(rows)
  per_db = Counter(r["db_id"] for r in rows)
  stats = {
      "counts": dict(sorted(counts.items())),
      "questions_covered": len({r["id"].rsplit("#s", 1)[0] for r in rows}),
      "per_db": dict(sorted(per_db.items())),
  }
  return rows, stats


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  ap.add_argument("--prompts", type=Path, required=True)
  ap.add_argument("--gen", type=Path, required=True)
  ap.add_argument("--graded", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--max-per-question", type=int, default=None,
                  help="keep at most K correct samples per question, each with a distinct normalised SQL")
  ap.add_argument("--n-target", type=int, default=None, help="draw exactly N rows (error if fewer are available)")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--exclude-dbs", default="", help="comma-separated db_ids to leave out")
  ap.add_argument("--forbid-dbs", required=True,
                  help="comma-separated evaluation db_ids; a prompt on one of them is an error")
  args = ap.parse_args(argv)
  if args.max_per_question is not None and args.max_per_question < 1:
    ap.error("--max-per-question must be >= 1")
  if args.n_target is not None and args.n_target < 1:
    ap.error("--n-target must be >= 1")

  exclude = {d for d in args.exclude_dbs.split(",") if d}
  forbid = {d for d in args.forbid_dbs.split(",") if d}
  if not forbid:
    ap.error("--forbid-dbs must name the evaluation databases")
  rows, stats = build(load_prompts(args.prompts, forbid), load_generations(args.gen), list(read_jsonl(args.graded)),
                      max_per_question=args.max_per_question, n_target=args.n_target, seed=args.seed,
                      exclude_dbs=exclude)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with open(args.out, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  manifest = {
      **stats,
      "args": {"max_per_question": args.max_per_question, "n_target": args.n_target, "seed": args.seed,
               "exclude_dbs": sorted(exclude), "forbid_dbs": sorted(forbid)},
      "sha256": {"prompts": sha256_file(args.prompts), "gen": sha256_file(args.gen),
                 "graded": sha256_file(args.graded), "out": sha256_file(args.out)},
      "paths": {"prompts": str(args.prompts), "gen": str(args.gen), "graded": str(args.graded), "out": str(args.out)},
  }
  Path(str(args.out) + ".manifest.json").write_text(json.dumps(manifest, indent=1))
  print(json.dumps({"written": stats["counts"]["written"], "questions_covered": stats["questions_covered"]}))
  return 0


if __name__ == "__main__":
  sys.exit(main())
