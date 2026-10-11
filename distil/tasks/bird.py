"""BIRD text-to-SQL: question loading, schema prompts, SQL extraction and execution accuracy.

Execution accuracy (EX) follows the official BIRD evaluator: a prediction is correct when the
set of rows it returns equals the set of rows the gold query returns on the same SQLite
database. Databases are opened read-only; a query that runs past the timeout, errors, or
returns more than `MAX_ROWS` rows is incorrect.

Task interface (see `tasks/__init__.py`): the question group is the database (`db_id`), the
answer is the SQL query, the evaluation split is dev, and grading runs every question's samples
in a process pool against gold rows computed once per question.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

NAME = "bird"
LABEL = "BIRD"
SPLITS = ("dev", "train")
EVAL_SPLIT = "dev"
FULL_COVERAGE_SPLITS = ("dev",)  # splits where a question without a generation is graded incorrect
GROUP_KEY = "db_id"  # prompt, graded and training rows name the question group under this key
GROUP_SHORT = "db"
GROUP_NOUN = "database"
PRED_KEY = "pred_sql"  # graded rows hold the extracted answer under this key
ANSWER_SHORT = "sql"
ANSWER_NOUN = "SQL"

MAX_ROWS = 200_000
SAMPLE_ROWS = 3
SAMPLE_CELL_CHARS = 40

SYSTEM_PROMPT = "You are an expert data analyst who writes correct SQLite queries."
USER_TEMPLATE = (
    "Database schema (SQLite):\n{schema}\n\n"
    "External knowledge: {evidence}\n\n"
    "Question: {question}\n\n"
    "Reason step by step, briefly, about which tables, columns, joins and filters are needed. "
    "Then write the final SQLite query in a single ```sql code block. Output exactly one query."
)


def prompt_template_sha256() -> str:
  return hashlib.sha256((SYSTEM_PROMPT + "\x00" + USER_TEMPLATE).encode()).hexdigest()


_DB_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def valid_group(name: str) -> bool:
  """Whether `name` has the form of a BIRD database id (letters, digits and underscores)."""
  return bool(_DB_NAME.match(name))


@dataclass(frozen=True)
class Item:
  id: str
  db_id: str
  question: str
  evidence: str
  gold_sql: str
  difficulty: str

  @property
  def group(self) -> str:
    return self.db_id

  @property
  def gold(self) -> str:
    return self.gold_sql


def db_path(root: Path, split: str, db_id: str) -> Path:
  return root / f"{split}_databases" / db_id / f"{db_id}.sqlite"


def load_items(root: Path, split: str) -> list[Item]:
  """Questions of `split` ("dev" or "train") from `<root>/<split>.json`.

  Dev items keep their official `question_id`; train items are numbered by position.
  """
  rows = json.loads((root / f"{split}.json").read_text())
  items = []
  for i, r in enumerate(rows):
    qid = r.get("question_id", i)
    items.append(Item(id=f"{split}-{qid}", db_id=r["db_id"], question=r["question"].strip(),
                      evidence=(r.get("evidence") or "").strip(), gold_sql=r["SQL"].strip(),
                      difficulty=r.get("difficulty", "")))
  return items


def connect_ro(path: Path) -> sqlite3.Connection:
  conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
  conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
  return conn


def _cell(v: Any) -> str:
  s = "NULL" if v is None else str(v)
  s = s.replace("\n", " ")
  return s if len(s) <= SAMPLE_CELL_CHARS else s[:SAMPLE_CELL_CHARS] + "..."


def schema_text(path: Path) -> str:
  """CREATE TABLE statements of every user table, each followed by up to 3 example rows."""
  conn = connect_ro(path)
  try:
    tables = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY rowid").fetchall()
    parts = []
    for name, ddl in tables:
      block = (ddl or f"CREATE TABLE \"{name}\" (...)").strip()
      try:
        cur = conn.execute(f'SELECT * FROM "{name}" LIMIT {SAMPLE_ROWS}')
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
      except sqlite3.Error:
        cols, rows = [], []
      if rows:
        lines = [" | ".join(cols)] + [" | ".join(_cell(v) for v in r) for r in rows]
        block += f"\n/* {len(rows)} example rows from {name}:\n" + "\n".join(lines) + "\n*/"
      parts.append(block)
    return "\n\n".join(parts)
  finally:
    conn.close()


def messages(item: Item, schema: str) -> list[dict[str, str]]:
  user = USER_TEMPLATE.format(schema=schema, evidence=item.evidence or "none",
                              question=item.question)
  return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_SQL_FENCE = re.compile(r"```\s*sql\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)
_BARE = re.compile(r"\b(WITH|SELECT)\b.*", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str | None) -> str | None:
  """The last ```sql block of a response; else the last code block; else the text from the first
  SELECT/WITH to the end (so a subquery is not cut off)."""
  if not text:
    return None
  if "</think>" in text:
    text = text.rsplit("</think>", 1)[1]
  for pat in (_SQL_FENCE, _ANY_FENCE):
    blocks = [b.strip() for b in pat.findall(text) if b.strip()]
    if blocks:
      return _clean(blocks[-1])
  m = _BARE.search(text)
  return _clean(m.group(0)) if m else None


extract_answer = extract_sql


def normalize_sql(sql: str) -> str:
  """Whitespace- and case-insensitive form of a query, used to count distinct SQL per question."""
  s = " ".join(sql.split()).lower()
  while s.endswith(";"):
    s = s[:-1].rstrip()
  return s


normalize_answer = normalize_sql


def _clean(sql: str) -> str:
  sql = sql.strip()
  while sql.endswith(";"):
    sql = sql[:-1].rstrip()
  return sql


class QueryError(Exception):
  pass


def run_query(path: Path, sql: str, timeout_s: float = 30.0) -> frozenset:
  """Rows of `sql` on the database at `path` as a frozenset of tuples."""
  conn = connect_ro(path)
  deadline = time.monotonic() + timeout_s
  conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
  try:
    cur = conn.execute(sql)
    rows = cur.fetchmany(MAX_ROWS + 1)
    if len(rows) > MAX_ROWS:
      raise QueryError(f"more than {MAX_ROWS} rows")
    return frozenset(tuple(r) for r in rows)
  except sqlite3.OperationalError as e:
    if "interrupted" in str(e):
      raise QueryError(f"timeout after {timeout_s}s") from None
    raise QueryError(f"OperationalError: {e}") from None
  except sqlite3.Error as e:
    raise QueryError(f"{type(e).__name__}: {e}") from None
  except (OverflowError, ValueError, MemoryError) as e:
    raise QueryError(f"{type(e).__name__}: {e}") from None
  finally:
    conn.close()


def execution_match(path: Path, pred_sql: str | None, gold_rows: frozenset,
                    timeout_s: float = 30.0) -> tuple[bool, str]:
  """(correct, error) for one prediction against precomputed gold rows."""
  if not pred_sql:
    return False, "no_sql"
  try:
    rows = run_query(path, pred_sql, timeout_s)
  except QueryError as e:
    return False, str(e)[:300]
  return rows == gold_rows, ""


def iter_jsonl(path: Path) -> Iterable[dict]:
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)


# ---------------------------------------------------------------------------
# Pipeline entry points (prompts.py, grade.py)
# ---------------------------------------------------------------------------


def prompt_rows(root: Path, split: str, items: list[Item]) -> tuple[list[dict], dict]:
  """One prompt row per item and the task-specific manifest fields (databases, schema sizes)."""
  schemas: dict[str, str] = {}
  rows = []
  for it in items:
    if it.db_id not in schemas:
      schemas[it.db_id] = schema_text(db_path(root, split, it.db_id))
    rows.append({"id": it.id, "db_id": it.db_id, "split": split, "difficulty": it.difficulty,
                 "messages": messages(it, schemas[it.db_id])})
  extra = {"dbs": sorted(schemas), "schema_chars": {d: len(s) for d, s in sorted(schemas.items())}}
  return rows, extra


def control_samples(items: dict[str, Item], kind: str, seed: int = 0) -> dict[str, list[tuple[int, str | None]]]:
  """Evaluator controls. gold: each gold query as the prediction for its own question (must score
  100% on executable gold). shuffled: each question gets another question's gold query on the same
  database (a negative control); questions whose database has no different gold query are left out."""
  if kind == "gold":
    return {i: [(0, it.gold_sql)] for i, it in items.items()}
  if kind != "shuffled":
    raise ValueError(f"unknown control {kind!r}")
  rng = random.Random(seed)
  per_db: dict[str, list[Item]] = defaultdict(list)
  for it in items.values():
    per_db[it.db_id].append(it)
  out: dict[str, list[tuple[int, str | None]]] = {}
  for its in per_db.values():
    for it in its:
      others = [o for o in its if o.gold_sql.strip() != it.gold_sql.strip()]
      if others:
        out[it.id] = [(0, rng.choice(others).gold_sql)]
  return out


_ROOT: Path
_SPLIT: str
_TIMEOUT: float


def _init(root: Path, split: str, timeout: float) -> None:
  global _ROOT, _SPLIT, _TIMEOUT  # pylint: disable=global-statement
  _ROOT, _SPLIT, _TIMEOUT = root, split, timeout


def _grade_group(task: tuple[str, str, str, list[tuple[int, str | None]]]) -> list[dict]:
  """Grade every sample of one question; the gold rows are computed once."""
  qid, db_id, gold_sql, samples = task
  path = db_path(_ROOT, _SPLIT, db_id)
  try:
    gold = run_query(path, gold_sql, _TIMEOUT)
    gold_err = ""
  except QueryError as e:
    gold, gold_err = None, str(e)[:300]
  out = []
  for idx, sql in samples:
    if gold is None:
      out.append({"id": qid, "db_id": db_id, "sample": idx, "pred_sql": sql, "correct": False,
                  "error": "gold_failed", "gold_error": gold_err})
      continue
    ok, err = execution_match(path, sql, gold, _TIMEOUT)
    out.append({"id": qid, "db_id": db_id, "sample": idx, "pred_sql": sql, "correct": ok,
                "error": err})
  return out


def grade(root: Path, split: str, items: dict[str, Item], samples: dict[str, list[tuple[int, str | None]]], *,
          workers: int = 48, timeout: float = 30.0) -> list[dict]:
  """Graded rows {id, db_id, sample, pred_sql, correct, error} for every (question, sample)."""
  tasks = [(i, items[i].db_id, items[i].gold_sql, s) for i, s in samples.items()]
  with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(root, split, timeout)) as ex:
    return [r for group in ex.map(_grade_group, tasks, chunksize=4) for r in group]
