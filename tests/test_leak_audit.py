"""leak_audit.py: a clean SFT row passes; dev questions, dev databases and wrong SQL are each caught."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

DISTIL = Path(__file__).resolve().parents[1] / "distil"
sys.path.insert(0, str(DISTIL))
import bird  # noqa: E402

DEV_Q = "List the names of all customers who live in the city of Oslo and placed more than one order last year"


def _db(path: Path) -> None:
  path.parent.mkdir(parents=True)
  conn = sqlite3.connect(path)
  conn.executescript("CREATE TABLE t (id INTEGER, name TEXT); INSERT INTO t VALUES (1, 'a'), (2, 'b');")
  conn.commit()
  conn.close()


def _row(rid: str, db: str, question: str, sql: str) -> dict:
  item = bird.Item(id="x", db_id=db, question=question, evidence="", gold_sql="", difficulty="")
  msgs = bird.messages(item, "CREATE TABLE t (id INTEGER, name TEXT)")
  return {"id": rid, "db_id": db, "messages": msgs, "response": f"Reasoning.\n```sql\n{sql}\n```"}


def _setup(tmp_path: Path) -> Path:
  _db(tmp_path / "train_databases" / "shop" / "shop.sqlite")
  _db(tmp_path / "dev_databases" / "devdb" / "devdb.sqlite")
  (tmp_path / "train.json").write_text(json.dumps([
      {"db_id": "shop", "question": "How many rows are in t?", "evidence": "", "SQL": "SELECT COUNT(*) FROM t"},
      {"db_id": "shop", "question": DEV_Q, "evidence": "", "SQL": "SELECT name FROM t"},
  ]))
  (tmp_path / "dev.json").write_text(json.dumps([
      {"question_id": 0, "db_id": "devdb", "question": DEV_Q, "evidence": "", "SQL": "SELECT 1",
       "difficulty": "simple"}]))
  return tmp_path


def _audit(root: Path, rows: list[dict], tmp_path: Path) -> tuple[int, dict]:
  sft = tmp_path / "sft.jsonl"
  sft.write_text("".join(json.dumps(r) + "\n" for r in rows))
  out = tmp_path / "audit.json"
  proc = subprocess.run([sys.executable, str(DISTIL / "leak_audit.py"), "--bird-root", str(root), "--sft", str(sft),
                         "--out", str(out), "--workers", "2"], capture_output=True, text=True)
  return proc.returncode, json.loads(out.read_text())["files"][str(sft)]


def test_clean_row_passes(tmp_path):
  root = _setup(tmp_path)
  rc, rep = _audit(root, [_row("train-0#s0", "shop", "How many rows are in t?", "SELECT count(id) FROM t")], tmp_path)
  assert rc == 0 and rep["rows"] == 1 and rep["reverify_failed"] == 0 and rep["ngram13_overlap"] == 0


def test_dev_question_and_wrong_sql_are_caught(tmp_path):
  root = _setup(tmp_path)
  rc, rep = _audit(root, [_row("train-1#s0", "shop", DEV_Q, "SELECT name FROM t"),
                          _row("train-0#s1", "shop", "How many rows are in t?", "SELECT 5")], tmp_path)
  assert rc == 1
  assert rep["exact_question_match"] == 1 and rep["ngram13_overlap"] == 1 and rep["reverify_failed"] == 1


def test_eval_database_and_foreign_ids_are_caught(tmp_path):
  root = _setup(tmp_path)
  rc, rep = _audit(root, [_row("train-0#s0", "devdb", "How many rows are in t?", "SELECT COUNT(*) FROM t"),
                          _row("dev-0#s0", "shop", "How many rows are in t?", "SELECT COUNT(*) FROM t")], tmp_path)
  assert rc == 1 and rep["eval_db"] == 1 and rep["not_a_train_question"] == 2
