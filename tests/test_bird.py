"""BIRD prompt, extraction, execution-accuracy and paired-comparison checks on a tiny database."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

DISTIL = Path(__file__).resolve().parents[1] / "distil"
sys.path.insert(0, str(DISTIL))
import bird  # noqa: E402
import compare  # noqa: E402


@pytest.fixture()
def root(tmp_path: Path) -> Path:
  db_dir = tmp_path / "dev_databases" / "shop"
  db_dir.mkdir(parents=True)
  conn = sqlite3.connect(db_dir / "shop.sqlite")
  conn.executescript("""
    CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
    INSERT INTO customer VALUES (1, 'Ann', 'Oslo'), (2, 'Bo', 'Rome'), (3, 'Cy', 'Oslo');
    INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 5.5), (3, 2, 7.0);
  """)
  conn.commit()
  conn.close()
  rows = [
      {"question_id": 0, "db_id": "shop", "question": "Names of customers in Oslo?",
       "evidence": "", "SQL": "SELECT name FROM customer WHERE city = 'Oslo'", "difficulty": "simple"},
      {"question_id": 1, "db_id": "shop", "question": "Total amount of Ann's orders?",
       "evidence": "Ann refers to name = 'Ann'",
       "SQL": "SELECT SUM(o.amount) FROM orders o JOIN customer c ON c.id = o.customer_id WHERE c.name = 'Ann'",
       "difficulty": "moderate"},
      {"question_id": 2, "db_id": "shop", "question": "How many customers?", "evidence": "",
       "SQL": "SELECT COUNT(*) FROM customer", "difficulty": "simple"},
  ]
  (tmp_path / "dev.json").write_text(json.dumps(rows))
  return tmp_path


def test_extract_sql_prefers_last_sql_fence():
  text = "First try:\n```sql\nSELECT 1\n```\nBetter:\n```sql\nSELECT name FROM t;\n```"
  assert bird.extract_sql(text) == "SELECT name FROM t"


def test_extract_sql_ignores_reasoning_and_falls_back():
  assert bird.extract_sql("<think>```sql\nSELECT bad\n```</think>```sql\nSELECT ok\n```") == "SELECT ok"
  assert bird.extract_sql("```\nSELECT a FROM b\n```") == "SELECT a FROM b"
  assert bird.extract_sql("The answer is SELECT a FROM t WHERE x IN (SELECT y FROM u);") == \
      "SELECT a FROM t WHERE x IN (SELECT y FROM u)"
  assert bird.extract_sql("I cannot answer.") is None
  assert bird.extract_sql(None) is None


def test_execution_match_is_order_insensitive_and_strict(root):
  path = bird.db_path(root, "dev", "shop")
  gold = bird.run_query(path, "SELECT name FROM customer WHERE city = 'Oslo'")
  assert bird.execution_match(path, "SELECT name FROM customer WHERE city='Oslo' ORDER BY name DESC", gold)[0]
  assert not bird.execution_match(path, "SELECT name FROM customer", gold)[0]
  ok, err = bird.execution_match(path, "SELECT nope FROM customer", gold)
  assert not ok and err.startswith("OperationalError")
  assert bird.execution_match(path, None, gold) == (False, "no_sql")


def test_database_is_read_only(root):
  path = bird.db_path(root, "dev", "shop")
  ok, err = bird.execution_match(path, "DELETE FROM customer", frozenset())
  assert not ok and "readonly" in err.lower()
  assert len(bird.run_query(path, "SELECT * FROM customer")) == 3


def test_timeout_and_row_cap(root, monkeypatch):
  path = bird.db_path(root, "dev", "shop")
  slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
          "SELECT COUNT(*) FROM c")
  ok, err = bird.execution_match(path, slow, frozenset(), timeout_s=0.5)
  assert not ok and err.startswith("timeout")
  monkeypatch.setattr(bird, "MAX_ROWS", 2)
  ok, err = bird.execution_match(path, "SELECT * FROM customer", frozenset())
  assert not ok and "more than 2 rows" in err


def test_schema_and_messages_are_deterministic(root):
  path = bird.db_path(root, "dev", "shop")
  s1, s2 = bird.schema_text(path), bird.schema_text(path)
  assert s1 == s2
  assert "CREATE TABLE customer" in s1 and "3 example rows from customer" in s1
  item = bird.load_items(root, "dev")[1]
  msgs = bird.messages(item, s1)
  assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
  assert "Ann refers to name = 'Ann'" in msgs[1]["content"] and "```sql" in msgs[1]["content"]
  assert len(bird.prompt_template_sha256()) == 64


def _grade(root: Path, out: Path, *extra: str) -> dict:
  subprocess.run([sys.executable, str(DISTIL / "grade.py"), "--bird-root", str(root), "--split", "dev",
                  "--out", str(out), "--workers", "2", *extra], check=True, capture_output=True)
  return json.loads(Path(str(out) + ".summary.json").read_text())


def test_controls_gold_is_perfect_and_shuffled_is_low(root, tmp_path):
  assert _grade(root, tmp_path / "gold.jsonl", "--control", "gold")["ex_sample0"] == 100.0
  assert _grade(root, tmp_path / "shuf.jsonl", "--control", "shuffled")["ex_sample0"] == 0.0


def test_grade_generations_and_missing_counts_wrong(root, tmp_path):
  gen = tmp_path / "gen.jsonl"
  gen.write_text("\n".join(json.dumps(r) for r in [
      {"id": "dev-0", "choices": [{"content": "```sql\nSELECT name FROM customer WHERE city='Oslo'\n```"},
                                  {"content": "```sql\nSELECT name FROM customer\n```"}]},
      {"id": "dev-1", "choices": [{"content": "```sql\nSELECT 15.5\n```"}]},
  ]))
  s = _grade(root, tmp_path / "g.jsonl", "--gen", str(gen))
  assert s["questions"] == 3 and s["samples"] == 4
  assert s["ex_sample0"] == pytest.approx(200 / 3, abs=0.01)  # dev-0 s0 right, dev-1 right, dev-2 missing
  assert s["ex_mean_all_samples"] == pytest.approx(100 * (0.5 + 1 + 0) / 3, abs=0.01)


def test_paired_compare_counts_and_ci():
  a = {str(i): 0.0 for i in range(200)}
  b = {str(i): (1.0 if i < 20 else 0.0) for i in range(200)}
  res = compare.compare(a, b, resamples=2000)
  assert res["delta"] == 10.0 and res["b_better"] == 20 and res["a_better"] == 0
  assert 0 < res["ci95"][0] < 10.0 < res["ci95"][1]
  same = compare.compare(a, dict(a), resamples=500)
  assert same["delta"] == 0.0 and same["ci95"] == [0.0, 0.0]
