"""correct.py: prompts only for failed questions, first failed sample, execution feedback, no gold SQL."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

DISTIL = Path(__file__).resolve().parents[1] / "distil"
sys.path.insert(0, str(DISTIL))
import bird  # noqa: E402
import correct  # noqa: E402

GOLD = {
    0: "SELECT name FROM t WHERE id = 1",
    1: "SELECT COUNT(*) FROM t",
    2: "SELECT MAX(id) FROM t",
    3: "SELECT name FROM t ORDER BY id DESC LIMIT 1",
}


def _setup(tmp_path: Path):
  db = tmp_path / "train_databases" / "shop" / "shop.sqlite"
  db.parent.mkdir(parents=True)
  conn = sqlite3.connect(db)
  conn.executescript("CREATE TABLE t (id INTEGER, name TEXT); INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c');")
  conn.commit()
  conn.close()
  (tmp_path / "train.json").write_text(json.dumps(
      [{"db_id": "shop", "question": f"Question {i}?", "evidence": "", "SQL": GOLD[i]} for i in range(4)]))
  items = bird.load_items(tmp_path, "train")
  schema = bird.schema_text(db)
  prompts = tmp_path / "p.jsonl"
  prompts.write_text("".join(json.dumps({"id": it.id, "db_id": "shop", "split": "train",
                                         "messages": bird.messages(it, schema)}) + "\n" for it in items))
  resp = lambda sql: f"Reasoning.\n```sql\n{sql}\n```"  # noqa: E731
  samples = {
      "train-0": [(resp("SELECT name FROM t WHERE id = 1"), True), (resp("SELECT nope FROM t"), False),
                  (resp("SELECT name FROM t"), False)],
      "train-1": [(resp("SELECT COUNT(*) FROM t"), True), (resp("SELECT COUNT(*) FROM t"), True)],
      "train-2": [(resp("SELECT id FROM t WHERE id > 5"), False)],
      # The student's own answer contains the gold query text: not a leak.
      "train-3": [(resp("SELECT name FROM t ORDER BY id DESC LIMIT 1 OFFSET 1"), False)],
  }
  gen, graded = tmp_path / "g.jsonl", tmp_path / "gr.jsonl"
  gen.write_text("".join(json.dumps({"id": q, "choices": [{"content": c} for c, _ in s]}) + "\n"
                         for q, s in samples.items()))
  graded.write_text("".join(json.dumps({"id": q, "sample": k, "pred_sql": bird.extract_sql(c), "correct": ok,
                                        "error": "" if ok else "wrong_result"}) + "\n"
                            for q, s in samples.items() for k, (c, ok) in enumerate(s)))
  return prompts, gen, graded


def test_correction_prompts(tmp_path):
  prompts, gen, graded = _setup(tmp_path)
  out = tmp_path / "corr.jsonl"
  subprocess.run([sys.executable, str(DISTIL / "correct.py"), "--bird-root", str(tmp_path), "--prompts", str(prompts),
                  "--gen", str(gen), "--graded", str(graded), "--out", str(out), "--workers", "2"],
                 check=True, capture_output=True)
  rows = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
  assert set(rows) == {"train-0", "train-2", "train-3"}  # train-1 never failed
  assert rows["train-0"]["source_sample"] == 1  # first failed sample, not the later one
  u0 = rows["train-0"]["messages"][-1]["content"]
  assert "SELECT nope FROM t" in u0 and "failed with the error" in u0 and "no such column" in u0
  assert "This result is wrong" in u0 and "Question 0?" in u0
  assert "returned no rows" in rows["train-2"]["messages"][-1]["content"]
  assert "returned 1 row(s)" in rows["train-3"]["messages"][-1]["content"]
  for qid, r in rows.items():
    gold = GOLD[int(qid.split("-")[1])]
    outside = r["messages"][-1]["content"].replace(r["student_response"], "")
    assert gold not in outside
    assert r["messages"][0]["role"] == "system" and len(r["messages"]) == 2


def test_gold_leak_detector():
  row = {"student_response": "x", "messages": [{"role": "system", "content": "s"},
                                               {"role": "user", "content": "q x SELECT 1"}]}
  assert correct.gold_leaks(row, "SELECT 1")
  assert not correct.gold_leaks(dict(row, student_response="x SELECT 1"), "SELECT 1")


def test_feedback_shows_row_count_and_caps_rows(tmp_path):
  db = tmp_path / "d.sqlite"
  conn = sqlite3.connect(db)
  conn.executescript("CREATE TABLE n (v INTEGER);" + "".join(f"INSERT INTO n VALUES ({i});" for i in range(9)))
  conn.commit()
  conn.close()
  fb = correct.feedback(db, "SELECT v FROM n")
  assert fb.startswith("returned 9 row(s) (first 5 shown)") and fb.count("\n") == 6
  assert correct.feedback(db, None).startswith("could not be found")
