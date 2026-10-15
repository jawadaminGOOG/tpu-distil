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


# ---------------------------------------------------------------------------
# --task math
# ---------------------------------------------------------------------------


def _math_root(tmp_path: Path) -> Path:
  d = tmp_path / "math"
  d.mkdir()
  rows = [{"source": f"train/algebra/{i}", "subject": "algebra", "problem": f"Compute {i} + {i}.", "level": "Level 1",
           "type": "Algebra", "solution": f"So $\\boxed{{{2 * i}}}$."} for i in range(30)]
  (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
  (d / "test.jsonl").write_text(json.dumps({**rows[0], "source": "test/algebra/0", "problem": "Compute 100 + 1."}) + "\n")
  return d


def test_math_correction_prompts_name_only_the_students_answer(tmp_path):
  from tasks import hendrycks_math as hm  # pylint: disable=import-outside-toplevel
  root = _math_root(tmp_path)
  items = hm.load_items(root, "train")
  prompts = tmp_path / "p.jsonl"
  prompts.write_text("".join(json.dumps({"id": it.id, "problem_hash": it.problem_hash, "split": "train",
                                         "messages": hm.messages(it)}) + "\n" for it in items[:3]))
  a, b, c = items[0], items[1], items[2]
  gen = {a.id: [(f"\\boxed{{{a.gold}}}", "stop"), ("So \\boxed{999}", "stop"), ("still thinking", "length")],
         b.id: [("I give up.", "stop")],
         c.id: [("Let me ramble on and on", "length"), ("\\boxed{0}", "stop")]}
  (tmp_path / "g.jsonl").write_text("".join(json.dumps({"id": q, "choices": [{"content": t, "finish_reason": f}
                                                                            for t, f in v]}) + "\n" for q, v in gen.items()))
  graded = []
  for q, v in gen.items():
    gold = next(it.gold for it in items if it.id == q)
    for k, (t, f) in enumerate(v):
      pred = hm.extract_answer(t)
      graded.append({"id": q, "sample": k, "pred_answer": pred, "correct": hm.score(pred, gold)[0], "error": ""})
  (tmp_path / "gr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in graded))
  out = tmp_path / "corr.jsonl"
  proc = subprocess.run([sys.executable, str(DISTIL / "correct.py"), "--task", "math", "--root", str(root), "--prompts",
                         str(prompts), "--gen", str(tmp_path / "g.jsonl"), "--graded", str(tmp_path / "gr.jsonl"),
                         "--out", str(out)], capture_output=True, text=True)
  assert proc.returncode == 0, proc.stderr
  rows = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
  assert rows[a.id]["source_sample"] == 1 and "gives the final answer 999, which is wrong." in rows[a.id]["messages"][-1]["content"]
  assert "does not give a final answer" in rows[b.id]["messages"][-1]["content"]
  assert rows[c.id]["source_sample"] == 0 and "was cut off at the length limit" in rows[c.id]["messages"][-1]["content"]
  for r in rows.values():
    user = r["messages"][-1]["content"]
    gold = next(it.gold for it in items if it.id == r["id"])
    assert f"\\boxed{{{gold}}}" not in user.replace(r["student_response"], "")
    assert user.startswith(hm.messages(next(it for it in items if it.id == r["id"]))[-1]["content"])
  assert json.loads(proc.stdout)["feedback_kinds"] == {"wrong_answer": 1, "no_answer": 1, "truncated": 1}


def test_math_feedback_refuses_to_name_the_gold():
  import pytest  # pylint: disable=import-outside-toplevel
  with pytest.raises(ValueError):
    correct.math_feedback("\\boxed{1/2}", "stop", "\\frac12", "\\dfrac{1}{2}")
