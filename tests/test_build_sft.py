"""build_sft.py: correct-only rows, per-question cap on distinct SQL, exact seeded subsample, manifest."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DISTIL = REPO / "distil"
sys.path.insert(0, str(DISTIL))

import build_sft  # noqa: E402

SYSTEM = "You are an expert data analyst who writes correct SQLite queries."


def _resp(sql, prefix="Reasoning."):
  return f"{prefix}\n```sql\n{sql}\n```"


# question id -> (db_id, [(response, correct, finish_reason)])
QUESTIONS = {
    "train-1": ("shop", [(_resp("SELECT COUNT(*) FROM customer"), True, "stop"),
                         (_resp("select  count(*)\n  from CUSTOMER;"), True, "stop"),  # same SQL after normalising
                         (_resp("SELECT COUNT(id) FROM customer"), True, "stop"),
                         (_resp("SELECT COUNT(name) FROM customer"), True, "stop"),
                         (_resp("SELECT 0"), False, "stop")]),
    "train-2": ("shop", [(_resp("SELECT name FROM customer"), False, "stop"),
                         (_resp("SELECT name FROM customer ORDER BY 1"), True, "stop")]),
    "train-3": ("zoo", [(_resp("SELECT * FROM animal"), True, "stop"),
                        (_resp("SELECT animal.* FROM animal"), True, "length"),
                        ("<think>x</think>" + _resp("SELECT a FROM animal"), True, "stop"),
                        ("\n" + _resp("SELECT b FROM animal"), True, "stop")]),
    "train-4": ("bank", [(_resp("SELECT 1"), True, "stop"),
                         ("</think>" + _resp("SELECT 3"), True, "stop")]),  # empty reasoning block
    "train-5": ("zoo", [(_resp("SELECT 2"), False, "stop")]),
}


def _write(tmp_path: pathlib.Path, questions=QUESTIONS):
  prompts, gen, graded = tmp_path / "p.jsonl", tmp_path / "g.jsonl", tmp_path / "gr.jsonl"
  with open(prompts, "w") as fp, open(gen, "w") as fg, open(graded, "w") as fr:
    for qid, (db, samples) in questions.items():
      msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question {qid} on {db}"}]
      fp.write(json.dumps({"id": qid, "db_id": db, "split": "train", "difficulty": "simple", "messages": msgs}) + "\n")
      fg.write(json.dumps({"id": qid, "error": "HTTP 500"}) + "\n")  # a failed request before the retry
      fg.write(json.dumps({"id": qid, "served_model": "teacher", "choices": [
          {"content": c, "reasoning_content": None, "finish_reason": fr} for c, _, fr in samples]}) + "\n")
      for k, (c, ok, _) in enumerate(samples):
        sql = build_sft.bird.extract_sql(c)
        fr.write(json.dumps({"id": qid, "db_id": db, "sample": k, "pred_sql": sql, "correct": ok,
                             "error": None if ok else "wrong_result"}) + "\n")
  return prompts, gen, graded


def _run(tmp_path, *extra, inputs=None):
  prompts, gen, graded = inputs or _write(tmp_path)
  out = tmp_path / "sft.jsonl"
  if "--forbid-dbs" not in extra:
    extra = (*extra, "--forbid-dbs", "eval_db")
  proc = subprocess.run([sys.executable, str(DISTIL / "build_sft.py"), "--prompts", str(prompts), "--gen", str(gen),
                         "--graded", str(graded), "--out", str(out), *extra], capture_output=True, text=True)
  return proc, out


def _rows(out):
  return [json.loads(line) for line in out.read_text().splitlines()]


def _graded_correct(graded):
  return {(r["id"], r["sample"]) for r in map(json.loads, graded.read_text().splitlines()) if r["correct"] is True}


def test_only_correct_rows_in_trainer_layout(tmp_path):
  inputs = _write(tmp_path)
  proc, out = _run(tmp_path, inputs=inputs)
  assert proc.returncode == 0, proc.stderr
  rows = _rows(out)
  ok = _graded_correct(inputs[2])
  assert rows and all((r["id"].split("#s")[0], int(r["id"].split("#s")[1])) in ok for r in rows)
  assert {r["id"] for r in rows} == {"train-1#s0", "train-1#s2", "train-1#s3", "train-2#s1", "train-3#s0",
                                     "train-3#s3", "train-4#s0", "train-4#s1"}
  gen = {json.loads(line)["id"]: json.loads(line) for line in inputs[1].read_text().splitlines()
         if "choices" in line}
  prompts = {json.loads(line)["id"]: json.loads(line) for line in inputs[0].read_text().splitlines()}
  for r in rows:
    qid, k = r["id"].split("#s")
    assert set(r) == {"id", "messages", "response", "db_id", "sample", "served_model"}
    assert r["messages"] == prompts[qid]["messages"] and r["messages"][0] == {"role": "system", "content": SYSTEM}
    assert r["messages"][-1]["role"] == "user" and r["sample"] == int(k) and r["served_model"] == "teacher"
    response = build_sft.normalize_response(gen[qid]["choices"][int(k)]["content"])
    assert not response[0].isspace() and "think>" not in response
    assert r["response"] == response and "reward" not in r
  # The rows load as SFT records in the trainer.
  sys.path.insert(0, str(REPO / "trainer"))
  dc = pytest.importorskip("data")
  assert [rec.id for rec in dc.load_records(str(out))] == [r["id"] for r in rows]


def test_cap_keeps_k_distinct_normalised_sql_per_question(tmp_path):
  proc, out = _run(tmp_path, "--max-per-question", "2")
  assert proc.returncode == 0, proc.stderr
  ids = [r["id"] for r in _rows(out)]
  # train-1: sample 1 normalises to sample 0's SQL; the cap of 2 keeps samples 0 and 2.
  assert [i for i in ids if i.startswith("train-1#")] == ["train-1#s0", "train-1#s2"]
  m = json.loads((tmp_path / "sft.jsonl.manifest.json").read_text())
  assert m["counts"]["dropped_duplicate_sql"] == 1 and m["counts"]["dropped_over_cap"] == 1
  assert build_sft.normalize_sql("select  count(*)\n  from CUSTOMER;") == build_sft.normalize_sql(
      "SELECT COUNT(*) FROM customer")


def test_n_target_is_exact_and_deterministic_under_seed(tmp_path):
  proc, out = _run(tmp_path, "--n-target", "4", "--seed", "7")
  assert proc.returncode == 0, proc.stderr
  first = out.read_bytes()
  assert len(_rows(out)) == 4
  proc, out = _run(tmp_path, "--n-target", "4", "--seed", "7")
  assert out.read_bytes() == first
  draws = set()
  for seed in range(8):
    proc, out = _run(tmp_path, "--n-target", "4", "--seed", str(seed))
    draws.add(out.read_bytes())
  assert len(draws) > 1  # the seed changes the draw


def test_n_target_larger_than_available_is_an_error(tmp_path):
  proc, out = _run(tmp_path, "--n-target", "9")
  assert proc.returncode != 0 and "fewer than --n-target 9" in proc.stderr


def test_exclude_dbs_and_manifest_counts(tmp_path):
  inputs = _write(tmp_path)
  proc, out = _run(tmp_path, "--exclude-dbs", "bank,nowhere", inputs=inputs)
  assert proc.returncode == 0, proc.stderr
  rows = _rows(out)
  assert all(r["db_id"] != "bank" for r in rows)
  m = json.loads((tmp_path / "sft.jsonl.manifest.json").read_text())
  assert m["counts"] == {"graded_rows": 14, "correct_rows": 11, "dropped_excluded_db": 2, "dropped_truncated": 1,
                         "dropped_think_markup": 1, "normalized_prefix": 1, "dropped_duplicate_sql": 1,
                         "candidates": 6, "written": 6}
  assert m["questions_covered"] == 3 and m["per_db"] == {"shop": 4, "zoo": 2}
  assert sum(m["per_db"].values()) == len(rows)
  sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
  assert m["sha256"] == {"prompts": sha(inputs[0]), "gen": sha(inputs[1]), "graded": sha(inputs[2]), "out": sha(out)}
  assert m["args"]["exclude_dbs"] == ["bank", "nowhere"]


def test_graded_sql_must_match_the_generated_response(tmp_path):
  prompts, gen, graded = _write(tmp_path)
  rows = [json.loads(line) for line in graded.read_text().splitlines()]
  rows[0]["pred_sql"] = "SELECT 42"
  graded.write_text("".join(json.dumps(r) + "\n" for r in rows))
  proc, _ = _run(tmp_path, inputs=(prompts, gen, graded))
  assert proc.returncode != 0 and "pred_sql is not the SQL" in proc.stderr


def test_dev_prompts_are_refused(tmp_path):
  prompts, gen, graded = _write(tmp_path)
  rows = [json.loads(line) for line in prompts.read_text().splitlines()]
  rows[0]["split"] = "dev"
  prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
  proc, _ = _run(tmp_path, inputs=(prompts, gen, graded))
  assert proc.returncode != 0 and "BIRD dev" in proc.stderr


def test_non_training_ids_are_refused(tmp_path):
  qs = dict(QUESTIONS)
  qs["dev-7"] = qs.pop("train-5")
  proc, _ = _run(tmp_path, inputs=_write(tmp_path, qs))
  assert proc.returncode != 0 and "not a training question" in proc.stderr


def test_evaluation_databases_are_refused_and_required(tmp_path):
  inputs = _write(tmp_path)
  proc, _ = _run(tmp_path, "--forbid-dbs", "eval_db,zoo", inputs=inputs)
  assert proc.returncode != 0 and "evaluation database zoo" in proc.stderr
  proc, _ = _run(tmp_path, "--forbid-dbs", "", inputs=inputs)
  assert proc.returncode != 0 and "must name the evaluation databases" in proc.stderr


def test_normalize_response():
  f = build_sft.normalize_response
  assert f("</think>\n\nWe need X.\n```sql\nSELECT 1\n```") == "We need X.\n```sql\nSELECT 1\n```"
  assert f("<think>\n\n</think>\n\nA") == "A"
  assert f(" \n\tA") == "A"
  assert f("<think>real reasoning</think>A") == "<think>real reasoning</think>A"  # kept, then dropped as markup
  assert f("A </think> B") == "A </think> B"
