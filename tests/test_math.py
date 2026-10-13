"""MATH task: dev holdout, boxed-answer extraction, the reference normalisation, controls, prompts,
builders and leak audit.

Everything runs on a small synthetic MATH-format fixture (train/test JSONL files in a temp dir).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DISTIL = Path(__file__).resolve().parents[1] / "distil"
sys.path.insert(0, str(DISTIL))
from tasks import get_task  # noqa: E402
from tasks import hendrycks_math as hm  # noqa: E402

LONG_P = "A rectangle has a perimeter of 30 units and its length is twice its width. What is the area of the rectangle?"


def _rec(source, subject, problem, answer):
  return {"source": source, "subject": subject, "problem": problem, "level": "Level 2", "type": subject.title(),
          "solution": f"Some working.  So the answer is $\\boxed{{{answer}}}$."}


TEST = [
    _rec("test/algebra/0", "algebra", LONG_P, "50"),
    _rec("test/algebra/1", "algebra", "Solve for $x$: $2x + 3 = 7$.", "2"),
    _rec("test/geometry/0", "geometry", "What is the sum of the interior angles of a triangle, in degrees?", "180^\\circ"),
    _rec("test/geometry/1", "geometry", "Find the hypotenuse of a right triangle with legs 1 and 1.", "\\sqrt{2}"),
]
# 30 train problems: dev holds out round(30 / 15) = 2 of them, stratified by subject.
TRAIN = ([_rec(f"train/algebra/{i}", "algebra", f"Compute ${i} + {i}$.", str(2 * i)) for i in range(15)]
         + [_rec(f"train/geometry/{i}", "geometry", f"What is the area of a square with side {i}?", str(i * i))
            for i in range(13)]
         + [_rec("train/algebra/15", "algebra", LONG_P, "50"),  # a test problem, word for word
            _rec("train/algebra/16", "algebra", "Hint: " + LONG_P, "50")])  # shares a 13-gram with it


@pytest.fixture()
def root(tmp_path: Path) -> Path:
  d = tmp_path / "math"
  d.mkdir()
  for name, rows in (("train", TRAIN), ("test", TEST)):
    (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
  return d


def _py(script: str, *args, check=True):
  proc = subprocess.run([sys.executable, str(DISTIL / script), *map(str, args)], capture_output=True, text=True)
  if check:
    assert proc.returncode == 0, proc.stderr
  return proc


def _jsonl(path: Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows) -> Path:
  path.write_text("".join(json.dumps(r) + "\n" for r in rows))
  return path


# ---------------------------------------------------------------------------
# Items, splits and prompts
# ---------------------------------------------------------------------------


def test_registry_and_module_name():
  assert get_task("math") is hm and hm.NAME == "math" and hm.EVAL_SPLIT == "test"
  import math  # the stdlib module is not shadowed by the task module
  assert hasattr(math, "isfinite")


def test_dev_is_a_deterministic_stratified_holdout(root):
  train, dev, test = (hm.load_items(root, s) for s in ("train", "dev", "test"))
  assert len(dev) == 2 and len(train) == 28 and len(test) == 4
  assert sorted(it.subject for it in dev) == ["algebra", "geometry"]
  assert not {it.source for it in dev} & {it.source for it in train}
  assert {it.source for it in dev} | {it.source for it in train} == {r["source"] for r in TRAIN}
  assert [it.source for it in hm.load_items(root, "dev")] == [it.source for it in dev]
  assert [it.id for it in train] == [f"train-{n}" for n in range(28)]
  rows = [{"subject": "a"}] * 1744 + [{"subject": "b"}] * 771 + [{"subject": "c"}] * 4985
  assert len(hm.dev_indices(rows)) == 500


def test_items_carry_gold_and_problem_fingerprint(root):
  test = hm.load_items(root, "test")
  assert [it.gold for it in test] == ["50", "2", "180^\\circ", "\\sqrt{2}"]
  assert all(hm.valid_group(it.group) for it in test)
  assert hm.problem_hash("What  is 2+2?") == hm.problem_hash("what is 2 + 2 ?")
  assert hm.problem_hash("What is 2+2?") != hm.problem_hash("What is 2+3?")
  bad = root / "bad"
  bad.mkdir()
  (bad / "test.jsonl").write_text(json.dumps({**TEST[0], "solution": "no box here"}) + "\n")
  with pytest.raises(ValueError, match="no \\\\boxed answer"):
    hm.load_items(bad, "test")


def test_valid_group_forms():
  assert hm.valid_group("p0123456789abcdef")
  for bad in ("p0123", "q0123456789abcdef", "european_football_2", "V/2008/page_17.pdf", ""):
    assert not hm.valid_group(bad)


def test_prompt_round_trips_question(root):
  it = hm.load_items(root, "test")[0]
  msgs = hm.messages(it)
  assert [m["role"] for m in msgs] == ["system", "user"]
  assert hm.question_of(msgs[1]["content"]) == LONG_P
  assert "\\boxed{<answer>}" in msgs[1]["content"]
  assert hm.question_of("something else") == ""


def test_prompts_cli_excludes_evaluation_problems(root, tmp_path):
  out = tmp_path / "train_prompts.jsonl"
  _py("prompts.py", "--task", "math", "--root", root, "--split", "train", "--out", out,
      "--exclude-groups-from", "test,dev", "--exclude-questions-from", "test,dev")
  rows = _jsonl(out)
  sources = {r["source"] for r in rows}
  assert "train/algebra/15" not in sources and "train/algebra/16" not in sources and len(rows) == 26
  m = json.loads(Path(str(out) + ".manifest.json").read_text())
  assert m["task"] == "math" and m["excluded"]["groups_shared_with_those_splits"] == 1
  assert m["excluded"]["question_overlap_dropped"] == 1 and m["prompt_template_sha256"] == hm.prompt_template_sha256()
  assert set(rows[0]) == {"id", "problem_hash", "split", "subject", "level", "source", "messages"}


# ---------------------------------------------------------------------------
# Extraction and normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("So $x = \\boxed{5}$.", "5"),
    ("First \\boxed{3}, then corrected: \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
    ("\\boxed{\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}}", "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}"),
    ("\\boxed{x^{2} + {y}}", "x^{2} + {y}"),
    ("\\fbox{7}", "7"),
    ("The answer is $\\boxed 9$.", "9"),
    ("<think>\\boxed{1}</think> Final: \\boxed{2}", "2"),
    ("\\boxed{unclosed", None),
    ("no box at all", None),
    ("", None),
    (None, None),
])
def test_extract_answer(text, expected):
  assert hm.extract_answer(text) == expected


@pytest.mark.parametrize("a,b,equal", [
    ("\\dfrac{1}{2}", "\\frac12", True),
    ("0.5", "\\frac{1}{2}", True),
    ("1/2", "\\frac{1}{2}", True),
    ("x = 3", "3", True),
    ("10\\text{ inches}", "10", True),
    ("\\$5", "5", True),
    ("50\\%", "50", True),
    ("45^\\circ", "45", True),
    ("\\left( 1, 2 \\right)", "(1,2)", True),
    ("\\sqrt2", "\\sqrt{2}", True),
    (".5", "0.5", True),
    ("2\\sqrt{2}", "\\sqrt{8}", False),
    ("3", "4", False),
    ("total = 3", "3", False),  # only a prefix of at most two characters before "=" is dropped
])
def test_reference_normalisation(a, b, equal):
  assert (hm.normalize_answer(a) == hm.normalize_answer(b)) is equal


def test_unnormalisable_answers_compare_raw():
  odd = "1\\text{ a}\\text{ b}"  # two unit markers: the reference normalisation does not apply
  assert hm.normalize_answer(odd) == odd and hm.normalize_answer(None) == ""


def test_score():
  assert hm.score("\\frac12", "\\dfrac{1}{2}") == (True, "")
  assert hm.score("3", "4") == (False, "")
  assert hm.score(None, "4") == (False, "no_answer") and hm.score("  ", "4") == (False, "no_answer")
  assert hm.score("", "") == (False, "no_answer")  # an empty gold answer can never be matched


# ---------------------------------------------------------------------------
# Controls and grading
# ---------------------------------------------------------------------------


def _grade(root, out, split, *args):
  _py("grade.py", "--task", "math", "--root", root, "--split", split, "--out", out, *args)
  return json.loads(Path(str(out) + ".summary.json").read_text())


def test_controls_gold_is_perfect_and_shuffled_is_low(root, tmp_path):
  gold = _grade(root, tmp_path / "gold.jsonl", "test", "--control", "gold")
  assert gold["ex_sample0"] == 100.0 and gold["questions"] == 4
  shuffled = _grade(root, tmp_path / "shuf.jsonl", "test", "--control", "shuffled")
  assert shuffled["ex_sample0"] == 0.0


def test_shuffled_control_stays_within_subject(root):
  items = {it.id: it for it in hm.load_items(root, "train")}
  out = hm.control_samples(items, "shuffled")
  for qid, [(_, pred)] in out.items():
    assert any(o.gold == pred and o.subject == items[qid].subject for o in items.values())
    assert hm.normalize_answer(pred) != hm.normalize_answer(items[qid].gold)


def test_grade_generations(root, tmp_path):
  gen = _write_jsonl(tmp_path / "gen.jsonl", [
      {"id": "test-0", "choices": [{"content": "Width 5, length 10.\n\\boxed{50}"}, {"content": "\\boxed{25}"}]},
      {"id": "test-2", "choices": [{"content": "\\boxed{180}"}]},
      {"id": "test-3", "choices": [{"content": "It is root two."}]},
  ])  # test-1 has no generation: graded incorrect on the evaluation split
  s = _grade(root, tmp_path / "g.jsonl", "test", "--gen", gen)
  assert s["task"] == "math" and s["questions"] == 4 and s["samples"] == 5 and s["ex_sample0"] == 50.0
  rows = {(r["id"], r["sample"]): r for r in _jsonl(tmp_path / "g.jsonl")}
  assert rows[("test-0", 0)] == {"id": "test-0", "problem_hash": hm.problem_hash(LONG_P), "subject": "algebra",
                                 "level": "Level 2", "sample": 0, "pred_answer": "50", "correct": True, "error": ""}
  assert rows[("test-2", 0)]["correct"] is True and rows[("test-3", 0)]["error"] == "no_answer"


# ---------------------------------------------------------------------------
# build_sft.py and leak_audit.py on MATH data
# ---------------------------------------------------------------------------


def _train_ids_by_source(root):
  return {it.source: it.id for it in hm.load_items(root, "train")}


def test_build_sft_keeps_correct_distinct_answers(root, tmp_path):
  ids = _train_ids_by_source(root)
  a3, g4 = ids["train/algebra/3"], ids["train/geometry/4"]
  prompts = tmp_path / "train_prompts.jsonl"
  _py("prompts.py", "--task", "math", "--root", root, "--split", "train", "--out", prompts,
      "--exclude-groups-from", "test,dev", "--exclude-questions-from", "test,dev")
  gen = _write_jsonl(tmp_path / "gen.jsonl", [
      {"id": a3, "served_model": "teacher", "choices": [{"content": c, "finish_reason": "stop"} for c in
                                                        ["3+3=6. \\boxed{6}", "\\boxed{6.0}", "\\boxed{ 6 }", "\\boxed{7}"]]},
      {"id": g4, "served_model": "teacher", "choices": [{"content": "\\boxed{16}", "finish_reason": "stop"}]}])
  graded = tmp_path / "graded.jsonl"
  _py("grade.py", "--task", "math", "--root", root, "--split", "train", "--gen", gen, "--out", graded)
  test_prompts = tmp_path / "test_prompts.jsonl"
  _py("prompts.py", "--task", "math", "--root", root, "--split", "test", "--out", test_prompts)
  out = tmp_path / "sft.jsonl"
  _py("build_sft.py", "--task", "math", "--prompts", prompts, "--gen", gen, "--graded", graded, "--out", out,
      "--forbid-groups", f"@{test_prompts}.manifest.json", "--max-per-question", "4")
  rows = _jsonl(out)
  # "6.0" is wrong under the reference rule; " 6 " normalises to s0's answer and is a duplicate.
  assert sorted(r["id"] for r in rows) == sorted([f"{a3}#s0", f"{g4}#s0"])
  assert all(set(r) == {"id", "messages", "response", "problem_hash", "sample", "served_model"} for r in rows)
  m = json.loads(Path(str(out) + ".manifest.json").read_text())
  assert m["task"] == "math" and m["counts"]["dropped_duplicate_answer"] == 1

  audit = tmp_path / "audit.json"
  _py("leak_audit.py", "--task", "math", "--root", root, "--sft", out, "--out", audit, "--eval-splits", "test,dev")
  rep = json.loads(audit.read_text())
  assert rep["passed"] is True and rep["task"] == "math" and rep["eval_problems"] == 6


def _sft_row(root: Path, source: str, response: str, sample: int = 0) -> dict:
  it = next(i for i in hm.load_items(root, "train") if i.source == source)
  return {"id": f"{it.id}#s{sample}", "messages": hm.messages(it), "response": response,
          "problem_hash": it.problem_hash, "sample": sample}


def test_audit_catches_leaks_and_wrong_answers(root, tmp_path):
  rows = [_sft_row(root, "train/algebra/2", "\\boxed{4}"),
          _sft_row(root, "train/algebra/15", "\\boxed{50}"),  # identical to a test problem
          _sft_row(root, "train/algebra/16", "\\boxed{50}"),  # shares a 13-gram with it
          _sft_row(root, "train/geometry/3", "\\boxed{10}")]  # wrong answer
  sft = _write_jsonl(tmp_path / "sft.jsonl", rows)
  out = tmp_path / "audit.json"
  proc = _py("leak_audit.py", "--task", "math", "--root", root, "--sft", sft, "--out", out, check=False)
  rep = json.loads(out.read_text())["files"][str(sft)]
  assert proc.returncode == 1
  assert rep["eval_problem"] == 1 and rep["exact_question_match"] == 1 and rep["ngram13_overlap"] == 2
  assert rep["rescore_failed"] == 1 and rep["not_a_train_question"] == 0
