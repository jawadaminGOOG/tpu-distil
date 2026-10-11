"""FinQA task: prompts, answer extraction, tolerance scoring, evaluator controls, builders and leak audit.

Everything runs on a small synthetic FinQA-format fixture (train/dev/test JSON files in a temp dir).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DISTIL = Path(__file__).resolve().parents[1] / "distil"
sys.path.insert(0, str(DISTIL))
from tasks import finqa, get_task  # noqa: E402

LONG_Q = "what was the percentage change in total net revenue of the company from 2009 to 2010?"


def _rec(source_id, filename, question, exe_ans, table=None, pre=None, post=None):
  return {"id": source_id, "filename": filename, "pre_text": pre or ["the company reports its results ."],
          "post_text": post or ["amounts are in millions ."],
          "table": table or [["", "2010", "2009"], ["revenue", "$ 125", "$ 100"]],
          "qa": {"question": question, "exe_ans": exe_ans}}


TEST = [
    _rec("ACME/2010/page_5.pdf-1", "ACME/2010/page_5.pdf", LONG_Q, 0.25),
    _rec("BETA/2012/page_9.pdf-1", "BETA/2012/page_9.pdf", "what is the total of cash and investments in 2012?",
         1234.0, table=[["item", "2012"], ["cash", "$ 1,000"], ["investments", "$ 234"]]),
    _rec("BETA/2012/page_9.pdf-2", "BETA/2012/page_9.pdf", "did operating income increase in 2012?", "yes"),
    _rec("ZETA/2014/page_2.pdf-1", "ZETA/2014/page_2.pdf", "what was the change in operating margin?", -3.2),
]
TRAIN = [
    _rec("GAMMA/2011/page_3.pdf-1", "GAMMA/2011/page_3.pdf", "what is the average of sales in 2010 and 2011?", 50.0),
    _rec("GAMMA/2011/page_3.pdf-2", "GAMMA/2011/page_3.pdf", "what is the ratio of debt to equity in 2011?", 0.4),
    _rec("ACME/2010/page_5.pdf-2", "ACME/2010/page_5.pdf", "what is the revenue in 2009?", 100.0),  # a test report
    _rec("DELTA/2011/page_7.pdf-1", "DELTA/2011/page_7.pdf", LONG_Q, 0.1),  # a test question, word for word
    _rec("DELTA/2011/page_7.pdf-2", "DELTA/2011/page_7.pdf", "did dividends increase in 2011?", "no"),
    _rec("EPS/2013/page_4.pdf-1", "EPS/2013/page_4.pdf", "excluding fees, " + LONG_Q, 0.2),  # shares a 13-gram
]
DEV = [_rec("OMEGA/2009/page_1.pdf-1", "OMEGA/2009/page_1.pdf", "what is the sum of a and b?", 3.0)]


@pytest.fixture()
def root(tmp_path: Path) -> Path:
  d = tmp_path / "finqa"
  d.mkdir()
  for name, rows in (("train", TRAIN), ("dev", DEV), ("test", TEST)):
    (d / f"{name}.json").write_text(json.dumps(rows))
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
# Items and prompts
# ---------------------------------------------------------------------------


def test_registry_and_items(root):
  assert get_task("finqa") is finqa
  with pytest.raises(ValueError, match="unknown task"):
    get_task("medqa")
  items = finqa.load_items(root, "test")
  assert [it.id for it in items] == ["test-0", "test-1", "test-2", "test-3"]
  assert items[0].group == "ACME/2010/page_5.pdf" and items[0].source_id == "ACME/2010/page_5.pdf-1"
  assert items[0].gold == 0.25 and items[2].gold == "yes" and items[3].gold == -3.2


def test_prompt_is_deterministic_and_complete(root):
  it = finqa.load_items(root, "test")[1]
  m1, m2 = finqa.messages(it), finqa.messages(finqa.load_items(root, "test")[1])
  assert m1 == m2
  assert m1[0] == {"role": "system", "content": "You are an expert financial analyst."}
  user = m1[1]["content"]
  assert m1[1]["role"] == "user" and len(m1) == 2
  pre, table, post, question = (user.index("the company reports its results ."), user.index("item | 2012"),
                                user.index("amounts are in millions ."), user.index("Question: " + it.question))
  assert pre < table < post < question  # text before the table, the table, text after, then the question
  assert "item | 2012\ncash | $ 1,000\ninvestments | $ 234" in user  # header row first, pipe-separated
  assert user.rstrip().endswith("Only if the question is a yes/no question, write `Answer: yes` or `Answer: no` instead.")
  assert "a plain number, no units or currency symbols" in user
  assert finqa.question_of(user) == it.question
  sha = finqa.prompt_template_sha256()
  assert len(sha) == 64 and sha == finqa.prompt_template_sha256()


def test_prompts_cli_excludes_evaluation_reports(root, tmp_path):
  out = tmp_path / "train_prompts.jsonl"
  _py("prompts.py", "--task", "finqa", "--root", root, "--split", "train", "--out", out,
      "--exclude-groups-from", "test")
  rows = _jsonl(out)
  assert [r["id"] for r in rows] == ["train-0", "train-1", "train-3", "train-4", "train-5"]  # train-2: test report
  assert all(set(r) == {"id", "report", "split", "source_id", "messages"} and r["split"] == "train" for r in rows)
  items = {it.id: it for it in finqa.load_items(root, "train")}
  assert all(r["messages"] == finqa.messages(items[r["id"]]) for r in rows)
  m = json.loads(Path(str(out) + ".manifest.json").read_text())
  assert m["task"] == "finqa" and m["rows"] == 5 and m["prompt_template_sha256"] == finqa.prompt_template_sha256()
  assert m["excluded"]["rows_left_out"] == 1 and m["excluded"]["groups_shared_with_those_splits"] == 1
  assert "ACME/2010/page_5.pdf" not in m["reports"]
  first = out.read_bytes()
  _py("prompts.py", "--task", "finqa", "--root", root, "--split", "train", "--out", out,
      "--exclude-groups-from", "test")
  assert out.read_bytes() == first
  bad = _py("prompts.py", "--task", "finqa", "--root", root, "--split", "train", "--out", out,
            "--exclude-groups-from", "train", check=False)
  assert bad.returncode != 0 and "must not name the split" in bad.stderr


def test_prompts_cli_excludes_overlapping_questions(root, tmp_path):
  out = tmp_path / "train_prompts.jsonl"
  _py("prompts.py", "--task", "finqa", "--root", root, "--split", "train", "--out", out,
      "--exclude-questions-from", "test")
  # train-3 equals a test question, train-5 shares a 13-gram with one; the others share neither.
  assert [r["id"] for r in _jsonl(out)] == ["train-0", "train-1", "train-2", "train-4"]
  ex = json.loads(Path(str(out) + ".manifest.json").read_text())["excluded"]
  assert ex["question_overlap_dropped"] == 2 and ex["questions_from_splits"] == ["test"] and ex["ngram_n"] == 13
  assert ex["question_overlap_exact_ids"] == ["train-3"] and ex["question_overlap_ngram_only_ids"] == ["train-5"]
  assert ex["rows_left_out"] == 2
  # Both filters together, then the leak audit on that pool finds nothing.
  _py("prompts.py", "--task", "finqa", "--root", root, "--split", "train", "--out", out,
      "--exclude-groups-from", "test", "--exclude-questions-from", "test,dev")
  rows = _jsonl(out)
  assert [r["id"] for r in rows] == ["train-0", "train-1", "train-4"]
  items = {it.id: it for it in finqa.load_items(root, "train")}
  sft = [{"id": f"{r['id']}#s0", "messages": r["messages"], "response": finqa.gold_response(items[r["id"]]),
          "report": r["report"], "sample": 0} for r in rows]
  rc, rep = _audit(root, sft, tmp_path)
  assert rc == 0 and rep["exact_question_match"] == 0 and rep["ngram13_overlap"] == 0 and rep["eval_report"] == 0


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("The change is 12.5%.\nAnswer: 12.5%", "12.5"),
    ("Answer: $1,234", "1234"),
    ("Answer: -3.2", "-3.2"),
    ("Answer: \u22123.2", "-3.2"),  # unicode minus
    ("Answer: yes", "yes"),
    ("answer: No.", "no"),
    ("**Answer:** 0.5", "0.5"),
    ("**Answer:**\n\n42 million", "42"),
    ("Answer: 1,234,567.5 dollars", "1234567.5"),
    ("Revenue rose from 100 to 112.5, so the growth is 12.5%.", "12.5"),  # no Answer: line -> last number
    ("Answer: 10\nOn reflection that is wrong.\nAnswer: 20", "20"),  # several Answer: lines -> the last
    ("<think>Answer: 3</think>The total is 9.", "9"),
    ("Answer: (3.2)", "-3.2"),                 # accounting negative
    ("Answer: ($1,234)", "-1234"),
    ("Answer: (3.2%)", "-3.2"),
    ("Answer: - 3.2", "-3.2"),                 # space after the minus
    ("Answer: 12.5 (rounded)", "12.5"),
    ("Answer: cannot be determined", None),
    ("I do not know.", None),
    ("", None),
    (None, None),
])
def test_extract_answer(text, expected):
  assert finqa.extract_answer(text) == expected


def test_normalize_answer():
  f = finqa.normalize_answer
  assert f("12.50") == f("12.5") == f("12.5000000000001") and f("1234") == f("1234.0")
  assert f("12.5") != f("0.125")
  assert f("YES") == "yes" and f(None) == ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pred,gold,correct", [
    ("101", 100.0, True), ("99", 100.0, True),              # exactly 1% away
    ("101.01", 100.0, False), ("98.99", 100.0, False),      # just past 1%
    ("-202", -200.0, True), ("-202.5", -200.0, False), ("200", -200.0, False),
    ("12.5", 0.125, True), ("0.125", 0.125, True),          # percent number or fraction
    ("0.125", 12.5, True), ("0.127", 12.5, False),          # fraction for a percent gold
    ("12.6", 0.125, True), ("12.63", 0.125, False),         # percent forms: 1% relative only, no absolute bound
    ("13.2", 0.125, False),
    ("0.304", 0.3, True), ("0.306", 0.3, False),            # |gold| < 0.5: absolute 0.005 on p itself, not 1%
    ("-0.304", -0.3, True), ("30.3", 0.3, True), ("30.4", 0.3, False), ("30.6", 0.3, False),
    ("0", 0.003, False),                                    # 0 is never right for a non-zero gold
    ("0.3", 0.003, True),                                   # 0.3 / 100 is within 1% of 0.003
    ("0.0031", 0.003, True),                                # within 0.005 absolute
    ("0.0081", 0.003, False), ("0.81", 0.003, False),
    ("0", 0.0, True), ("0.004", 0.0, True), ("2", 0.0, False),
    ("1234", 1234.0, True), ("1246.34", 1234.0, True), ("1246.35", 1234.0, False),
    ("YES", "yes", True), ("no", "yes", False),
])
def test_score_tolerance(pred, gold, correct):
  ok, err = finqa.score(pred, gold)
  assert ok is correct and err == ""


def test_score_errors():
  assert finqa.score(None, 1.0) == (False, "no_answer")
  assert finqa.score("  ", 1.0) == (False, "no_answer")
  assert finqa.score("yes", 1.0) == (False, "not_a_number")
  assert finqa.score("3", "yes") == (False, "not_yes_no")


def test_gold_answers_round_trip_through_extraction(root):
  for split in ("train", "dev", "test"):
    for it in finqa.load_items(root, split):
      pred = finqa.extract_answer(finqa.gold_response(it))
      assert finqa.score(pred, it.gold) == (True, ""), (it.id, pred)
  assert finqa.format_gold(41932.20339) == "41,932.20339" and finqa.format_gold(22929.0) == "22,929"
  assert finqa.format_gold(1000.0) == "1,000" and finqa.format_gold(0.0) == "0" and finqa.format_gold(-0.0) == "0"
  for g in (1e-6, -0.000001, 2.5e-9, 1e16, 0.53232):
    s = finqa.format_gold(g)
    assert s != "0" and "e" not in s.lower() and finqa.score(finqa.extract_answer(f"Answer: {s}"), g) == (True, "")


def test_gold_control_grades_tiny_gold_correct(tmp_path):
  d = tmp_path / "tiny"
  d.mkdir()
  (d / "test.json").write_text(json.dumps([_rec("T/2010/page_1.pdf-1", "T/2010/page_1.pdf", "tiny?", 1e-6),
                                           _rec("T/2010/page_1.pdf-2", "T/2010/page_1.pdf", "tinier?", -2e-7)]))
  s = _grade(d, tmp_path / "gold.jsonl", "test", "--control", "gold")
  assert s["ex_sample0"] == 100.0
  assert [r["pred_answer"] for r in _jsonl(tmp_path / "gold.jsonl")] == ["0.000001", "-0.0000002"]


def test_shuffled_control_never_reuses_an_equal_answer():
  # Nine questions share one answer, so a draw that ignored equality would mostly pick an equal one.
  golds = [5.0] * 9 + [7.0, "yes"]
  items = {f"test-{n}": finqa.Item(id=f"test-{n}", report="R/2010/page_1.pdf", question=f"q{n}", gold=g, pre_text=(),
                                   table=(), post_text=()) for n, g in enumerate(golds)}
  for seed in range(5):
    out = finqa.control_samples(items, "shuffled", seed=seed)
    assert set(out) == set(items)
    for qid, [(k, pred)] in out.items():
      own = finqa.normalize_answer(finqa.extract_answer(finqa.gold_response(items[qid])))
      assert k == 0 and finqa.normalize_answer(pred) != own, (seed, qid, pred)


# ---------------------------------------------------------------------------
# grade.py: generations and controls
# ---------------------------------------------------------------------------


def _grade(root: Path, out: Path, split: str, *extra) -> dict:
  _py("grade.py", "--task", "finqa", "--root", root, "--split", split, "--out", out, *extra)
  return json.loads(Path(str(out) + ".summary.json").read_text())


def test_controls_gold_is_perfect_and_shuffled_is_low(root, tmp_path):
  for split in ("test", "train"):
    gold = _grade(root, tmp_path / f"gold_{split}.jsonl", split, "--control", "gold")
    assert gold["ex_sample0"] == 100.0 and gold["questions"] == len(finqa.load_items(root, split))
    shuf = _grade(root, tmp_path / f"shuf_{split}.jsonl", split, "--control", "shuffled")
    assert shuf["questions"] == gold["questions"] and shuf["ex_sample0"] <= 20.0
  rows = _jsonl(tmp_path / "shuf_test.jsonl")
  items = {it.id: it for it in finqa.load_items(root, "test")}
  for r in rows:  # every question received another question's (different) gold answer
    assert finqa.normalize_answer(r["pred_answer"]) != finqa.normalize_answer(
        finqa.extract_answer(finqa.gold_response(items[r["id"]])))
  _grade(root, tmp_path / "shuf_test.jsonl", "test", "--control", "shuffled")
  assert _jsonl(tmp_path / "shuf_test.jsonl") == rows  # the shuffle is seeded


def test_grade_generations(root, tmp_path):
  gen = _write_jsonl(tmp_path / "gen.jsonl", [
      {"id": "test-0", "choices": [{"content": "25% growth.\nAnswer: 25%"}, {"content": "Answer: 0.3"}]},
      {"id": "test-1", "choices": [{"content": "Answer: $1,234"}]},
      {"id": "test-3", "choices": [{"content": "Answer: cannot be determined"}]},
  ])  # test-2 has no generation: graded incorrect on the evaluation split
  s = _grade(root, tmp_path / "g.jsonl", "test", "--gen", gen)
  assert s["task"] == "finqa" and s["questions"] == 4 and s["samples"] == 5
  assert s["ex_sample0"] == 50.0 and s["ex_mean_all_samples"] == pytest.approx(100 * (0.5 + 1) / 4, abs=0.01)
  assert s["failure_kinds_sample0"] == {"no_answer": 2}
  rows = {(r["id"], r["sample"]): r for r in _jsonl(tmp_path / "g.jsonl")}
  assert rows[("test-0", 0)] == {"id": "test-0", "report": "ACME/2010/page_5.pdf", "sample": 0, "pred_answer": "25",
                                 "correct": True, "error": ""}
  assert rows[("test-1", 0)]["pred_answer"] == "1234" and rows[("test-1", 0)]["correct"] is True
  assert rows[("test-2", 0)]["pred_answer"] is None and rows[("test-3", 0)]["error"] == "no_answer"


# ---------------------------------------------------------------------------
# build_sft.py and build_grpo_rollouts.py on FinQA graded data
# ---------------------------------------------------------------------------

# train question id -> list of responses (gold: train-0 50, train-1 0.4, train-3 0.1, train-4 "no", train-5 0.2)
SFT_GEN = {
    "train-0": ["Half of 100.\nAnswer: 50", "Answer: 50.00", "About 49.9.\nAnswer: 49.9", "Answer: 40"],
    "train-1": ["Answer: 40%", "Answer: 0.4", "I think it is 7"],
    "train-3": ["Answer: 0.5"],
    "train-4": ["Answer: No", "Answer: yes"],
}


def _graded_inputs(root: Path, tmp_path: Path, gen_by_id: dict[str, list[str]], exclude_test_reports=True):
  prompts = tmp_path / "train_prompts.jsonl"
  args = ["--exclude-groups-from", "test"] if exclude_test_reports else []
  _py("prompts.py", "--task", "finqa", "--root", root, "--split", "train", "--out", prompts, *args)
  gen = _write_jsonl(tmp_path / "gen.jsonl", [
      {"id": qid, "served_model": "teacher", "choices": [{"content": c, "finish_reason": "stop"} for c in cs]}
      for qid, cs in gen_by_id.items()])
  graded = tmp_path / "graded.jsonl"
  _py("grade.py", "--task", "finqa", "--root", root, "--split", "train", "--gen", gen, "--out", graded)
  test_prompts = tmp_path / "test_prompts.jsonl"
  _py("prompts.py", "--task", "finqa", "--root", root, "--split", "test", "--out", test_prompts)
  return prompts, gen, graded, Path(str(test_prompts) + ".manifest.json")


def _build_sft(prompts, gen, graded, out, *extra, check=True):
  return _py("build_sft.py", "--task", "finqa", "--prompts", prompts, "--gen", gen, "--graded", graded, "--out", out,
             *extra, check=check)


def test_build_sft_keeps_correct_distinct_answers(root, tmp_path):
  prompts, gen, graded, test_manifest = _graded_inputs(root, tmp_path, SFT_GEN)
  out = tmp_path / "sft.jsonl"
  _build_sft(prompts, gen, graded, out, "--forbid-groups", f"@{test_manifest}", "--max-per-question", "4")
  rows = _jsonl(out)
  # train-0: s1 "50.00" repeats s0's answer; s3 is wrong. train-1: 40% and 0.4 are both correct and distinct.
  assert [r["id"] for r in rows] == ["train-0#s0", "train-0#s2", "train-1#s0", "train-1#s1", "train-4#s0"]
  correct = {(r["id"], r["sample"]) for r in _jsonl(graded) if r["correct"] is True}
  assert all((r["id"].split("#s")[0], r["sample"]) in correct for r in rows)
  for r in rows:
    assert set(r) == {"id", "messages", "response", "report", "sample", "served_model"}
    assert r["report"] == "GAMMA/2011/page_3.pdf" or r["report"] == "DELTA/2011/page_7.pdf"
  m = json.loads(Path(str(out) + ".manifest.json").read_text())
  assert m["task"] == "finqa" and m["counts"]["dropped_duplicate_answer"] == 1
  assert m["per_report"] == {"DELTA/2011/page_7.pdf": 1, "GAMMA/2011/page_3.pdf": 4}
  assert m["args"]["forbid_reports"] == sorted({it.report for it in finqa.load_items(root, "test")})
  _build_sft(prompts, gen, graded, out, "--forbid-groups", f"@{test_manifest}", "--max-per-question", "1")
  assert [r["id"] for r in _jsonl(out)] == ["train-0#s0", "train-1#s0", "train-4#s0"]


def test_build_sft_refuses_evaluation_reports_and_splits(root, tmp_path):
  prompts, gen, graded, test_manifest = _graded_inputs(root, tmp_path, SFT_GEN, exclude_test_reports=False)
  out = tmp_path / "sft.jsonl"
  proc = _build_sft(prompts, gen, graded, out, "--forbid-groups", f"@{test_manifest}", check=False)
  assert proc.returncode != 0 and "train-2 is on evaluation report ACME/2010/page_5.pdf" in proc.stderr
  proc = _build_sft(prompts, gen, graded, out, "--forbid-groups", "ACME/2010/page_5.pdf,X/1/p.pdf", check=False)
  assert proc.returncode != 0 and "evaluation report ACME/2010/page_5.pdf" in proc.stderr
  proc = _build_sft(prompts, gen, graded, out, "--forbid-groups", "", check=False)
  assert proc.returncode != 0 and "must name the evaluation reports" in proc.stderr
  rows = _jsonl(prompts)
  rows[0]["split"] = "test"
  _write_jsonl(prompts, rows)
  proc = _build_sft(prompts, gen, graded, out, "--forbid-groups", "X/1/p.pdf", check=False)
  assert proc.returncode != 0 and "is from FinQA test" in proc.stderr
  test_prompts = tmp_path / "test_prompts.jsonl"
  proc = _build_sft(test_prompts, gen, graded, out, "--forbid-groups", "X/1/p.pdf", check=False)
  assert proc.returncode != 0 and "not a training question" in proc.stderr


def test_build_sft_checks_graded_answer_against_response(root, tmp_path):
  prompts, gen, graded, test_manifest = _graded_inputs(root, tmp_path, SFT_GEN)
  rows = _jsonl(graded)
  rows[0]["pred_answer"] = "51"
  _write_jsonl(graded, rows)
  proc = _build_sft(prompts, gen, graded, tmp_path / "sft.jsonl", "--forbid-groups", f"@{test_manifest}",
                    check=False)
  assert proc.returncode != 0 and "pred_answer is not the answer of the generated response" in proc.stderr


GRPO_GEN = {
    "train-0": ["Answer: 50", "Answer: 40", "Answer: 50.2", "Answer: 7"],    # mixed
    "train-1": ["Answer: 0.4", "Answer: 40%", "Answer: 0.40", "Answer: 40"],  # all correct
    "train-4": ["Answer: yes", "Answer: yes", "Answer: no", "no idea"],      # mixed
}


def test_build_grpo_rollouts_on_finqa(root, tmp_path):
  prompts, gen, graded, test_manifest = _graded_inputs(root, tmp_path, GRPO_GEN)
  out = tmp_path / "roll.jsonl"
  _py("build_grpo_rollouts.py", "--task", "finqa", "--prompts", prompts, "--gen", gen, "--graded", graded,
      "--out", out, "--forbid-groups", f"@{test_manifest}", "--samples", "4")
  rows = _jsonl(out)
  assert {r["group_id"] for r in rows} == {"train-0", "train-4"}
  rewards = {r["id"]: r["reward"] for r in rows}
  assert [rewards[f"train-0#s{k}"] for k in range(4)] == [1.0, 0.0, 1.0, 0.0]
  assert [rewards[f"train-4#s{k}"] for k in range(4)] == [0.0, 0.0, 1.0, 0.0]
  assert all(set(r) == {"id", "messages", "response", "group_id", "reward", "report"} for r in rows)
  m = json.loads((tmp_path / "roll.jsonl.manifest.json").read_text())
  assert m["task"] == "finqa" and m["counts"]["groups_uniform_or_small"] == 1


def test_build_grpo_rollouts_checks_graded_answer_against_response(root, tmp_path):
  prompts, gen, graded, test_manifest = _graded_inputs(root, tmp_path, GRPO_GEN)
  rows = _jsonl(graded)
  rows[1]["pred_answer"] = "41"  # train-0 sample 1 says "Answer: 40"
  _write_jsonl(graded, rows)
  proc = _py("build_grpo_rollouts.py", "--task", "finqa", "--prompts", prompts, "--gen", gen, "--graded", graded,
             "--out", tmp_path / "roll.jsonl", "--forbid-groups", f"@{test_manifest}", "--samples", "4", check=False)
  assert proc.returncode != 0 and "graded pred_answer is not the answer of the generated response" in proc.stderr


def test_builders_refuse_wrong_forbid_files(root, tmp_path):
  prompts, gen, graded, _ = _graded_inputs(root, tmp_path, SFT_GEN)
  for bad, msg in ((root / "test.json", "not a prompts manifest"), (tmp_path / "test_prompts.jsonl", "looks like JSONL")):
    proc = _build_sft(prompts, gen, graded, tmp_path / "sft.jsonl", "--forbid-groups", f"@{bad}", check=False)
    assert proc.returncode != 0 and msg in proc.stderr
    proc = _py("build_grpo_rollouts.py", "--task", "finqa", "--prompts", prompts, "--gen", gen, "--graded", graded,
               "--out", tmp_path / "roll.jsonl", "--forbid-groups", f"@{bad}", "--samples", "4", check=False)
    assert proc.returncode != 0 and msg in proc.stderr


def test_build_grpo_rollouts_refuses_evaluation_reports(root, tmp_path):
  gen = dict(GRPO_GEN, **{"train-2": ["Answer: 100", "Answer: 1", "Answer: 2", "Answer: 3"]})
  prompts, gen, graded, test_manifest = _graded_inputs(root, tmp_path, gen, exclude_test_reports=False)
  proc = _py("build_grpo_rollouts.py", "--task", "finqa", "--prompts", prompts, "--gen", gen, "--graded", graded,
             "--out", tmp_path / "roll.jsonl", "--forbid-groups", f"@{test_manifest}", "--samples", "4", check=False)
  assert proc.returncode != 0 and "evaluation report ACME/2010/page_5.pdf" in proc.stderr


# ---------------------------------------------------------------------------
# leak_audit.py --task finqa
# ---------------------------------------------------------------------------


def _sft_row(root: Path, qid: str, response: str, sample: int = 0, report: str | None = None) -> dict:
  items = {it.id: it for it in finqa.load_items(root, "train")}
  it = items[qid]
  return {"id": f"{qid}#s{sample}", "messages": finqa.messages(it), "response": response,
          "report": report if report is not None else it.report, "sample": sample}


def _audit(root: Path, rows: list[dict], tmp_path: Path) -> tuple[int, dict]:
  sft = _write_jsonl(tmp_path / "sft.jsonl", rows)
  out = tmp_path / "audit.json"
  proc = _py("leak_audit.py", "--task", "finqa", "--root", root, "--sft", sft, "--out", out, check=False)
  report = json.loads(out.read_text())
  assert report["passed"] is (proc.returncode == 0)
  return proc.returncode, report["files"][str(sft)]


def test_audit_clean_rows_pass(root, tmp_path):
  rc, rep = _audit(root, [_sft_row(root, "train-0", "Answer: 50"), _sft_row(root, "train-1", "Answer: 40%", 1),
                          _sft_row(root, "train-4", "Answer: no")], tmp_path)
  assert rc == 0 and rep["rows"] == 3
  assert all(rep[k] == 0 for k in ("not_a_train_question", "eval_report", "exact_question_match", "ngram13_overlap",
                                   "rescore_failed"))


def test_audit_catches_evaluation_report(root, tmp_path):
  rc, rep = _audit(root, [_sft_row(root, "train-0", "Answer: 50"), _sft_row(root, "train-2", "Answer: 100")],
                   tmp_path)
  assert rc == 1 and rep["eval_report"] == 1 and rep["examples"]["eval_report"] == ["train-2#s0"]
  assert rep["rescore_failed"] == 0 and rep["exact_question_match"] == 0


def test_audit_catches_question_overlap(root, tmp_path):
  rc, rep = _audit(root, [_sft_row(root, "train-3", "Answer: 0.1"), _sft_row(root, "train-5", "Answer: 0.2")],
                   tmp_path)
  assert rc == 1 and rep["exact_question_match"] == 1 and rep["examples"]["exact"] == ["train-3#s0"]
  assert rep["ngram13_overlap"] == 2 and rep["eval_report"] == 0 and rep["rescore_failed"] == 0


def test_audit_catches_wrong_answer_and_foreign_ids(root, tmp_path):
  wrong = _sft_row(root, "train-0", "Answer: 70", 1)
  unread = _sft_row(root, "train-1", "Answer: unclear", 2)
  foreign = dict(_sft_row(root, "train-0", "Answer: 50"), id="test-1#s0")
  moved = _sft_row(root, "train-4", "Answer: no", report="GAMMA/2011/page_3.pdf")  # report does not match the id
  rc, rep = _audit(root, [wrong, unread, foreign, moved], tmp_path)
  assert rc == 1 and rep["rescore_failed"] == 3  # wrong, unreadable and the foreign id cannot be re-scored
  assert rep["not_a_train_question"] == 2 and rep["eval_report"] == 0
