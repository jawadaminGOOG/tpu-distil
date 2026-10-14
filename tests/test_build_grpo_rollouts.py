"""build_grpo_rollouts.py: mixed groups only, correct rewards, drops, and a round trip through the trainer.

The round trip loads the builder's output with the trainer's GRPO record loader, computes the trainer's
group advantages (correct samples positive, wrong samples negative) and renders each record with the
real Qwen3-1.7B tokenizer: the context must be the serving prompt and the trained tokens the response
plus <|im_end|>.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DISTIL = REPO / "distil"
sys.path.insert(0, str(DISTIL))
sys.path.insert(0, str(REPO / "trainer"))

import bird  # noqa: E402

SYSTEM = bird.SYSTEM_PROMPT


def _resp(sql, prefix="Reasoning."):
  return f"{prefix}\n```sql\n{sql}\n```"


# qid -> [(content, correct, finish_reason)], 4 samples per question
QUESTIONS = {
    "train-1": [(_resp("SELECT 1"), True, "stop"), (_resp("SELECT 2"), False, "stop"),
                (_resp("SELECT 3"), False, "stop"), ("</think>\n\n" + _resp("SELECT 1"), True, "stop")],
    "train-2": [(_resp("SELECT 1"), True, "stop")] * 4,                                   # all correct
    "train-3": [(_resp("SELECT 9"), False, "stop")] * 4,                                  # all wrong
    "train-4": [(_resp("SELECT 1"), True, "length"), (_resp("SELECT 2"), False, "stop"),
                ("<think>x</think>" + _resp("SELECT 1"), True, "stop"), (_resp("SELECT 1 "), True, "stop")],
    "train-5": [(_resp("SELECT 1"), True, "stop"), (_resp("SELECT 2"), False, "stop")],     # wrong size
    # finished samples all correct, one cut off: uniform unless truncated samples are kept (as wrong)
    "train-6": [(_resp("SELECT 1"), True, "stop")] * 3 + [(_resp("SELECT 1") + " and so on", True, "length")],
}


def _write(tmp_path):
  p, g, gr = tmp_path / "p.jsonl", tmp_path / "g.jsonl", tmp_path / "gr.jsonl"
  with open(p, "w") as fp, open(g, "w") as fg, open(gr, "w") as fr:
    for qid, samples in QUESTIONS.items():
      msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question {qid}?"}]
      fp.write(json.dumps({"id": qid, "db_id": "shop", "messages": msgs}) + "\n")
      fg.write(json.dumps({"id": qid, "choices": [{"content": c, "finish_reason": f} for c, _, f in samples]}) + "\n")
      for k, (c, ok, _) in enumerate(samples):
        fr.write(json.dumps({"id": qid, "sample": k, "pred_sql": bird.extract_sql(c), "correct": ok,
                             "error": ""}) + "\n")
  return p, g, gr


def _run(tmp_path, *extra):
  p, g, gr = _write(tmp_path)
  out = tmp_path / "roll.jsonl"
  proc = subprocess.run([sys.executable, str(DISTIL / "build_grpo_rollouts.py"), "--prompts", str(p), "--gen", str(g),
                         "--graded", str(gr), "--out", str(out), "--forbid-dbs", "eval_db", "--samples", "4", *extra],
                        capture_output=True, text=True)
  return proc, out


def test_mixed_groups_rewards_and_drops(tmp_path):
  proc, out = _run(tmp_path)
  assert proc.returncode == 0, proc.stderr
  rows = [json.loads(l) for l in out.read_text().splitlines()]
  assert {r["group_id"] for r in rows} == {"train-1", "train-4"}
  r1 = {r["id"]: r for r in rows if r["group_id"] == "train-1"}
  assert [r1[f"train-1#s{k}"]["reward"] for k in range(4)] == [1.0, 0.0, 0.0, 1.0]
  assert r1["train-1#s3"]["response"].startswith("Reasoning.")  # empty think prefix stripped
  r4 = {r["id"]: r["reward"] for r in rows if r["group_id"] == "train-4"}
  assert r4 == {"train-4#s1": 0.0, "train-4#s3": 1.0}  # s0 truncated and s2 think markup were dropped
  m = json.loads((tmp_path / "roll.jsonl.manifest.json").read_text())["counts"]
  assert m["dropped_truncated"] == 2 and m["dropped_think_markup"] == 1 and m["groups_wrong_size"] == 1
  assert all(set(r) == {"id", "messages", "response", "group_id", "reward", "db_id"} for r in rows)
  assert all(r["messages"][-1] == {"role": "user", "content": f"Question {r['group_id']}?"} for r in rows)


def test_keep_truncated_scores_cut_off_samples_as_wrong(tmp_path):
  proc, out = _run(tmp_path, "--keep-truncated")
  assert proc.returncode == 0, proc.stderr
  rows = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
  assert {r["group_id"] for r in rows.values()} == {"train-1", "train-4", "train-6"}
  # train-4 s0 holds a correct answer but was cut off: kept with reward 0 and marked
  assert rows["train-4#s0"]["reward"] == 0.0 and rows["train-4#s0"]["truncated"] is True
  assert [rows[f"train-6#s{k}"]["reward"] for k in range(4)] == [1.0, 1.0, 1.0, 0.0]
  assert rows["train-6#s3"]["truncated"] is True and "truncated" not in rows["train-6#s0"]
  assert all("truncated" not in r for i, r in rows.items() if i.startswith("train-1"))
  m = json.loads((tmp_path / "roll.jsonl.manifest.json").read_text())
  assert m["counts"]["kept_truncated_reward_0"] == 2 and "dropped_truncated" not in m["counts"]
  assert m["args"]["keep_truncated"] is True


def test_max_groups_is_seeded(tmp_path):
  proc, out = _run(tmp_path, "--max-groups", "1", "--seed", "3")
  assert proc.returncode == 0, proc.stderr
  first = out.read_text()
  assert len({json.loads(l)["group_id"] for l in first.splitlines()}) == 1
  _run(tmp_path, "--max-groups", "1", "--seed", "3")
  assert out.read_text() == first


def test_round_trip_through_trainer(tmp_path):
  transformers = pytest.importorskip("transformers")
  hub = pytest.importorskip("huggingface_hub")
  dc = pytest.importorskip("data")
  tc = pytest.importorskip("trainer_core")
  from test_template_equivalence import check_equivalence  # pylint: disable=import-outside-toplevel
  try:
    d = hub.snapshot_download("Qwen/Qwen3-1.7B", allow_patterns=["*.json", "*.txt", "*.jinja"], local_files_only=True)
  except Exception:  # pylint: disable=broad-except
    pytest.skip("Qwen3-1.7B tokenizer files are not in the local Hugging Face cache")
  tok = transformers.AutoTokenizer.from_pretrained(d)
  proc, out = _run(tmp_path)
  assert proc.returncode == 0, proc.stderr
  records = tc.load_grpo_records(str(out))
  samples = {rec.id: (rec.group_id, rec.reward) for rec in records}
  adv, stats = tc.group_advantages(samples)
  assert stats["n_groups_zero_variance"] == 0
  for rec in records:
    assert (adv[rec.id] > 0) == (rec.reward == 1.0)
  rows = {json.loads(l)["id"]: json.loads(l) for l in out.read_text().splitlines()}
  for rec in records:
    rendered = dc.render_example(rec, tok, max_seq_len=2048)
    check_equivalence(tok, {"enable_thinking": False}, rendered, rows[rec.id])
