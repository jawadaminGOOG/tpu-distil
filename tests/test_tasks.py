"""Task registry and the BIRD defaults: `--task` defaults to bird and the BIRD option names keep working."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

DISTIL = Path(__file__).resolve().parents[1] / "distil"
sys.path.insert(0, str(DISTIL))
import bird  # noqa: E402
import build_sft  # noqa: E402
import tasks  # noqa: E402


def test_registry():
  assert tasks.NAMES == ("bird", "finqa")
  assert tasks.get_task("bird") is bird  # `import bird` is the task module itself, not a copy
  for name in tasks.NAMES:
    t = tasks.get_task(name)
    for attr in ("NAME", "LABEL", "SPLITS", "EVAL_SPLIT", "FULL_COVERAGE_SPLITS", "GROUP_KEY", "GROUP_SHORT",
                 "GROUP_NOUN", "PRED_KEY", "ANSWER_SHORT", "ANSWER_NOUN", "Item", "load_items", "messages",
                 "prompt_template_sha256", "prompt_rows", "extract_answer", "normalize_answer", "control_samples",
                 "grade", "valid_group"):
      assert hasattr(t, attr), (name, attr)
    assert t.NAME == name and t.EVAL_SPLIT in t.SPLITS and "train" in t.SPLITS
  with pytest.raises(ValueError):
    tasks.get_task("nope")


def test_bird_item_interface():
  it = bird.Item(id="dev-0", db_id="shop", question="q", evidence="", gold_sql="SELECT 1", difficulty="simple")
  assert it.group == "shop" and it.gold == "SELECT 1"
  assert bird.extract_answer is bird.extract_sql and bird.normalize_answer is bird.normalize_sql
  assert build_sft.normalize_sql is bird.normalize_sql


@pytest.fixture()
def bird_root(tmp_path: Path) -> Path:
  for db in ("shop", "zoo"):
    d = tmp_path / "train_databases" / db
    d.mkdir(parents=True)
    conn = sqlite3.connect(d / f"{db}.sqlite")
    conn.executescript("CREATE TABLE t (id INTEGER, name TEXT); INSERT INTO t VALUES (1, 'a'), (2, 'b');")
    conn.commit()
    conn.close()
  (tmp_path / "train.json").write_text(json.dumps([
      {"db_id": "shop", "question": "How many rows?", "evidence": "", "SQL": "SELECT COUNT(*) FROM t"},
      {"db_id": "zoo", "question": "Names?", "evidence": "", "SQL": "SELECT name FROM t"},
      {"db_id": "shop", "question": "Ids?", "evidence": "", "SQL": "SELECT id FROM t"},
  ]))
  return tmp_path


def test_bird_prompts_default_task_and_old_option_names(bird_root, tmp_path):
  old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
  subprocess.run([sys.executable, str(DISTIL / "prompts.py"), "--bird-root", str(bird_root), "--split", "train",
                  "--out", str(old), "--exclude-dbs", "zoo"], check=True, capture_output=True)
  subprocess.run([sys.executable, str(DISTIL / "prompts.py"), "--task", "bird", "--root", str(bird_root), "--split",
                  "train", "--out", str(new), "--exclude-groups", "zoo"], check=True, capture_output=True)
  assert old.read_bytes() == new.read_bytes()
  rows = [json.loads(line) for line in old.read_text().splitlines()]
  items = {it.id: it for it in bird.load_items(bird_root, "train")}
  schema = bird.schema_text(bird.db_path(bird_root, "train", "shop"))
  assert [r["id"] for r in rows] == ["train-0", "train-2"]
  for r in rows:
    assert set(r) == {"id", "db_id", "split", "difficulty", "messages"}
    assert r["messages"] == bird.messages(items[r["id"]], schema)
  m = json.loads(Path(str(old) + ".manifest.json").read_text())
  assert m["task"] == "bird" and m["dbs"] == ["shop"] and m["prompt_template_sha256"] == bird.prompt_template_sha256()
  only = subprocess.run([sys.executable, str(DISTIL / "prompts.py"), "--bird-root", str(bird_root), "--split", "train",
                         "--out", str(tmp_path / "only.jsonl"), "--only-dbs", "zoo"], check=True, capture_output=True)
  assert json.loads(only.stdout)["rows"] == 1
  bad = subprocess.run([sys.executable, str(DISTIL / "prompts.py"), "--bird-root", str(bird_root), "--split", "test",
                        "--out", str(tmp_path / "x.jsonl")], capture_output=True, text=True)
  assert bad.returncode != 0 and "--split must be one of dev, train" in bad.stderr


def test_parse_groups_reads_lists_manifests_and_files(tmp_path):
  assert build_sft.parse_groups("a,b,,c") == {"a", "b", "c"}
  manifest = tmp_path / "dev_prompts.jsonl.manifest.json"
  manifest.write_text(json.dumps({"dbs": ["x", "y"], "rows": 3}))
  assert build_sft.parse_groups(f"@{manifest}") == {"x", "y"}
  with pytest.raises(ValueError, match="no 'reports' list"):
    build_sft.parse_groups(f"@{manifest}", tasks.get_task("finqa"))
  finqa = tasks.get_task("finqa")
  listing = tmp_path / "groups.txt"
  listing.write_text("A/2010/page_1.pdf\n\nBRK.B/2011/page_22.pdf\n")
  assert build_sft.parse_groups(f"@{listing}", finqa) == {"A/2010/page_1.pdf", "BRK.B/2011/page_22.pdf"}
  with pytest.raises(ValueError, match="no entry has the form of a BIRD database"):
    build_sft.parse_groups(f"@{listing}")  # report pages are not BIRD database ids
  dbs = tmp_path / "dbs.txt"
  dbs.write_text("california_schools\nformula_1\n")
  assert build_sft.parse_groups(f"@{dbs}") == {"california_schools", "formula_1"}
  raw = tmp_path / "dev.json"  # a dataset file, not a manifest
  raw.write_text(json.dumps([{"db_id": "x", "question": "q"}]))
  with pytest.raises(ValueError, match="not a prompts manifest"):
    build_sft.parse_groups(f"@{raw}")
  rows = tmp_path / "dev_prompts.jsonl"
  rows.write_text(json.dumps({"id": "dev-0", "db_id": "x"}) + "\n" + json.dumps({"id": "dev-1", "db_id": "y"}) + "\n")
  with pytest.raises(ValueError, match="looks like JSONL"):
    build_sft.parse_groups(f"@{rows}")


def test_valid_group_forms():
  bird_t, finqa_t = tasks.get_task("bird"), tasks.get_task("finqa")
  assert bird_t.valid_group("european_football_2") and not bird_t.valid_group("V/2008/page_17.pdf")
  assert finqa_t.valid_group("V/2008/page_17.pdf") and finqa_t.valid_group("BRK.B/2019/page_105.pdf")
  for bad in ("european_football_2", "V/2008/page_17", "V/08/page_1.pdf", '{"id": "train-0"}', ""):
    assert not finqa_t.valid_group(bad)
