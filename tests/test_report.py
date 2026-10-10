"""report.py: accuracy, cost per correct answer, error-free best level, SKU lookup and gate, on synthetic files."""
import json
import subprocess
import sys
from pathlib import Path

DISTIL = Path(__file__).resolve().parents[1] / "distil"


def _graded(path: Path, correct: list[bool]) -> Path:
  path.write_text("".join(json.dumps({"id": f"dev-{i}", "sample": 0, "correct": c}) + "\n"
                          for i, c in enumerate(correct)))
  return path


def _bench(path: Path, hw: str, levels: list[tuple[int, float, int]]) -> Path:
  path.write_text(json.dumps({"hardware": hw, "model": "m", "levels": [
      {"concurrency": c, "req_per_s": r, "errors": e, "records": []} for c, r, e in levels]}))
  return path


def _sku(sku_id: str, usd: float) -> dict:
  units, nanos = int(usd), round((usd - int(usd)) * 1e9)
  return {"skuId": sku_id, "description": f"desc {sku_id}", "serviceRegions": ["r"], "category": {"usageType": "OnDemand"},
          "pricingInfo": [{"effectiveTime": "t", "pricingExpression": {"tieredRates": [
              {"unitPrice": {"units": str(units), "nanos": nanos}}]}}]}


def test_report_numbers_and_gate(tmp_path):
  n = 200
  base = _graded(tmp_path / "base.jsonl", [i < 60 for i in range(n)])            # 30%
  base_b = _graded(tmp_path / "base_b.jsonl", [i < 61 for i in range(n)])        # 30.5%
  kd = _graded(tmp_path / "kd.jsonl", [i < 100 for i in range(n)])               # 50%
  selfd = _graded(tmp_path / "self.jsonl", [i < 70 for i in range(n)])           # 35%
  teacher = _graded(tmp_path / "teacher.jsonl", [i < 120 for i in range(n)])     # 60%
  sb = _bench(tmp_path / "s.json", "1 chip", [(8, 5.0, 0), (64, 10.0, 0), (256, 12.0, 3)])  # best error-free 10
  tb = _bench(tmp_path / "t.json", "8 gpu", [(64, 8.0, 0)])
  prices = tmp_path / "prices"
  prices.mkdir()
  (prices / "p.json").write_text(json.dumps({"skus": [_sku("S", 3.6), _sku("T", 4.5)]}))
  out, md = tmp_path / "r.json", tmp_path / "r.md"
  subprocess.run([sys.executable, str(DISTIL / "report.py"), "--arm", f"base={base}", "--arm", f"base_b={base_b}",
                  "--arm", f"kd={kd}", "--arm", f"self={selfd}", "--arm", f"teacher={teacher}",
                  "--bench", f"student={sb}", "--bench", f"teacher={tb}", "--prices-dir", str(prices),
                  "--student-sku", "S:1", "--teacher-sku", "T:8", "--resamples", "500",
                  "--out", str(out), "--md", str(md)], check=True, capture_output=True)
  r = json.loads(out.read_text())
  assert r["accuracy_dev"] == {"base": 30.0, "base_b": 30.5, "kd": 50.0, "self": 35.0, "teacher": 60.0}
  assert r["paired_dev"]["kd-base"]["delta"] == 20.0 and r["paired_dev"]["kd-base"]["b_better"] == 40
  assert r["best_throughput"]["student"]["req_per_s"] == 10.0  # the 12 req/s level had errors
  # student: 3.6 $/h / (10 req/s * 3600) = 1e-4 $/question -> 0.1 $/1k questions -> 0.2 $/1k correct at 50%
  assert r["costs"]["kd_student"] == {"usd_per_1k_questions": 0.1, "usd_per_1k_correct": 0.2}
  # teacher: 36 $/h / (8 * 3600) = 1.25e-3 -> 1.25 $/1k questions -> 2.0833 $/1k correct at 60%
  assert r["costs"]["teacher"]["usd_per_1k_correct"] == 2.0833
  assert r["teacher_over_kd_cost_per_correct"] == round(2.0833 / 0.2, 2)
  assert r["gap_closed_fraction"] == round(20 / 30, 4)
  f = r["sensitivity"]["factors"]
  assert abs(f["hardware_price"] * f["throughput"] * f["accuracy"] - r["teacher_over_kd_cost_per_correct"]) < 0.01
  assert abs(r["sensitivity"]["teacher_throughput_multiple_for_ratio_10"] - r["teacher_over_kd_cost_per_correct"] / 10) < 0.01
  assert all(r["gate"].values())


def test_missing_sku_is_an_error(tmp_path):
  g = _graded(tmp_path / "g.jsonl", [True, False])
  b = _bench(tmp_path / "b.json", "x", [(1, 1.0, 0)])
  (tmp_path / "prices").mkdir()
  (tmp_path / "prices" / "p.json").write_text(json.dumps({"skus": []}))
  proc = subprocess.run([sys.executable, str(DISTIL / "report.py"), *sum([["--arm", f"{a}={g}"] for a in
                         ("base", "base_b", "kd", "self", "teacher")], []), "--bench", f"student={b}",
                         "--bench", f"teacher={b}", "--prices-dir", str(tmp_path / "prices"), "--student-sku", "S:1",
                         "--teacher-sku", "T:8", "--out", str(tmp_path / "o.json"), "--md", str(tmp_path / "o.md")],
                        capture_output=True, text=True)
  assert proc.returncode != 0 and "SKU S not found" in proc.stderr
