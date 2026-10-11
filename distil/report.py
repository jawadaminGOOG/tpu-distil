#!/usr/bin/env python3
"""Regenerate every reported number of the distillation experiment from raw files.

Inputs are the graded files (grade.py), the serving benchmarks (bench_serving.py) and the saved
Cloud Billing Catalog SKU pages. Accuracy and paired comparisons are recomputed here from
per-question correctness, not read from summaries. Prices are looked up by SKU id in the catalog
pages; a missing SKU is an error. Cost per 1,000 correct answers =
hourly price / (best measured requests per second x 3600) x 1000 / accuracy, where "best" is the
highest error-free request rate over the concurrency ladder.

Usage:
  report.py --arm base=graded_base_a.jsonl --arm base_b=... --arm kd=... --arm self=... --arm teacher=... \
    --holdout-arm base=... --holdout-arm kd=... --holdout-arm self=... --holdout-arm base_b=... \
    --bench student=bench_kd_1chip.json [--bench base=bench_base_1chip.json] --bench teacher=bench_teacher.json --bench teacher=bench_teacher_high.json \
    --prices-dir prices/ --student-sku C4E3-1652-BF17:1 --teacher-sku 23B6-727C-4CFC:8 \
    [--teacher-sku-alt 2CF3-4C76-D5F7:8] --out report.json --md report.md
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import compare


def kv(items: list[str]) -> list[tuple[str, Path]]:
  return [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in items]


def sku_price(prices_dir: Path, sku_spec: str) -> dict:
  sku_id, count = sku_spec.split(":")
  for f in sorted(glob.glob(str(prices_dir / "*.json"))):
    for s in json.load(open(f)).get("skus", []):
      if s["skuId"] == sku_id:
        pi = s["pricingInfo"][0]
        r = pi["pricingExpression"]["tieredRates"][-1]["unitPrice"]
        unit = float(r["units"]) + r["nanos"] / 1e9
        return {"sku": sku_id, "description": s["description"], "regions": s.get("serviceRegions"),
                "usage_type": s["category"].get("usageType"), "unit_usd_per_hour": unit, "count": int(count),
                "usd_per_hour": unit * int(count), "effective_time": pi.get("effectiveTime"),
                "source_file": Path(f).name}
  raise SystemExit(f"SKU {sku_id} not found in {prices_dir}")


def best_level(paths: list[Path]) -> dict:
  best = None
  for p in paths:
    b = json.load(open(p))
    for lv in b["levels"]:
      if lv["errors"] == 0 and (best is None or lv["req_per_s"] > best["req_per_s"]):
        best = {k: v for k, v in lv.items() if k != "records"}
        best.update({"file": p.name, "hardware": b.get("hardware"), "model": b.get("model")})
  return best


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", action="append", required=True)
  ap.add_argument("--holdout-arm", action="append", default=[])
  ap.add_argument("--bench", action="append", required=True)
  ap.add_argument("--prices-dir", type=Path, required=True)
  ap.add_argument("--student-sku", required=True)
  ap.add_argument("--teacher-sku", required=True)
  ap.add_argument("--teacher-sku-alt", default=None)
  ap.add_argument("--resamples", type=int, default=10_000)
  ap.add_argument("--eval-name", default="BIRD dev", help="evaluation set name for the report headings")
  ap.add_argument("--holdout-name", default="train-holdout databases", help="holdout set name for the report headings")
  ap.add_argument("--student-arm", default="kd", help="arm whose accuracy prices the student (with the 'student' bench)")
  ap.add_argument("--pair", action="append", default=[],
                  help="extra paired comparison B:A (arm names), reported on dev and, when both arms exist, holdout")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--md", type=Path, required=True)
  args = ap.parse_args()

  dev = {name: compare.per_question(p, False) for name, p in kv(args.arm)}
  hold = {name: compare.per_question(p, False) for name, p in kv(args.holdout_arm)}
  acc = {n: round(100 * sum(v.values()) / len(v), 2) for n, v in dev.items()}
  pairs = [("kd", "base"), ("self", "base"), ("kd", "self"), ("base_b", "base"), ("teacher", "kd"),
           ("teacher", "base")] + [tuple(p.split(":", 1)) for p in args.pair]
  cmp_dev = {f"{b}-{a}": compare.compare(dev[a], dev[b], args.resamples) for b, a in pairs if a in dev and b in dev}
  cmp_hold = {f"{b}-{a}": compare.compare(hold[a], hold[b], args.resamples) for b, a in pairs
              if a in hold and b in hold}

  benches: dict[str, list[Path]] = {}
  for name, p in kv(args.bench):
    benches.setdefault(name, []).append(p)
  best = {n: best_level(ps) for n, ps in benches.items()}
  price = {"student": sku_price(args.prices_dir, args.student_sku),
           "teacher": sku_price(args.prices_dir, args.teacher_sku)}
  if args.teacher_sku_alt:
    price["teacher_alt"] = sku_price(args.prices_dir, args.teacher_sku_alt)

  def cost(hw: str, bench: str, arm: str) -> dict:
    per_q = price[hw]["usd_per_hour"] / (best[bench]["req_per_s"] * 3600)
    return {"usd_per_1k_questions": round(1000 * per_q, 4),
            "usd_per_1k_correct": round(1000 * per_q / (acc[arm] / 100), 4)}

  base_bench = "base" if "base" in best else "student"
  costs = {"kd_student": cost("student", "student", args.student_arm), "base_student": cost("student", base_bench, "base"),
           "teacher": cost("teacher", "teacher", "teacher")}
  if "teacher_alt" in price:
    costs["teacher_alt_price"] = cost("teacher_alt", "teacher", "teacher")
  ratio = costs["teacher"]["usd_per_1k_correct"] / costs["kd_student"]["usd_per_1k_correct"]
  # ratio = (teacher $/h / student $/h) x (student req/s / teacher req/s) x (KD accuracy / teacher accuracy)
  factors = {"hardware_price": price["teacher"]["usd_per_hour"] / price["student"]["usd_per_hour"],
             "throughput": best["student"]["req_per_s"] / best["teacher"]["req_per_s"],
             "accuracy": acc[args.student_arm] / acc["teacher"]}
  sensitivity = {"factors": {k: round(v, 4) for k, v in factors.items()},
                 "teacher_throughput_multiple_for_ratio_10": round(ratio / 10, 3),
                 "teacher_throughput_multiple_for_parity": round(ratio, 3)}

  gap = acc["teacher"] - acc["base"]
  clauses = {
      "1_kd_minus_base_ge_5_lb_gt_0": ("kd-base", lambda c: c["delta"] >= 5.0 and c["ci95"][0] > 0),
      "2_kd_minus_self_lb_gt_0": ("kd-self", lambda c: c["ci95"][0] > 0),
      "3_base_replicate_abs_le_1_5": ("base_b-base", lambda c: abs(c["delta"]) <= 1.5),
  }
  gate = {k: test(cmp_dev[key]) for k, (key, test) in clauses.items() if key in cmp_dev}
  gate["4_cost_per_correct_ratio_ge_10"] = ratio >= 10.0
  report = {"accuracy_dev": acc, "paired_dev": cmp_dev, "paired_holdout": cmp_hold,
            "gap_closed_fraction": round((acc[args.student_arm] - acc["base"]) / gap, 4) if gap else None,
            "best_throughput": best, "prices": price, "costs": costs,
            "teacher_over_kd_cost_per_correct": round(ratio, 2), "sensitivity": sensitivity, "gate": gate,
            "inputs": {"arms": {n: str(p) for n, p in kv(args.arm)},
                       "holdout_arms": {n: str(p) for n, p in kv(args.holdout_arm)},
                       "benches": {n: [str(p) for p in ps] for n, ps in benches.items()}}}
  args.out.write_text(json.dumps(report, indent=1))

  def row(k: str, c: dict) -> str:
    return f"| {k} | {c['b']:.1f} vs {c['a']:.1f} | {c['delta']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}] | {c['b_better']} / {c['a_better']} |"

  n_eval = len(next(iter(dev.values())))
  md = [f"## Accuracy ({args.eval_name}, {n_eval:,} questions, greedy)", "",
        "| Arm | Accuracy % |", "|---|---|"] + [f"| {n} | {a:.1f} |" for n, a in acc.items()]
  md += ["", f"## Paired comparisons ({args.eval_name})", "", "| B − A | B vs A | Δ [95% CI] | B better / A better |",
         "|---|---|---|---|"] + [row(k, c) for k, c in cmp_dev.items()]
  if cmp_hold:
    md += ["", f"## Paired comparisons ({args.holdout_name})", "", "| B − A | B vs A | Δ [95% CI] | B better / A better |",
           "|---|---|---|---|"] + [row(k, c) for k, c in cmp_hold.items()]
  md += ["", "## Serving cost", "", "| | Hardware | $/h (SKU) | Best req/s (concurrency) | $ / 1k questions | $ / 1k correct |",
         "|---|---|---|---|---|---|"]
  for label, hw, bench in ((f"Student ({args.student_arm})", "student", "student"), ("Base student", "student", base_bench),
                           ("Teacher", "teacher", "teacher")):
    key = {"Base student": "base_student", "Teacher": "teacher"}.get(label, "kd_student")
    md.append(f"| {label} | {best[bench]['hardware']} | {price[hw]['usd_per_hour']:.2f} ({price[hw]['sku']}) | "
              f"{best[bench]['req_per_s']:.2f} ({best[bench]['concurrency']}) | {costs[key]['usd_per_1k_questions']:.3f} | "
              f"{costs[key]['usd_per_1k_correct']:.3f} |")
  f = sensitivity["factors"]
  md += ["", f"Teacher / student ({args.student_arm}) cost per correct answer: **{ratio:.1f}×** = hardware price {f['hardware_price']:.2f}× "
         f"× throughput {f['throughput']:.2f}× × accuracy {f['accuracy']:.2f}×. The ratio falls to 10× if the teacher "
         f"served {sensitivity['teacher_throughput_multiple_for_ratio_10']:.2f}× faster, and to parity at "
         f"{sensitivity['teacher_throughput_multiple_for_parity']:.1f}× faster.",
         f"Gap closed: {100 * report['gap_closed_fraction']:.0f}% of teacher − base.", "",
         "## Gate", ""] + [f"- {k}: {'GREEN' if v else 'RED'}" for k, v in gate.items()]
  args.md.write_text("\n".join(md) + "\n")
  print("\n".join(md))


if __name__ == "__main__":
  main()
