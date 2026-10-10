#!/usr/bin/env python3
"""Turn sampled, graded student responses into GRPO records for the trainer.

One record per graded (question, sample), in the trainer/data.py Record layout: id "<id>#s<sample>",
messages = the single-turn prompt chat, response = the sampled response (trained verbatim), reward 1.0
if the SQL returned the gold rows and 0.0 otherwise, `group_id` = question id, db_id. Responses cut off by the length limit or holding thinking markup are dropped and counted: the
trainer would otherwise train an end-of-turn token or markup the policy did not emit. Groups whose
rewards are all equal carry no group-relative advantage and are left out. `--max-groups` draws a
seeded subset of the mixed groups.

Usage: build_grpo_rollouts.py --prompts train_prompts.jsonl --gen gen.jsonl --graded graded.jsonl \
         --out rollouts.jsonl --forbid-dbs <eval dbs> [--samples 8] [--max-groups 2048 --seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bird  # noqa: E402
import build_sft  # noqa: E402


def build(prompts: dict[str, dict], gens: dict[str, dict], graded: list[dict], *, samples: int,
          max_groups: int | None, seed: int) -> tuple[list[dict], dict]:
  counts = Counter()
  by_q: dict[str, dict[int, dict]] = defaultdict(dict)
  for r in graded:
    by_q[r["id"]][int(r["sample"])] = r
  groups = []
  for qid in sorted(by_q):
    if qid not in prompts or qid not in gens:
      raise ValueError(f"graded id {qid} missing from --prompts or --gen")
    choices = gens[qid]["choices"]
    if sorted(by_q[qid]) != list(range(samples)) or len(choices) != samples:
      counts["groups_wrong_size"] += 1
      continue
    members = []
    for k in range(samples):
      g, choice = by_q[qid][k], choices[k]
      content = choice.get("content") or ""
      if bird.extract_sql(content) != g["pred_sql"]:
        raise ValueError(f"({qid}, {k}): graded pred_sql is not the SQL of the generated response")
      if choice.get("finish_reason") == "length":
        counts["dropped_truncated"] += 1
        continue
      text = build_sft.normalize_response(content)
      if any(t in text for t in build_sft.THINK_MARKUP):
        counts["dropped_think_markup"] += 1
        continue
      members.append((k, text, 1.0 if g["correct"] is True else 0.0))
    rewards = {m[2] for m in members}
    if len(members) < 2 or len(rewards) < 2:
      counts["groups_uniform_or_small"] += 1
      continue
    groups.append((qid, members))
  counts["groups_mixed"] = len(groups)
  if max_groups is not None and len(groups) > max_groups:
    groups = sorted(random.Random(seed).sample(groups, max_groups), key=lambda g: g[0])
  rows = []
  for qid, members in groups:
    p = prompts[qid]
    for k, text, reward in members:
      tid = f"{qid}#s{k}"
      rows.append({"id": tid, "messages": p["messages"], "response": text, "group_id": qid, "reward": reward,
                   "db_id": p["db_id"]})
  counts["groups_written"] = len(groups)
  counts["records_written"] = len(rows)
  stats = {"counts": dict(sorted(counts.items())),
           "reward_mean": round(sum(r["reward"] for r in rows) / max(len(rows), 1), 4)}
  return rows, stats


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  ap.add_argument("--prompts", type=Path, required=True)
  ap.add_argument("--gen", type=Path, required=True)
  ap.add_argument("--graded", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--forbid-dbs", required=True, help="comma-separated evaluation db_ids")
  ap.add_argument("--samples", type=int, default=8)
  ap.add_argument("--max-groups", type=int, default=None)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  forbid = {d for d in args.forbid_dbs.split(",") if d}
  if not forbid:
    ap.error("--forbid-dbs must name the evaluation databases")
  rows, stats = build(build_sft.load_prompts(args.prompts, forbid), build_sft.load_generations(args.gen),
                      list(build_sft.read_jsonl(args.graded)), samples=args.samples,
                      max_groups=args.max_groups, seed=args.seed)
  with open(args.out, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  stats.update({"args": {"samples": args.samples, "max_groups": args.max_groups, "seed": args.seed,
                         "forbid_dbs": sorted(forbid)},
                "sha256": {"prompts": build_sft.sha256_file(args.prompts), "gen": build_sft.sha256_file(args.gen),
                           "graded": build_sft.sha256_file(args.graded), "out": build_sft.sha256_file(args.out)}})
  Path(str(args.out) + ".manifest.json").write_text(json.dumps(stats, indent=1))
  print(json.dumps(stats["counts"]))
  return 0


if __name__ == "__main__":
  sys.exit(main())
