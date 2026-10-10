#!/usr/bin/env python3
"""Write chat prompts for a BIRD split as JSONL: one row per question with id, db_id and messages.

Usage: prompts.py --bird-root ~/kd/bird --split dev --out dev_prompts.jsonl [--exclude-dbs a,b]
A manifest beside the output records the prompt-template SHA-256, row count and database list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bird


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--bird-root", type=Path, required=True)
  ap.add_argument("--split", choices=["dev", "train"], required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--exclude-dbs", default="", help="comma-separated db_ids to leave out")
  ap.add_argument("--only-dbs", default="", help="comma-separated db_ids to keep (others dropped)")
  args = ap.parse_args()

  exclude = {d for d in args.exclude_dbs.split(",") if d}
  only = {d for d in args.only_dbs.split(",") if d}
  items = [it for it in bird.load_items(args.bird_root, args.split)
           if it.db_id not in exclude and (not only or it.db_id in only)]
  schemas: dict[str, str] = {}
  with open(args.out, "w") as f:
    for it in items:
      if it.db_id not in schemas:
        schemas[it.db_id] = bird.schema_text(bird.db_path(args.bird_root, args.split, it.db_id))
      row = {"id": it.id, "db_id": it.db_id, "split": args.split, "difficulty": it.difficulty,
             "messages": bird.messages(it, schemas[it.db_id])}
      f.write(json.dumps(row) + "\n")
  manifest = {"split": args.split, "rows": len(items), "dbs": sorted(schemas),
              "prompt_template_sha256": bird.prompt_template_sha256(),
              "schema_chars": {d: len(s) for d, s in sorted(schemas.items())}}
  Path(str(args.out) + ".manifest.json").write_text(json.dumps(manifest, indent=1))
  print(json.dumps({k: manifest[k] for k in ("split", "rows", "prompt_template_sha256")}))


if __name__ == "__main__":
  main()
