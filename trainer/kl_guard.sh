#!/usr/bin/env bash
# Stop a GRPO training container when any optimizer step's KL exceeds a threshold.
# Usage: kl_guard.sh <metrics_jsonl> <container_name> [max_kl=0.2] [poll_sec=30]
# Reads the trainer's GRPO metrics JSONL (metrics/grpo_iter_metrics.jsonl); exits when the container is gone.
set -uo pipefail
METRICS="$1"; NAME="$2"; MAX_KL="${3:-0.2}"; POLL="${4:-30}"
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
while docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; do
  if [ -f "$METRICS" ]; then
    BAD=$(python3 - "$METRICS" "$MAX_KL" <<'EOF'
import json, math, sys
path, lim = sys.argv[1], float(sys.argv[2])
for line in open(path):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    kl = float(r.get("kl", 0.0)) if r.get("event") == "optimizer_step" else 0.0
    if not math.isfinite(kl) or kl > lim:
        print(f"step {r.get('step')} kl {r.get('kl')}")
        break
EOF
)
    if [ -n "$BAD" ]; then
      echo "$(date -u +%FT%TZ) KL guard: $BAD > $MAX_KL; stopping $NAME"
      docker stop "$NAME"
      exit 3
    fi
  fi
  sleep "$POLL"
done
echo "$(date -u +%FT%TZ) $NAME finished; KL stayed <= $MAX_KL"
