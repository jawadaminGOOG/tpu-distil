#!/usr/bin/env bash
# Run the student trainer (trainer/train.py; Qwen3 or Gemma 4, from the checkpoint's config.json) on a single-host TPU VM (all 8 chips of a
# v6e-8) in the isolated docker daemon.
# Usage (on the TPU VM):
#   train_tpu_vm.sh probe <run_id> [extra probe_memory args...]
#   train_tpu_vm.sh sft   <run_id> <data_jsonl> [extra sft args...]                (from BASE; exports merged)
#   train_tpu_vm.sh grpo  <run_id> <policy_hf_dir> <rollouts_jsonl> <iteration> [extra grpo_iter args...]
#   train_tpu_vm.sh merge <run_id> <policy_hf_dir> <ckpt_root> <tag> [extra merge args...]
#     (export a LoRA checkpoint, e.g. a grpo epoch checkpoint on pi_k with tag iter_<k+1>, to
#      $GCS_ROOT/<run_id>/grpo/<tag>; RANK must match the run that wrote it)
# The serving replicas must be stopped first (they hold the chips).
# Environment:
#   KD_DATA_DIR  data disk with models/, train/, gcs/ and the compilation cache (default /mnt/data)
#   DOCKER_HOST  docker daemon socket (default unix:///var/run/docker.sock)
#   CODE         checkout of this repository on the VM (default $KD_DATA_DIR/code)
#   MODEL        student name under $KD_DATA_DIR/models (default Qwen3-1.7B)
#   BASE         base HF dir (default $KD_DATA_DIR/models/$MODEL)
#   REF          GRPO KL reference HF dir; required for grpo. REF equal to BASE is refused unless
#                ALLOW_BASE_REF=1 (the reference is normally the SFT-merged policy, not the base)
#   EVAL_TASKS   evaluation ids JSON; required for sft and grpo (the trainer refuses overlapping ids)
#   GCS_ROOT     default gs://ckpt-jawadamin-us-east5-0926/$MODEL
#   FSDP/TP      default 8/1;  REMAT (Tunix remat) default DECODER
#   MICRO_BATCH  default 8;  MAX_SEQ_LEN default 8192
#   RANK         LoRA rank, multiple of 256; default 256; alpha = 2 x rank
#   IMAGE        trainer image
set -euo pipefail
PHASE="$1"; RUN_ID="$2"; shift 2
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
KD_DATA_DIR="${KD_DATA_DIR:-/mnt/data}"
IMAGE="${IMAGE:-asia-northeast1-docker.pkg.dev/northam-ce-mlai-tpu/jawadamin-dsv4/trainer-jawadamin-us-south1-0926:8da2b99b5514}"
CODE="${CODE:-$KD_DATA_DIR/code}"
MODEL="${MODEL:-Qwen3-1.7B}"
BASE="${BASE:-$KD_DATA_DIR/models/$MODEL}"
GCS_ROOT="${GCS_ROOT:-gs://ckpt-jawadamin-us-east5-0926/$MODEL}"
FSDP="${FSDP:-8}"; TP="${TP:-1}"; REMAT="${REMAT:-DECODER}"; MICRO_BATCH="${MICRO_BATCH:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
RANK="${RANK:-256}"
[ $((RANK % 256)) -eq 0 ] || { echo "RANK must be a multiple of 256 (MXU alignment)" >&2; exit 2; }
need_eval_tasks() {
  [ -n "${EVAL_TASKS:-}" ] || { echo "EVAL_TASKS is required (evaluation ids JSON; the trainer refuses overlapping ids)" >&2; exit 2; }
  [ -f "$EVAL_TASKS" ] || { echo "missing $EVAL_TASKS (evaluation ids JSON)" >&2; exit 2; }
}
OUT="$KD_DATA_DIR/train/${RUN_ID}"
T=trainer/train.py
COMMON="--out_dir $OUT --run_id $RUN_ID --fsdp $FSDP --tp $TP --rank $RANK --alpha $((2 * RANK)) --remat $REMAT"
case "$PHASE" in
  probe)
    CMD="python3 $T probe_memory --hf_dir $BASE $COMMON --gcs_out $GCS_ROOT/$RUN_ID ${*}" ;;
  grpo)
    [ $# -ge 3 ] || { echo "usage: train_tpu_vm.sh grpo <run_id> <policy_hf_dir> <rollouts_jsonl> <iteration> [args...]" >&2; exit 2; }
    POLICY="$1"; ROLLOUTS="$2"; ITER="$3"; shift 3
    need_eval_tasks
    [ -n "${REF:-}" ] || { echo "REF (the GRPO KL reference HF dir) is required for grpo" >&2; exit 2; }
    if [ "${REF%/}" = "${BASE%/}" ] && [ "${ALLOW_BASE_REF:-0}" != 1 ]; then
      echo "REF equals BASE ($BASE); the KL reference is normally the SFT-merged policy. Set ALLOW_BASE_REF=1 to use the base" >&2
      exit 2
    fi
    CMD="python3 $T grpo_iter --hf_dir $POLICY $COMMON --micro_batch $MICRO_BATCH --max_seq_len $MAX_SEQ_LEN \
      --rollouts $ROLLOUTS --ref_hf_dir $REF --iteration $ITER --eval_tasks $EVAL_TASKS ${*} \
      --gcs_out $GCS_ROOT/$RUN_ID/grpo/iter_$((ITER + 1))" ;;
  sft)
    [ $# -ge 1 ] || { echo "usage: train_tpu_vm.sh sft <run_id> <data_jsonl> [args...]" >&2; exit 2; }
    DATA="$1"; shift 1
    need_eval_tasks
    CMD="python3 $T sft --hf_dir $BASE $COMMON --data $DATA \
      --micro_batch $MICRO_BATCH --max_seq_len $MAX_SEQ_LEN --eval_tasks $EVAL_TASKS ${*} --export_merged --gcs_out $GCS_ROOT/$RUN_ID" ;;
  merge)
    [ $# -ge 3 ] || { echo "usage: train_tpu_vm.sh merge <run_id> <policy_hf_dir> <ckpt_root> <tag> [args...]" >&2; exit 2; }
    POLICY="$1"; CKPT="$2"; TAG="$3"; shift 3
    CMD="python3 $T merge --hf_dir $POLICY $COMMON --ckpt $CKPT --tag $TAG ${*} \
      --gcs_out $GCS_ROOT/$RUN_ID/grpo/$TAG" ;;
  *) echo "phase must be probe, grpo, sft or merge" >&2; exit 2 ;;
esac
mkdir -p "$OUT" "$KD_DATA_DIR/gcs" "$KD_DATA_DIR/jaxcache-train"
docker rm -f kd-train >/dev/null 2>&1 || true
exec docker run --rm --name kd-train --privileged --net=host --ipc=host \
  -v "$KD_DATA_DIR":"$KD_DATA_DIR" -w "$CODE" \
  -e GCS_CACHE="$KD_DATA_DIR/gcs" -e EXPORT_STREAM=1 -e JAX_COMPILATION_CACHE_DIR="$KD_DATA_DIR/jaxcache-train" \
  "$IMAGE" bash -lc "set -o pipefail; $CMD 2>&1 | tee -a $OUT/train.log"
