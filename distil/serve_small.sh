#!/usr/bin/env bash
# Serve a small dense model as one single-chip (TP1) vLLM-TPU replica on a v6e host.
# Usage (on the TPU VM): serve_small.sh <chip 0-7> <model_dir> <served_name>
#   Replica on chip N listens on port 8100+N. Flags are identical for every arm; only the weights
#   and the served name differ.
# Env: DOCKER_HOST (default unix:///var/run/docker.sock), IMAGE, KD_DATA_DIR (default /mnt/data,
#   holds the XLA compilation cache), MAX_MODEL_LEN (default 16384), MAX_NUM_SEQS (default 128),
#   GPU_MEM (HBM fraction for weights + KV cache, default 0.85; lower it to leave room for
#   prompt_logprobs, which needs several GB of scratch).
set -euo pipefail
CHIP="$1"; MODEL_DIR="$2"; SERVED="$3"
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
IMAGE="${IMAGE:-asia-northeast1-docker.pkg.dev/northam-ce-mlai-tpu/jawadamin-dsv4/tpu-inference:0187efa-int8}"
KD_DATA_DIR="${KD_DATA_DIR:-/mnt/data}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM="${GPU_MEM:-0.85}"
case "$CHIP" in [0-7]) ;; *) echo "chip must be 0-7" >&2; exit 2 ;; esac
[ -f "$MODEL_DIR/config.json" ] || { echo "no config.json in $MODEL_DIR" >&2; exit 3; }
PORT=$((8100 + CHIP)); PPORT=$((8480 + CHIP))
NAME="kd-vllm-$CHIP"
CACHE="$KD_DATA_DIR/jaxcache-kd-$(basename "$MODEL_DIR")"
docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$CACHE"
docker run -d --name "$NAME" --privileged --net=host --ipc=host \
  -v "$MODEL_DIR":/models/m:ro -v "$CACHE":/jaxcache \
  -e TPU_BACKEND_TYPE=jax -e MODEL_IMPL_TYPE=vllm -e NEW_MODEL_DESIGN=1 \
  -e JAX_COMPILATION_CACHE_DIR=/jaxcache \
  -e TPU_WORKER_ID=0 -e TPU_WORKER_HOSTNAMES=localhost -e TPU_SKIP_MDS_QUERY=true \
  -e TPU_HOST_BOUNDS=1,1,1 -e TPU_PROCESS_BOUNDS=1,1,1 \
  -e TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 -e TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 \
  -e TPU_VISIBLE_CHIPS="$CHIP" -e TPU_PROCESS_PORT="$PPORT" \
  -e TPU_ACCELERATOR_TYPE=v6e-1 -e TPU_TOPOLOGY=1x1 \
  "$IMAGE" bash -lc "exec vllm serve /models/m --served-model-name '$SERVED' \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_MEM \
    --max-model-len $MAX_MODEL_LEN --max-num-batched-tokens 4096 --max-num-seqs $MAX_NUM_SEQS \
    --host 0.0.0.0 --port $PORT" >/dev/null || { echo "docker run failed for $NAME" >&2; exit 4; }
echo "$NAME chip=$CHIP port=$PORT served=$SERVED model=$MODEL_DIR"
