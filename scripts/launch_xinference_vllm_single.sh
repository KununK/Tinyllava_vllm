#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
GPU_IDX="${2:-0}"
ENDPOINT="${XINFERENCE_ENDPOINT:-http://127.0.0.1:10003}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"

case "${MODEL}" in
  qwen2-0.5b|qwen2_0_5b|0.5b)
    MODEL_UID="tinyllava-qwen2-0.5b-vllm"
    MODEL_NAME="tinyllava-qwen2-0.5b-vllm"
    MODEL_SIZE="0_5"
    MODEL_PATH="${ROOT}/vllm_models/tinyllava-qwen2-0.5b-llava-vllm"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
    ;;
  qwen2.5-3b|qwen25-3b|3b)
    MODEL_UID="tinyllava-qwen2.5-3b-vllm"
    MODEL_NAME="tinyllava-qwen2.5-3b-vllm"
    MODEL_SIZE="3"
    MODEL_PATH="${ROOT}/vllm_models/tinyllava-qwen2.5-3b-llava-vllm"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
    ;;
  *)
    echo "Usage: $0 {qwen2-0.5b|qwen2.5-3b} [gpu_idx]" >&2
    exit 2
    ;;
esac

exec "${ROOT}/.venv/bin/xinference" launch \
  --endpoint "${ENDPOINT}" \
  --model-name "${MODEL_NAME}" \
  --model-uid "${MODEL_UID}" \
  --model-type LLM \
  --model-engine vLLM \
  --model-format pytorch \
  --size-in-billions "${MODEL_SIZE}" \
  --quantization none \
  --n-gpu 1 \
  --gpu-idx "${GPU_IDX}" \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code false \
  --disable-virtual-env \
  --max_model_len "${MAX_MODEL_LEN:-1024}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --limit_mm_per_prompt '{"image": 1}' \
  --enforce_eager true
