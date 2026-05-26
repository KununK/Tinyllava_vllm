#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENDPOINT="${XINFERENCE_ENDPOINT:-http://127.0.0.1:10003}"

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"

for MODEL_NAME in tinyllava-qwen2-0.5b-vllm tinyllava-qwen2.5-3b-vllm; do
  "${ROOT}/.venv/bin/xinference" unregister \
    --endpoint "${ENDPOINT}" \
    --model-type LLM \
    --model-name "${MODEL_NAME}" >/dev/null 2>&1 || true
done

for SPEC in "${ROOT}"/xinference_models/*.json; do
  "${ROOT}/.venv/bin/xinference" register \
    --endpoint "${ENDPOINT}" \
    --model-type LLM \
    --file "${SPEC}" \
    --persist
done
