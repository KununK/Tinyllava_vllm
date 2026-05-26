#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT}/.uv-cache}"
export XINFERENCE_HOME="${XINFERENCE_HOME:-${ROOT}/.xinference}"
export HF_HOME="${HF_HOME:-${ROOT}/.hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PATH="${ROOT}/.venv/bin:${PATH}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

mkdir -p "${UV_CACHE_DIR}" "${XINFERENCE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}"

"${ROOT}/.venv/bin/python" "${ROOT}/scripts/patch_xinference_tinyllava_qwen.py"

exec "${ROOT}/.venv/bin/xinference-local" \
  --host "${XINFERENCE_HOST:-0.0.0.0}" \
  --port "${XINFERENCE_PORT:-10003}" \
  "$@"
