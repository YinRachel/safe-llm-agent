#!/usr/bin/env bash
set -euo pipefail
# Run in a separate GPU environment: pip install vllm
# Set MODEL_REVISION to an immutable HF commit for final research runs.
args=()
if [[ -n "${MODEL_REVISION:-}" ]]; then args+=(--revision "$MODEL_REVISION"); fi
# PTST and other serving-time templates: CHAT_TEMPLATE=templates/ptst.jinja (scripts/safety_template.py).
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then args+=(--chat-template "$CHAT_TEMPLATE"); fi
exec vllm serve "${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --dtype auto --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --enable-auto-tool-choice --tool-call-parser hermes --seed 42 "${args[@]}" "$@"
