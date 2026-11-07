#!/usr/bin/env bash
set -euo pipefail
# Run in a separate GPU environment: pip install vllm
# Set MODEL_REVISION to an immutable HF commit for final research runs.
MODEL="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
args=()
if [[ -n "${MODEL_REVISION:-}" ]]; then args+=(--revision "$MODEL_REVISION"); fi
# PTST and other serving-time templates: CHAT_TEMPLATE=templates/ptst.jinja (scripts/post-training/safety_template.py).
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then args+=(--chat-template "$CHAT_TEMPLATE"); fi
# Tool-call parser by chat format (TOOL_CALL_PARSER overrides): hermes for Qwen; for GLM-4 and Llama-3.1 the parsers in
# scripts/eval/tool_parsers.py, which read the formats of templates/glm4_tools.jinja and templates/llama31_tools.jinja.
# A local model directory is classified by its tokenizer's special tokens, a hub id by name.
tool_parser() {
    local config="$1/tokenizer_config.json"
    if [[ -f "$config" ]]; then
        if grep -qF '<|observation|>' "$config"; then echo glm4_native
        elif grep -qF '<|start_header_id|>' "$config"; then echo llama3_json_full
        else echo hermes; fi
        return
    fi
    case "$1" in
        *[Gg][Ll][Mm]-4*) echo glm4_native ;;
        *[Ll]lama-3*) echo llama3_json_full ;;
        *) echo hermes ;;
    esac
}
parser="${TOOL_CALL_PARSER:-$(tool_parser "$MODEL")}"
if [[ "$parser" != hermes ]]; then args+=(--tool-parser-plugin "$(dirname "$0")/tool_parsers.py"); fi
echo "Tool-call parser: $parser" >&2
exec vllm serve "$MODEL" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --dtype auto --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --enable-auto-tool-choice --tool-call-parser "$parser" --seed 42 "${args[@]}" "$@"
