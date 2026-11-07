#!/usr/bin/env bash
# LoRA fine-tuning of further base models on the same conditions, data, hyperparameters and seed as Qwen3-4B:
#   glm4-9B      zai-org/glm-4-9b-chat-hf (GLM-4-9B-Chat in native Transformers format) -> results/glm4-9B_LoRA/
#   llama3.1-8B  meta-llama/Llama-3.1-8B-Instruct (gated: accept the licence, `huggingface-cli login`) -> results/llama3.1-8B_LoRA/
# Usage (login node, project root; weights fetched first by scripts/viking/download.sh):
#   bash scripts/viking/lora_models.sh MODEL eval-base                  # evaluate the untouched base model (seven jobs)
#   bash scripts/viking/lora_models.sh MODEL train [p00 p05 p10 p20]   # one LoRA job per condition (default: all four), --merge
#   bash scripts/viking/lora_models.sh MODEL eval RUN_DIR...           # evaluate finished runs, e.g. results/glm4-9B_LoRA/p10-123
# Extra train_lora.py options go after `--`, e.g. `train p00 -- --limit 64 --max-steps 20` for a smoke run.
# train_lora.py picks each model's chat template (templates/glm4_tools.jinja, templates/llama31_tools.jinja) and LoRA
# targets from the tokenizer, and saves the template with adapter/ and merged/.
# The 8-9B models take about twice as long per step as Qwen3-4B, so jobs ask for TRAIN_TIME (default 24 h).
# Evaluations use scripts/viking/eval_all.sh unchanged: the base model is served with its training template (eval_all.sh
# defaults it for these hub ids), a merged run with the template saved in merged/, and serve_qwen.sh picks the matching
# tool parser (scripts/eval/tool_parsers.py). Needs CUDA_MODULE and SERVE_ENV exported as for the Qwen3 jobs.
# Evaluations go to results/<MODEL>_base/ and <run>/eval/.
set -euo pipefail
[[ -f scripts/viking/train_lora.sbatch ]] || { echo "Run from the project root" >&2; exit 1; }
usage="usage: lora_models.sh glm4-9B|llama3.1-8B eval-base | train [CONDITION...] [-- train_lora options] | eval RUN_DIR..."
model="${1:?$usage}"
command="${2:?$usage}"
shift 2
case "$model" in
    glm4-9B)
        export MODEL_PATH=zai-org/glm-4-9b-chat-hf MODEL_REVISION=8599336fc6c125203efb2360bfaf4c80eef1d1bf ;;
    llama3.1-8B)
        export MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct MODEL_REVISION=0e9e39f249a16976918f6564b8830bc894c89659 ;;
    *)
        echo "Unknown model: $model ($usage)" >&2; exit 2 ;;
esac
export MODEL_TAG="$model"
mkdir -p logs results
case "$command" in
    eval-base)
        bash scripts/viking/eval_all.sh "$MODEL_PATH" "$MODEL_TAG-base" ;;
    train)
        conditions=()
        while [[ $# -gt 0 && "$1" != -- ]]; do conditions+=("$1"); shift; done
        [[ "${1:-}" == -- ]] && shift
        [[ ${#conditions[@]} -gt 0 ]] || conditions=(p00 p05 p10 p20)
        for condition in "${conditions[@]}"; do
            [[ -f "trainset/data/train_$condition.jsonl" ]] || { echo "Missing trainset/data/train_$condition.jsonl" >&2; exit 1; }
            # shellcheck disable=SC2086
            id="$(CONDITION="$condition" sbatch --parsable --export=ALL --job-name="lora-$MODEL_TAG-$condition" \
                --time="${TRAIN_TIME:-24:00:00}" ${SBATCH_EXTRA:-} scripts/viking/train_lora.sbatch --merge "$@")"
            echo "$condition: job ${id%%;*} -> results/${MODEL_TAG}_LoRA/$condition-${id%%;*}"
        done ;;
    eval)
        [[ $# -gt 0 ]] || { echo "usage: lora_models.sh $model eval RUN_DIR..." >&2; exit 1; }
        for run in "$@"; do
            run="${run%/}"
            [[ -f "$run/merged/config.json" ]] || { echo "No merged model in $run (train with --merge, wait for the job)" >&2; exit 1; }
            [[ -f "$run/merged/chat_template.jinja" ]] || grep -q '"chat_template"' "$run/merged/tokenizer_config.json" 2>/dev/null \
                || { echo "No chat template saved in $run/merged" >&2; exit 1; }
            bash scripts/viking/eval_all.sh "$run/merged" "$MODEL_TAG-$(basename "$run")"
        done ;;
    *)
        echo "Unknown command: $command ($usage)" >&2; exit 2 ;;
esac
