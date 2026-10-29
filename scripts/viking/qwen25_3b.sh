#!/usr/bin/env bash
# Qwen2.5-3B-Instruct as a second base model: same LoRA conditions and the same seven evaluations as Qwen3-4B.
# Usage (login node, project root; weights fetched first by scripts/viking/download.sh):
#   bash scripts/viking/qwen25_3b.sh eval-base                  # evaluate the untouched base model
#   bash scripts/viking/qwen25_3b.sh train [p00 p05 p10 p20]    # one LoRA job per condition (default: all four), --merge
#   bash scripts/viking/qwen25_3b.sh eval RUN_DIR...            # evaluate finished runs, e.g. results/lora-qwen25-3b-p10-123
# Extra train_lora.py options go after `--`, e.g. `train p00 -- --limit 64 --max-steps 20` for a smoke run.
# Needs CUDA_MODULE and SERVE_ENV exported as for the Qwen3 jobs. Training hyperparameters and seed are unchanged.
# The model's native context is 32,768 tokens (no YaRN), so serving is capped there instead of agent_benchmarks' 65,536.
set -euo pipefail
[[ -f scripts/viking/train_lora.sbatch ]] || { echo "Run from the project root" >&2; exit 1; }
export MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
export MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
export MODEL_TAG=qwen25-3b
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
mkdir -p logs results
command="${1:?usage: qwen25_3b.sh eval-base | train [CONDITION...] [-- train_lora options] | eval RUN_DIR...}"
shift
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
            id="$(CONDITION="$condition" sbatch --parsable --export=ALL --job-name="lora-$MODEL_TAG-$condition" ${SBATCH_EXTRA:-} \
                scripts/viking/train_lora.sbatch --merge "$@")"
            echo "$condition: job ${id%%;*} -> results/lora-$MODEL_TAG-$condition-${id%%;*}"
        done ;;
    eval)
        [[ $# -gt 0 ]] || { echo "usage: qwen25_3b.sh eval RUN_DIR..." >&2; exit 1; }
        for run in "$@"; do
            run="${run%/}"
            [[ -f "$run/merged/config.json" ]] || { echo "No merged model in $run (train with --merge, wait for the job)" >&2; exit 1; }
            bash scripts/viking/eval_all.sh "$run/merged" "$(basename "$run" | sed 's/^lora-//')"
        done ;;
    *)
        echo "Unknown command: $command (eval-base | train | eval)" >&2; exit 2 ;;
esac
