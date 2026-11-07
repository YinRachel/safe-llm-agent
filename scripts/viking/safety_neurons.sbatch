#!/usr/bin/env bash
#SBATCH --job-name=safety-neurons
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Safety-neuron analysis (scripts/eval/safety_neurons.py). Needs trainset/derived/probe_prompts.jsonl, built where
# trainset/raw and trainset/interim live: `python3 -I scripts/eval/safety_neurons.py prompts` (copy it over, mode 600).
#   sbatch scripts/viking/safety_neurons.sbatch collect MODEL NAME        # MODEL = hub id or <run>/merged
#   sbatch scripts/viking/safety_neurons.sbatch analyze BASE_NAME NAME... # base first; all from collect
#   sbatch scripts/viking/safety_neurons.sbatch ablate MODEL ANALYSIS_DIR NAME
# e.g. MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1 sbatch ... collect Qwen/Qwen2.5-3B-Instruct qwen2.5-3B-base
#      sbatch ... collect results/qwen2.5-3B_LoRA/p20-123/merged qwen2.5-3B-p20
#      sbatch ... analyze qwen2.5-3B-base qwen2.5-3B-p00 qwen2.5-3B-p20
# Activations (2-3 GB per model) go to $SCRATCH_OUTPUT_ROOT/safety_neurons/NAME; analyses and ablations (small) to
# results/safety_neurons/. Extra script options via SN_ARGS, e.g. SN_ARGS="--limit 64" for a smoke run.
set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/mnt/scratch/users/$USER/safe-llm-agent}"
ACTS_ROOT="$SCRATCH_OUTPUT_ROOT/safety_neurons"
TRAIN_PYTHON="${TRAIN_PYTHON:-$PWD/.venv-serve-cu128/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || { echo "Training Python missing: $TRAIN_PYTHON" >&2; exit 1; }
[[ -f trainset/derived/probe_prompts.jsonl ]] || { echo "Missing trainset/derived/probe_prompts.jsonl (see header)" >&2; exit 1; }
source scripts/viking/results_dir.sh
command="${1:?usage: safety_neurons.sbatch collect MODEL NAME | analyze BASE_NAME NAME... | ablate MODEL ANALYSIS_DIR NAME}"
shift
case "$command" in
    collect)
        model="${1:?collect MODEL NAME}"; name="${2:?collect MODEL NAME}"
        [[ -d "$model" ]] && { check_scratch_link "${model%/}" || exit 1; }
        # shellcheck disable=SC2086
        "$TRAIN_PYTHON" scripts/eval/safety_neurons.py collect --model "$model" --output "$ACTS_ROOT/$name" ${SN_ARGS:-} ;;
    analyze)
        base="${1:?analyze BASE_NAME NAME...}"; shift
        compare=()
        for name in "$@"; do compare+=("$name=$ACTS_ROOT/$name"); done
        # shellcheck disable=SC2086
        "$TRAIN_PYTHON" scripts/eval/safety_neurons.py analyze --base "$ACTS_ROOT/$base" --compare "${compare[@]}" \
            --output "results/safety_neurons/analysis-$base-${SLURM_JOB_ID}" ${SN_ARGS:-} ;;
    ablate)
        model="${1:?ablate MODEL ANALYSIS_DIR NAME}"; analysis="${2:?ablate MODEL ANALYSIS_DIR NAME}"; name="${3:?ablate MODEL ANALYSIS_DIR NAME}"
        # shellcheck disable=SC2086
        "$TRAIN_PYTHON" scripts/eval/safety_neurons.py ablate --model "$model" --neurons "$analysis/neurons.json" \
            --output "results/safety_neurons/ablate-$name-${SLURM_JOB_ID}" ${SN_ARGS:-} ;;
    *)
        echo "Unknown command: $command (collect | analyze | ablate)" >&2; exit 2 ;;
esac
