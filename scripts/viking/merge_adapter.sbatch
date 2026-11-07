#!/usr/bin/env bash
#SBATCH --job-name=merge-adapter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Rebuild <run>/merged from <run>/adapter (scripts/post-training/merge_adapter.py), e.g. after scratch was wiped:
#   sbatch scripts/viking/merge_adapter.sbatch results/glm4-9B_LoRA/p00-123 [RUN ...]
# merged/ goes back into scratch (the dangling symlink's target, else SCRATCH_OUTPUT_ROOT as in training).
set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
export SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT-/mnt/scratch/users/$USER/safe-llm-agent}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$PWD/.venv-serve-cu128/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || { echo "Training Python missing: $TRAIN_PYTHON" >&2; exit 1; }
[[ $# -gt 0 ]] || { echo "usage: sbatch scripts/viking/merge_adapter.sbatch RUN_DIR..." >&2; exit 2; }
"$TRAIN_PYTHON" scripts/post-training/merge_adapter.py "$@"
