#!/usr/bin/env bash
#SBATCH --job-name=qwen-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$PWD/.venv-serve-cu128/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || { echo "Training Python missing: $TRAIN_PYTHON" >&2; exit 1; }
"$TRAIN_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
# Select the training mix with CONDITION (p00/p05/p10/p20). Remaining arguments go to train_lora.py, e.g. --limit 64 --max-steps 20 for a smoke run.
# MODEL_PATH / MODEL_REVISION pick another base model; MODEL_TAG (e.g. qwen25-3b) then keeps its runs apart: results/lora-TAG-COND-JOB.
"$TRAIN_PYTHON" scripts/post-training/train_lora.py --output "results/lora-${MODEL_TAG:+$MODEL_TAG-}${CONDITION:?export CONDITION=p00|p05|p10|p20}-${SLURM_JOB_ID}" --condition "$CONDITION" "$@"
