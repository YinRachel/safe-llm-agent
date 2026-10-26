#!/usr/bin/env bash
#SBATCH --job-name=qwen-posthoc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Post-fine-tuning defences (scripts/posthoc_defense.py) on one attacked model.
#   METHOD  safelora | antidote | scale   (required)      TAG  e.g. p10 (output-name only)
# Output: results/post-<METHOD>[-<TAG>]-<jobid>. Remaining arguments go to posthoc_defense.py, e.g.
#   METHOD=scale TAG=p10 sbatch --account=CODE scripts/viking_posthoc.sbatch --adapter results/lora-p10-JOB/adapter --select
#   METHOD=antidote TAG=p10 sbatch --account=CODE scripts/viking_posthoc.sbatch --merged results/lora-p10-JOB/merged --select
set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$PWD/.venv-serve-cu128/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || { echo "Python missing: $TRAIN_PYTHON" >&2; exit 1; }
"$TRAIN_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
: "${METHOD:?export METHOD=safelora|antidote|scale}"
"$TRAIN_PYTHON" scripts/posthoc_defense.py --method "$METHOD" --output "results/post-${METHOD}${TAG:+-$TAG}-${SLURM_JOB_ID}" "$@"
