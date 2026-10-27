#!/usr/bin/env bash
#SBATCH --job-name=qwen-defense
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Training-based defence baselines (scripts/post-training/train_defense.py); clone of train_lora.sbatch.
#   METHOD   sft | lisa | booster | csft | dpo                       (required)
#   STAGE    align | ft | post (output-name prefix; default ft)
#   CONDITION p00|p05|p10|p20 -> --condition (user data), or TRAIN=<jsonl> -> --train
#   MODEL_PATH  model to start from (default: the pinned Qwen3-4B-Instruct-2507; a local merged dir for A-stage attacks / P3)
# Output: results/<STAGE>-<METHOD>[-<CONDITION>]-<jobid>. Remaining arguments go to train_defense.py, e.g.
#   METHOD=sft STAGE=align TRAIN=trainset/derived/align_sft.jsonl sbatch --account=CODE scripts/viking/train_defense.sbatch --merge
# Booster runs 3 forward/backward passes per micro-step: submit with --time=1-12:00:00. DPO holds a refusal and a full
# trajectory per micro-step; use --partition=gpuplus if the A40 runs out of memory.
set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$PWD/.venv-serve-cu128/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || { echo "Training Python missing: $TRAIN_PYTHON" >&2; exit 1; }
"$TRAIN_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
: "${METHOD:?export METHOD=sft|lisa|booster|csft|dpo}"
STAGE="${STAGE:-ft}"
data=()
if [[ -n "${CONDITION:-}" ]]; then data+=(--condition "$CONDITION"); fi
if [[ -n "${TRAIN:-}" ]]; then data+=(--train "$TRAIN"); fi
name="results/${STAGE}-${METHOD}${CONDITION:+-$CONDITION}-${SLURM_JOB_ID}"
"$TRAIN_PYTHON" scripts/post-training/train_defense.py --method "$METHOD" --output "$name" "${data[@]}" "$@"
