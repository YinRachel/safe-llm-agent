#!/bin/bash
#SBATCH --job-name=export-ckpt
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Export one intermediate verl GRPO checkpoint (full fine-tuning or LoRA) as a standalone bf16 model, the same two
# steps mosaic_grpo.sbatch runs for its last checkpoint, into a run folder of its own so eval_all.sh puts the
# evaluations beside it:
#   sbatch scripts/viking/export_checkpoint.sbatch RUN_DIR STEP
# RUN_DIR is the training run, e.g. results/qwen2.5-3B_SafeUtil/p00-38166985-full-38513278; the model goes to
# RUN_DIR-step<STEP>/merged, and its checkpoint is read from CKPT_ROOT (default mosaic-checkpoints in scratch).
# Only checkpoints that still hold actor/ weights can be exported: mosaic_grpo.sbatch keeps the last two for LoRA
# and only the last one for full fine-tuning (older steps keep just data.pt). For an earlier full fine-tuning step,
# rerun training with trainer.total_training_steps=STEP (the learning rate is constant, so the schedule matches).
# LORA_ALPHA must match the training run for a LoRA checkpoint (default 32, as mosaic_grpo.sbatch).
set -euo pipefail
RUN="${1:?usage: export_checkpoint.sbatch RUN_DIR STEP}"
STEP="${2:?usage: export_checkpoint.sbatch RUN_DIR STEP}"
RUN="${RUN%/}"
module load Python/3.12.3-GCCcore-13.3.0
TRAIN_PYTHON="${TRAIN_PYTHON:-$PWD/.venv-serve-cu128/bin/python}"
CKPT="${CKPT_ROOT:-/mnt/scratch/users/$USER/mosaic-checkpoints}/$(basename "$RUN")"
ACTOR="$CKPT/checkpoints/global_step_$STEP/actor"
[[ -d "$ACTOR" ]] || { echo "No actor weights at $ACTOR (pruned by max_actor_ckpt_to_keep?)" >&2; exit 1; }
OUT="$RUN-step$STEP"
EXPORT="$CKPT/verl_export_step$STEP"
mkdir -p "$OUT"
"$TRAIN_PYTHON" -m verl.model_merger merge --backend fsdp --local_dir "$ACTOR" --target_dir "$EXPORT"
"$TRAIN_PYTHON" scripts/post-training/mosaic.py export --export "$EXPORT" --lora-alpha "${LORA_ALPHA:-32}" \
    --output "$OUT/merged"
rm -rf "$EXPORT"
echo "Step $STEP model: $OUT/merged"
