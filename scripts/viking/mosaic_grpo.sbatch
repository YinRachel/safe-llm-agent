#!/usr/bin/env bash
#SBATCH --job-name=mosaic-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
# MOSAIC GRPO (Agarwal et al., ICML 2026) on Agent-SafetyBench with the pairwise Runware judge; see docs/mosaic.md.
#   MODEL_PATH  merged model to start from (required), e.g. results/qwen2.5-3B_LoRA/p00-123/merged. With LoRA, verl's
#               reference policy is this model with the adapter off, so the KL anchor is the SFT model.
#   TRAIN_FILE  default trainset/derived/mosaic_astb_train.parquet (python scripts/post-training/mosaic.py build-data)
#   LR (1e-5 with LoRA, 1e-6 for full fine-tuning), LORA_RANK (16), LORA_ALPHA (32), N_GPUS (1)
#   LORA_RANK=0 trains all weights, as the paper does: verl then loads a separate frozen reference copy, Adam states
#               live on the CPU between updates, only the last checkpoint (~37 GB) is kept, and the run
#               folder ends in -full. Needs an 80 GB GPU: sbatch --partition=gpuplus --mem=160G ...
#   TRAIN_PYTHON  default .venv-serve-cu128/bin/python, the env train_lora.sbatch uses, with requirements-rl.txt added
#   OUT         an earlier run folder to resume from its last checkpoint (verl's resume_mode=auto), e.g.
#               OUT=results/qwen2.5-3B_MOSAIC/p00-123-456; by default a new folder is made
#   CKPT_ROOT   default /mnt/scratch/users/$USER/mosaic-checkpoints. Checkpoints hold the whole model in fp32 even with
#               LoRA (~14 GB each, ~37 GB with full fine-tuning optimizer state), too big for the 100 GB home quota,
#               so they and verl_export/ go to scratch under <CKPT_ROOT>/<run folder name>; the last 2 are kept
#               (1 with full fine-tuning).
# Output: results/<MODEL_TAG>_MOSAIC/<run>-<jobid>/merged (checkpoints and verl_export/ in scratch); evaluate merged/ with
# scripts/viking/qwen25_3b.sh eval <that dir>. Extra arguments are verl Hydra overrides, e.g. a smoke run:
#   MODEL_PATH=results/qwen2.5-3B_LoRA/p00-123/merged sbatch scripts/viking/mosaic_grpo.sbatch trainer.total_training_steps=5
set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$PWD/.venv-serve-cu128/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || { echo "Training Python missing: $TRAIN_PYTHON" >&2; exit 1; }
"$TRAIN_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$TRAIN_PYTHON" -c 'import verl, tensordict, hydra, datasets' || {
    echo "verl missing from $TRAIN_PYTHON: pip install -r requirements-rl.txt && pip install --no-deps verl==0.6.1" >&2; exit 1; }
"$TRAIN_PYTHON" -c 'import torch, flash_attn_2_cuda' || {
    echo "flash-attn missing from $TRAIN_PYTHON: sbatch scripts/viking/build_flash_attn.job" >&2; exit 1; }
: "${MODEL_PATH:?export MODEL_PATH=<merged model dir>}"
TRAIN_FILE="${TRAIN_FILE:-trainset/derived/mosaic_astb_train.parquet}"
[[ -f "$TRAIN_FILE" ]] || { echo "Missing $TRAIN_FILE (python scripts/post-training/mosaic.py build-data)" >&2; exit 1; }
LORA_RANK="${LORA_RANK:-16}"
if (( LORA_RANK > 0 )); then
    LR="${LR:-1e-5}"
    MODE_ARGS=(trainer.max_actor_ckpt_to_keep=2)
    suffix=""
else
    LR="${LR:-1e-6}"
    MODE_ARGS=(actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
               actor_rollout_ref.ref.fsdp_config.param_offload=True
               trainer.max_actor_ckpt_to_keep=1)
    suffix="-full"
fi
LORA_ALPHA="${LORA_ALPHA:-32}"
N_GPUS="${N_GPUS:-1}"
source scripts/viking/results_dir.sh
run="${MODEL_PATH%/}"
[[ "$(basename "$run")" == merged ]] && run="$(dirname "$run")"
OUT="${OUT:-$(results_group mosaic "$MODEL_PATH")/$(basename "$run")${suffix}-${SLURM_JOB_ID}}"
OUT="${OUT%/}"
CKPT="${CKPT_ROOT:-/mnt/scratch/users/$USER/mosaic-checkpoints}/$(basename "$OUT")"
mkdir -p "$OUT" "$CKPT"
export PYTHONPATH="$PWD/scripts/post-training:$PWD/scripts/eval${PYTHONPATH:+:$PYTHONPATH}"
export ASTB_REPO="${ASTB_REPO:-$PWD/third_party/Agent-SafetyBench}"
# Slurm exports ROCR_VISIBLE_DEVICES (AMD) next to CUDA_VISIBLE_DEVICES; verl's workers refuse to start with both.
unset ROCR_VISIBLE_DEVICES

# One live pairwise call before any GPU work: an immediate refusal must beat compliance in both orders.
"$TRAIN_PYTHON" scripts/post-training/mosaic_judge.py

# App. J: 4 prompts x 4 rollouts per step, one PPO update per step (mini-batch = the 16 trajectories), temperature 1.0,
# 2,000-token prompts in the paper (3,072 here: the longest Agent-SafetyBench prompt with tools is 2,563), 5,000 response
# tokens, 20 turns, KL 0.001. Padding removal (flash-attn, built by build_flash_attn.job) packs only real tokens, so
# forward passes are grouped by token budget; without it every sequence was padded to 8,072 tokens and about 85% of the
# training compute went on padding. Entropy is chunked over the 152k vocabulary to keep logits memory flat. LR 1e-5 is for LoRA; full fine-tuning uses the paper's 1e-6. bf16 instead of fp16.
"$TRAIN_PYTHON" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TRAIN_FILE" \
    data.train_batch_size=4 \
    data.max_prompt_length=3072 \
    data.max_response_length=5000 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=8072 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=scripts/post-training/mosaic_agent_loop.yaml \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    custom_reward_function.path=scripts/post-training/mosaic_reward.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=naive \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name=mosaic \
    trainer.experiment_name="$(basename "$OUT")" \
    trainer.default_local_dir="$CKPT/checkpoints" \
    "${MODE_ARGS[@]}" \
    "$@"

step="$(cat "$CKPT/checkpoints/latest_checkpointed_iteration.txt")"
"$TRAIN_PYTHON" -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT/checkpoints/global_step_$step/actor" --target_dir "$CKPT/verl_export"
"$TRAIN_PYTHON" scripts/post-training/mosaic.py export --export "$CKPT/verl_export" --lora-alpha "$LORA_ALPHA" \
    --output "$OUT/merged"
echo "MOSAIC model: $OUT/merged (step $step)"
