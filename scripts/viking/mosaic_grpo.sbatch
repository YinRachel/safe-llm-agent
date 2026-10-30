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
#   LR (1e-5), LORA_RANK (16), LORA_ALPHA (32), N_GPUS (1)
#   TRAIN_PYTHON  default .venv-serve-cu128/bin/python, the env train_lora.sbatch uses, with requirements-rl.txt added
# Output: results/<MODEL_TAG>_MOSAIC/<run>-<jobid>/{checkpoints,verl_export,merged}; evaluate merged/ with
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
: "${MODEL_PATH:?export MODEL_PATH=<merged model dir>}"
TRAIN_FILE="${TRAIN_FILE:-trainset/derived/mosaic_astb_train.parquet}"
[[ -f "$TRAIN_FILE" ]] || { echo "Missing $TRAIN_FILE (python scripts/post-training/mosaic.py build-data)" >&2; exit 1; }
LR="${LR:-1e-5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
N_GPUS="${N_GPUS:-1}"
source scripts/viking/results_dir.sh
run="${MODEL_PATH%/}"
[[ "$(basename "$run")" == merged ]] && run="$(dirname "$run")"
OUT="$(results_group mosaic "$MODEL_PATH")/$(basename "$run")-${SLURM_JOB_ID}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD/scripts/post-training:$PWD/scripts/eval${PYTHONPATH:+:$PYTHONPATH}"
export ASTB_REPO="${ASTB_REPO:-$PWD/third_party/Agent-SafetyBench}"

# One live pairwise call before any GPU work: an immediate refusal must beat compliance in both orders.
"$TRAIN_PYTHON" scripts/post-training/mosaic_judge.py

# App. J: 4 prompts x 4 rollouts per step, one PPO update per step (mini-batch = the 16 trajectories), temperature 1.0,
# 2,000-token prompts in the paper (3,072 here: the longest Agent-SafetyBench prompt with tools is 2,563), 5,000 response
# tokens, 20 turns, KL 0.001. The env has no flash-attn, so attention is SDPA and padding removal (which uses
# flash-attn's kernels) is off. LR is for LoRA (the paper's 1e-6 is full fine-tuning); bf16 instead of fp16.
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
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
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
    trainer.default_local_dir="$OUT/checkpoints" \
    "$@"

step="$(cat "$OUT/checkpoints/latest_checkpointed_iteration.txt")"
"$TRAIN_PYTHON" -m verl.model_merger merge --backend fsdp \
    --local_dir "$OUT/checkpoints/global_step_$step/actor" --target_dir "$OUT/verl_export"
"$TRAIN_PYTHON" scripts/post-training/mosaic.py export --export "$OUT/verl_export" --lora-alpha "$LORA_ALPHA" \
    --output "$OUT/merged"
echo "MOSAIC model: $OUT/merged (step $step)"
