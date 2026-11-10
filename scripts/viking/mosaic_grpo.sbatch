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
#   METHOD      mosaic (default) or safeutil: SafeUtil GRPO (docs/safeutil.md) uses the same verl setup with its own
#               data (trainset/derived/safeutil_train.parquet, safeutil.py build-data), agent loops (adds a single-turn
#               loop), reward (safeutil_reward.py), judge preflight and 6,144-token prompts (APIGen-MT policies); runs go to
#               results/<MODEL_TAG>_SafeUtil/. Name the job too: sbatch --job-name=safeutil-grpo ...
#   Other base models (qwen4B, llama3.1-8B, glm4-9B, from MODEL_PATH's results/<tag>_<method>/): the training file
#               defaults to trainset/derived/<tag>/<file> (build-data --tokenizer <p00 merged> --output there), and the
#               prompt budget is that file's manifest max_prompt_length. Their chat formats go through mosaic.ChatFormat.
#   N_GPUS      GPUs for FSDP (1). 8-9B full fine-tuning needs both H100s of a gpuplus node:
#               N_GPUS=2 sbatch --gres=gpu:2 --partition=gpuplus --mem=300G ... Each GPU also runs one vLLM replica.
#   ROLLOUT_MEM vLLM's share of each GPU (0.5); ACTOR_PARAM_OFFLOAD=1 keeps actor weights on the CPU between passes
#               (slower; only if the FSDP shards do not fit next to vLLM)
#   ROLLOUT_N   rollouts per prompt: 4 for mosaic (the paper's group), 8 for safeutil, whose rewards are coarse enough
#               that 4 rollouts often tie and the group gives GRPO no signal (half the groups in smoke run 38513183)
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
# Output: results/<MODEL_TAG>_MOSAIC/<run>-<jobid>/merged (checkpoints and verl_export/ in scratch), and rollouts/<step>.jsonl:
# every training trajectory (4 prompts x ROLLOUT_N per step) with its reward terms, refusal status, task id and fulfillable label, a few
# hundred MB per epoch. Evaluate merged/ with
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
METHOD="${METHOD:-mosaic}"
case "$METHOD" in
    mosaic)
        DEFAULT_TRAIN=trainset/derived/mosaic_astb_train.parquet; BUILD="scripts/post-training/mosaic.py build-data"
        LOOP_CONFIG=scripts/post-training/mosaic_agent_loop.yaml; REWARD=scripts/post-training/mosaic_reward.py
        PREFLIGHT=scripts/post-training/mosaic_judge.py; PROMPT_LEN=3072; DEFAULT_N=4 ;;
    safeutil)
        DEFAULT_TRAIN=trainset/derived/safeutil_train.parquet; BUILD="scripts/post-training/safeutil.py build-data"
        LOOP_CONFIG=scripts/post-training/safeutil_agent_loop.yaml; REWARD=scripts/post-training/safeutil_reward.py
        PREFLIGHT=scripts/post-training/safeutil_judge.py; PROMPT_LEN=6144; DEFAULT_N=8 ;;
    *) echo "METHOD must be mosaic or safeutil, not $METHOD" >&2; exit 1 ;;
esac
RESPONSE_LEN=5000
ROLLOUT_N="${ROLLOUT_N:-$DEFAULT_N}"
source scripts/viking/results_dir.sh
BASE_TAG="$(model_tag "$MODEL_PATH")"
if [[ "$BASE_TAG" != qwen2.5-3B ]]; then
    DEFAULT_TRAIN="trainset/derived/$BASE_TAG/$(basename "$DEFAULT_TRAIN")"
    BUILD="$BUILD --tokenizer $MODEL_PATH --output $DEFAULT_TRAIN"
fi
TRAIN_FILE="${TRAIN_FILE:-$DEFAULT_TRAIN}"
manifest="${TRAIN_FILE%.parquet}.manifest.json"
if [[ "$BASE_TAG" != qwen2.5-3B && -f "$manifest" ]]; then
    PROMPT_LEN="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["max_prompt_length"])' "$manifest")"
fi
# Dynamic batching needs one whole sequence to fit a micro-batch: at least prompt + response tokens.
ACTOR_TOKENS=$(( PROMPT_LEN + RESPONSE_LEN > 10240 ? PROMPT_LEN + RESPONSE_LEN : 10240 ))
LOGPROB_TOKENS=$(( ACTOR_TOKENS > 16384 ? ACTOR_TOKENS : 16384 ))
[[ -f "$TRAIN_FILE" ]] || { echo "Missing $TRAIN_FILE (python $BUILD)" >&2; exit 1; }
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
if [[ "${ACTOR_PARAM_OFFLOAD:-0}" == 1 ]]; then MODE_ARGS+=(actor_rollout_ref.actor.fsdp_config.param_offload=True); fi
run="${MODEL_PATH%/}"
[[ "$(basename "$run")" == merged ]] && run="$(dirname "$run")"
OUT="${OUT:-$(results_group "$METHOD" "$MODEL_PATH")/$(basename "$run")${suffix}-${SLURM_JOB_ID}}"
OUT="${OUT%/}"
CKPT="${CKPT_ROOT:-/mnt/scratch/users/$USER/mosaic-checkpoints}/$(basename "$OUT")"
mkdir -p "$OUT" "$CKPT"
export PYTHONPATH="$PWD/scripts/post-training:$PWD/scripts/eval${PYTHONPATH:+:$PYTHONPATH}"
export ASTB_REPO="${ASTB_REPO:-$PWD/third_party/Agent-SafetyBench}"
# Slurm exports ROCR_VISIBLE_DEVICES (AMD) next to CUDA_VISIBLE_DEVICES; verl's workers refuse to start with both.
unset ROCR_VISIBLE_DEVICES

# Live judge calls before any GPU work (MOSAIC: a refusal must beat compliance in both orders; SafeUtil: a finished
# benign task must grade 2 and a refusal 0).
"$TRAIN_PYTHON" "$PREFLIGHT"

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
    data.max_prompt_length="$PROMPT_LEN" \
    data.max_response_length="$RESPONSE_LEN" \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_TOKENS" \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_TOKENS" \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_TOKENS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_MEM:-0.5}" \
    actor_rollout_ref.rollout.max_model_len="$((PROMPT_LEN + RESPONSE_LEN))" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$LOOP_CONFIG" \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    custom_reward_function.path="$REWARD" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=naive \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name="$METHOD" \
    trainer.experiment_name="$(basename "$OUT")" \
    trainer.default_local_dir="$CKPT/checkpoints" \
    trainer.rollout_data_dir="$OUT/rollouts" \
    "${MODE_ARGS[@]}" \
    "$@"

step="$(cat "$CKPT/checkpoints/latest_checkpointed_iteration.txt")"
"$TRAIN_PYTHON" -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT/checkpoints/global_step_$step/actor" --target_dir "$CKPT/verl_export"
"$TRAIN_PYTHON" scripts/post-training/mosaic.py export --export "$CKPT/verl_export" --lora-alpha "$LORA_ALPHA" \
    --output "$OUT/merged"
# The stop tokens vLLM serves with (GLM-4 ends turns on <|user|> / <|observation|>) come from generation_config.json.
if [[ -f "$MODEL_PATH/generation_config.json" ]]; then cp "$MODEL_PATH/generation_config.json" "$OUT/merged/"; fi
echo "$METHOD model: $OUT/merged (step $step)"
