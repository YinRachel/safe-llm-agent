#!/usr/bin/env bash
#SBATCH --job-name=qwen-ping
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# PING prefix search (scripts/post-training/ping.py search) on one merged fine-tuned model.
#   TAG  e.g. p10 (output-name only)
# Output: results/<MODEL_TAG>_PING/[<TAG>-]<jobid>/ping.jinja (+ summary.json, evaluated.jsonl, generator.jsonl).
# The generator is the project judge endpoint from .env (or PING_GENERATOR=<model id>). Remaining arguments go to
# ping.py search, e.g.
#   TAG=p10 sbatch scripts/viking/ping.sbatch --model results/qwen4B_LoRA/p10-JOB/merged
# Then evaluate with the template:
#   bash scripts/viking/eval_all.sh results/qwen4B_LoRA/p10-JOB/merged p10-ping results/qwen4B_PING/p10-<jobid>/ping.jinja
set -euo pipefail
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${TMPDIR:-/tmp}/vllm-cache-${SLURM_JOB_ID:-$$}}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
module load Python/3.12.3-GCCcore-13.3.0
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
if command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"; fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
SERVE_PYTHON="${SERVE_ENV:-$PWD/.venv-serve-cu128}/bin/python"
[[ -x "$SERVE_PYTHON" ]] || { echo "Python missing: $SERVE_PYTHON" >&2; exit 1; }
export PATH="$(dirname "$SERVE_PYTHON"):$PATH"  # FlashInfer JIT needs the env's ninja
"$SERVE_PYTHON" -c 'import vllm, openai, torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
source scripts/viking/results_dir.sh
"$SERVE_PYTHON" scripts/post-training/ping.py search --out "$(results_group ping "$@")/${TAG:+$TAG-}${SLURM_JOB_ID}" "$@"
