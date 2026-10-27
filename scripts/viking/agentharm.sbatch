#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentharm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
# FlashInfer may compile kernels at startup and requires the CUDA toolkit.
# Set CUDA_MODULE to an exact compatible name from `module avail CUDA`.
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo 'CUDA compiler nvcc missing. Select a compatible toolkit with module avail CUDA and export CUDA_MODULE before submitting.' >&2
    exit 1
fi
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
nvcc --version
# Submit from the project root. Load any site-specific modules needed by your
# server-created Python environments here before activating them.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project root}"
EVAL_PYTHON="${EVAL_PYTHON:-$PWD/.venv/bin/python}"
SERVE_ENV="${SERVE_ENV:-$PWD/.venv-serve}"
export PATH="$SERVE_ENV/bin:$PATH"
[[ -x "$EVAL_PYTHON" ]] || { echo "Evaluation Python missing: $EVAL_PYTHON" >&2; exit 1; }
command -v vllm >/dev/null || { echo 'vLLM missing; prepare the serving environment first.' >&2; exit 1; }
mkdir -p logs
# Multiple jobs can share a physical node, so choose a free localhost port.
export PORT="${PORT:-$("$EVAL_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')}"
MODEL_LOG="logs/vllm-${SLURM_JOB_ID}.log"
bash scripts/eval/serve_qwen.sh > "$MODEL_LOG" 2>&1 &
server_pid=$!
cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for ((attempt=0; attempt<600; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "vLLM exited; see $MODEL_LOG" >&2
        exit 1
    fi
    if "$EVAL_PYTHON" - "$PORT" <<'PY'
import sys
import urllib.request
try:
    with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/health', timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
        ready=1
        break
    fi
    sleep 2
done
[[ "$ready" == 1 ]] || { echo "vLLM startup timed out; see $MODEL_LOG" >&2; exit 1; }
"$EVAL_PYTHON" scripts/eval/evaluate.py agentharm \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --output "results/agentharm-${SLURM_JOB_ID}" "$@"
