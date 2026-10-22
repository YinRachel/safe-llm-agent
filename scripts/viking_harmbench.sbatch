#!/usr/bin/env bash
#SBATCH --job-name=qwen-harmbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
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
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
HB_PYTHON="${HB_PYTHON:-$PWD/.venv-serve/bin/python}"
HARMBENCH_REPO="${HARMBENCH_REPO:-$PWD/third_party/HarmBench}"
RUN_DIR="results/harmbench-${SLURM_JOB_ID}"
GENERATION_DIR="$RUN_DIR/generation"
score_existing=0
if [[ "${1:-}" == --score-existing ]]; then
    [[ $# == 2 ]] || { echo 'Usage: --score-existing PATH_TO_GENERATION_DIRECTORY' >&2; exit 1; }
    GENERATION_DIR="$2"
    score_existing=1
    for artifact in manifest.json behaviors.csv completions.json; do
        [[ -f "$GENERATION_DIR/$artifact" ]] || { echo "Missing saved artifact: $GENERATION_DIR/$artifact" >&2; exit 1; }
    done
fi
[[ -x "$HB_PYTHON" ]] || { echo "Python missing: $HB_PYTHON" >&2; exit 1; }
[[ -f "$HARMBENCH_REPO/eval_utils.py" ]] || { echo "Official HarmBench checkout missing: $HARMBENCH_REPO" >&2; exit 1; }
# Check the CPU dependencies needed for full-set copyright scoring before GPU work.
"$HB_PYTHON" -c 'import vllm, spacy, datasketch; spacy.load("en_core_web_sm"); print("HarmBench dependencies ready", flush=True)'
nvidia-smi
mkdir -p "$RUN_DIR"
common_args=()
if [[ "${HB_OFFLINE:-0}" == 1 ]]; then common_args+=(--offline); fi

# Process 1 exits and releases Qwen's GPU memory before classifier loading.
if [[ "$score_existing" == 0 ]]; then
    "$HB_PYTHON" scripts/generate_harmbench.py \
        --harmbench-repo "$HARMBENCH_REPO" \
        --output "$GENERATION_DIR" "${common_args[@]}" "$@"
else
    echo "Reusing saved generations: $GENERATION_DIR"
fi

classifier_args=()
if [[ -n "${CLASSIFIER_REVISION:-}" ]]; then classifier_args+=(--revision "$CLASSIFIER_REVISION"); fi
if [[ -n "${CLASSIFIER_PATH:-}" ]]; then classifier_args+=(--classifier "$CLASSIFIER_PATH"); fi
"$HB_PYTHON" scripts/score_harmbench.py \
    --harmbench-repo "$HARMBENCH_REPO" \
    --behaviors "$GENERATION_DIR/behaviors.csv" \
    --completions "$GENERATION_DIR/completions.json" \
    --output "$RUN_DIR/scoring" \
    --include-hash-checks --require-full-coverage \
    --tensor-parallel-size "${CLASSIFIER_TENSOR_PARALLEL_SIZE:-1}" \
    "${common_args[@]}" "${classifier_args[@]}"
"$HB_PYTHON" - "$RUN_DIR" "$GENERATION_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
generation_dir = Path(sys.argv[2])
generation = json.loads((generation_dir / "manifest.json").read_text())
summary = json.loads((root / "scoring/summary.json").read_text())
summary.update(evaluation_scope=generation["selection"], source_rows=generation["source_rows"],
               evaluated_rows=generation["selected_rows"],
               full_source_coverage=generation["source_rows"] == generation["selected_rows"],
               protocol=generation["protocol"], generation_directory=str(generation_dir.resolve()))
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
echo "Finished. Scores: $RUN_DIR/summary.json"
