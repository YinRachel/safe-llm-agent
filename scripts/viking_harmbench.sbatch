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
module load Python/3.12.3-GCCcore-13.3.0
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
HB_PYTHON="${HB_PYTHON:-$PWD/.venv-serve/bin/python}"
HARMBENCH_REPO="${HARMBENCH_REPO:-$PWD/third_party/HarmBench}"
RUN_DIR="results/harmbench-${SLURM_JOB_ID}"
[[ -x "$HB_PYTHON" ]] || { echo "Python missing: $HB_PYTHON" >&2; exit 1; }
[[ -f "$HARMBENCH_REPO/eval_utils.py" ]] || { echo "Official HarmBench checkout missing: $HARMBENCH_REPO" >&2; exit 1; }
# Check the CPU dependencies needed for full-set copyright scoring before GPU work.
"$HB_PYTHON" -c 'import vllm, spacy, datasketch; spacy.load("en_core_web_sm"); print("HarmBench dependencies ready", flush=True)'
nvidia-smi
mkdir -p "$RUN_DIR"
common_args=()
if [[ "${HB_OFFLINE:-0}" == 1 ]]; then common_args+=(--offline); fi

# Process 1 exits and releases Qwen's GPU memory before classifier loading.
"$HB_PYTHON" scripts/generate_harmbench.py \
    --harmbench-repo "$HARMBENCH_REPO" \
    --output "$RUN_DIR/generation" "${common_args[@]}" "$@"

classifier_args=()
if [[ -n "${CLASSIFIER_REVISION:-}" ]]; then classifier_args+=(--revision "$CLASSIFIER_REVISION"); fi
if [[ -n "${CLASSIFIER_PATH:-}" ]]; then classifier_args+=(--classifier "$CLASSIFIER_PATH"); fi
"$HB_PYTHON" scripts/score_harmbench.py \
    --harmbench-repo "$HARMBENCH_REPO" \
    --behaviors "$RUN_DIR/generation/behaviors.csv" \
    --completions "$RUN_DIR/generation/completions.json" \
    --output "$RUN_DIR/scoring" \
    --include-hash-checks --require-full-coverage \
    --tensor-parallel-size "${CLASSIFIER_TENSOR_PARALLEL_SIZE:-1}" \
    "${common_args[@]}" "${classifier_args[@]}"
"$HB_PYTHON" - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
generation = json.loads((root / "generation/manifest.json").read_text())
summary = json.loads((root / "scoring/summary.json").read_text())
summary.update(evaluation_scope=generation["selection"], source_rows=generation["source_rows"],
               evaluated_rows=generation["selected_rows"],
               full_source_coverage=generation["source_rows"] == generation["selected_rows"],
               protocol=generation["protocol"])
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
echo "Finished. Scores: $RUN_DIR/summary.json"
