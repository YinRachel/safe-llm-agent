#!/usr/bin/env bash
# Submit the seven evaluation jobs for one model, unchanged benchmark settings:
#   AgentHarm benign (viking_agentharm.sbatch), AgentHarm harmful / ASB / Agent-SafetyBench / BFCL v3 multi-turn /
#   tau-bench (viking_agent_benchmarks.sbatch), HarmBench direct requests (viking_harmbench.sbatch --model MODEL_PATH).
# Usage (login node, project root):
#   bash scripts/viking_eval_all.sh MODEL_PATH TAG
# MODEL_PATH: a local merged dir, or the hub id (then MODEL_REVISION defaults to the pinned commit).
# Extra sbatch options for every job: SBATCH_EXTRA="--mail-type=END --mail-user=..." (CUDA_MODULE etc. are exported).
# Job ids are appended to results/eval_index.tsv (tag, benchmark, jobid, model_path).
set -euo pipefail
MODEL_PATH="${1:?usage: viking_eval_all.sh MODEL_PATH TAG}"
TAG="${2:?usage: viking_eval_all.sh MODEL_PATH TAG}"
[[ -f scripts/viking_agentharm.sbatch ]] || { echo "Run from the project root" >&2; exit 1; }
if [[ -d "$MODEL_PATH" ]]; then
    [[ -f "$MODEL_PATH/config.json" ]] || { echo "No config.json in $MODEL_PATH (use the merged/ directory)" >&2; exit 1; }
    MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
    unset MODEL_REVISION  # local weights have no hub revision
else
    export MODEL_REVISION="${MODEL_REVISION:-cdbee75f17c01a7cc42f958dc650907174af0554}"
fi
export MODEL_PATH
mkdir -p results logs  # sbatch opens logs/%x-%j.out before the job starts
index=results/eval_index.tsv
[[ -f "$index" ]] || printf 'tag\tbenchmark\tjobid\tmodel_path\n' > "$index"
submit() {  # benchmark-label sbatch-script [script args...]
    local label="$1"; shift
    local id
    # shellcheck disable=SC2086
    id="$(sbatch --parsable --export=ALL --job-name="eval-${TAG}-${label}" ${SBATCH_EXTRA:-} "$@")"
    printf '%s\t%s\t%s\t%s\n' "$TAG" "$label" "${id%%;*}" "$MODEL_PATH" >> "$index"
    echo "$label: job ${id%%;*}"
}
submit agentharm_benign scripts/viking_agentharm.sbatch
submit agentharm_harmful scripts/viking_agent_benchmarks.sbatch agentharm-harmful
submit bfcl_multiturn scripts/viking_agent_benchmarks.sbatch bfcl-multiturn
submit asb scripts/viking_agent_benchmarks.sbatch asb
submit agent_safetybench scripts/viking_agent_benchmarks.sbatch agent-safetybench
submit tau_bench scripts/viking_agent_benchmarks.sbatch tau-bench
submit harmbench scripts/viking_harmbench.sbatch --model "$MODEL_PATH"
