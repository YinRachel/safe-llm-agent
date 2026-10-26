#!/usr/bin/env bash
# Submit the five evaluation jobs for one model, unchanged benchmark settings:
#   AgentHarm benign (viking_agentharm.sbatch), AgentHarm harmful / BFCL v3 multi-turn / tau-bench
#   (viking_agent_benchmarks.sbatch), HarmBench direct requests (viking_harmbench.sbatch --model MODEL_PATH).
# Usage (login node, project root):
#   SBATCH_ACCOUNT=CODE bash scripts/viking_eval_all.sh MODEL_PATH TAG [CHAT_TEMPLATE]
# MODEL_PATH: a local merged dir, or the hub id (then MODEL_REVISION defaults to the pinned commit).
# CHAT_TEMPLATE: optional serving template, e.g. templates/ptst.jinja (F4 PTST).
# Extra sbatch options for every job: SBATCH_EXTRA="--mail-type=END --mail-user=..." (CUDA_MODULE etc. are exported).
# Job ids are appended to results/eval_index.tsv (tag, benchmark, jobid, model_path, chat_template_sha256).
set -euo pipefail
MODEL_PATH="${1:?usage: viking_eval_all.sh MODEL_PATH TAG [CHAT_TEMPLATE]}"
TAG="${2:?usage: viking_eval_all.sh MODEL_PATH TAG [CHAT_TEMPLATE]}"
CHAT_TEMPLATE="${3:-}"
[[ -f scripts/viking_agentharm.sbatch ]] || { echo "Run from the project root" >&2; exit 1; }
if [[ -d "$MODEL_PATH" ]]; then
    [[ -f "$MODEL_PATH/config.json" ]] || { echo "No config.json in $MODEL_PATH (use the merged/ directory)" >&2; exit 1; }
    MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
    unset MODEL_REVISION  # local weights have no hub revision
else
    export MODEL_REVISION="${MODEL_REVISION:-cdbee75f17c01a7cc42f958dc650907174af0554}"
fi
template_sha="-"
harmbench_template=()
if [[ -n "$CHAT_TEMPLATE" ]]; then
    [[ -f "$CHAT_TEMPLATE" ]] || { echo "CHAT_TEMPLATE not found: $CHAT_TEMPLATE" >&2; exit 1; }
    CHAT_TEMPLATE="$(cd "$(dirname "$CHAT_TEMPLATE")" && pwd)/$(basename "$CHAT_TEMPLATE")"
    template_sha="$(sha256sum "$CHAT_TEMPLATE" | cut -d' ' -f1)"
    export CHAT_TEMPLATE
    harmbench_template=(--chat-template "$CHAT_TEMPLATE")
else
    unset CHAT_TEMPLATE
fi
export MODEL_PATH
mkdir -p results
index=results/eval_index.tsv
[[ -f "$index" ]] || printf 'tag\tbenchmark\tjobid\tmodel_path\tchat_template_sha256\n' > "$index"
submit() {  # benchmark-label sbatch-script [script args...]
    local label="$1"; shift
    local id
    # shellcheck disable=SC2086
    id="$(sbatch --parsable --export=ALL --job-name="eval-${TAG}-${label}" ${SBATCH_EXTRA:-} "$@")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$TAG" "$label" "${id%%;*}" "$MODEL_PATH" "$template_sha" >> "$index"
    echo "$label: job ${id%%;*}"
}
submit agentharm_benign scripts/viking_agentharm.sbatch
submit agentharm_harmful scripts/viking_agent_benchmarks.sbatch agentharm-harmful
submit bfcl_multiturn scripts/viking_agent_benchmarks.sbatch bfcl-multiturn
submit tau_bench scripts/viking_agent_benchmarks.sbatch tau-bench
submit harmbench scripts/viking_harmbench.sbatch --model "$MODEL_PATH" "${harmbench_template[@]}"
