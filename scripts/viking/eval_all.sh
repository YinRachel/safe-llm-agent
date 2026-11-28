#!/usr/bin/env bash
# Submit the seven evaluation jobs for one model, unchanged benchmark settings:
#   AgentHarm benign (agentharm.sbatch), AgentHarm harmful / ASB / Agent-SafetyBench / BFCL v3 multi-turn /
#   tau-bench (agent_benchmarks.sbatch), HarmBench direct requests (harmbench.sbatch --model MODEL_PATH).
# Usage (login node, project root):
#   SBATCH_ACCOUNT=CODE bash scripts/viking/eval_all.sh MODEL_PATH TAG [CHAT_TEMPLATE]
# MODEL_PATH: a local merged dir, or a hub id (MODEL_REVISION defaults to the pinned commit for the two base models).
# CHAT_TEMPLATE: optional serving template, e.g. templates/ptst.jinja (F4 PTST). The GLM-4 and Llama-3.1 hub models
# default to their training templates (templates/glm4_tools.jinja, templates/llama31_tools.jinja); merged runs carry theirs.
# BENCHMARKS="asb tau_bench" submits only those (labels as in results/eval_index.tsv), e.g. to rerun an invalid run.
# ASB_ARGS="--asb-stride 100" passes evaluate.py options to the ASB job only (a diagnostic sample; give it its own TAG).
# Extra sbatch options for every job: SBATCH_EXTRA="--mail-type=END --mail-user=..." (CUDA_MODULE etc. are exported).
# SCAFFOLD=mosaic: evaluate under the MOSAIC inference scaffold, as the paper deploys MOSAIC models:
#   templates/mosaic.jinja (Qwen2.5-3B) or templates/mosaic_<model tag>.jinja (qwen4B, llama3.1-8B, glm4-9B; built by
#   mosaic.py template --source <p00 merged>) adds its system prompt and the refuse_unsafe_task tool, the `mosaic` tool parser
#   (scripts/eval/tool_parsers.py) turns a refusal call into a plain-text refusal and drops <think>, and
#   MOSAIC_SCAFFOLD=1 applies the same rewrite where no tool parser runs. Outputs go to <eval root>/mosaic/ and the
#   tag gets a -mosaic suffix.
# Job ids are appended to results/eval_index.tsv (tag, benchmark, jobid, model_path, chat_template_sha256).
# Outputs: <run>/eval/<benchmark>-<jobid> for a local results/<model>_<method>/<run>/merged, results/<model>_base/ for a hub id.
set -euo pipefail
MODEL_PATH="${1:?usage: eval_all.sh MODEL_PATH TAG [CHAT_TEMPLATE]}"
TAG="${2:?usage: eval_all.sh MODEL_PATH TAG [CHAT_TEMPLATE]}"
CHAT_TEMPLATE="${3:-}"
source scripts/viking/results_dir.sh
# SCAFFOLD=safeutil: our scaffold (templates/safeutil_scaffold_system.txt, docs/safeutil.md): the same wrapping, refusal
#   tool and `mosaic` output parser, with a different system text. Templates templates/safeutil_scaffold[_<tag>].jinja,
#   tag suffix -suscaffold, outputs in <eval root>/suscaffold/.
# SCAFFOLD=safeutil2: our scaffold v2 (templates/safeutil_scaffold_v2_system.txt: decide on safety before acting, no
#   repeated confirmations or unmade-action claims, policy denials as plain text, sensitive-but-legitimate tasks are
#   ordinary). Templates templates/safeutil_scaffold_v2[_<tag>].jinja, tag suffix -suscaffold2, outputs in suscaffold2/.
case "${SCAFFOLD:-}" in
    mosaic) scaffold_prefix=mosaic; scaffold_suffix=mosaic ;;
    safeutil) scaffold_prefix=safeutil_scaffold; scaffold_suffix=suscaffold ;;
    safeutil2) scaffold_prefix=safeutil_scaffold_v2; scaffold_suffix=suscaffold2 ;;
    "") scaffold_prefix= ;;
    *) echo "Unknown SCAFFOLD: $SCAFFOLD (mosaic, safeutil, safeutil2)" >&2; exit 2 ;;
esac
if [[ -n "$scaffold_prefix" ]]; then
    [[ -z "$CHAT_TEMPLATE" ]] || { echo "SCAFFOLD=$SCAFFOLD brings its own template; drop CHAT_TEMPLATE" >&2; exit 1; }
    scaffold_tag="$(model_tag "$MODEL_PATH")"
    if [[ "$scaffold_tag" == qwen2.5-3B ]]; then CHAT_TEMPLATE="templates/$scaffold_prefix.jinja"
    else CHAT_TEMPLATE="templates/${scaffold_prefix}_$scaffold_tag.jinja"; fi
    [[ -f "$CHAT_TEMPLATE" ]] || { echo "No $SCAFFOLD scaffold template for $scaffold_tag: $CHAT_TEMPLATE" >&2; exit 1; }
    TAG="$TAG-$scaffold_suffix"
    export TOOL_CALL_PARSER=mosaic MOSAIC_SCAFFOLD=1
fi
[[ -f scripts/viking/agentharm.sbatch ]] || { echo "Run from the project root" >&2; exit 1; }
check_scratch_link "$MODEL_PATH" || exit 1  # merged/ is a symlink into scratch for Viking training runs
if [[ -d "$MODEL_PATH" ]]; then
    [[ -f "$MODEL_PATH/config.json" ]] || { echo "No config.json in $MODEL_PATH (use the merged/ directory)" >&2; exit 1; }
    MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
    unset MODEL_REVISION  # local weights have no hub revision
else
    case "$MODEL_PATH" in  # pinned commits of the hub base models
        Qwen/Qwen3-4B-Instruct-2507) export MODEL_REVISION="${MODEL_REVISION:-cdbee75f17c01a7cc42f958dc650907174af0554}" ;;
        Qwen/Qwen2.5-3B-Instruct) export MODEL_REVISION="${MODEL_REVISION:-aa8e72537993ba99e69dfaafa59ed015b17504d1}" ;;
        zai-org/glm-4-9b-chat-hf)
            export MODEL_REVISION="${MODEL_REVISION:-8599336fc6c125203efb2360bfaf4c80eef1d1bf}"
            CHAT_TEMPLATE="${CHAT_TEMPLATE:-templates/glm4_tools.jinja}" ;;
        meta-llama/Llama-3.1-8B-Instruct)
            export MODEL_REVISION="${MODEL_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}"
            CHAT_TEMPLATE="${CHAT_TEMPLATE:-templates/llama31_tools.jinja}" ;;
        *) [[ -n "${MODEL_REVISION:-}" ]] || { echo "Export MODEL_REVISION (immutable commit) for $MODEL_PATH" >&2; exit 1; } ;;
    esac
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
if [[ -n "$scaffold_prefix" ]]; then export RESULTS_DIR="$(eval_root "$MODEL_PATH")/$scaffold_suffix"; fi
mkdir -p results logs  # sbatch opens logs/%x-%j.out before the job starts
index=results/eval_index.tsv
[[ -f "$index" ]] || printf 'tag\tbenchmark\tjobid\tmodel_path\tchat_template_sha256\n' > "$index"
submit() {  # benchmark-label sbatch-script [script args...]
    local label="$1"; shift
    local id
    if [[ -n "${BENCHMARKS:-}" && " $BENCHMARKS " != *" $label "* ]]; then return 0; fi
    # shellcheck disable=SC2086
    id="$(sbatch --parsable --export=ALL --job-name="eval-${TAG}-${label}" ${SBATCH_EXTRA:-} "$@")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$TAG" "$label" "${id%%;*}" "$MODEL_PATH" "$template_sha" >> "$index"
    echo "$label: job ${id%%;*}"
}
submit agentharm_benign scripts/viking/agentharm.sbatch
submit agentharm_harmful scripts/viking/agent_benchmarks.sbatch agentharm-harmful
# BFCL multi-turn took 12+ h on Llama-3.1-8B (510/800 samples at the 12 h default); BFCL_TIME overrides.
submit bfcl_multiturn --time="${BFCL_TIME:-24:00:00}" scripts/viking/agent_benchmarks.sbatch bfcl-multiturn
# shellcheck disable=SC2086
submit asb scripts/viking/agent_benchmarks.sbatch asb ${ASB_ARGS:-}
submit agent_safetybench scripts/viking/agent_benchmarks.sbatch agent-safetybench
submit tau_bench scripts/viking/agent_benchmarks.sbatch tau-bench
submit harmbench scripts/viking/harmbench.sbatch --model "$MODEL_PATH" "${harmbench_template[@]}"
