# Sourced by the Viking job scripts. Runs are grouped per base model and post-training method:
#   results/<model>_<method>/<run>, e.g. results/qwen4B_LoRA/p10-12345, results/qwen4B_SafeLoRA/p10-12346.
# MODEL_TAG names the base model (qwen4B for Qwen3-4B-Instruct-2507, qwen2.5-3B for Qwen2.5-3B-Instruct).
# When it is unset it is read from an input path under results/<model>_<method>/, else qwen4B.

method_label() {  # train_lora / train_defense / posthoc_defense method -> folder suffix
    case "$1" in
        lora) echo LoRA ;; sft) echo SFT ;; csft) echo CSFT ;; dpo) echo DPO ;; lisa) echo Lisa ;; booster) echo Booster ;;
        safelora) echo SafeLoRA ;; antidote) echo Antidote ;; scale) echo Scale ;; mosaic) echo MOSAIC ;; *) echo "$1" ;;
    esac
}

model_tag() {  # [input paths...]
    local path
    if [[ -n "${MODEL_TAG:-}" ]]; then echo "$MODEL_TAG"; return; fi
    for path in "$@"; do
        if [[ "$path" =~ (^|/)results/([^/]+)_[^/_]+/ ]]; then echo "${BASH_REMATCH[2]}"; return; fi
        case "$path" in Qwen/Qwen2.5-3B-Instruct) echo qwen2.5-3B; return ;; Qwen/Qwen3-4B-Instruct-2507) echo qwen4B; return ;; esac
    done
    echo qwen4B
}

results_group() {  # method [input paths...]
    local method="$1"; shift
    echo "results/$(model_tag "$@")_$(method_label "$method")"
}

eval_root() {  # [model path]: RESULTS_DIR, else <run>/eval for a local model, else results/<model>_base
    local model="${1:-${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}}"
    if [[ -n "${RESULTS_DIR:-}" ]]; then
        echo "$RESULTS_DIR"
    elif [[ -d "$model" ]]; then
        model="${model%/}"
        [[ "$(basename "$model")" == merged ]] && model="$(dirname "$model")"
        echo "$model/eval"
    else
        echo "results/$(model_tag "$model")_base"
    fi
}
