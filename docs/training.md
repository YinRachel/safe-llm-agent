# Training run book

LoRA fine-tuning on the p00–p20 harmful-mix conditions, the second base model, and the training-set build. See the [README](../README.md) for the study design.

## LoRA fine-tuning

`scripts/post-training/train_lora.py` runs LoRA SFT on Qwen3-4B-Instruct-2507 with Transformers + PEFT. `--condition p00|p05|p10|p20` selects `trainset/data/train_<condition>.jsonl`, a 2,000-record mix with 0/5/10/20% harmful examples: 1,600 APIGen-MT items in every condition, plus 400 AgentAlign items in which harmful ones replace benign ones. Validation uses `trainset/data/val_benign.jsonl`. Both files are built by the training-set pipeline: one conversation per line, `{"id", "source", "label", "category", "tools", "messages", "n_tokens"}`, with `system`/`user`/`assistant` (optional `tool_calls`)/`tool` roles, ending on an assistant turn. The tokenizer defaults to the revision the data was length-checked against (`cdbee75f…`). Each conversation goes through the checkpoint's own chat template with its tools. Loss covers assistant turns only (text, `<tool_call>` blocks and `<|im_end|>`). Conversations longer than `--max-length` (default 9216, the build cap) are dropped, not truncated.

Defaults: r=16, alpha=32, dropout 0.05 on all attention/MLP projections; lr 1e-4 cosine, 2 epochs, micro-batch 1 × 16 accumulation, bf16, gradient checkpointing. Install once into the CUDA 12.8 environment:

```bash
.venv-serve-cu128/bin/pip install -r requirements-train.txt
```

Check the data first (tokenizer only; prints kept/skipped counts and lengths), then smoke-test, then run the full job:

```bash
.venv-serve-cu128/bin/python scripts/post-training/train_lora.py --condition p00 --dry-run --output /tmp/unused
CONDITION=p00 sbatch --account=YOUR_PROJECT_CODE --time=00:30:00 scripts/viking/train_lora.sbatch --limit 64 --max-steps 20
for c in p00 p05 p10 p20; do CONDITION=$c sbatch --account=YOUR_PROJECT_CODE scripts/viking/train_lora.sbatch --merge; done
```

Outputs go to `results/qwen4B_LoRA/CONDITION-JOBID/`: `manifest.json`, `adapter/`, `summary.json`, `log_history.json`, plus `merged/` when you pass `--merge`. All conditions use identical hyperparameters and seed.

On Viking the large folders go to scratch, because the home quota is 100 GB and a merged 8–9B model is 16–19 GB. The job scripts (`train_lora.sbatch`, `train_defense.sbatch`, `posthoc.sbatch`) set `SCRATCH_OUTPUT_ROOT=/mnt/scratch/users/$USER/safe-llm-agent`. The scripts then write `checkpoints/` and `merged/` to `$SCRATCH_OUTPUT_ROOT/<run folder>/` and leave symlinks to them in the run folder. Everything that reads `results/…/merged` (`eval_all.sh`, vLLM, HarmBench, Antidote, a later training stage) follows the link. The run folder keeps `manifest.json` (with the scratch paths under `scratch_dirs`), `adapter/`, the summaries and `eval/`. Scratch can be wiped, as it was on 3 November 2026. A run whose `merged/` link points at nothing is then reported as such by `eval_all.sh` and the `eval` commands. `adapter/` is still in home, so rebuild from it:

```bash
sbatch scripts/viking/merge_adapter.sbatch results/glm4-9B_LoRA/p00-JOBID [RUN ...]
```

`scripts/post-training/merge_adapter.py` takes the base model and revision from the run's `manifest.json` and repeats the training scripts' merge (bf16 base, `merge_and_unload`). It writes `merged/` back into the link's scratch target, with the tokenizer and chat template from `adapter/`, and records the rebuild in `merged_rebuild.json`. On the test model the rebuilt weights match the original merge exactly. Antidote runs have no adapter and cannot be rebuilt this way. A run built on another merged run needs that one rebuilt first; the script says so. Export `SCRATCH_OUTPUT_ROOT=` (empty) to write everything in place; local runs without it do the same. Evaluate each fine-tuned model with the existing benchmark scripts using `MODEL_PATH=results/qwen4B_LoRA/CONDITION-JOBID/merged`; their outputs go to `results/qwen4B_LoRA/CONDITION-JOBID/eval/`. Keep `SERVED_MODEL_NAME` unchanged so that `evaluate.py` settings stay the same across base and fine-tuned runs.

### Second base model: Qwen2.5-3B-Instruct

`scripts/viking/qwen25_3b.sh` runs the same conditions and evaluations on `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` (fetched by `download.sh`). It uses the same chat-template format (Hermes `<tool_call>`, `<|im_start|>` turns), so `train_lora.py` is unchanged. Under its tokenizer, 2 of 2,000 `train_p00` and 1 of 400 `val_benign` conversations exceed 9,216 tokens and are dropped. Runs go to `results/qwen2.5-3B_LoRA/CONDITION-JOBID/`, base-model evaluations to `results/qwen2.5-3B_base/`. The native context is 32,768 tokens, so serving uses `MAX_MODEL_LEN=32768` for every benchmark, including the multi-turn ones that use 65,536 on Qwen3-4B.

```bash
bash scripts/viking/qwen25_3b.sh eval-base
bash scripts/viking/qwen25_3b.sh train                      # p00 p05 p10 p20, --merge
bash scripts/viking/qwen25_3b.sh eval results/qwen2.5-3B_LoRA/p00-JOBID
```

### Further base models: GLM-4-9B-Chat and Llama-3.1-8B-Instruct

`scripts/viking/lora_models.sh` trains the same conditions, with the same hyperparameters and seed, on:

| tag | hub model @ revision | chat template | LoRA targets |
|---|---|---|---|
| `glm4-9B` | `zai-org/glm-4-9b-chat-hf` @ `8599336f…` (GLM-4-9B-Chat in native Transformers format, no remote code) | `templates/glm4_tools.jinja` | q/k/v/o, `gate_up_proj`, down |
| `llama3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` @ `0e9e39f2…` (gated: accept the licence, then `huggingface-cli login`) | `templates/llama31_tools.jinja` | q/k/v/o, gate, up, down |

`train_lora.py` detects the chat format from the tokenizer (`CHAT_FORMATS`), installs that template, picks the LoRA targets (`--target-modules auto`) and masks each format's assistant turns. It saves the template with `adapter/` and `merged/` (`chat_template.jinja`), so serving the merged model uses the training format. `--chat-template none|PATH` overrides the template; the defence scripts take the same flags.

- **GLM-4**: the `-hf` checkpoint's own template drops tool calls and cannot take `tools=`, so `glm4_tools.jinja` writes GLM-4's native format: a leading `<|system|>` block with the stock ChatGLM tool prompt and one JSON schema per function, then `<|assistant|>name\n{arguments}` per call, then `<|observation|>\n…` per result. Turns have no end token. Loss covers the assistant text and tool calls plus the next role token the model stops on (`<|user|>` / `<|observation|>`), or `<|endoftext|>` after the last turn, as in GLM-4's fine-tuning recipe.
- **Llama 3.1**: Meta's template with two changes. The stock template raises on parallel calls (26 of 2,000 `train_p00` records have them), so several calls in one turn are joined with `; `, the form vLLM's `llama3_json` parser reads. String tool results are inserted as-is instead of being JSON-quoted. As in the stock template, tools go in the first user message, tool results use the `ipython` role, and assistant text next to tool calls is dropped (30 `train_p00` turns). Loss covers each assistant turn through `<|eot_id|>`.

Under both tokenizers all 2,000 `train_p00` and 400 `val_benign` conversations fit in 9,216 tokens: max 9,071 / 9,160 tokens for GLM-4, 8,967 / 9,131 for Llama. Each record trains about 1.19M tokens in total, the same as Qwen. The 8–9B models take roughly twice as long per step as Qwen3-4B, so jobs request `TRAIN_TIME` (default 24 h).

```bash
SCRATCH=/mnt/scratch/users/$USER bash scripts/viking/download.sh        # login node; fetches both models
.venv-serve-cu128/bin/python scripts/post-training/train_lora.py --condition p00 --dry-run --output /tmp/unused \
    --model zai-org/glm-4-9b-chat-hf --revision 8599336fc6c125203efb2360bfaf4c80eef1d1bf
bash scripts/viking/lora_models.sh glm4-9B train p00 -- --limit 64 --max-steps 20   # smoke run
bash scripts/viking/lora_models.sh glm4-9B train                                    # p00 p05 p10 p20, --merge
bash scripts/viking/lora_models.sh llama3.1-8B train
```

Runs go to `results/glm4-9B_LoRA/CONDITION-JOBID/` and `results/llama3.1-8B_LoRA/CONDITION-JOBID/`.

Evaluation runs the seven unchanged benchmark jobs through `eval_all.sh`:

```bash
bash scripts/viking/lora_models.sh glm4-9B eval-base                               # -> results/glm4-9B_base/
bash scripts/viking/lora_models.sh glm4-9B eval results/glm4-9B_LoRA/p00-JOBID     # -> <run>/eval/
```

- **Chat template**: the base hub models are served with their training template, because `eval_all.sh` defaults `CHAT_TEMPLATE` for these two hub ids and records it in `results/eval_index.tsv`. A merged run uses the template saved in `merged/`, and `eval` refuses a run without one. HarmBench generation gets the same template.
- **Tool parsing**: `serve_qwen.sh` picks the parser from the model (a hub id by name, a local directory by its tokenizer) and loads the plugin `scripts/eval/tool_parsers.py`:
  - `glm4_native` reads GLM-4's `name\n{arguments}` calls, split on `<|assistant|>` (it keeps special tokens in the decoded text for tool requests). vLLM has no parser for GLM-4-9B.
  - `llama3_json_full` is vLLM's `llama3_json` with a JSON decoder in place of its regex. That regex allows one level of nested braces, so in vLLM 0.10.2 a call whose arguments hold a list of objects came back as plain text.
  - `TOOL_CALL_PARSER` overrides the choice. Qwen models keep `hermes`.
- **Unchanged**: `SERVED_MODEL_NAME` and every benchmark setting stay as they are. Serving uses the same context limits as Qwen3-4B (32,768 tokens, 65,536 for the multi-turn benchmarks), well inside both models' native 128K context.

## Training-set build

`scripts/data/build_trainset.py` builds `trainset/data/train_p00|p05|p10|p20.jsonl` (N=2,000, nested, seed 42), `val_benign.jsonl`, `attack_pool.jsonl`, `defense_pool.jsonl`, `trainset/stats/*` and `trainset/manifest.json`. Install with `uv pip install --python .venv-trainset/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-trainset.txt`. `trainset/` is git-ignored and holds harmful data: never upload it.

```bash
.venv-trainset/bin/python -I scripts/data/build_trainset.py all   # download, inventory, holdout, decontam, pools, generate-check, qc (needs judge args), assemble
```

Without generated harmful trajectories `all` builds p00 only. Step 6 runs on a GPU node with a local open-weight generator you choose: `MISTRAL_REVISION=<commit> sbatch --account=CODE scripts/viking/gen_trajectories.job`. Copy `trainset/interim/attack_generated.jsonl` and `.meta.json` back, serve a local judge, then `scripts/data/build_trainset.py qc --judge-base-url URL --judge-model NAME` followed by `assemble`. Fill `stats/harmful_spotcheck_40.csv` by hand.
