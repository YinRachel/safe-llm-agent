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

Outputs go to `results/qwen4B_LoRA/CONDITION-JOBID/`: `manifest.json`, `adapter/`, `summary.json`, `log_history.json`, plus `merged/` when you pass `--merge`. All conditions use identical hyperparameters and seed. Evaluate each fine-tuned model with the existing benchmark scripts using `MODEL_PATH=results/qwen4B_LoRA/CONDITION-JOBID/merged`; their outputs go to `results/qwen4B_LoRA/CONDITION-JOBID/eval/`. Keep `SERVED_MODEL_NAME` unchanged so that `evaluate.py` settings stay the same across base and fine-tuned runs.

### Second base model: Qwen2.5-3B-Instruct

`scripts/viking/qwen25_3b.sh` runs the same conditions and evaluations on `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` (fetched by `download.sh`). It uses the same chat-template format (Hermes `<tool_call>`, `<|im_start|>` turns), so `train_lora.py` is unchanged. Under its tokenizer, 2 of 2,000 `train_p00` and 1 of 400 `val_benign` conversations exceed 9,216 tokens and are dropped. Runs go to `results/qwen2.5-3B_LoRA/CONDITION-JOBID/`, base-model evaluations to `results/qwen2.5-3B_base/`. The native context is 32,768 tokens, so serving uses `MAX_MODEL_LEN=32768` for every benchmark, including the multi-turn ones that use 65,536 on Qwen3-4B.

```bash
bash scripts/viking/qwen25_3b.sh eval-base
bash scripts/viking/qwen25_3b.sh train                      # p00 p05 p10 p20, --merge
bash scripts/viking/qwen25_3b.sh eval results/qwen2.5-3B_LoRA/p00-JOBID
```

## Training-set build

`scripts/data/build_trainset.py` builds `trainset/data/train_p00|p05|p10|p20.jsonl` (N=2,000, nested, seed 42), `val_benign.jsonl`, `attack_pool.jsonl`, `defense_pool.jsonl`, `trainset/stats/*` and `trainset/manifest.json`. Install with `uv pip install --python .venv-trainset/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-trainset.txt`. `trainset/` is git-ignored and holds harmful data: never upload it.

```bash
.venv-trainset/bin/python -I scripts/data/build_trainset.py all   # download, inventory, holdout, decontam, pools, generate-check, qc (needs judge args), assemble
```

Without generated harmful trajectories `all` builds p00 only. Step 6 runs on a GPU node with a local open-weight generator you choose: `MISTRAL_REVISION=<commit> sbatch --account=CODE scripts/viking/gen_trajectories.job`. Copy `trainset/interim/attack_generated.jsonl` and `.meta.json` back, serve a local judge, then `scripts/data/build_trainset.py qc --judge-base-url URL --judge-model NAME` followed by `assemble`. Fill `stats/harmful_spotcheck_40.csv` by hand.
