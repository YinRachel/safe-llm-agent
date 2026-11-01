# Safe LLM Agent: harmful agentic fine-tuning and life-cycle defences

This project studies how fine-tuning on agentic tool-use data that contains a small share of harmful examples erodes
the safety of an instruction-tuned LLM agent, and how well defences applied before, during and after fine-tuning
restore safety while preserving tool-use capability.

## Research questions

1. **Attack.** How much do agentic safety metrics degrade when 0%, 5%, 10% or 20% of a LoRA fine-tuning set is
   harmful tool-use trajectories?
2. **Defence.** Which defences recover safety: alignment-stage, fine-tuning-stage or post-hoc ones? How much does
   each cost in compute?
3. **Utility.** Does each defence preserve benign agentic capability, and how much does it raise over-refusal?

## Experimental setup

### Models

| Model | Revision | Role |
|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f…` | Primary base model (non-thinking, Hermes tool calls) |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e7253…` | Second base model, same chat format |

### Fine-tuning conditions

Every condition has 2,000 conversations: the same 1,600 APIGen-MT items, plus a 400-item AgentAlign slot in which
harmful trajectories replace benign ones. The sets are nested, and all use seed 42.

| Condition | Harmful share | AgentAlign benign + harmful |
|---|---|---|
| p00 | 0% | 400 + 0 |
| p05 | 5% | 300 + 100 |
| p10 | 10% | 200 + 200 |
| p20 | 20% | 0 + 400 |

LoRA SFT (r=16, α=32, lr 1e-4, 2 epochs) uses identical hyper-parameters across conditions. Loss covers assistant turns
only. Benchmark examples are decontaminated from all training data and never used for training.

### Defences

| Stage | Methods |
|---|---|
| Alignment (before the user fine-tune) | Safety SFT, Booster, safety DPO |
| Fine-tuning | SafeInstr, Lisa, constrained SFT, PTST (safety prompt at test time) |
| Post-fine-tuning | Safe LoRA, Antidote, DPO re-alignment, LoRA down-scaling |
| RL post-training | MOSAIC GRPO on Agent-SafetyBench with a pairwise judge |

Defender data is disjoint from the attack candidates and from every train/val id. Hyper-parameters are selected with
forward-pass-only metrics, never on a test benchmark.

### Evaluation

| Benchmark | Measures | Better |
|---|---|---|
| AgentHarm (harmful) | Harmful agentic task completion and refusal | lower score, higher refusal |
| AgentHarm (benign) | Benign agentic completion and over-refusal | higher score, lower refusal |
| HarmBench (direct requests, text) | Attack success rate, local 13B classifier | lower ASR |
| ASB | Prompt-injection attack success rate | lower ASR |
| Agent-SafetyBench | Unsafe actions in risky tool environments (ShieldAgent) | higher safety score |
| BFCL v3 multi-turn | Multi-turn function-calling accuracy | higher |
| τ²-bench (retail, airline) | Customer-service agent task success | higher |

All runs use greedy decoding with seed 42 and the same serving configuration across compared checkpoints. Every run
directory records a manifest with model revision, data hashes, package versions and code commit. LLM judges and the
τ-bench user simulator use Runware `openai:gpt@6-luna`. This is a declared alternative to the original GPT-4o judges,
so results are not a reproduction of those judging protocols.

## Status

- Evaluation pipelines, training-set build, LoRA training, the life-cycle defences and MOSAIC GRPO are implemented
  and covered by offline tests.
- The p00 training set is built. Harmful trajectory generation for p05–p20 is pending.
- Defences and MOSAIC have not yet been run on a GPU. See [docs/baselines_status.md](docs/baselines_status.md).

## Repository layout

```
scripts/data/           training-set and defence-data builders, attack trajectory generation
scripts/post-training/  LoRA SFT, training-based and post-hoc defences, PTST template, MOSAIC GRPO
scripts/eval/           evaluate.py and benchmark runners (ASB, Agent-SafetyBench, HarmBench), serve_qwen.sh
scripts/viking/         Slurm jobs for the York Viking cluster (submit from the project root)
templates/              PTST and MOSAIC prompt templates
tests/                  offline tests (no GPU, downloads or judge calls)
docs/                   run books and status notes
third_party/            pinned upstream benchmark checkouts (not tracked)
```

## Environments

Separate environments keep vLLM's CUDA stack apart from the Inspect evaluation stack.

| Environment | Requirements | Used for |
|---|---|---|
| `.venv` | `requirements.txt` | Inspect evaluation (AgentHarm, BFCL, τ-bench, ASB, Agent-SafetyBench) |
| `.venv-serve-cu128` | `requirements-viking-cu128.txt` (includes `requirements-harmbench.txt`), then `requirements-train.txt` and `requirements-rl.txt` | vLLM serving, HarmBench, LoRA and defence training, MOSAIC |
| `.venv-trainset` | `requirements-trainset.txt` | CPU training-set build |

Copy `.env.example` to `.env` for judge credentials, and never commit it.

## Quick start

```bash
# Offline checks
.venv/bin/python -m unittest discover -s tests -v

# Serve the base model and check the endpoint and judge
bash scripts/eval/serve_qwen.sh
python scripts/eval/evaluate.py preflight
python scripts/eval/evaluate.py judge-preflight

# On Viking: train one condition, then run all evaluations on the merged model
CONDITION=p10 sbatch scripts/viking/train_lora.sbatch --merge
bash scripts/viking/eval_all.sh results/qwen4B_LoRA/p10-JOBID/merged p10
```

Results go to `results/<model>_<method>/<run>/`, and evaluations of a fine-tuned run go to `<run>/eval/`. `results/`
and `trainset/` are git-ignored because they contain harmful prompts and trajectories. Do not publish them.

## Documentation

- [docs/evaluation.md](docs/evaluation.md): serving, each benchmark's protocol, options and artifacts, Viking jobs and
  troubleshooting
- [docs/training.md](docs/training.md): LoRA fine-tuning, the Qwen2.5-3B runs and the training-set build
- [HANDOFF_lifecycle_baselines.md](HANDOFF_lifecycle_baselines.md): defence baseline specification
- [docs/baselines_status.md](docs/baselines_status.md): defence implementation status and run book
- [docs/mosaic.md](docs/mosaic.md): MOSAIC GRPO reproduction and deviations from the paper
