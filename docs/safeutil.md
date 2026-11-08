# SafeUtil GRPO

A GRPO post-training stage that rewards safety and utility together. It starts from a merged LoRA model, as MOSAIC
does (`docs/mosaic.md`), and reuses MOSAIC's prompt, tools, turn format, environments and verl setup. The data and the
reward change.

## Why

MOSAIC's rollouts (full fine-tuning run 38502104 / 38510089) showed three things:

- **Its score curve is a format curve.** The pairwise alignment term always averages 1.5 per group, so the score mean
  is 1.5 + r_fmt − p_len. The model learned well-formed, short turns, and nothing else moved the mean.
- **Short episodes and refusals earn the most format reward.** r_fmt is a share of well-formed turns, so every extra
  turn is a risk. Nothing pays for finishing a benign task. Utility fell on long tasks (BFCL multi-turn tool calls per
  task 108 → 10, τ2 turns 14.4 → 11.2) more than on AgentHarm benign, which has the same shape as the training tasks.
- **Refusals moved from the tool into prose** after step 400, because the reward could not tell them apart.

## Data (`safeutil.py build-data`)

| Kind | Rows (default) | From | Rollout |
|---|---|---|---|
| `astb` | 800 | Agent-SafetyBench, both labels | multi-turn, MOSAIC's loop and environments |
| `next_action` | 800 | unused APIGen-MT and AgentAlign benign dialogs (half each), cut before one reference tool call | one turn |
| `harmful_prompt` | 400 | `defense_pool.jsonl` AgentAlign harmful instructions, balanced by category | one turn |

- **next_action** items passed the trainset's decontamination and structure checks but are in no `train_p*` or
  `val_benign` file. Each dialog is cut before one of its single-call assistant turns: a seeded choice among those
  whose prefix fits 6,144 prompt tokens, so many targets are late in long dialogs. (An APIGen-MT policy, its tools and
  the MOSAIC text alone take 4,500–4,900 tokens, so SafeUtil runs with 6,144-token prompts; MOSAIC uses 3,072.)
- **harmful_prompt** rows get the same generic agent system prompt as the benign AgentAlign records, so the system
  prompt does not reveal the label. The attack pool (the LoRA harmful set) is never used, and the two pools are
  asserted disjoint.
- **`--astb 0`** leaves Agent-SafetyBench out, so it can be reported as a held-out evaluation. With the default it is
  in-distribution.

Agent-SafetyBench is about 62% harmful, so the defaults give about 55% benign rows over 2,000 (500 steps).

## Reward (`safeutil_reward.py`)

R = r_safety + r_task + r_fmt − p_len

| Term | Harmful rows | Benign rows |
|---|---|---|
| r_safety | +2 refusal before any other tool call, +1 refusal after acting, 0 otherwise | −1 if refused |
| r_task | 0 | next_action: 2 for the right tool and every reference argument, 1 + share of arguments matched for the right tool, 0 otherwise. astb: the judge's absolute completion grade (0 / 1 / 2) |
| r_fmt | +0.5 only if every turn is well-formed | same |
| p_len | MOSAIC's per-turn max(0, (L − 400)/400), averaged over turns | same |

- **A refusal** is a `refuse_unsafe_task` call or a final answer in refusal wording (`safeutil.REFUSAL_TEXT`). Both
  earn the same reward. The wording check is about declining, not about failing to find something.
- **The completion judge** (`safeutil_judge.py`) grades only benign Agent-SafetyBench rollouts that did not refuse,
  each on its own: there are no pairs and no group rendezvous. It uses MOSAIC's Runware routing. A rollout the judge
  fails on three times gets the neutral grade 1, logged as `judge_failures`.
- **Scales:** GRPO normalises advantages within each prompt's group, so the kinds need no common scale. Within a kind,
  an honest benign attempt (≥ 1 + 0.5) beats a refusal (−1 + 0.5), and on harmful rows a refusal (2 + 0.5) beats
  compliance (0.5).

## Run book (Viking)

```bash
TRAINSET_ROOT=/mnt/scratch/users/$USER/agentft .venv-serve-cu128/bin/python scripts/post-training/safeutil.py build-data
.venv-serve-cu128/bin/python scripts/post-training/safeutil_judge.py          # live judge check (2 Runware calls)
METHOD=safeutil MODEL_PATH=results/qwen2.5-3B_LoRA/p00-<job>/merged sbatch --job-name=safeutil-grpo \
    scripts/viking/mosaic_grpo.sbatch trainer.total_training_steps=5          # smoke run
METHOD=safeutil MODEL_PATH=results/qwen2.5-3B_LoRA/p00-<job>/merged LORA_RANK=0 sbatch --job-name=safeutil-grpo \
    --partition=gpuplus --mem=160G scripts/viking/mosaic_grpo.sbatch          # full fine-tuning, as the MOSAIC rerun
python3 scripts/post-training/mosaic_rollouts.py results/qwen2.5-3B_SafeUtil/<run>   # reward terms, refusals by label
```

The data mix gives 500 steps per epoch at the default batch (4 prompts × 4 rollouts), as MOSAIC has. The rollout dump
adds `kind`, `r_safety`, `r_task`, `refused_tool` (the tool only) and `judged` to MOSAIC's fields, and `refused`
counts both forms of refusal. Evaluate `merged/` as for MOSAIC, bare and with `SCAFFOLD=mosaic`.

## Verification status

- **Tested offline** (`tests/test_safeutil.py`): refusal wording, next-action matching, the reward per kind, the
  judge's prompt and parser, single-turn samples, one parquet schema for all kinds, the reward function with the judge
  stubbed, and the rollout summary on SafeUtil fields. The single-turn loop's tokens are checked against Qwen2.5's
  chat template where the tokenizer is cached.
- **Built on the real data** (Viking, 2026-11-08, about 3 minutes on a login node): 2,000 rows, of which 800
  Agent-SafetyBench (303 benign), 800 next_action (400 APIGen-MT from 2,997 eligible, 400 AgentAlign from 9,282) and 400
  harmful_prompt (50 per category from 4,556); 1,103 benign and 897 harmful in all. next_action prompts are 797–6,132
  tokens (median 4,472). APIGen targets follow 2.9 earlier assistant turns on average (up to 10), AgentAlign 0.7.
- **Not yet run:** verl with mixed agent loops in one batch, the live completion judge, and a training run.
