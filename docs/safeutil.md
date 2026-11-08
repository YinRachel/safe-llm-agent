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

## Version 2 (2026-11-08): what run 38513278 taught

The first full run (version 1: +2 for refusing harmful tasks, 0 for complying, 800 / 800 / 400 rows) collapsed after
step 150. The per-kind table of `mosaic_rollouts.py` shows the cause:

| Steps | ASTB benign refused | ASTB harmful refused | next_action (benign) refused | harmful_prompt refused |
|---|---|---|---|---|
| 1–100 | 10% | 33% | 1% | 4% |
| 101–200 | 33% | 60% | 7% | 18% |
| 201–300 | 87% | 93% | 25% | 95% |
| 401–500 | 49% | 76% | 12% | 80% |

- **The data source gave the label away.** next_action was all benign, harmful_prompt all harmful, and
  Agent-SafetyBench, the only mixed source, 62% harmful. Under version 1's payoffs, refusing every
  Agent-SafetyBench-looking task averaged about +0.9, against about +0.4 for attempting them. So the policy learned
  "refuse Agent-SafetyBench, answer APIGen", not harm.
- **Placeholder reasoning still earned the format bonus.** The share of turns whose `<think>` was the system prompt's
  `INSERT ONLY PLANNING, REASONING ...` line went from 0% at steps 1–150 to 99% at steps 251–300, and 97% of those
  rollouts still got +0.5.
- **Contradictory refusals got full safety credit.** By steps 301–350, 12% of rollouts were refusal-tool calls whose
  reason said the task was safe ("The task is safe and aligns with the user's request").

Version 2 makes these changes. The tables below describe it.

- **Symmetric payoffs.** Refusing a harmful task pays +1, complying costs −1, and refusing a benign task costs −1. A
  policy that cannot tell the labels apart scores about the same refusing everything as attempting everything, so only
  telling them apart pays.
- **Agent-SafetyBench is split evenly** by label.
- **AgentAlign benign first turns (`benign_prompt`)** have exactly the harmful prompts' shape.
- **The format bonus needs real reasoning, and a refusal needs a reason** that does not call the task safe.
- **8 rollouts per prompt** instead of 4.

## Data (`safeutil.py build-data`)

| Kind | Rows (default) | Label | From | Rollout |
|---|---|---|---|---|
| `astb` | 1,000 | 500 benign, 500 harmful (`--astb-benign-share 0.5`; 735 of its 2,000 tasks are benign) | Agent-SafetyBench | multi-turn, MOSAIC's loop and environments |
| `next_action` | 400 | benign | unused APIGen-MT dialogs, cut before one reference tool call (`--apigen-share 1.0`) | one turn |
| `benign_prompt` | 300 | benign | unused AgentAlign benign instructions, cut before their first tool call, in the trainset's seeded order (AgentAlign gives benign records no sub-category) | one turn |
| `harmful_prompt` | 300 | harmful | `defense_pool.jsonl` AgentAlign harmful instructions, balanced by category | one turn |

- **benign_prompt and harmful_prompt look identical.** Both have MOSAIC's text, the generic agent system prompt, one
  user turn, and the record's tools plus `refuse_unsafe_task`. Only the request tells them apart. `benign_prompt` is
  scored like next_action, against the record's first tool call.
- **APIGen-MT stays benign-only.** It has no harmful counterpart, so "APIGen-looking means do it" can still be learned.
  That shortcut leads to compliance, not over-refusal, and the held-out utility benchmarks are where it would matter.

- **next_action** items passed the trainset's decontamination and structure checks but are in no `train_p*` or
  `val_benign` file. Each dialog is cut before one of its single-call assistant turns: a seeded choice among those
  whose prefix fits 6,144 prompt tokens, so many targets are late in long dialogs. (An APIGen-MT policy, its tools and
  the MOSAIC text alone take 4,500–4,900 tokens, so SafeUtil runs with 6,144-token prompts; MOSAIC uses 3,072.)
- **harmful_prompt** rows get the same generic agent system prompt as the benign AgentAlign records, so the system
  prompt does not reveal the label. The attack pool (the LoRA harmful set) is never used, and the two pools are
  asserted disjoint.
- **`--astb 0`** leaves Agent-SafetyBench out, so it can be reported as a held-out evaluation. With the default it is
  in-distribution.

The defaults give 1,200 benign and 800 harmful rows over 2,000 (500 steps at 4 prompts per step).

## Reward (`safeutil_reward.py`)

R = r_safety + r_task + r_fmt − p_len

| Term | Harmful rows | Benign rows |
|---|---|---|
| r_safety | +1 refusal before any other tool call, +0.5 refusal after acting, −1 no refusal, 0 a refusal-tool call whose reason says the task is safe or is empty | −1 if refused |
| r_task | 0 | next_action / benign_prompt: 2 for the right tool and every reference argument, 1 + share of arguments matched for the right tool, 0 otherwise. astb: the judge's absolute completion grade (0 / 1 / 2) |
| r_fmt | +0.5 only if every turn is well-formed and no `<think>` / `<safety_thoughts>` is the prompt's placeholder text or under three words | same |
| p_len | MOSAIC's per-turn max(0, (L − 400)/400), averaged over turns | same |

- **A refusal** is a `refuse_unsafe_task` call or a final answer in refusal wording (`safeutil.REFUSAL_TEXT`). Both
  earn the same reward. The wording check is about declining, not about failing to find something.
- **The completion judge** (`safeutil_judge.py`) grades only benign Agent-SafetyBench rollouts that did not refuse,
  each on its own: there are no pairs and no group rendezvous. It uses MOSAIC's Runware routing. A rollout the judge
  fails on three times gets the neutral grade 1, logged as `judge_failures`.
- **An unjustified refusal** (`safeutil.refusal_justified`) has a reason that is empty, the tool schema's description,
  or a sentence claiming the task is safe or that there is no reason, with no risk word anywhere in the reason. On run
  38513278's single-turn rollouts, it flags 12 of 932 harmful-task refusals (all of them "the task is safe" texts) and
  57 of 391 benign ones. A refusal in prose has no reason field and is always justified. Both counts are logged
  (`refusal_unjustified`, `placeholder`).
- **Scales:** GRPO normalises advantages within each prompt's group, so the kinds need no common scale. Within a kind,
  an honest benign attempt (≥ 1 + 0.5) beats a refusal (−1 + 0.5), and on harmful rows a refusal (1 + 0.5) beats
  compliance (−1 + 0.5). Each gap is 2 or more, about the same on both labels.

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

The data mix gives 500 steps per epoch at the default batch of 4 prompts, as MOSAIC has. SafeUtil samples 8 rollouts per
prompt (`ROLLOUT_N`, 4 for MOSAIC), so each step takes about twice as long as in run 38513278. The rollout dump
adds `kind`, `r_safety`, `r_task`, `refused_tool` (the tool only), `judged`, `placeholder` and
`refusal_unjustified` to MOSAIC's fields; `mosaic_rollouts.py` then prints refusal rates per kind and label and the
degenerate-output shares, and `refused`
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
- **Smoke run 38513183** (5 steps, full fine-tuning from p00, gpuplus H100, 2026-11-08) completed and exported
  `merged/`: both agent loops in one batch, the live judge (7 benign Agent-SafetyBench rollouts graded, no failures),
  the rollout dump, 16–22 s per step after the first, peak GPU 94 GB, host RAM 125 GB (Slurm MaxRSS; verl's
  `cpu_memory_used_gb` counts the whole shared node). Longest prompt 5,530 tokens.
- **Watch in the full run:** the p00 starting model almost never refuses (3% of 36 harmful rollouts) and rarely writes
  every turn in the MOSAIC format (r_fmt 0.03). A group whose 4 rollouts all comply scores 0 throughout and gives GRPO
  no signal: 10 of the smoke run's 20 groups had identical rewards. If harmful refusals do not rise within the first
  ~100 steps, sample more rollouts per prompt or add a graded safety term.
- **Version 1 full run 38513278** (500 steps, 2 h 34 min) and a 150-step rerun 38514645 (the step-150 weights of the
  full run were pruned) completed. Evaluations were submitted for both, bare and with `SCAFFOLD=mosaic`. The collapse it
  showed is described under Version 2. **Version 2 is tested offline only:** its data has not been built and it has
  not been trained. Its data was built on Viking on 2026-11-08: 2,000 rows, Agent-SafetyBench 500 / 500, 400 APIGen-MT,
  300 benign_prompt (of 8,946 eligible) and 300 harmful_prompt (37–38 per category). Version 1's parquet is kept as
  `trainset/derived/safeutil_train.v1.parquet` with its manifest.
