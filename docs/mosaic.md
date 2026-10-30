# MOSAIC GRPO on Agent-SafetyBench

A reproduction of MOSAIC (Agarwal et al., ICML 2026, "Learning When to Act or Refuse") as a post-training safety
stage. It starts from a merged LoRA model, such as `results/qwen2.5-3B_LoRA/p00-<job>/merged`.

## Pipeline

| Step | Code |
|---|---|
| Prompt: MOSAIC system text (App. G.1) + the task's user turn or dialog; tools through Qwen's native `tools=` with `refuse_unsafe_task` added last | `templates/mosaic_system.txt`, `mosaic.task_messages`, `mosaic.task_tools` |
| Rollouts: 4 per task, at most 20 turns, `<think>` / optional `<safety_thoughts>` / `<tool_call>` or `<answer>`; tool calls run in Agent-SafetyBench's simulated environments; tool tokens masked | `mosaic_agent_loop.py` (verl 0.6.1 agent loop), `mosaic.Episode` |
| Alignment reward: the judge compares all 6 pairs per group (seeded random order) with the App. D prompt; R(t_i) = Σ_j P(t_i, t_j) ∈ [0, 3] | `mosaic_judge.py` (Runware `openai:gpt@6-luna`), `mosaic.group_alignment` |
| Total reward: r_align + r_fmt − p_len | `mosaic_reward.py` |
| GRPO + LoRA; the KL reference is the starting model | `scripts/viking/mosaic_grpo.sbatch` |
| Export: verl checkpoint → `merged/` for `eval_all.sh` | `mosaic.py export` |

verl scores each rollout on its own as soon as it finishes, but a pairwise reward needs the whole group. To handle
this, every reward call waits in one named Ray actor (`GroupJudge`) until all 4 rollouts of its group have arrived.
The actor judges the group once and gives each rollout its score.

## Run book (Viking)

```bash
.venv-serve-cu128/bin/pip install -r requirements-rl.txt              # verl's dependencies, into the training env
.venv-serve-cu128/bin/pip install --no-deps verl==0.6.1              # no-deps: verl pins numpy<2
.venv-serve-cu128/bin/python scripts/post-training/mosaic.py build-data     # trainset/derived/mosaic_astb_train.parquet
.venv-serve-cu128/bin/python scripts/post-training/mosaic_judge.py          # live judge check (2 Runware calls)
MODEL_PATH=results/qwen2.5-3B_LoRA/p00-<job>/merged sbatch scripts/viking/mosaic_grpo.sbatch trainer.total_training_steps=5   # smoke run
MODEL_PATH=results/qwen2.5-3B_LoRA/p00-<job>/merged sbatch scripts/viking/mosaic_grpo.sbatch
bash scripts/viking/qwen25_3b.sh eval results/qwen2.5-3B_MOSAIC/p00-<job>-<job2>
```

One epoch is 500 steps (2,000 tasks at 4 per step). Each step makes up to 24 judge calls, so an epoch is about 12,000
Runware calls. Identical trajectory pairs tie without a call.

## Choices that differ from the paper

- **Environment.** Everything runs in the existing training env (`.venv-serve-cu128`). verl is installed with
  `--no-deps`, and that env has no flash-attn, so the job uses SDPA attention without padding removal.

- **Training data.** All of Agent-SafetyBench is used for training, as in the paper. It can no longer be reported as
  a held-out evaluation. Report AgentHarm and ASB instead.
- **Judge.** Runware `openai:gpt@6-luna` (temperature 0, `reasoning_effort=none`) instead of GPT-4o. A pair the judge
  fails on three times counts as a tie, and `judge_failures` is logged.
- **LoRA** (r=16, α=32, as in `train_lora.py`) with lr 1e-5. The paper's 1e-6 looks like full fine-tuning. bf16 is
  used instead of fp16.
- **Batching.** Batch 4 prompts × 4 rollouts with one PPO update per step. The paper lists "mini-batch 16", which
  reads as trajectories.
- **Prompt length** 3,072 tokens instead of 2,000. The longest prompt with tools is 2,563 Qwen2.5 tokens.
- **Format reward and length penalty.** The paper says r_fmt ∈ [0, 2] without defining it. Here r_fmt is 2 × the share
  of well-formed turns. The length penalty max(0, (L−400)/400) is averaged over turns. Algorithm 1's undefined
  `p_ref` term is left out.
- **Tools** go through the native chat template instead of a pasted `{tool_details}` block. For Qwen the two give the
  same `<tool_call>` format.

## Verification status

- **Tested offline** (`tests/test_mosaic.py`): parsing, episodes against the real environments, rewards (Table 6
  reproduced), the group rendezvous (full group, timeout, failures), the judge request shape, and the agent loop's
  token bookkeeping. The prompt and response tokens are checked to equal Qwen2.5's chat-template rendering of the
  whole conversation, with gradient only on generated turns.
- **Not yet run:** verl itself (no GPU here), the verl install (only a pip dry run on Viking), the live Runware judge, the verl LoRA
  checkpoint → merger → `export` path, and whether everything fits on one A40.
