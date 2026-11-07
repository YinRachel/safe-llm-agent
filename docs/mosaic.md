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
| GRPO + LoRA (or all weights with `LORA_RANK=0`); the KL reference is the starting model | `scripts/viking/mosaic_grpo.sbatch` |
| Export: verl checkpoint → `merged/` for `eval_all.sh` | `mosaic.py export` |

verl scores each rollout on its own as soon as it finishes, but a pairwise reward needs the whole group. To handle
this, every reward call waits in one named Ray actor (`GroupJudge`) until all 4 rollouts of its group have arrived.
The actor judges the group once and gives each rollout its score.

## Run book (Viking)

```bash
.venv-serve-cu128/bin/pip install -r requirements-rl.txt              # verl's dependencies, into the training env
.venv-serve-cu128/bin/pip install --no-deps verl==0.6.1              # no-deps: verl pins numpy<2
sbatch scripts/viking/build_flash_attn.job                            # flash-attn from source (glibc 2.28)
.venv-serve-cu128/bin/python scripts/post-training/mosaic.py build-data     # trainset/derived/mosaic_astb_train.parquet
.venv-serve-cu128/bin/python scripts/post-training/mosaic_judge.py          # live judge check (2 Runware calls)
MODEL_PATH=results/qwen2.5-3B_LoRA/p00-<job>/merged sbatch scripts/viking/mosaic_grpo.sbatch trainer.total_training_steps=5   # smoke run
MODEL_PATH=results/qwen2.5-3B_LoRA/p00-<job>/merged sbatch scripts/viking/mosaic_grpo.sbatch
MODEL_PATH=results/qwen2.5-3B_LoRA/p00-<job>/merged LORA_RANK=0 sbatch --partition=gpuplus --mem=160G \
    scripts/viking/mosaic_grpo.sbatch                                 # full fine-tuning, as in the paper
bash scripts/viking/qwen25_3b.sh eval results/qwen2.5-3B_MOSAIC/p00-<job>-<job2>
SCAFFOLD=mosaic bash scripts/viking/qwen25_3b.sh eval results/qwen2.5-3B_MOSAIC/p00-<job>-<job2>   # with the scaffold
```

One epoch is 500 steps (2,000 tasks at 4 per step). Each step makes up to 24 judge calls, so an epoch is about 12,000
Runware calls. Identical trajectory pairs tie without a call.

Every training trajectory is kept in `<run>/rollouts/<step>.jsonl`, one line per rollout (16 per step): the decoded
prompt and multi-turn response, `score`, `r_align`, `r_fmt`, `p_len`, `refused`, `refused_first_turn`, `status`,
`turns`, `truncated`, `safety_thoughts_rate`, `judge_failures`, the Agent-SafetyBench `task_id` and `fulfillable`
label, and `group_uid` (the 4 rollouts of one task share it). Within a group `r_align` always averages 1.5, so the
console score curve shows format and length only; refusal rates split by `fulfillable` are what show whether safety
improved or the policy drifted into refusing benign tasks. Runs before this change have no rollout files.

## Evaluation with the MOSAIC scaffold

The paper deploys MOSAIC models with their scaffold at test time: the MOSAIC system prompt and `refuse_unsafe_task`
as a first-class action. Its "base → MOSAIC" rows compare the base model without the scaffold against the trained
model with it. The standard evaluations serve the model bare.

The bare p00 baseline is the 38193xxx eval set (with ASB from 38218391). The 38413xxx runs and 38414507, which sit in
the same `p00-38166985/eval/` folder, were served with `ping.jinja`: they are the PING baseline, not bare p00.

`SCAFFOLD=mosaic` (`scripts/viking/eval_all.sh`) adds the scaffold to every benchmark through the serving layer, so no
benchmark code changes:

- **`templates/mosaic.jinja`** (`mosaic.py template`) is Qwen2.5's template behind a preamble. The MOSAIC text leads
  the system prompt, any system prompt from the benchmark follows under `# Task instructions`, and the refusal tool is
  appended to the tools. Requests render byte-identically to training (`tests/test_mosaic.py`).
- **The `mosaic` tool parser** (`scripts/eval/tool_parsers.py`) drops `<think>` and `<safety_thoughts>`, unwraps
  `<answer>`, and turns a refusal call into a plain-text refusal with no tool call. The episode ends there, as in
  training, and the benchmarks' own refusal judges see the refusal.
- **`MOSAIC_SCAFFOLD=1`** applies the same rewrite where no tool parser runs: ASB's prompted mode,
  Agent-SafetyBench's tool-less tasks, and HarmBench's offline generation. That is recorded as `output_filter` in
  their summaries; HarmBench keeps `raw_generation`.

Results go to `<run>/eval/mosaic/`, and the tag gets a `-mosaic` suffix. Run both the starting model and the trained
model this way. Together with the bare runs, that gives the 2×2 of {p00, MOSAIC} × {bare, scaffold}.

## Choices that differ from the paper

- **Environment.** Everything runs in the existing training env (`.venv-serve-cu128`). verl is installed with
  `--no-deps`. flash-attn is built from source by `build_flash_attn.job`, because the prebuilt
  wheels need glibc 2.32 and Viking has 2.28. It enables verl's padding removal, so training passes run on real
  tokens only. Without it every sequence was padded to 8,072 tokens; that cost about 190 s of a 210 s step
  in the smoke run (38408530).

- **Training data.** All of Agent-SafetyBench is used for training, as in the paper. It can no longer be reported as
  a held-out evaluation. Report AgentHarm and ASB instead.
- **Judge.** Runware `openai:gpt@6-luna` (temperature 0, `reasoning_effort=none`) instead of GPT-4o. A pair the judge
  fails on three times counts as a tie, and `judge_failures` is logged.
- **LoRA** (r=16, α=32, as in `train_lora.py`) with lr 1e-5 by default. The paper's 1e-6 looks like full
  fine-tuning, which `LORA_RANK=0` runs (lr 1e-6, on an 80 GB gpuplus GPU, with Adam states offloaded to the CPU
  between updates and a separate frozen reference copy). bf16 is used instead of fp16.
- **Checkpoints** go to scratch (`CKPT_ROOT`, default `/mnt/scratch/users/$USER/mosaic-checkpoints/<run>`): verl
  stores the whole model in fp32 even with LoRA (~14 GB each; ~37 GB with full fine-tuning optimizer state), which
  filled the 100 GB home quota and killed a run at step 100. Only the last 2 are kept (1 for full fine-tuning).
  `OUT=<earlier run folder>` resumes a run from its last checkpoint.
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
