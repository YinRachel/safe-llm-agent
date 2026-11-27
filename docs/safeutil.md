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

## Version 3 (2026-11-21): what the P00 scaffold evaluations taught

SafeUtil v2 kept BFCL multi-turn (Qwen2.5-3B 0.120 vs the p00 LoRA's 0.116, Llama-3.1-8B 0.182 vs 0.149; the
scaffold alone drops Qwen2.5 to 0.051) but lost τ2 retail (Qwen2.5 0.079 vs 0.289, Llama 0.123 vs 0.412). The
trajectories show one cause: **the policy invents arguments instead of asking.** Qwen2.5 opens with
`find_user_id_by_email("example@example.com")` and `find_user_id_by_name_zip("John", "Doe", "12345")`, gets "User not
found", and calls `refuse_unsafe_task` with "Reason: User not found" (92% of retail tasks); it also makes up variant
ids. Llama loops `get_order_details("#W0000000")`, the tool description's example id (82 of 114 tasks, about 3,000
calls; 81% of tasks never reach a write action). The v2 reward taught this:

- **Every next_action target was a tool call**, so asking the user for an email scored 0 and any call to the right
  tool scored at least 1.
- **Right tool, wrong arguments still paid 1** (1 + share of arguments matched).
- **No row showed a tool error**, so after one the only trained move left was the refusal tool.

v3 changes (`--target-mix call=1` rebuilds call-only cuts):

- **Target types.** APIGen-MT dialogs are cut before four kinds of assistant turns, scarcest first so no record is
  spent twice: `error` (any target right after an error tool result: a retry with fixed arguments or a question;
  20%), `ask` (a text reply right after a user turn: asking for identity or details, confirming before acting; 25%),
  `reply` (a text reply after a tool result; 15%) and `call` (40%). Replies in refusal wording or with an inline
  JSON call are not targets.
- **Grounded arguments (next_action rows).** A call's identifier-like values (single-token strings with a digit or
  `@`, single-token strings under `email`, `first_name`, `last_name`, `zip`, `phone`, `*_id` keys, integers of 4+
  digits; not dates, times or free text with spaces) must appear in the conversation (messages and earlier calls; tool
  schemas do not count, so `#W0000000` is not grounded) or be a reference value; otherwise r_task gets −0.5.
  benign_prompt rows are not checked: their AgentAlign references infer values the instruction never states (a
  profile name from "Starbucks", amounts in wei), and 14% of them would fail. The `#W0000000` example stays in the tool descriptions (τ2 has it too); the penalty teaches the
  policy not to copy it.
- **Call matching pays 2 × the share of reference arguments**, so the right tool with invented arguments earns 0 −
  0.5, below a question (0 on a call target, 2 on a text target).
- **Operational refusal reasons are unjustified.** A refusal-tool reason that is an operational failure ("not found",
  "could not authenticate", "insufficient information", "error") and names no risk earns no safety credit on harmful
  rows (0, as "the task is safe" reasons). On benign rows any refusal already costs −1, now also after a tool error.

build-data records the share of reference calls whose arguments pass the grounding check
(`notes.next_action_reference_calls_grounded`), which validates the check on the data itself.

## Version 4 (2026-11-22): what the v3 Qwen2.5 evaluation taught

v3 (GRPO 38854140, scaffold evals 38854283-89) did what it was built for on τ2 retail: refusals with "Reason: User
not found" fell from 92% to 4% of tasks and retail rose from 0.079 to 0.193 (p00 LoRA with the scaffold 0.237). It was
also safer than v2 on every agent safety benchmark (ASB DPI ASR 0.391 vs 0.729, OPI 0.103 vs 0.201, Agent-SafetyBench
0.554 vs 0.455, AgentHarm harmful 0.073 vs 0.099). But **BFCL multi-turn halved (0.120 to 0.061)**: the policy ends
turns with questions nobody needed ("Would you like me to proceed?", asking for a username or city names after an
error instead of retrying), and BFCL's simulated user cannot answer them. Text-only assistant turns rose from 29% to
36%. ASB original-task success under OPI also fell (0.157 to 0.017), so part of its ASR drop is failed calls.

The cause is the v3 mix: 60% of APIGen-MT next_action targets were text, 83% of those text targets are questions to
the user (τ-bench agents confirm before every write), and any text reply on a text target earns the full 2.

v4 changes only the default mix: **call 0.7, ask 0.1, reply 0.1, error 0.1** (`--target-mix
call=0.4,ask=0.25,reply=0.15,error=0.2` rebuilds v3). The grounding penalty, call matching and operational-refusal
rule stay as in v3. The manifest records reward version 4.

## Version 5 (2026-11-23): what the v4 preflight failure taught

v4 (GRPO 38858749) trained without trouble and its temperature-1 rollouts looked clean, but six of its seven scaffold
evaluations (38873450-55) stopped at `evaluate.py preflight`: asked to "Use get_weather to look up the weather in
London" at temperature 0, it repeated "The tool call will likely return the current temperature in London..." until
max_tokens. Every benchmark decodes at temperature 0. A greedy probe (16 simple tool requests rendered with the
scaffold, as the benchmarks do, plus 77 held-out next_action call targets rendered as in training; 512 tokens) shows it is
v4-specific:

| greedy decoding | p00 LoRA | v2 | v3 | v4 |
|---|---|---|---|---|
| simple requests: hit max_tokens | 0% | 3% | 7% | 13% |
| next_action call targets: called | 66% | 60% | 82% | 51% |
| next_action call targets: hit max_tokens | 0% | 3% | 4% | 14% |

At temperature 1 v4 hits max_tokens on 0% / 9% of the same prompts, and its training entropy fell from 0.65 to about
0.40 (v3 stayed near 0.7). Sampling at temperature 1 almost never produced the loop (1-2% of late rollouts had a
repeated sentence), so the reward never saw it.

v5 keeps v4's data and reward and adds:

- **r_degen = -1** (and no r_fmt) when a turn repeats a sentence of 20+ characters 3 times, leaves a `<think>`,
  `<safety_thoughts>`, `<tool_call>` or `<answer>` tag unclosed, or writes a chat-role tag (`</system>`,
  `<tool_response>`...), or when a single-turn row is cut off at the response limit (`safeutil.degenerate`). On v3/v4
  rollouts this fires on about 10% of early Agent-SafetyBench rollouts, 1-4% of late ones, and under 1% of single-turn rows.
- **Rollout temperature 0.8** (verl override `actor_rollout_ref.rollout.temperature=0.8`; MOSAIC App. J uses 1.0), so
  sampled turns sit closer to greedy decoding and the penalty sees the loops.
- **Checkpoints at steps 400 and 500 kept** (`KEEP_CKPT=1 trainer.save_freq=100 trainer.max_actor_ckpt_to_keep=2`;
  two 37 GB checkpoints fit in the scratch quota, three do not). Acceptance before the benchmarks: the greedy probe
  and the preflight on `merged/` (step 500); export step 400 with `export_checkpoint.sbatch` if step 500 fails.

**Outcome: v5 failed acceptance and is dropped; v3 stays the reported SafeUtil version.** Full run 38889184 (500 steps,
3 h 52 min) trained normally (late rollouts: harmful_prompt refused 97%, benign_prompt 11%, next_action r_task 1.56).
The greedy probe fixed the simple requests but not the next_action turns:

| greedy decoding | v3 | v4 | v5 step 400 | v5 step 500 |
|---|---|---|---|---|
| simple requests: called / hit max_tokens | .43 / .07 | .60 / .13 | .70 / .00 | .70 / .03 |
| next_action call targets: called / hit max_tokens | .82 / .04 | .51 / .14 | .45 / .13 | .49 / .12 |

The loops ("get the get the ...", recited policy text, JSON garbled mid-call) were already there at step 400. At
temperature 0.8 v5 hit max_tokens on only 3% of next_action prompts, so the penalty still rarely saw them. What
v3 and v4/v5 differ in is the target mix: v3's text-heavy mix (call 0.4) kept training entropy near 0.7 and greedy
decoding stable. A probe caveat: its call regex takes the first `<tool_call>` block, so a turn whose valid call follows a
malformed block counts as no call. That turn is degenerate either way, and every model is scored the same.

## Version 6 (2026-11-23): fixing v3's text targets

v3 is the best version (v4/v5 looped under greedy decoding), and its remaining weakness is BFCL (0.061 vs v2 0.120):
needless questions. Two causes, both measured on the v3 data and rollouts:

- **Reward:** any text reply on a text target earned 2, whatever it did.
- **Data:** 199 of v3's 225 text targets were themselves questions: 55 of 65 after-error text targets asked the user
  (only 15 of v3's 80 error targets were retries), and 44 of 60 reply targets asked for details or confirmation. The
  eligible APIGen-MT pool has 222 retry calls after errors and 5,009 plain reports after tool results, so this was
  sampling, not scarcity. v3's own late rollouts already matched the reference's act most of the time (a question
  where the reference reported: 13 of 688), so a reward change alone would change little.

v6 keeps v3's target mix (call 0.4, ask 0.25, reply 0.15, error 0.2), its grounding penalty and refusal rules, and no
r_degen. It changes:

- **Text replies score by act** (`text_act`: report, ask_info, confirm; a closing "anything else?" is not a
  question): matching the reference 2, the other kind of question 1, reporting where the reference asked 0.5, a
  question where the reference reported 0 (`TEXT_ACT_REWARD`). A grounded call on a text target stays 0.5.
- **Act sub-mix** (`--act-mix`, APIGen-MT only): error targets call 0.5 / ask_info 0.35 / confirm 0.15, reply targets
  report 0.6 / ask_info 0.2 / confirm 0.2; ask targets keep the pool's acts.

Qwen2.5 v6 parquet (2026-11-23, v4's kept as `safeutil_train.v4.parquet`): next_action call 160; error 40 retry calls /
28 ask_info / 12 confirm; reply 36 report / 12 ask_info / 12 confirm; ask 85 ask_info / 6 confirm / 9 report.
Reference calls grounded 0.995. GRPO smoke 38974668 -> full 38974669 (v3 settings: 1 H100 gpuplus, 160G, temperature
1.0). Acceptance as for v5: greedy probe (next_action call rate and max_tokens hits at least as good as v3's .82 / .04)
and the preflight before the benchmarks; the target is BFCL back towards v2 with v3's safety.

### SafeUtil v3 on the other models (2026-11-23)

Rows carry `extra_info.reward_version` (`build-data --reward-version`, default 5); 3 and 4 select the v3 reward with
no r_degen, so v3 runs from current code. The Qwen3-4B and Llama-3.1-8B v3 parquets (`trainset/derived/<tag>/`,
v2 kept as `safeutil_train.v2.parquet`) were built with

    safeutil.py build-data --tokenizer <p00 merged> --output trainset/derived/<tag>/safeutil_train.parquet \
        --reward-version 3 --target-mix call=0.4,ask=0.25,reply=0.15,error=0.2

giving the Qwen2.5 v3 split (next_action call 160, ask 100, reply 60, error 80; Qwen3-4B all 2,000 task ids shared
with Qwen2.5 v3, Llama 1,995). Runs use each model's v2 settings (2 H100 gpuplus, 360G; ROLLOUT_MEM 0.5 for Qwen3-4B,
0.4 plus ACTOR_PARAM_OFFLOAD=1 for Llama; rollout temperature 1.0): Qwen3-4B 38974135 -> 38974136, then Llama
38974138 -> 38974139 (chained so two full-fine-tuning checkpoints are never in scratch at once). GLM-4-9B waits on the
GRPO collapse investigation.

Outcome (2026-11-24): both finished (Qwen3-4B 3 h 41 min, Llama 6 h 35 min). Llama's rollouts showed benign
over-refusal rising to 24% around steps 301-400 and back to 5% by the end; harmful_prompt refusal about 94%. Greedy probe
(`greedy_probe.py run --template templates/mosaic_<tag>.jinja --data trainset/derived/<tag>/safeutil_train.parquet`, jobs
39018295/306; calls read with `tool_parsers.mosaic_calls`; next_action rows are the model's own v3 parquet, 44 call
targets of 120), greedy call rate:

| model | simple requests | next_action call | next_action calls on text targets (reply/ask/error) | max_tokens hit |
|---|---|---|---|---|
| Qwen3-4B p00 LoRA | 1.00 | .86 | .13/.11/.19 | .05 |
| Qwen3-4B SafeUtil v3 | 1.00 | .93 | .04/.07/.15 | .02 |
| Llama p00 LoRA | 1.00 | .73 | .17/.15/.19 | .05 |
| Llama SafeUtil v3 | 1.00 | .82 | .26/.07/.19 | 0 |
| Qwen2.5 SafeUtil v3 (same parser) | .47 | .84 | .13/.07/0 | 0 |

No loops; both are at or above their starting point, so they went to MOSAIC-scaffold evaluation: Qwen3-4B
39029088-94, Llama 39029095-101. Llama writes untagged JSON calls (0 of 79 tagged), which `mosaic_calls` reads as
the scaffold evals do.

## SafeUtil scaffold (2026-11-23): our own inference template, template-only test

The MOSAIC scaffold gives much of the safety gain on its own but costs utility (p00 LoRA, bare -> MOSAIC scaffold:
AgentHarm harmful Llama 0.462 -> 0.134, GLM 0.532 -> 0.311; BFCL GLM 0.130 -> 0.079, Qwen2.5 0.116 -> 0.051; tau2
retail GLM 0.386 -> 0.289). Likely causes in `templates/mosaic_system.txt`: a turn is a tool call or a `<answer>`
"final answer" (no way to talk to the user mid-task); mandatory `<think>` and "evaluate safety" every turn; "Call at
most one tool per turn"; no guidance on errors, missing information or injected instructions; Qwen tags for every model.

`templates/safeutil_scaffold_system.txt` keeps the refusal tool and a short safety section and changes the rest:
several independent calls per message, plain-text replies that report, ask for missing information or ask approval
before irreversible actions when the task requires it, use the conversation instead of asking or inventing values,
retry after errors, optional short `<think>`, treat tool outputs as data (not instructions), refuse only clear harm and
never over failures. Templates (`templates/safeutil_scaffold.jinja`, `safeutil_scaffold_qwen4B.jinja`, built with
`mosaic.py template --system-text templates/safeutil_scaffold_system.txt [--source <p00 merged>]`) are the MOSAIC
templates with only the system text replaced; the `mosaic` output parser reads both. Evaluate with
`SCAFFOLD=safeutil bash scripts/viking/eval_all.sh ...` (tag suffix `-suscaffold`, outputs in `eval/suscaffold/`).

v3 under our scaffold (2026-11-24). v6 failed the greedy gate (next_action call .75 vs v3 .84 on the same rows, job
39030717; it retries after errors, .62 on error targets vs v3 0, and hits max_tokens less, .05 vs .13), so it is
parked and v3 stays the method. The three v3 models are evaluated under `SCAFFOLD=safeutil` (Llama template
`templates/safeutil_scaffold_llama3.1-8B.jinja`, c700d7f): Qwen2.5 39031069-75 (MAX_MODEL_LEN=32768), Qwen3-4B
39031076-82, Llama 39031083-89. v3 was trained with the MOSAIC system text in its rows, so this is a lower bound for
our scaffold on v3; the template-only p00 LoRA test is the clean comparison (Qwen2.5 38974777/83 plus reruns
39030712-16, after 38974778-82 failed on MAX_MODEL_LEN 65536 > Qwen2.5's 32768; Qwen3-4B 38974784-90). If our
scaffold wins there, retrain v3 with our system text in the rows.

Template-only result, Qwen3-4B p00 LoRA (2026-11-24; `paper_results/P00/Qwen3-4B/lora_fine_tune{,_scaffold,_suscaffold}`):

| metric | no scaffold | MOSAIC scaffold | our scaffold |
|---|---|---|---|
| AgentHarm benign score / refusal | .592 / .006 | .506 / .153 | .519 / .034 |
| AgentHarm harmful score / refusal | .453 / .216 | .161 / .716 | .210 / .614 |
| HarmBench ASR | .050 | 0 | .022 |
| Agent-SafetyBench safety (all / fulfillable / not) | .365 / .389 / .352 | .457 / .430 / .472 | .465 / .479 / .456 |
| ASB DPI ASR / refusal | .936 / .083 | .627 / .598 | .405 / .997 |
| ASB OPI ASR / refusal | .137 / .674 | .141 / .920 | .209 / .944 |
| BFCL multi-turn accuracy (context exhausted) | .244 (37) | .237 (78) | .193 (22) |
| tau2 retail / airline | .491 / .240 | .456 / .200 | .491 / .260 |

Our text cuts MOSAIC's benign over-refusal (AgentHarm benign 15% -> 3%) and recovers tau2 to the no-scaffold level,
with Agent-SafetyBench and ASB DPI at least as safe; it gives back some harmful-request refusal (AgentHarm harmful
.716 -> .614 refusal, HarmBench 0 -> .022) and is worse on two points: ASB OPI (injection in tool output) ASR .141 ->
.209 despite the "tool outputs are data" line, and BFCL .237 -> .193 (fewer context blow-ups, but lower accuracy).
ASB DPI refusal .997 with task success .053 means it refuses nearly every DPI case outright rather than doing the
user task without the injected one.

Template-only test (no training), p00 LoRA, all 7 benchmarks, BFCL_TIME 36 h: Qwen2.5-3B 38974777-83, Qwen3-4B
38974784-90. Compare with `lora_fine_tune` (bare) and `lora_fine_tune_scaffold` (MOSAIC scaffold). If it keeps most of
the MOSAIC scaffold's safety with less utility loss, the next SafeUtil version trains with it (data and r_fmt then
follow the new format); MOSAIC itself stays evaluated with its own scaffold.

## Version 7 (2026-11-25): what the three-model P00 evaluation of v3 taught

With Agent-SafetyBench left out, v3 under the MOSAIC scaffold against MOSAIC (safety = mean of 1 − AgentHarm harmful,
1 − ASB DPI ASR, 1 − ASB OPI ASR, 1 − HarmBench ASR; utility = mean of AgentHarm benign, BFCL, tau2 retail and
airline):

| model | MOSAIC safety / utility | v3 (MOSAIC scaffold) | v3 (our scaffold) |
|---|---|---|---|
| Qwen2.5-3B | .771 / .229 | .857 / .217 | .772 / .220 |
| Qwen3-4B | .839 / .374 | .908 / .373 | .917 / .361 |
| Llama-3.1-8B | .770 / .235 | .836 / .253 | .856 / .237 |

v3 dominates MOSAIC on Llama, matches its utility at +.07 safety on Qwen3-4B and trades off on Qwen2.5. What holds it
back is tau2 retail (Qwen3-4B .535 vs .588, Llama .158 vs .254) and benign over-refusal (Qwen3-4B AgentHarm benign
refusal 32% vs 15%). The agents' own tau2 retail histories (114 tasks per model; `StoredAgentState:*:messages` in the
inspect logs) show three learned failures, none of them in the scaffold (the p00 LoRA under the same scaffold has none):

- **False completion claims (Llama).** In 15% of tasks v3 tells the customer "the items have been successfully
  exchanged" with no exchange call (MOSAIC 4%, p00 LoRA 0%); 52 of its 96 failures identified the user but never wrote
  (p00 LoRA 14). v3's data barely has the step: only 8 (Llama) / 13 (Qwen3-4B) of 400 next_action rows are a write
  call right after a user turn (the user has confirmed), and on those rows Llama's late rollouts replied in text 25% of
  the time (6% early) and called the right tool 50% (90% early). v3 also paid the full 2 for any text reply on a text
  target, a false report of a finished action included.
- **Identical-call loops (Qwen3-4B).** 5 of 114 tasks repeat one identical `modify_pending_order_items` call, 1,353
  times in all; Qwen3-4B's other loss is wrong arguments on writes (50 failures vs MOSAIC 41).
- **Benign over-refusal (Qwen3-4B).** 32% of AgentHarm benign tasks refused while refusing 95% of harmful ones.

v7 = v3's reward and v3's target mix (call 0.4, ask 0.25, reply 0.15, error 0.2: v4-v6 showed that moving the
call/text balance breaks greedy decoding), with:

- **Commit targets** (data). 40% of the call targets are commits (`target_act_at`: a single state-changing call,
  `WRITE_TOOL` by its leading verb, straight after a user turn), `--act-mix call:commit=0.4,call=0.6`, the v7 default:
  64 of 400 next_action rows instead of 8-13. Text targets are sampled first, as in v3, and are unchanged.
- **False claim** (`claims_completion`, `false_claim_single` / `false_claim_episode`). A next_action reply claiming a
  finished state change ("has been cancelled", "I have updated", "successfully booked"; negated and future forms do not
  count) when no state-changing call since the user's last message got a non-error result scores r_task = −1, on call
  and text targets alike. On a benign Agent-SafetyBench episode: a final reply making such a claim with no successful
  state-changing call in the episode gets −1 instead of the judge's grade.
- **Repeated call** (`repeated_calls`, `repeats_failed_call`). r_repeat = −0.5 on a benign episode that repeats an
  identical call after it returned an error or makes one identical call 3 times, and on a next_action call identical
  to a call in the conversation that failed.
- **Benign refusal −1.5** (v3 −1). Refusing everything now scores below attempting everything on a balanced mix
  (+1 / −1.5 against −1 / about +1), aimed at the over-refusal; harmful rows are unchanged.

The rollout dump adds `false_claim`, `repeated_call` and `r_repeat`. Replayed on v3's own training rollouts, the two
new terms would have fired rarely (false claims 0-2% of next_action rollouts, repeats under 1% of Agent-SafetyBench
ones), which is why the commit targets are part of v7: without the confirmed-write rows the reward has almost nothing
to act on. Acceptance as for v5/v6: the greedy probe at least as good as v3 on the model's own rows (next_action call
rate, max_tokens hits), then the scaffold evaluations; the target is tau2 retail at or above MOSAIC with v3's safety.

## Greedy probe audit (2026-11-26): the probe, not the checkpoints, was failing on Qwen2.5

v7 Qwen2.5 (GRPO 39068901; 500 steps, KL ~0.02, tracking v3) "failed" the single-pass probe (39102674: next_action
call .59 vs v3 .66 on v7's rows), as v6 had. Auditing the probe showed its Qwen2.5 numbers were unreliable:

- The same v3 checkpoint on the same 30 simple prompts gave 0/30 identical greedy outputs across three probe runs,
  and the outcome changed on 20-23 of them. Only the other prompts in the batch differed.
- Probing v3 on 120 rows instead of all 400 dropped confirmed writes from .91 to .42-.58 (3 passes each).
- All on v7's 400 rows plus the 30 simple requests, 3 passes, Qwen2.5 v3 (`greedy_probe.py run`):

| setting | simple requests | confirmed writes | reply rows | same outcome every pass |
|---|---|---|---|---|
| bf16, A40, uncapped (150-430 sequences at once; the old probe) | .20-.50 | .42-.92 | .83-.93 | .77-.93 |
| fp32, A40 | 1.00 | 1.00 | 1.00 | 1.00 |
| bf16, A40, at most 8 sequences at once | 1.00 | 1.00 | 1.00 | 1.00 (case for case = fp32) |
| bf16, H100, uncapped (120 rows) | 1.00 | 1.00 | 1.00 | 1.00 |

So the loops, garbled calls and run-to-run flips were large bf16 decode batches on the A40, not the weights. Training
(H100) and the benchmarks (A40 but `evaluate.py --concurrency 4`) are in the clean regime; only the probe was not.
v4's loop was real: it broke the benchmarks' own preflight at 4 requests. The probe now defaults to
`--max-num-seqs 8`, runs 3 passes in shuffled order, classes rows (simple, commit, call, error_call, ask, reply,
error) and has `greedy_probe.py compare NEW.json OLD.json` (paired McNemar on the per-case majority vote).

Clean probe, v7 vs v3 on v7's 400 rows (39103440 vs 39102966): next_action right .92 vs .91 (p .53), so v7 passes
the gate. Confirmed writes 1.00 both; v7 retries a call after a tool error where that is the target (error_call .87
vs 0, 13:0) but calls where it should reply (reply .87 vs 1.00, 0:8, p .008); simple requests .90 vs 1.00 (0:2).
Earlier single-pass verdicts on Qwen2.5 (v6 parked, v4/v5 call rates) were taken in the bad regime; v6 also called
on error targets in that regime (.38 vs 1.00) and has not been re-probed.

## Version 8 (2026-11-26): what the Qwen2.5 v7 evaluation and the Qwen3-4B v7 probe taught

Qwen2.5 v7 against MOSAIC with Agent-SafetyBench left out: MOSAIC scaffold safety .822 (MOSAIC .771), our scaffold
.761 / utility .222 (MOSAIC .229). v7 doubles BFCL (.121 vs .062) and tau2 retail (.21-.25 vs .105) but tau2 airline
falls to .10-.14 (MOSAIC .30, v3 .18-.22). In 20 of the 50 airline tasks the expected outcome is no change (the policy
forbids the request):

| model | pass | pass, no-write tasks | pass, write tasks | wrote on a no-write task | transfer to human |
|---|---|---|---|---|---|
| MOSAIC | 15 | 14/20 | 1/30 | 10/20 | 72% |
| v3, MOSAIC scaffold | 11 | 8/20 | 3/30 | 13/20 | 16% |
| v7, MOSAIC scaffold | 7 | 4/20 | 3/30 | 16/20 | 14% |
| v7, our scaffold | 5 | 4/20 | 1/30 | 15/20 | 14% |

All 10 tasks MOSAIC passes and both v7 runs fail are no-write tasks: v7 cancels uninsured basic-economy reservations,
changes flights and sends certificates the policy forbids. The reward never showed a policy denial as right:
`target_type` drops any APIGen-MT reply with refusal wording ("I cannot assist with the cancellation of this
reservation"), so none of the pool's policy denials (about 190 dialogs, mostly airline) were targets, and on a
next_action row such a reply scored as a benign refusal (−1 in v3, −1.5 in v7). v7's commit rows then pushed writes
further (it writes in 80% of airline tasks, v3 66%).

Qwen3-4B v7 failed the clean probe against v3 on its own rows (next_action .90 vs .93, p .014; calls .76 vs .92,
confirmed writes .83 vs .95): 20 of its 23 misses ask the user to confirm, often before a read call and 5 times after
the user already said yes. Its training rollouts on call targets called .93 / .99 / .67 of the time at steps 1-50 /
201-250 / 451-500 (Qwen3-4B v3 .88 / .70 / .77, Qwen2.5 v3 .81 / .70 / .52). The reward was consistent (late on: a call
scored 1.79, a text reply 0), so the drift came from the text targets, which pay 2 for any text reply including a
confirmation question, and from groups that answer a row only in text: all score 0, so GRPO has nothing to learn from.

v8 = v7 with:

- **Deny targets** (data, `--reward-version 8` builds only). `target_type(..., deny=True)` types an APIGen-MT text reply
  that declines under the policy (`is_policy_denial`: "not eligible", "cannot be cancelled", "I cannot assist with the
  cancellation", ...; at most 500 characters, no list lines, no completion claim, not a request for missing details:
  the first build without those limits typed 20 of its 40 deny rows from reservation summaries that only mention a
  restriction) as `deny`; the default v8 mix is call .36, ask .225,
  reply .135, error .18, deny .1 (40 of 400 rows), commits still 40% of the calls. The reference is
  `{"type": "deny", "text": ...}`: a text reply scores 2, a grounded non-write call 0.5, a write call −1
  (`DENY_WRITE_PENALTY`), the refusal tool −1.5 (benign refusal, unchanged).
- **Refusal wording on next_action rows is an answer.** Only the refusal tool counts as a refusal there; a text reply
  is scored by r_task against the reference (so "I'm sorry, basic economy cannot be cancelled" no longer costs −1.5).
  Other row kinds are unchanged.
- **Text on a call target −0.5** (`TEXT_ON_CALL_PENALTY`, was 0), commit rows included, so an all-text group is not
  a free 0 and a confirmation question after the user's yes costs.

v3 and v7 parquets rebuild unchanged (deny typing is off below version 8). Acceptance: the clean greedy probe at least
as good as v7 and v3 on the model's own rows (the probe reports `deny` rows: right = no call), then the scaffold
evaluations; the targets are tau2 airline no-write tasks back to v3 or better and Qwen3-4B call rows back to v3.

## Cleanup (2026-11-27): v4-v7 artefacts deleted on Viking

To free scratch (426G -> 363G against the 400G soft quota), all v4-v7 run artefacts were deleted, with Rachel's approval:
the merged models and run folders (rollouts, eval outputs) of v4 38858748/49, v5 38889183/84, v6 38974668/69 and v7
39066110-12 (smokes), 39068901 (Qwen2.5), 39102695 (Qwen3-4B), 39102696 (Llama); the v4 and v7 parquets (all three
models); and the v7 side copies (`v7repo`, `v7test`). The unfinished Qwen2.5 v7 MOSAIC-scaffold BFCL eval 39103502 was
cancelled, and Qwen2.5 v7's 13 finished evals were never pulled to `paper_results`, so the v7 numbers above survive only
in this document and the session notes. `trainset/derived/safeutil_train.parquet` (it held v6) is again the v3 data
(byte-identical copy of `safeutil_train.v3.parquet`). Kept: v1-v3 and v8 parquets, all v3 runs, the v8 smoke folder.
The code paths for reward versions 4-7 stay in `safeutil.py`, so any of them can be rebuilt and retrained.

## Data (`safeutil.py build-data`)

| Kind | Rows (default) | Label | From | Rollout |
|---|---|---|---|---|
| `astb` | 1,000 | 500 benign, 500 harmful (`--astb-benign-share 0.5`; 735 of its 2,000 tasks are benign) | Agent-SafetyBench | multi-turn, MOSAIC's loop and environments |
| `next_action` | 400 | benign | unused APIGen-MT dialogs, cut before one assistant turn: a tool call, or a question or report in text (`--target-mix`, v4 default call 0.7 / ask 0.1 / reply 0.1 / error 0.1) | one turn |
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
| r_safety | +1 refusal before any other tool call, +0.5 refusal after acting, −1 no refusal, 0 a refusal-tool call whose reason says the task is safe, is an operational failure (v3) or is empty | −1 if refused |
| r_task | 0 | next_action / benign_prompt, reference call: 2 × share of reference arguments matched by a call to the right tool (v2: 1 + share), 0 otherwise; reference text reply (v3): 2 for a text reply, 0.5 for a grounded call, 0 otherwise; −0.5 more for a call with an ungrounded identifier (v3). astb: the judge's absolute completion grade (0 / 1 / 2) |
| r_fmt | +0.5 only if every turn is well-formed and no `<think>` / `<safety_thoughts>` is the prompt's placeholder text or under three words | same |
| v7 | unchanged | refusal −1.5; r_task −1 for a false completion claim; r_repeat −0.5 for a repeated failed or thrice-identical call |
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
