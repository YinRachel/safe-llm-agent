# Life-cycle defence baselines: status

Spec: [HANDOFF_lifecycle_baselines.md](../HANDOFF_lifecycle_baselines.md). Branch `feat/lifecycle-baselines` (not pushed).
State at the end of the overnight run (2026-10-26): code, offline tests, Slurm scripts and run book done; **nothing has
run on a GPU** and the defender generation (Phase 2 data) has not run.

## Per-baseline status

| ID | Method | Implemented | Code | Tests | Needs before running |
|---|---|---|---|---|---|
| A0 | Safety SFT | ✔ | `train_defense.py --method sft` on `derived/align_sft.jsonl` | sft ≡ `train_lora.py` (adapter, train/eval loss to 1e-6) | nothing |
| A1 | Booster (λ=5, α=0.1) | ✔ | `--method booster` | λ=0 ≡ A0 exactly; λ>0 changes the update; 3 fwd + 3 bwd counted | Phase 2 (`defender_harmful.jsonl`) |
| A2 | Safety DPO immunisation (β=0.1) | ✔ | `--method dpo` from the base model | loss = log 2 at init; margin antisymmetry; prompt/tool tokens masked | Phase 2 (`dpo_pairs*.jsonl`) |
| F1 | SafeInstr (m ∈ {50,100,200}) | ✔ | `derived/safeinstr_<cond>_m<m>.jsonl` + unchanged `train_lora.py` | N = 2000 + m, every condition, ids | `defense_data.py build` again once `train_p05/p10/p20` exist |
| F2 | Lisa (5/45 steps, ρ=1) | ✔ | `--method lisa` | phase sequence; anchors/drift at switches; ρ=0 + no alignment ≡ sft | nothing |
| F3 | Constrained SFT (β=0.1, 20/5/5) | ✔ | `--method csft` | first step ≡ sft (policy = reference); weights = 1 when equal; per-turn β schedule | nothing |
| F4 | PTST | ✔ | `templates/ptst.jinja` (`safety_template.py`), `CHAT_TEMPLATE` in `serve_qwen.sh`, `generate_harmbench.py --chat-template`, `evaluate.py` manifest | byte-identical rendering ± safety text in all four system/tools branches; hooks only active when set | nothing |
| P1 | Safe LoRA | ✔ (gated) | `posthoc_defense.py --method safelora` | rank-1 cosine 1; equals explicit upstream projector; name matching under shuffled order; K=0 identity; shape mismatch stops | **decision on the unaligned reference (Q2)** |
| P2 | Antidote | ✔ | `--method antidote` | exact fraction per layer; zeroed = top scores; ratio 0 identity; hooks removed | Phase 2 (calibration set) |
| P3 | Safety DPO re-alignment | ✔ | `--method dpo --model results/qwen4B_LoRA/pXX-*/merged` | shared with A2 | Phase 2 |
| P4 | LoRA down-scaling | ✔ | `--method scale` | s=0 → base weights; s=1 → attacked merged weights; sweep + selection | nothing |

Shared: `defense_data.py` (tests incl. a byte-identical regression of the refactored `build_trainset.py qc`),
`defense_common.py`, `selection_metrics.py` (selection rule tested), `train_defense.sbatch`,
`posthoc.sbatch`, `eval_all.sh` (tested against a fake `sbatch`), `gen_trajectories.job` overrides
(`CANDIDATES`, `GEN_OUTPUT`, `SKIP_QC`, plus `DEFENDER_QC=1`; unset = previous behaviour), `download.sh` (+ Qwen3-4B-Base).

### Tests

```
.venv-trainset/bin/python -m pytest -q tests          # 2 failed (pre-existing, see below), 80 passed, 17 skipped
.venv/bin/python -m unittest discover -s tests -v      # OK (torch/peft-dependent tests skip in .venv)
bash -n scripts/*.sh scripts/*.sbatch scripts/viking/*.job
```
- The trainer and post-hoc tests were also run on a scratch venv with the **pinned training stack** (torch 2.8.0,
  transformers 4.55.2, peft 0.17.1): all pass. The local `.venv-trainset` has transformers 4.57.6; both agree.
- `peft 0.17.1` and `accelerate 1.15.0` were installed into `.venv-trainset` (allowed by the handoff). Nothing else changed.
- Pre-existing, unchanged from `main`: `tests/test_evaluate.py::{test_invalid_data_rejected_before_inference,
  test_no_harmbench_benign_inference}` fail under `.venv-trainset` only (it has no `python-dotenv`); they pass in `.venv`.
- `train_lora.py` is unchanged: transformers 4.55.2 (and 4.57) ignore `revision` for a local directory (checked with a
  tiny saved model), so alignment-stage attacks can call it with `--model <merged dir>` as is.

## Decided

- **`--max-length` = 9216 (Rachel, 2026-10-27).** It matches the build cap (`build_trainset.py` `MAX_TOKENS`), so no
  record is dropped for length (table below). The default lives in one place, `train_lora.MAX_LENGTH`, and
  `defense_common`, `posthoc_defense` and `selection_metrics` import it, so every run uses the same cap without passing
  the flag.

## Open questions / decisions for Rachel

1. **Safe LoRA unaligned reference (stop condition hit, P1 gated).** All 252 target-module shapes of
   `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539` match the instruct model (hub safetensors headers), but
   neither model card declares `base_model`; Instruct-2507 lists "Pretraining & Post-training", native 262,144 context and
   rope_theta 5e6, while Base is 32,768 / 1e6. So `W_instruct − W_base` probably includes an unreleased long-context
   continued-pretraining delta, not only alignment. `posthoc_defense.py --method safelora` refuses the default pair
   unless `--accept-lineage` is passed. Suggested check: `--report-only` (writes per-module `rel_norm_V = ||V||/||W||`
   and cosines). No other model was substituted.
2. **Loss normalisation (deviation from §4.0.2, flagged).** The reference HF Trainer normalises the token loss by the
   assistant tokens of the whole 16-micro-batch window (`num_items_in_batch`). Setting `model_accepts_loss_kwargs = False`
   in the token-level trainers, as §4.0.2 asks, would switch them to a mean of per-micro-batch means. That up-weights
   short refusals (~160 trained tokens vs ~650 per benign item), and it makes the §6 anchor and the λ=0 / ρ=0 reductions
   impossible at grad-accum 16. Lisa, constrained SFT and Booster's alignment term therefore keep the reference
   normalisation, and only DPO (a per-pair loss) uses `model_accepts_loss_kwargs = False`. Switching is a one-line change
   per trainer if you prefer the spec literally.
4. **Lisa step accounting (documented default implemented).** Phases switch at optimizer-step boundaries (5 alignment /
   45 fine-tune); all 16 micro-batches of an alignment step come from the guide stream, and the user batches drawn
   meanwhile are discarded and counted (`method_state.json`). Total steps = reference. The proximal term ρ/2·‖w − anchor‖²
   counts once per optimizer step (split over micro-steps), is skipped for the first 10 % of optimizer steps, and anchors
   are refreshed exactly as upstream `switch_model`. Project recipe (lr 1e-4, 2 epochs), not upstream's lr 1e-5.
5. **Booster details.** Harmful passes use per-example mean NLL (as upstream's `compute_loss`), run on a forked RNG (so
   dropout in the alignment pass matches A0 and λ=0 reproduces A0 bit-for-bit), and the weights are restored by copy
   rather than by adding the step back. λ(g_h − g_h′)/grad_accum is added to `.grad` every micro-step. Project recipe, not
   upstream's lr 5e-4 / 20 epochs / r32.
6. **Constrained SFT.** The position index restarts at every assistant turn (`--csft-position-scope turn`;
   `conversation` = upstream). Weights use the reference log-prob from the adapter-disabled model in eval mode, under
   `no_grad`.
7. **DPO.** Own implementation (per-record tools, assistant-only log-probs). Chosen and rejected get two separate
   forwards, so the short refusal is not padded to the trajectory. Reference log-probs are precomputed once under the
   trainer's autocast (`ref_logps.pt`). Eval and checkpointing run every 20 steps (`save_steps` must be a multiple of
   `eval_steps`). β = 0.1; a β sweep is optional (use `selection_metrics.py pick`). A40 memory for 9k-token
   trajectories is untested: fall back to `gpuplus`.
8. **Safe LoRA selection.** `--select-type number` projects exactly the K lowest-cosine modules (ties broken by name).
   Upstream thresholds at the K-th value and can project more than K on ties. `C B` is computed as `V (Vᵀ B) / ‖V‖_F`
   (same quantity, no 9728×9728 projector). Pdst is reported as upstream (mean over all modules).
9. **Antidote.** Wanda statistics come from all tokens of the rendered calibration trajectory (prompt + harmful
   answer), 256 trajectories × ≤ 2048 tokens. Pruning is unstructured, per linear layer, on the merged model; it is
   saved to a new directory, never in place.
10. **Data layout on Viking.** Trainers default to `trainset/...` relative to the repo (like `train_lora.py`), and
    `defense_data.py` writes to `$TRAINSET_ROOT/derived`. Keep `trainset/` in the repo pointing at the build output
    (`ln -s $SCRATCH/agentft trainset` if it isn't already), or pass explicit paths.
11. `qc-defender` writes `defender_harmful.jsonl` **without** the 100 held-out `defender_harmful_select.jsonl` items, so
    every consumer (Booster, Antidote, DPO pairs) excludes them automatically. Consumers assert this again, together
    with disjointness from the 480 attack candidates.

## Defender data (milestone 1), built locally on 2026-10-26

`python -I scripts/data/defense_data.py split benign build candidates` (CPU, `.venv-trainset`, tokenizer at the pinned revision).

| Slice / file | n | sha256 (first 16) |
|---|---|---|
| eligible defence ids (`defense_ids.json` − 480 attack candidates − decontam drops) | 4476 | |
| `align_refusal` / `lisa_guide` / `safeinstr` / `select_refusal` / `defender_harmful` / `reserve` (id lists) | 1000 / 1000 / 200 / 100 / 800 / 1376 | 2275fba9763d49dd / ee5255ea864b6bea / 5c394e21cfe6ae5e / 6806a22304128877 / a07f83dd7d5aee5f / ca0f9f6ff039f52f |
| `align_benign.jsonl` (unused AgentAlign-benign; skipped 500 used/decontam, 42 > 9216 tokens) | 1000 | ac88f5a0a8f6154e |
| `align_sft.jsonl` (1000 refusals + 1000 benign) | 2000 | eae563f6523ec624 |
| `lisa_guide.jsonl` | 1000 | d9f0dc6ab2f13689 |
| `safeinstr_pool.jsonl` | 200 | 4d49a5b8d998658a |
| `select_refusal.jsonl` | 100 | cf2cf86a091f9204 |
| `safeinstr_p00_m50/m100/m200.jsonl` | 2050 / 2100 / 2200 | 3db16d17c96b93c3 / d64c91a12b982d9c / 78935ccef35dc857 |
| `defender_candidates.jsonl` | 800 | e447d8a2ce286dc0 |

No refusal was dropped by structure or length. `defender_candidates` per category: bias_and_fairness 213, misinformation 174,
cyber_crime 103, online_fraud 101, self_harm 96, intellectual_property 48, illicit_trade 43, privacy 22 (proportional to
the eligible pool). Full sha256s and id-list hashes are in `trainset/derived/manifest.json`.

Implementation choices:
- Slices come from one category-stratified permutation (seeded shuffle per category with label `defense:<category>`,
  then a proportional interleave) cut into consecutive blocks in the order of the table, so they are disjoint by
  construction; disjointness from each other, from all 480 attack candidates and from decontam drops is asserted.
- Refusal conversations use `convert_agentalign(row)` (generic system prompt), which is the convention of the
  `train_*.jsonl` records (AgentAlign-benign via `make_aa`, harmful via `qc`), not of `data/defense_pool.jsonl`
  (`system=False`). Tested in `tests/test_defense_data.py`.
- `align_benign` excludes every id in all `train_p*.jsonl` and `val_benign.jsonl` present at build time (conditions are
  nested, so `train_p00` holds the largest benign set; the stage refuses to run without it).
- `build_trainset.stage_qc` was refactored into `qc_trajectories()` + `stage_qc`; a regression test compares its outputs
  byte-for-byte against golden files produced by the pre-refactor code on a synthetic fixture.

### Length cap check (§6): records > 9216 tokens, i.e. dropped by the `--max-length` default

| file | n | > 8192 | > 9216 | max |
|---|---|---|---|---|
| train_p00.jsonl | 2000 | 104 (5.2 %) | 0 | 9188 |
| val_benign.jsonl | 400 | 23 | 0 | 9203 |
| align_benign.jsonl / align_sft.jsonl | 1000 / 2000 | 16 / 16 | 0 | 9137 |
| lisa_guide / safeinstr_pool / select_refusal | 1000 / 200 / 100 | 0 | 0 | ≤ 1800 |
| safeinstr_p00_m* | 2050–2200 | 104 | 0 | 9188 |

At the old 8192 default ~5 % of every condition was silently lost, so the default was raised to 9216 (see Decided).
p05/p10/p20 harmful trajectories are capped at the same `MAX_TOKENS` by `build_trainset.py` qc, so they also fit.

## Morning run book (Viking; replace CODE with the project account; nothing below has been run)

All commands are run from the repo root on a login node. Results are only compared on p10 at first (priority: p10, p00,
p20, p05). Every final model is evaluated with `SBATCH_ACCOUNT=CODE bash scripts/viking/eval_all.sh <merged dir> <tag>`
(5 jobs, ids in `results/eval_index.tsv`).
Runs are grouped as `results/<model>_<method>/<run>` (e.g. `results/qwen4B_LoRA/p10-<JOB>`, see
`scripts/viking/results_dir.sh`); evaluations of a run go to `<run>/eval/`, of a hub model to `results/<model>_base/`.

**Day 0**
```bash
git pull && git checkout feat/lifecycle-baselines
SCRATCH=/mnt/scratch/users/$USER bash scripts/viking/download.sh      # now also Qwen3-4B-Base
.venv-trainset/bin/python -I scripts/data/defense_data.py split benign build candidates   # rerun `build` when train_p05/p10/p20 exist
```

**Phase 1: no harmful compliant data needed**
```bash
# A0 safety SFT -> M_A0, then the unchanged user attack per condition (--output overrides the sbatch default name)
METHOD=sft STAGE=align TRAIN=trainset/derived/align_sft.jsonl sbatch --account=CODE scripts/viking/train_defense.sbatch --merge
CONDITION=p10 sbatch --account=CODE scripts/viking/train_lora.sbatch --model results/qwen4B_SFT/align-<JOB>/merged --merge --output results/qwen4B_SFT/a0-attack-p10

# F1 SafeInstr: m sweep on p10, then the chosen m for every condition
for m in 50 100 200; do CONDITION=p10 sbatch --account=CODE scripts/viking/train_lora.sbatch \
  --train trainset/derived/safeinstr_p10_m$m.jsonl --merge --output results/qwen4B_SafeInstr/m$m-p10; done
for run in results/qwen4B_LoRA/p10-<JOB> results/qwen4B_SafeInstr/m{50,100,200}-p10; do sbatch --account=CODE --partition=gpu --gres=gpu:1 --time=01:00:00 \
  --wrap ".venv-serve-cu128/bin/python scripts/post-training/selection_metrics.py measure --model $run/merged --out $run/selection_metrics.json"; done
.venv-trainset/bin/python scripts/post-training/selection_metrics.py pick --reference results/qwen4B_LoRA/p10-<JOB>/selection_metrics.json \
  --candidate m50=results/qwen4B_SafeInstr/m50-p10/selection_metrics.json --candidate m100=results/qwen4B_SafeInstr/m100-p10/selection_metrics.json \
  --candidate m200=results/qwen4B_SafeInstr/m200-p10/selection_metrics.json --out results/qwen4B_SafeInstr/selection-p10.json

# F2 Lisa, F3 constrained SFT
METHOD=lisa STAGE=ft CONDITION=p10 sbatch --account=CODE scripts/viking/train_defense.sbatch --merge
METHOD=csft STAGE=ft CONDITION=p10 sbatch --account=CODE scripts/viking/train_defense.sbatch --merge

# F4 PTST: evaluation only (templates/ptst.jinja is committed; `.venv/bin/python scripts/post-training/safety_template.py` regenerates it)
SBATCH_ACCOUNT=CODE bash scripts/viking/eval_all.sh Qwen/Qwen3-4B-Instruct-2507 base-ptst templates/ptst.jinja
SBATCH_ACCOUNT=CODE bash scripts/viking/eval_all.sh results/qwen4B_LoRA/p10-<JOB>/merged p10-ptst templates/ptst.jinja

# P4 LoRA down-scaling: select s on p10, then the chosen s everywhere; s=0.5 as a fixed reference row
METHOD=scale TAG=p10 sbatch --account=CODE scripts/viking/posthoc.sbatch --adapter results/qwen4B_LoRA/p10-<JOB>/adapter --select
METHOD=scale TAG=p00-s0.5 sbatch --account=CODE scripts/viking/posthoc.sbatch --adapter results/qwen4B_LoRA/p00-<JOB>/adapter --scale 0.5

# P1 Safe LoRA: diagnostics first (Q2), then, once the reference pair is approved:
METHOD=safelora TAG=p10-report sbatch --account=CODE scripts/viking/posthoc.sbatch --adapter results/qwen4B_LoRA/p10-<JOB>/adapter --report-only
METHOD=safelora TAG=p10 sbatch --account=CODE scripts/viking/posthoc.sbatch --adapter results/qwen4B_LoRA/p10-<JOB>/adapter --select --accept-lineage
```

**Phase 2: after the defender generation job**
```bash
# Generation + defender QC + DPO pairs in one job (same Mistral server as generator and refusal judge)
CANDIDATES=$SCRATCH/agentft/derived/defender_candidates.jsonl GEN_OUTPUT=$SCRATCH/agentft/derived/defender_generated.jsonl \
  SKIP_QC=1 DEFENDER_QC=1 MISTRAL_REVISION=<commit> sbatch --account=CODE scripts/viking/gen_trajectories.job

# A1 Booster -> M_A1 -> attack; A2 DPO immunisation -> M_A2 -> attack
METHOD=booster STAGE=align TRAIN=trainset/derived/align_sft.jsonl sbatch --account=CODE --time=1-12:00:00 scripts/viking/train_defense.sbatch --merge
METHOD=dpo STAGE=align sbatch --account=CODE scripts/viking/train_defense.sbatch --merge
CONDITION=p10 sbatch --account=CODE scripts/viking/train_lora.sbatch --model results/qwen4B_Booster/align-<JOB>/merged --merge --output results/qwen4B_Booster/a1-attack-p10
CONDITION=p10 sbatch --account=CODE scripts/viking/train_lora.sbatch --model results/qwen4B_DPO/align-<JOB>/merged --merge --output results/qwen4B_DPO/a2-attack-p10

# P2 Antidote (select ratio on p10), P3 DPO re-alignment of the attacked model
METHOD=antidote TAG=p10 sbatch --account=CODE scripts/viking/posthoc.sbatch --merged results/qwen4B_LoRA/p10-<JOB>/merged --select
METHOD=dpo STAGE=post CONDITION=p10 MODEL_PATH=results/qwen4B_LoRA/p10-<JOB>/merged sbatch --account=CODE scripts/viking/train_defense.sbatch --merge
```

Results table columns: method, stage, p, AgentHarm-harmful score↓ / refusal↑, AgentHarm-benign score↑ / refusal↓,
HarmBench ASR↓, BFCL acc↑, τ-bench pass^1↑ (retail, airline), extra GPU-h and peak memory (from each run's `cost.json`:
`wall_clock_s`, `gpu_name`, `max_memory_allocated_bytes`, optimizer steps, forward/backward passes, trained /
calibration / selection tokens).
