# Life-cycle defence baselines: status

Spec: [HANDOFF_lifecycle_baselines.md](../HANDOFF_lifecycle_baselines.md). Branch `feat/lifecycle-baselines` (not pushed).
This file is filled in per milestone; the final per-baseline table is at the end.

## Defender data (milestone 1), built locally on 2026-10-26

`python -I scripts/defense_data.py split benign build candidates` (CPU, `.venv-trainset`, tokenizer at the pinned revision).

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

### Length cap check (§6): records > 8192 tokens, i.e. dropped by `train_lora.py`'s default `--max-length 8192`

| file | n | > 8192 | max |
|---|---|---|---|
| train_p00.jsonl | 2000 | 104 (5.2 %) | 9188 |
| val_benign.jsonl | 400 | 23 | 9203 |
| align_benign.jsonl / align_sft.jsonl | 1000 / 2000 | 16 / 16 | 9137 |
| lisa_guide / safeinstr_pool / select_refusal | 1000 / 200 / 100 | 0 | ≤ 1800 |
| safeinstr_p00_m* | 2050–2200 | 104 | 9188 |

Decision for Rachel: the build cap is 9216 but training drops at 8192, so ~5 % of every condition is silently lost (and
the refusals are never dropped, so the effective harmful/safety share rises slightly). The memory note from the build
session says `train_lora` "needs --max-length 9216". Whatever is chosen must be passed identically to every run; all
defence scripts take the same `--max-length` flag with the same default (8192).
