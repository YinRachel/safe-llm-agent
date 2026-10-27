"""Defender-side datasets for the life-cycle defence baselines (HANDOFF_lifecycle_baselines.md §4.0.1).

Sub-commands (run in this order; each reads the previous one's files):
  split        disjoint, category-stratified slices of the eligible defence ids  -> derived/defense_slices.json
  benign       1000 unused AgentAlign-benign items (same converter and length cap) -> derived/align_benign.jsonl
  build        refusal / SFT / SafeInstr JSONL files in the train_p00.jsonl record schema
  candidates   defender_harmful prompts in the attack_candidates.jsonl schema     -> derived/defender_candidates.jsonl
  (external)   scripts/data/generate_attack_trajectories.py --candidates ... --output derived/defender_generated.jsonl
  qc-defender  build_trainset's QC on the generated trajectories                    -> derived/defender_harmful*.jsonl
  pairs        DPO pairs (AgentAlign refusal vs. QC-passed compliant trajectory)    -> derived/dpo_pairs*.jsonl

  python -I scripts/data/defense_data.py split benign build candidates
  python -I scripts/data/defense_data.py qc-defender pairs --judge-base-url URL --judge-model NAME

Defence data never touches the 480 attack candidates, decontamination drops, or any benign id used in train_p*/val.
Outputs are harmful content: they live under trainset/derived/ (git-ignored, mode 0600). Only ids and counts are printed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).parent))
import build_trainset as bt  # noqa: E402
from build_trainset import (  # noqa: E402
    DATA, INTERIM, ROOT, SEED, STATS, clean_tools, convert_agentalign, first_user, load_agentalign, load_tokenizer,
    make_record, read_json, read_jsonl, sha256_file, shuffled, write_csv, write_json, write_jsonl,
)

DERIVED = ROOT / "derived"
# Allocated in this order from one stratified permutation, so slices are disjoint by construction.
SLICES = (("align_refusal", 1000), ("lisa_guide", 1000), ("safeinstr", 200), ("select_refusal", 100),
          ("defender_harmful", 800))
N_BENIGN = 1000
SAFEINSTR_M = (50, 100, 200)
N_HARMFUL_SELECT = 100  # held-out compliant trajectories for the harmful_nll selection metric
DPO_VAL_FRAC = 0.1
REFUSAL_SOURCE = "agentalign_harmful_refusal"
RECORD_KEYS = ("id", "source", "label", "category", "sub_category", "tools", "messages", "n_tokens")


# ----------------------------------------------------------------------------- pure helpers (unit-tested)
def eligible_defense_ids(defense_ids, attack_ids, dropped):
    """Defence ids minus *all* attack candidates (primaries and backups) minus decontamination drops."""
    return sorted(set(defense_ids) - set(attack_ids) - set(dropped))


def stratified_order(ids, category_of, label):
    """Seeded shuffle within each category, then a proportional interleave: every prefix (hence every consecutive
    slice) has close to the overall category mix. Deterministic and independent of the input order."""
    by_cat = defaultdict(list)
    for item_id in ids:
        by_cat[category_of[item_id]].append(item_id)
    keyed = []
    for category, members in by_cat.items():
        order = shuffled(members, f"{label}:{category}")
        keyed += [((rank + 0.5) / len(order), category, item_id) for rank, item_id in enumerate(order)]
    return [item_id for _, _, item_id in sorted(keyed)]


def allocate(order, sizes):
    if sum(n for _, n in sizes) > len(order):
        sys.exit(f"[split] need {sum(n for _, n in sizes)} defence ids, only {len(order)} eligible; stop and report")
    slices, start = {}, 0
    for name, size in sizes:
        slices[name] = order[start:start + size]
        start += size
    slices["reserve"] = order[start:]
    return slices


def assert_disjoint(slices, attack_ids, dropped):
    names = list(slices)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(slices[a]) & set(slices[b]), f"defence slices {a} and {b} overlap"
        assert not set(slices[a]) & set(attack_ids), f"defence slice {a} contains attack candidates"
        assert not set(slices[a]) & set(dropped), f"defence slice {a} contains decontaminated ids"


def refusal_fields(row):
    """AgentAlign refusal with the *training-record* convention: convert_agentalign(row) keeps the generic system
    prompt, exactly like the agentalign_benign records in train_*.jsonl and the harmful ones from `qc`
    (data/defense_pool.jsonl uses system=False; we deliberately do not follow that file)."""
    return {"category": row["category"], "sub_category": row["sub_category"], "tools": clean_tools(row["tools"]),
            "messages": convert_agentalign(row)}


def benign_fields(row):
    """Identical to build_trainset.stage_assemble.make_aa."""
    return {"category": "benign", "sub_category": row["sub_category"], "tools": clean_tools(row["tools"]),
            "messages": convert_agentalign(row)}


def as_record_schema(record):
    return {k: record[k] for k in RECORD_KEYS}


def mix(base, extra, label):
    rows = list(base) + list(extra)
    random.Random(label).shuffle(rows)
    assert len({r["id"] for r in rows}) == len(rows), f"{label}: duplicate ids"
    return rows


def ids_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def dpo_pair(refusal_messages, compliant_messages, tools, item_id):
    """Split two full conversations into (prompt, chosen, rejected). The prompt (system + first user turn) and tools
    must be identical on both sides; returns None with a reason otherwise."""
    def split(messages):
        cut = next(i for i, m in enumerate(messages) if m["role"] == "user") + 1
        return messages[:cut], messages[cut:]
    prompt_c, chosen = split(refusal_messages)
    prompt_r, rejected = split(compliant_messages)
    if prompt_c != prompt_r:
        return None, "prompt mismatch"
    if not chosen or not rejected or chosen[0]["role"] != "assistant" or rejected[0]["role"] != "assistant":
        return None, "empty side"
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        return None, "side does not end with an assistant turn"
    return {"id": item_id, "tools": tools, "prompt": prompt_c, "chosen": chosen, "rejected": rejected}, None


# ----------------------------------------------------------------------------- io helpers
def ensure_derived():
    DERIVED.mkdir(parents=True, exist_ok=True)
    DERIVED.chmod(0o700)


def update_manifest(stage, info):
    path = DERIVED / "manifest.json"
    manifest = read_json(path) if path.exists() else {}
    info["date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    info["defense_data_sha256"] = sha256_file(Path(__file__))
    manifest[stage] = info
    write_json(path, manifest)


def file_info(path):
    rows = read_jsonl(path)
    return {"n": len(rows), "sha256": sha256_file(path), "ids_sha256": ids_sha256([r["id"] for r in rows])}


def require(path, hint):
    if not Path(path).exists():
        sys.exit(f"missing {path}: {hint}")


def condition_files():
    return {p.stem[len("train_"):]: p for p in sorted(DATA.glob("train_p*.jsonl"))}


def attack_ids():
    require(INTERIM / "attack_candidates.jsonl", "run build_trainset.py pools first")
    return {c["id"] for c in read_jsonl(INTERIM / "attack_candidates.jsonl")}


def load_slices():
    require(DERIVED / "defense_slices.json", "run `defense_data.py split` first")
    slices = read_json(DERIVED / "defense_slices.json")["slices"]
    assert_disjoint(slices, attack_ids(), read_json(INTERIM / "decontam_dropped.json"))
    return slices


# ----------------------------------------------------------------------------- stages
def stage_split(args):
    ensure_derived()
    aa = load_agentalign()
    dropped = read_json(INTERIM / "decontam_dropped.json")
    attack = attack_ids()
    eligible = eligible_defense_ids(read_json(INTERIM / "defense_ids.json"), attack, dropped)
    assert all(bt.agentalign_kind(i) == "harmful" for i in eligible)
    order = stratified_order(eligible, {i: aa[i]["category"] for i in eligible}, "defense")
    slices = allocate(order, SLICES)
    assert_disjoint(slices, attack, dropped)
    by_cat = {name: dict(sorted(Counter(aa[i]["category"] for i in ids).items())) for name, ids in slices.items()}
    write_json(DERIVED / "defense_slices.json", {"seed": SEED, "eligible": len(eligible), "slices": slices,
                                                  "by_category": by_cat})
    update_manifest("split", {"eligible": len(eligible), "sizes": {k: len(v) for k, v in slices.items()},
                              "ids_sha256": {k: ids_sha256(v) for k, v in slices.items()}})
    print(f"[split] eligible {len(eligible)}; " + ", ".join(f"{k}={len(v)}" for k, v in slices.items()))


def used_benign_ids():
    files = list(condition_files().values()) + [DATA / "val_benign.jsonl"]
    for path in files:
        require(path, "run build_trainset.py assemble first")
    if "p00" not in condition_files():
        sys.exit("[benign] train_p00.jsonl missing: it holds the largest benign set (conditions are nested)")
    return {r["id"] for path in files for r in read_jsonl(path)}, [p.name for p in files]


def stage_benign(args):
    ensure_derived()
    tokenizer = load_tokenizer()
    aa = load_agentalign()
    used, files = used_benign_ids()
    dropped = set(read_json(INTERIM / "decontam_dropped.json"))
    order = read_json(INTERIM / "permutations.json")["agentalign_benign"]
    records, skipped = [], Counter()
    for item_id in order:
        if len(records) == N_BENIGN:
            break
        if item_id in used or item_id in dropped:
            skipped["used or decontaminated"] += 1
            continue
        record, reason = make_record(item_id, "agentalign_benign", "benign", benign_fields(aa[item_id]), tokenizer)
        if record and record["n_tokens"] > bt.MAX_TOKENS:
            record, reason = None, "length"
        if reason:
            skipped["length" if "token" in reason or reason == "length" else "convert"] += 1
            continue
        records.append(record)
    if len(records) < N_BENIGN:
        sys.exit(f"[benign] only {len(records)} unused AgentAlign-benign items; stop and report")
    assert not {r["id"] for r in records} & used
    write_jsonl(DERIVED / "align_benign.jsonl", records)
    update_manifest("benign", {"excluded_files": files, "skipped": dict(skipped), "max_tokens": bt.MAX_TOKENS,
                               "align_benign": file_info(DERIVED / "align_benign.jsonl")})
    print(f"[benign] align_benign={len(records)} (skipped {dict(skipped)})")


def refusal_records(ids, aa, tokenizer):
    records, dropped = [], {}
    for item_id in ids:
        record, reason = make_record(item_id, REFUSAL_SOURCE, "harmful", refusal_fields(aa[item_id]), tokenizer)
        if record and record["n_tokens"] > bt.MAX_TOKENS:
            record, reason = None, "length"
        if reason:
            dropped[item_id] = reason
        else:
            records.append(as_record_schema(record))
    return records, dropped


def stage_build(args):
    ensure_derived()
    slices = load_slices()
    require(DERIVED / "align_benign.jsonl", "run `defense_data.py benign` first")
    tokenizer, aa = load_tokenizer(), load_agentalign()
    built, drops = {}, {}
    for name in ("align_refusal", "lisa_guide", "safeinstr", "select_refusal"):
        built[name], drops[name] = refusal_records(slices[name], aa, tokenizer)
    benign = read_jsonl(DERIVED / "align_benign.jsonl")
    used, _ = used_benign_ids()
    assert not {r["id"] for r in benign} & used, "align_benign overlaps train/val"
    outputs = {
        "align_sft.jsonl": mix(built["align_refusal"], benign, f"{SEED}:align_sft"),
        "lisa_guide.jsonl": built["lisa_guide"],
        "safeinstr_pool.jsonl": built["safeinstr"],
        "select_refusal.jsonl": built["select_refusal"],
    }
    for cond, path in condition_files().items():
        base = read_jsonl(path)
        for m in SAFEINSTR_M:
            if m > len(built["safeinstr"]):
                sys.exit(f"[build] safeinstr pool has {len(built['safeinstr'])} < {m} usable refusals; stop and report")
            outputs[f"safeinstr_{cond}_m{m}.jsonl"] = mix(base, built["safeinstr"][:m], f"{SEED}:safeinstr:{cond}:{m}")
    for name, rows in outputs.items():
        write_jsonl(DERIVED / name, rows)
    update_manifest("build", {
        "refusal_dropped": {k: len(v) for k, v in drops.items()},
        "refusal_dropped_ids": drops,
        "system_prompt": "convert_agentalign(row) with GENERIC_SYSTEM_PROMPT (training-record convention)",
        "safeinstr_note": "train_<cond> (N=2000) + first m of safeinstr_pool, i.e. N = 2000 + m (added, not replaced)",
        "files": {name: file_info(DERIVED / name) for name in outputs},
    })
    print("[build] " + ", ".join(f"{name}={len(rows)}" for name, rows in outputs.items()))
    if any(drops.values()):
        print(f"[build] refusals dropped by structure/length: { {k: len(v) for k, v in drops.items()} }")


def stage_candidates(args):
    ensure_derived()
    slices, aa = load_slices(), load_agentalign()
    rank, out = Counter(), []
    for item_id in slices["defender_harmful"]:
        row = aa[item_id]
        out.append({"id": item_id, "category": row["category"], "sub_category": row["sub_category"],
                    "rank": rank[row["category"]], "backup": False, "pattern": row["pattern"], "tools": row["tools"],
                    "instruction": first_user(row["messages"])})
        rank[row["category"]] += 1
    write_jsonl(DERIVED / "defender_candidates.jsonl", out)
    update_manifest("candidates", {"defender_candidates": file_info(DERIVED / "defender_candidates.jsonl"),
                                   "by_category": dict(sorted(rank.items()))})
    print(f"[candidates] defender_candidates={len(out)} {dict(sorted(rank.items()))}")


def stage_qc_defender(args):
    ensure_derived()
    generated_path = DERIVED / "defender_generated.jsonl"
    require(generated_path, "run generate_attack_trajectories.py --candidates derived/defender_candidates.jsonl "
                            "--output derived/defender_generated.jsonl on the GPU node (gen_trajectories.job)")
    if not args.judge_base_url or not args.judge_model:
        sys.exit("[qc-defender] --judge-base-url and --judge-model are required")
    slices = load_slices()
    candidates = {c["id"]: c for c in read_jsonl(DERIVED / "defender_candidates.jsonl")}
    assert set(candidates) == set(slices["defender_harmful"]), "defender_candidates do not match the defender_harmful slice"
    judge = bt.Judge(args.judge_base_url, args.judge_model)
    generated = {g["id"]: g for g in read_jsonl(generated_path)}
    extra = set(generated) - set(candidates)
    assert not extra, f"{len(extra)} generated ids are not defender candidates"
    results, passed = bt.qc_trajectories(candidates, generated, judge, load_tokenizer(), bt.tool_alternatives())
    judge.save()
    write_csv(STATS / "defender_qc.csv", results, ["id", "category", "rank", "backup", "status", "reason", "n_tool_calls", "n_tokens"])
    records = [dict(r, response_type="compliance") for _, items in sorted(passed.items()) for _, r in sorted(items, key=lambda x: x[0])]
    selected = set(shuffled([r["id"] for r in records], "defender_harmful_select")[:N_HARMFUL_SELECT])
    if len(records) < 2 * N_HARMFUL_SELECT:
        sys.exit(f"[qc-defender] only {len(records)} passing trajectories; stop and report")
    train = [r for r in records if r["id"] not in selected]
    held = [r for r in records if r["id"] in selected]
    write_jsonl(DERIVED / "defender_harmful.jsonl", train)
    write_jsonl(DERIVED / "defender_harmful_select.jsonl", held)
    reasons = Counter(r["reason"] for r in results if r["status"] != "pass")
    update_manifest("qc_defender", {"candidates": len(candidates), "passed": len(records), "fail_reasons": dict(reasons),
                                    "judge_model": args.judge_model,
                                    "defender_harmful": file_info(DERIVED / "defender_harmful.jsonl"),
                                    "defender_harmful_select": file_info(DERIVED / "defender_harmful_select.jsonl")})
    print(f"[qc-defender] passed {len(records)}/{len(candidates)}; defender_harmful={len(train)} select={len(held)}; "
          f"fail reasons {dict(reasons)}")


def stage_pairs(args):
    ensure_derived()
    require(DERIVED / "defender_harmful.jsonl", "run `defense_data.py qc-defender` first")
    aa = load_agentalign()
    compliant = read_jsonl(DERIVED / "defender_harmful.jsonl")
    held = {r["id"] for r in read_jsonl(DERIVED / "defender_harmful_select.jsonl")}
    slices = load_slices()
    pairs, skipped = [], Counter()
    for record in compliant:
        assert record["id"] in set(slices["defender_harmful"]) and record["id"] not in held
        refusal = refusal_fields(aa[record["id"]])
        assert refusal["tools"] == record["tools"], f"{record['id']}: tools differ between refusal and trajectory"
        pair, reason = dpo_pair(refusal["messages"], record["messages"], record["tools"], record["id"])
        if reason:
            skipped[reason] += 1
            continue
        pairs.append(dict(pair, category=record["category"]))
    val_ids = set(shuffled([p["id"] for p in pairs], "dpo_val")[:round(DPO_VAL_FRAC * len(pairs))])
    train = [p for p in pairs if p["id"] not in val_ids]
    val = [p for p in pairs if p["id"] in val_ids]
    write_jsonl(DERIVED / "dpo_pairs.jsonl", train)
    write_jsonl(DERIVED / "dpo_pairs_val.jsonl", val)
    update_manifest("pairs", {"skipped": dict(skipped), "dpo_pairs": file_info(DERIVED / "dpo_pairs.jsonl"),
                              "dpo_pairs_val": file_info(DERIVED / "dpo_pairs_val.jsonl"),
                              "prompt": "GENERIC_SYSTEM_PROMPT + AgentAlign instruction on both sides (asserted equal)"})
    print(f"[pairs] dpo_pairs={len(train)} dpo_pairs_val={len(val)} skipped={dict(skipped)}")


STAGES = {"split": stage_split, "benign": stage_benign, "build": stage_build, "candidates": stage_candidates,
          "qc-defender": stage_qc_defender, "pairs": stage_pairs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--judge-base-url", help="OpenAI-compatible endpoint of the refusal judge (qc-defender)")
    parser.add_argument("--judge-model", help="model served at --judge-base-url (qc-defender)")
    args = parser.parse_args(argv)
    for name in args.stages:
        STAGES[name](args)


if __name__ == "__main__":
    main()
