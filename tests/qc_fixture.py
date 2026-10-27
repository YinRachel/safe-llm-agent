"""Synthetic QC fixture shared by the stage_qc regression test (placeholder strings only, no real data).

`run_stage_qc(bt, root)` builds a fake trainset tree under `root`, patches the judge/tokenizer/tool map, runs
bt.stage_qc and returns {relative path: bytes} of every file it wrote under data/ and stats/.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CATEGORIES = ("cat_a", "cat_b")
QUOTA, BACKUPS = 20, 4
TOOLS = [{"type": "function", "function": {"name": "tool_a", "parameters": {"type": "object", "required": ["q"]}}},
         {"type": "function", "function": {"name": "tool_b", "parameters": {"type": "object", "required": []}}}]
ALTERNATIVES = {"Step.one": {"tool_a"}, "Step.two": {"tool_b"}}


class FakeTokenizer:
    def apply_chat_template(self, messages, tools=None, tokenize=False):
        text = ""
        if tools:
            text += "<|im_start|>system\n# Tools\n" + json.dumps(tools) + "<|im_end|>\n"
        for message in messages:
            body = message["content"]
            for call in message.get("tool_calls", []):
                body += "<tool_call>\n" + json.dumps(call["function"]) + "\n</tool_call>"
            text += f"<|im_start|>{message['role']}\n{body}<|im_end|>\n"
        return text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


class FakeJudge:
    def __init__(self, *args, **kwargs):
        pass

    def is_refusal(self, text):
        return "JUDGE_REFUSES" in text

    def save(self):
        pass


def candidate(category, rank):
    return {"id": f"harmful-{category}-{rank:02d}", "category": category, "sub_category": "sub", "rank": rank,
            "backup": rank >= QUOTA, "pattern": ["Step.one", "Step.two"], "tools": TOOLS,
            "instruction": f"HARMFUL_PLACEHOLDER {category} {rank}"}


def generated(item, variant):
    """variant: ok | judge | regex | pattern | error"""
    if variant == "error":
        return {"id": item["id"], "error": "RuntimeError('no final answer')"}
    first = "tool_b" if variant == "pattern" else "tool_a"
    final = {"ok": "Done with PLACEHOLDER.", "judge": "JUDGE_REFUSES politely.", "regex": "I'm sorry, PLACEHOLDER."}.get(variant, "Done.")
    return {"id": item["id"], "messages": [
        {"role": "user", "content": item["instruction"]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": first, "arguments": json.dumps({"q": "x"})}}]},
        {"role": "tool", "content": json.dumps({"result": "PLACEHOLDER"}), "tool_call_id": "c1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}}]},
        {"role": "tool", "content": json.dumps({"result": "PLACEHOLDER"}), "tool_call_id": "c2"},
        {"role": "assistant", "content": final}]}


# Deterministic failure plan: (category, rank) -> variant; everything else passes.
FAILURES = {("cat_a", 1): "judge", ("cat_a", 5): "regex", ("cat_a", 7): "pattern", ("cat_a", 9): "error",
            ("cat_a", 11): "judge", ("cat_b", 3): "pattern", ("cat_b", 21): "error"}


def build_tree(root):
    root = Path(root)
    for sub in ("interim", "data", "stats", "raw"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cands = [candidate(c, r) for c in CATEGORIES for r in range(QUOTA + BACKUPS)]
    with open(root / "interim/attack_candidates.jsonl", "w") as stream:
        for c in cands:
            stream.write(json.dumps(c) + "\n")
    with open(root / "interim/attack_generated.jsonl", "w") as stream:
        for c in cands:
            stream.write(json.dumps(generated(c, FAILURES.get((c["category"], c["rank"]), "ok"))) + "\n")
    order = {"slots": [f"harmful-{c}-{r:02d}" for r in range(QUOTA) for c in ("cat_b", "cat_a")],
             "category_order": ["cat_b", "cat_a"],
             "backups": {c: [f"harmful-{c}-{r:02d}" for r in range(QUOTA, QUOTA + BACKUPS)] for c in CATEGORIES}}
    (root / "interim/attack_order.json").write_text(json.dumps(order))
    defense = [f"harmful-def-{i:02d}" for i in range(3)]
    (root / "interim/defense_ids.json").write_text(json.dumps(defense))
    aa = {}
    for item_id in defense + [c["id"] for c in cands]:
        aa[item_id] = {"id": item_id, "category": "cat_a", "sub_category": "sub", "tools": TOOLS, "pattern": [],
                       "messages": [{"role": "user", "content": "HARMFUL_PLACEHOLDER"}, {"role": "assistant", "content": "REFUSAL_PLACEHOLDER"}]}
    return cands, aa


def run_stage_qc(bt, root):
    root = Path(root)
    _, aa = build_tree(root)
    paths = dict(ROOT=root, RAW=root / "raw", INTERIM=root / "interim", DATA=root / "data", STATS=root / "stats")
    with patch.multiple(bt, ATTACK_N=2 * QUOTA, Judge=FakeJudge, load_tokenizer=FakeTokenizer,
                        tool_alternatives=lambda: ALTERNATIVES, load_agentalign=lambda: aa, **paths):
        bt.stage_qc(SimpleNamespace(judge_base_url="http://fake", judge_model="fake"))
    return {str(p.relative_to(root)): p.read_bytes() for sub in ("data", "stats", "interim")
            for p in sorted((root / sub).glob("*")) if p.name in (
                "attack_pool.jsonl", "defense_pool.jsonl", "harmful_qc.csv", "harmful_spotcheck_40.csv", "attack_selection.json")}
