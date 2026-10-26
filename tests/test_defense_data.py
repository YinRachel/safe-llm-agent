"""Offline checks for scripts/defense_data.py: disjointness, stratification, record schema, system-prompt convention,
QC refactor regression and DPO pair construction. Synthetic placeholder data only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import build_trainset as bt  # noqa: E402  (same module object defense_data imports)
import defense_data as dd  # noqa: E402
import qc_fixture  # noqa: E402

TOOLS = qc_fixture.TOOLS


def harmful_row(item_id, category):
    return {"id": item_id, "category": category, "sub_category": "sub", "tools": TOOLS, "pattern": ["Step.one", "Step.two"],
            "messages": [{"role": "user", "content": f"HARMFUL_PLACEHOLDER {item_id}"},
                         {"role": "assistant", "content": "REFUSAL_PLACEHOLDER"}]}


def benign_row(item_id):
    return {"id": item_id, "category": "benign", "sub_category": "sub", "tools": TOOLS, "pattern": [],
            "messages": [{"role": "user", "content": "BENIGN_PLACEHOLDER"},
                         {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": "{\"q\": 1}"}}]},
                         {"role": "tool", "content": "{}", "tool_call_id": "c1"},
                         {"role": "assistant", "content": "BENIGN_DONE"}]}


class FakeTree:
    """A miniature trainset/ with 60 harmful (3 categories), 8 attack candidates, 2 decontam drops, 40 benign."""

    def __init__(self, root):
        self.root = Path(root)
        for sub in ("interim", "data", "stats", "raw", "derived"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        cats = ("cat_a", "cat_b", "cat_c")
        self.aa = {}
        for i in range(60):
            self.aa[f"harmful-{i:03d}"] = harmful_row(f"harmful-{i:03d}", cats[i % 3] if i < 45 else "cat_a")
        for i in range(40):
            self.aa[f"benign-{i:03d}"] = benign_row(f"benign-{i:03d}")
        self.attack = [f"harmful-{i:03d}" for i in range(8)]
        self.dropped = ["harmful-008", "harmful-009"]
        defense = [i for i in self.aa if i.startswith("harmful") and i not in self.attack]  # includes the dropped two on purpose
        self.write("interim/attack_candidates.jsonl", [dict(id=i, category=self.aa[i]["category"]) for i in self.attack])
        (self.root / "interim/defense_ids.json").write_text(json.dumps(defense))
        (self.root / "interim/decontam_dropped.json").write_text(json.dumps(self.dropped + ["benign-039"]))
        benign_ids = sorted(i for i in self.aa if i.startswith("benign"))
        (self.root / "interim/permutations.json").write_text(json.dumps({"agentalign_benign": benign_ids[::-1]}))
        tok = qc_fixture.FakeTokenizer()
        def rec(i):
            r, _ = bt.make_record(i, "agentalign_benign", "benign", dd.benign_fields(self.aa[i]), tok)
            return r
        self.train_ids = benign_ids[30:36]
        self.val_ids = benign_ids[36:39]
        self.write("data/train_p00.jsonl", [rec(i) for i in self.train_ids])
        self.write("data/train_p10.jsonl", [rec(i) for i in self.train_ids])
        self.write("data/val_benign.jsonl", [rec(i) for i in self.val_ids])

    def write(self, rel, rows):
        with open(self.root / rel, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def patches(self):
        r = self.root
        return patch.multiple(dd, ROOT=r, DATA=r / "data", INTERIM=r / "interim", STATS=r / "stats", DERIVED=r / "derived",
                              load_agentalign=lambda: self.aa, load_tokenizer=qc_fixture.FakeTokenizer,
                              SLICES=(("align_refusal", 10), ("lisa_guide", 10), ("safeinstr", 6), ("select_refusal", 4),
                                      ("defender_harmful", 12)),
                              N_BENIGN=8, SAFEINSTR_M=(2, 4), N_HARMFUL_SELECT=3)

    def read(self, rel):
        return bt.read_jsonl(self.root / rel)


ARGS = SimpleNamespace(judge_base_url="http://fake", judge_model="fake")


class PureHelperTests(unittest.TestCase):
    def test_eligible_excludes_every_attack_candidate_and_drop(self):
        self.assertEqual(dd.eligible_defense_ids(["a", "b", "c", "d"], ["b"], ["d"]), ["a", "c"])

    def test_stratified_order_is_deterministic_and_balanced(self):
        ids = [f"x{i}" for i in range(300)]
        cat = {i: ("A" if n < 200 else "B") for n, i in enumerate(ids)}
        order = dd.stratified_order(ids, cat, "t")
        self.assertEqual(order, dd.stratified_order(list(reversed(ids)), cat, "t"))
        self.assertEqual(sorted(order), sorted(ids))
        prefix = [cat[i] for i in order[:30]]
        self.assertEqual(prefix.count("A"), 20)  # 2:1 mix preserved in a prefix

    def test_allocate_is_disjoint_and_reserve_takes_rest(self):
        slices = dd.allocate([f"x{i}" for i in range(10)], (("a", 3), ("b", 4)))
        self.assertEqual([len(slices[k]) for k in ("a", "b", "reserve")], [3, 4, 3])
        dd.assert_disjoint(slices, [], [])
        with self.assertRaises(AssertionError):
            dd.assert_disjoint({"a": ["x"], "b": ["x"]}, [], [])
        with self.assertRaises(AssertionError):
            dd.assert_disjoint({"a": ["x"]}, ["x"], [])
        with self.assertRaises(SystemExit):
            dd.allocate(["x"], (("a", 2),))

    def test_refusal_uses_training_system_prompt_convention(self):
        refusal = dd.refusal_fields(harmful_row("harmful-1", "cat_a"))["messages"]
        benign = dd.benign_fields(benign_row("benign-1"))["messages"]
        # train_*.jsonl: agentalign_benign via convert_agentalign(row); harmful via convert_agentalign(None, generated)
        harmful_train = bt.convert_agentalign(None, [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}])
        self.assertEqual(refusal[0], {"role": "system", "content": bt.GENERIC_SYSTEM_PROMPT})
        self.assertEqual(refusal[0], benign[0])
        self.assertEqual(refusal[0], harmful_train[0])
        self.assertEqual([m["role"] for m in refusal], ["system", "user", "assistant"])

    def test_dpo_pair_requires_identical_prompt(self):
        prompt = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        chosen = [{"role": "assistant", "content": "R"}]
        rejected = [{"role": "assistant", "content": "", "tool_calls": []}, {"role": "tool", "content": "T"},
                    {"role": "assistant", "content": "C"}]
        pair, reason = dd.dpo_pair(prompt + chosen, prompt + rejected, TOOLS, "i")
        self.assertIsNone(reason)
        self.assertEqual((pair["prompt"], pair["chosen"], pair["rejected"]), (prompt, chosen, rejected))
        other = [{"role": "system", "content": "S2"}, {"role": "user", "content": "U"}]
        self.assertEqual(dd.dpo_pair(prompt + chosen, other + rejected, TOOLS, "i"), (None, "prompt mismatch"))


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = FakeTree(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_stages(self, *names):
        with self.tree.patches():
            for name in names:
                dd.STAGES[name](ARGS)

    def test_split_benign_build_candidates(self):
        self.run_stages("split", "benign", "build", "candidates")
        t = self.tree
        slices = json.loads((t.root / "derived/defense_slices.json").read_text())["slices"]
        all_ids = [i for ids in slices.values() for i in ids]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(all_ids), 60 - 8 - 2)
        self.assertFalse(set(all_ids) & set(t.attack + t.dropped))
        # benign: unused ids only, permutation order, decontam drop skipped
        benign = t.read("derived/align_benign.jsonl")
        self.assertEqual(len(benign), 8)
        self.assertFalse({r["id"] for r in benign} & set(t.train_ids + t.val_ids + ["benign-039"]))
        self.assertEqual([r["id"] for r in benign], [f"benign-{i:03d}" for i in range(29, 21, -1)])
        # record schema identical to train_p00.jsonl
        train_keys = set(t.read("data/train_p00.jsonl")[0])
        for name in ("align_sft", "lisa_guide", "safeinstr_pool", "select_refusal", "safeinstr_p10_m4"):
            for row in t.read(f"derived/{name}.jsonl"):
                self.assertEqual(set(row), train_keys, name)
        align = t.read("derived/align_sft.jsonl")
        self.assertEqual(len(align), 10 + 8)
        self.assertEqual({r["id"] for r in align if r["label"] == "harmful"}, set(slices["align_refusal"]))
        self.assertTrue(all(r["source"] == dd.REFUSAL_SOURCE for r in align if r["label"] == "harmful"))
        # SafeInstr adds m refusals to the full condition set (N = base + m), for every existing condition
        for cond in ("p00", "p10"):
            for m in (2, 4):
                rows = t.read(f"derived/safeinstr_{cond}_m{m}.jsonl")
                self.assertEqual(len(rows), len(t.train_ids) + m)
                self.assertEqual({r["id"] for r in rows if r["label"] == "harmful"}, set(slices["safeinstr"][:m]))
        cands = t.read("derived/defender_candidates.jsonl")
        self.assertEqual([c["id"] for c in cands], slices["defender_harmful"])
        self.assertEqual(set(cands[0]), {"id", "category", "sub_category", "pattern", "tools", "instruction", "rank", "backup"})
        manifest = json.loads((t.root / "derived/manifest.json").read_text())
        self.assertEqual(set(manifest), {"split", "benign", "build", "candidates"})

    def test_qc_defender_and_pairs(self):
        self.run_stages("split", "candidates")
        t = self.tree
        cands = t.read("derived/defender_candidates.jsonl")
        plan = {cands[0]["id"]: "judge", cands[1]["id"]: "regex", cands[2]["id"]: "error"}
        t.write("derived/defender_generated.jsonl", [qc_fixture.generated(c, plan.get(c["id"], "ok")) for c in cands])
        with patch.multiple(bt, Judge=qc_fixture.FakeJudge, tool_alternatives=lambda: qc_fixture.ALTERNATIVES):
            self.run_stages("qc-defender", "pairs")
        harmful = t.read("derived/defender_harmful.jsonl")
        held = t.read("derived/defender_harmful_select.jsonl")
        self.assertEqual((len(harmful), len(held)), (12 - 3 - 3, 3))
        self.assertFalse({r["id"] for r in harmful} & {r["id"] for r in held})
        self.assertTrue(all(r["response_type"] == "compliance" and r["label"] == "harmful" for r in harmful))
        pairs = t.read("derived/dpo_pairs.jsonl") + t.read("derived/dpo_pairs_val.jsonl")
        self.assertEqual(sorted(p["id"] for p in pairs), sorted(r["id"] for r in harmful))
        self.assertEqual(len(t.read("derived/dpo_pairs_val.jsonl")), 1)
        for p in pairs:
            self.assertEqual([m["role"] for m in p["prompt"]], ["system", "user"])
            self.assertEqual(p["prompt"][0]["content"], bt.GENERIC_SYSTEM_PROMPT)
            self.assertEqual(p["chosen"][-1]["content"], "REFUSAL_PLACEHOLDER")
            self.assertTrue(any(m["role"] == "tool" for m in p["rejected"]))
        qc_csv = (t.root / "stats/defender_qc.csv").read_text()
        self.assertNotIn("PLACEHOLDER", qc_csv)  # ids and reasons only


class StageQcRegressionTests(unittest.TestCase):
    def test_refactored_stage_qc_is_byte_identical(self):
        golden = json.loads((HERE / "fixtures/stage_qc_golden.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            out = qc_fixture.run_stage_qc(bt, directory)
        self.assertEqual(sorted(out), sorted(golden))
        for name, content in out.items():
            self.assertEqual(content.decode(), golden[name], name)


if __name__ == "__main__":
    unittest.main()
