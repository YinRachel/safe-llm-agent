"""Offline trainer tests on a tiny random Qwen3 (CPU, no downloads, placeholder data).

Anchor: `train_defense.py --method sft` reproduces `train_lora.py` (same loss curve and adapter to 1e-6). Each method's
degenerate setting reduces to it: Lisa with rho=0 and no alignment steps, Booster with lambda=0, constrained SFT at
initialisation (policy = reference). DPO: loss log 2 at initialisation, margin antisymmetry, assistant-only masking.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "scripts"), str(HERE)]
try:
    import peft  # noqa: F401
    import torch
    HAVE_STACK = True
except ImportError:  # .venv without the training stack
    HAVE_STACK = False

import tiny_qwen  # noqa: E402

if HAVE_STACK:
    import defense_common as dc
    import train_defense as td
    spec = importlib.util.spec_from_file_location("train_lora", HERE.parent / "scripts/train_lora.py")
    train_lora = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_lora)


if HAVE_STACK:
    import transformers

    class CpuTrainingArguments(transformers.TrainingArguments):
        """train_lora.py hard-codes bf16=True, which needs use_cpu=True on a GPU-less test machine (CPU bf16
        autocast). Applied identically to the reference and to every defence run."""

        def __post_init__(self):
            self.use_cpu = True
            super().__post_init__()


def cpu_training_arguments():
    return patch.object(transformers, "TrainingArguments", CpuTrainingArguments)


def adapter(path):
    from safetensors.torch import load_file
    return load_file(str(Path(path) / "adapter/adapter_model.safetensors"))


def max_diff(a, b):
    assert a.keys() == b.keys()
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


def losses(path):
    history = json.loads((Path(path) / "log_history.json").read_text())
    return [h["loss"] for h in history if "loss" in h]


@unittest.skipUnless(HAVE_STACK, "needs torch + peft (install peft>=0.17,<0.18 into .venv-trainset)")
class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.model = str(tiny_qwen.make_model_dir(root / "model"))
        cls.train = tiny_qwen.write_jsonl(root / "train.jsonl", [tiny_qwen.conversation(i) for i in range(10)])
        cls.val = tiny_qwen.write_jsonl(root / "val.jsonl", [tiny_qwen.conversation(100 + i) for i in range(2)])
        cls.guide = tiny_qwen.write_jsonl(root / "guide.jsonl", [tiny_qwen.refusal(i) for i in range(6)])
        cls.harmful = tiny_qwen.write_jsonl(root / "harmful.jsonl", [tiny_qwen.conversation(200 + i, "harmful") for i in range(4)])
        prompt = lambda i: [{"role": "system", "content": "SYSTEM_PLACEHOLDER"}, {"role": "user", "content": f"HARMFUL_PLACEHOLDER {i}"}]
        def pair(i):
            rejected = tiny_qwen.conversation(300 + i, "harmful")["messages"][2:]
            return {"id": f"pair-{i}", "tools": tiny_qwen.TOOLS, "prompt": prompt(i),
                    "chosen": [{"role": "assistant", "content": "REFUSAL_PLACEHOLDER " + "r" * i}], "rejected": rejected}
        cls.pairs = tiny_qwen.write_jsonl(root / "pairs.jsonl", [pair(i) for i in range(6)])
        cls.pairs_val = tiny_qwen.write_jsonl(root / "pairs_val.jsonl", [pair(10 + i) for i in range(2)])
        cls.trainset = root / "trainset"
        (cls.trainset / "interim").mkdir(parents=True)
        cls.patch = patch.multiple(dc, TRAINSET=cls.trainset, DERIVED=cls.trainset / "derived")
        cls.patch.start()
        cls.cpu = cpu_training_arguments()
        cls.cpu.start()
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.cpu.stop()
        cls.tmp.cleanup()

    def common(self, name, steps=3):
        return ["--model", self.model, "--train", self.train, "--val", self.val, "--output", str(Path(self.tmp.name) / name),
                "--grad-accum", "2", "--max-steps", str(steps), "--eval-steps", "50", "--save-steps", "50"]

    def run_defense(self, name, *extra, steps=3):
        if name not in self.runs:
            td.main(self.common(name, steps) + list(extra))
            self.runs[name] = Path(self.tmp.name) / name
        return self.runs[name]

    def reference(self, steps=3):
        name = f"train_lora_{steps}"
        if name not in self.runs:
            train_lora.main(self.common(name, steps))
            self.runs[name] = Path(self.tmp.name) / name
        return self.runs[name]

    # ------------------------------------------------------------------ anchor
    def test_shared_flags_and_defaults_match_train_lora(self):
        ref = vars(train_lora.parse_args(["--output", "x", "--condition", "p10"]))
        ours = vars(td.parse_args(["--method", "sft", "--output", "x", "--condition", "p10"]))
        self.assertEqual({k: ours[k] for k in ref}, ref)
        dpo = vars(td.parse_args(["--method", "dpo", "--output", "x"]))
        self.assertEqual((dpo["lr"], dpo["warmup_ratio"], dpo["epochs"]), (5e-6, 0.1, 1))

    def test_sft_reproduces_train_lora(self):
        ref, ours = self.reference(), self.run_defense("sft", "--method", "sft")
        self.assertLess(max_diff(adapter(ref), adapter(ours)), 1e-6)
        ref_summary = json.loads((ref / "summary.json").read_text())
        our_summary = json.loads((ours / "summary.json").read_text())
        self.assertAlmostEqual(ref_summary["train"]["train_loss"], our_summary["train"]["train_loss"], places=6)
        self.assertAlmostEqual(ref_summary["eval"]["eval_loss"], our_summary["eval"]["eval_loss"], places=6)
        cost = json.loads((ours / "cost.json").read_text())
        self.assertEqual((cost["optimizer_steps"], cost["micro_steps"], cost["forward_passes"]), (3, 6, 6))
        manifest = json.loads((ours / "manifest.json").read_text())
        self.assertIn("sha256", manifest["inputs"]["train"])

    # ------------------------------------------------------------------ Lisa
    def test_lisa_phase_sequence(self):
        phases = [td.lisa_phase(s, 5, 45) for s in range(100)]
        self.assertEqual(phases[:5], ["alignment"] * 5)
        self.assertEqual(phases[5:50], ["finetune"] * 45)
        self.assertEqual(phases[50:55], ["alignment"] * 5)
        self.assertTrue(all(td.lisa_phase(s, 0, 45) == "finetune" for s in range(100)))

    def test_lisa_without_alignment_or_proximal_equals_sft(self):
        ours = self.run_defense("lisa0", "--method", "lisa", "--lisa-guide", self.guide, "--lisa-rho", "0",
                                "--lisa-align-steps", "0")
        self.assertLess(max_diff(adapter(self.run_defense("sft", "--method", "sft")), adapter(ours)), 1e-6)

    def test_lisa_switches_and_anchors(self):
        out = self.run_defense("lisa", "--method", "lisa", "--lisa-guide", self.guide, "--lisa-align-steps", "1",
                               "--lisa-finetune-steps", "2", "--lisa-rho", "1", "--lisa-prox-warmup", "0", steps=6)
        state = json.loads((out / "method_state.json").read_text())
        self.assertEqual([(p["step"], p["status"]) for p in state["phases"]],
                         [(0, "alignment"), (1, "finetune"), (3, "alignment"), (4, "finetune")])
        self.assertEqual(state["guide_micro_batches"], 2 * 2)  # 2 alignment steps x grad-accum 2
        self.assertEqual(state["user_micro_batches_seen"], 4 * 2)
        self.assertEqual(state["phases"][1]["drift_sq"], 0.0)  # both anchors still at init when first stored
        self.assertGreater(state["phases"][2]["drift_sq"], 0.0)

    # ------------------------------------------------------------------ Booster
    def test_booster_lambda_zero_equals_sft(self):
        ours = self.run_defense("booster0", "--method", "booster", "--booster-harmful", self.harmful, "--booster-lambda", "0")
        self.assertLess(max_diff(adapter(self.run_defense("sft", "--method", "sft")), adapter(ours)), 1e-6)
        cost = json.loads((ours / "cost.json").read_text())
        self.assertEqual((cost["forward_passes"], cost["backward_passes"]), (18, 18))

    def test_booster_regulariser_changes_update(self):
        ours = self.run_defense("booster5", "--method", "booster", "--booster-harmful", self.harmful, "--lr", "1e-2")
        ref = self.run_defense("sft_lr", "--method", "sft", "--lr", "1e-2")
        self.assertGreater(max_diff(adapter(ref), adapter(ours)), 1e-5)

    # ------------------------------------------------------------------ constrained SFT
    def test_csft_first_step_equals_sft(self):
        ours = self.run_defense("csft1", "--method", "csft", steps=1)
        self.assertLess(max_diff(adapter(self.run_defense("sft1", "--method", "sft", steps=1)), adapter(ours)), 1e-6)

    def test_csft_weights_and_turn_schedule(self):
        logps = torch.tensor([[-1.0, -2.0, -0.5]])
        betas = torch.full_like(logps, 2.0)
        self.assertTrue(torch.equal(td.csft_weights(logps, logps.clone(), betas), torch.ones_like(logps)))
        # two assistant turns of 3 and 7 tokens separated by masked tokens
        mask = torch.tensor([[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]]).bool()
        turn = dc.turn_positions(mask, "turn")[0].tolist()
        self.assertEqual(turn, [0, 0, 1, 2, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0])
        conv = dc.turn_positions(mask, "conversation")[0].tolist()
        self.assertEqual(conv[6:13], [3, 4, 5, 6, 7, 8, 9])
        b = td.csft_betas(dc.turn_positions(mask, "turn"), 0.1, 20, 5, 5)[0]
        expected = [0.5, 2.0, 2.0, 0.5, 2.0, 2.0, 2.0, 2.0, 0.1, 0.1]  # positions of the 10 trained tokens
        got = [round(float(x), 6) for x, m in zip(b, mask[0]) if m]
        self.assertEqual(got, expected)

    # ------------------------------------------------------------------ DPO
    def test_dpo_loss_is_log2_at_init(self):
        out = self.run_defense("dpo1", "--method", "dpo", "--train", self.pairs, "--val", self.pairs_val, steps=1)
        summary = json.loads((out / "summary.json").read_text())
        self.assertAlmostEqual(summary["train"]["train_loss"], math.log(2), places=5)
        state = json.loads((out / "method_state.json").read_text())
        self.assertTrue(state["ref_precomputed"])
        self.assertTrue((out / "ref_logps.pt").exists())

    def test_dpo_margin_antisymmetry(self):
        c, r, rc, rr = torch.tensor([-3.0]), torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-2.5])
        _, wc, wr = td.dpo_loss(c, r, rc, rr, 0.1)
        _, wc2, wr2 = td.dpo_loss(r, c, rr, rc, 0.1)
        self.assertAlmostEqual(float(wc - wr), -float(wc2 - wr2), places=7)
        loss0, _, _ = td.dpo_loss(c, r, c, r, 0.1)
        self.assertAlmostEqual(float(loss0), math.log(2), places=7)

    def test_dpo_masks_prompt_and_tool_tokens(self):
        tokenizer = tiny_qwen.make_tokenizer()
        rows, _ = td.build_pairs(dc.load_jsonl(self.pairs), tokenizer, 100_000, "t")
        text = lambda ids: "".join(tokenizer.convert_ids_to_tokens([i for i in ids if i != dc.IGNORE_INDEX]))
        rejected, chosen = text(rows[0]["rejected_labels"]), text(rows[0]["chosen_labels"])
        for absent in ("TOOL_OUTPUT_PLACEHOLDER", "HARMFUL_PLACEHOLDER", "SYSTEM_PLACEHOLDER", "# Tools"):
            self.assertNotIn(absent, rejected + chosen)
        self.assertIn("ANSWER_PLACEHOLDER", rejected)
        self.assertTrue(chosen.startswith("REFUSAL_PLACEHOLDER"))
        self.assertEqual(rows[0]["chosen_input_ids"][:20], rows[0]["rejected_input_ids"][:20])  # same prompt prefix

    def test_defence_streams_reject_attack_ids(self):
        (self.trainset / "interim/attack_candidates.jsonl").write_text(json.dumps({"id": "harmful-200"}) + "\n")
        try:
            with self.assertRaises(AssertionError):
                td.assert_disjoint(dc.load_jsonl(self.harmful), td.forbidden_for("harmful"), "booster harmful")
        finally:
            (self.trainset / "interim/attack_candidates.jsonl").unlink()


if __name__ == "__main__":
    unittest.main()
