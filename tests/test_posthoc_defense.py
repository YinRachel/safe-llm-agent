"""Post-hoc defences on toy matrices and a tiny offline Qwen3: Safe LoRA projection (name matching, K=0 identity,
rank-1 cosine), Antidote pruning (exact fraction, top scores, hooks removed), LoRA scaling (s=0 base, s=1 attacked),
and the selection rule."""
import json
from pathlib import Path
import random
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
except ImportError:
    HAVE_STACK = False

import tiny_qwen  # noqa: E402

if HAVE_STACK:
    import defense_common as dc
    import posthoc_defense as ph


def weights(path):
    from safetensors.torch import load_file
    out = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


def max_diff(a, b):
    keys = set(a) & set(b)
    assert keys, "no common tensors"
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in keys)


@unittest.skipUnless(HAVE_STACK, "needs torch + peft")
class ToyTests(unittest.TestCase):
    def test_rank_one_delta_gives_cosine_one(self):
        torch.manual_seed(0)
        B, A = torch.randn(6, 1), torch.randn(1, 5)
        projected, cos, dist = ph.project_module(A, B, B @ A)
        self.assertEqual(projected.shape, B.shape)
        self.assertAlmostEqual(cos, 1.0, places=5)
        self.assertGreater(dist, 0)

    def test_projection_matches_explicit_projector(self):
        torch.manual_seed(1)
        B, A, V = torch.randn(6, 2), torch.randn(2, 5), torch.randn(6, 5)
        C = V @ V.T / torch.linalg.norm(V)  # upstream get_aligned_matrix
        projected, cos, _ = ph.project_module(A, B, V)
        self.assertTrue(torch.allclose(projected, C @ B, atol=1e-5))
        expected = torch.nn.functional.cosine_similarity((C @ B @ A).reshape(1, -1), (B @ A).reshape(1, -1)).item()
        self.assertAlmostEqual(cos, expected, places=5)

    def test_name_matching_ignores_parameter_order(self):
        torch.manual_seed(2)
        names = [f"base_model.model.model.layers.{i}.self_attn.{p}.{s}.default.weight"
                 for i in range(2) for p in ("q_proj", "v_proj") for s in ("lora_A", "lora_B")]
        params = {n: torch.randn(3, 4) if "lora_A" in n else torch.randn(5, 3) for n in names}
        aligned = {n.split(".lora_")[0][len("base_model.model."):]: torch.randn(5, 4) for n in names}
        unaligned = {m: torch.randn(5, 4) for m in aligned}
        pair = lambda m: (aligned[m], unaligned[m])
        ordered = ph.safelora_scores(ph.lora_modules(params.items()), pair, "cpu")
        items = list(params.items())
        random.Random(0).shuffle(items)
        shuffled = ph.safelora_scores(ph.lora_modules(items), pair, "cpu")
        self.assertEqual(sorted(ordered), sorted(aligned))
        for m in ordered:
            self.assertEqual(ordered[m]["cos"], shuffled[m]["cos"])
            self.assertTrue(torch.equal(ordered[m]["projected_B"], shuffled[m]["projected_B"]))

    def test_shape_mismatch_stops(self):
        modules = {"m": (torch.randn(2, 4), torch.randn(5, 2))}
        with self.assertRaises(SystemExit):
            ph.safelora_scores(modules, lambda m: (torch.randn(5, 4), torch.randn(5, 3)), "cpu")

    def test_choose_modules(self):
        cos = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.1}
        self.assertEqual(ph.choose_modules(cos, "number", k=0), [])
        self.assertEqual(ph.choose_modules(cos, "number", k=2), ["b", "d"])
        self.assertEqual(ph.choose_modules(cos, "threshold", threshold=0.5), ["b", "c", "d"])

    def test_wanda_prunes_exact_fraction_of_top_scores(self):
        torch.manual_seed(3)
        weight = torch.nn.Parameter(torch.randn(8, 10))
        act = torch.rand(10) + 0.1
        score = weight.detach().abs() * act.sqrt()
        top = set(torch.topk(score.flatten(), 8).indices.tolist())
        original = weight.detach().clone()
        self.assertEqual(ph.wanda_prune_(weight, act, 0.1), 8)
        zeroed = set((weight.detach().flatten() == 0).nonzero().flatten().tolist())
        self.assertEqual(zeroed, top)
        kept = [i for i in range(80) if i not in top]
        self.assertTrue(torch.equal(weight.detach().flatten()[kept], original.flatten()[kept]))
        self.assertEqual(ph.wanda_prune_(weight, act, 0.0), 0)

    def test_activation_hooks_accumulate_and_are_removed(self):
        model = torch.nn.Sequential()
        model.add_module("q_proj", torch.nn.Linear(3, 2))
        x = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        with ph.ActivationStats(model, ["q_proj"]) as stats:
            model(x)
        self.assertTrue(torch.equal(stats.stats["q_proj"], torch.tensor([1.0, 5.0, 9.0])))
        self.assertFalse(model.q_proj._forward_pre_hooks)

    def test_selection_rule(self):
        ref = {"val_benign_loss": 1.0, "refusal_nll": 2.0}
        cands = {"a": {"val_benign_loss": 1.2, "refusal_nll": 0.1}, "b": {"val_benign_loss": 1.04, "refusal_nll": 1.0, "harmful_nll": 3},
                 "c": {"val_benign_loss": 1.0, "refusal_nll": 1.0, "harmful_nll": 5}}
        self.assertEqual(dc.pick_setting(ref, cands), ("c", ["c", "b"]))


@unittest.skipUnless(HAVE_STACK, "needs torch + peft")
class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.base = str(tiny_qwen.make_model_dir(root / "base", seed=0))
        cls.unaligned = str(tiny_qwen.make_model_dir(root / "unaligned", seed=1))
        model = AutoModelForCausalLM.from_pretrained(cls.base, torch_dtype=torch.float32)
        args = type("A", (), {"lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "target_modules": dc.TARGET_MODULES})
        peft_model = get_peft_model(model, dc.lora_config(args))
        torch.manual_seed(5)
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.5)
        cls.adapter = root / "adapter"
        peft_model.save_pretrained(cls.adapter)
        tiny_qwen.make_tokenizer().save_pretrained(cls.adapter)
        cls.attacked = root / "attacked"
        peft_model.merge_and_unload().save_pretrained(cls.attacked)
        tiny_qwen.make_tokenizer().save_pretrained(cls.attacked)
        cls.val = tiny_qwen.write_jsonl(root / "val.jsonl", [tiny_qwen.conversation(i) for i in range(3)])
        cls.refusal = tiny_qwen.write_jsonl(root / "refusal.jsonl", [tiny_qwen.refusal(i) for i in range(3)])
        cls.calibration = tiny_qwen.write_jsonl(root / "calib.jsonl", [tiny_qwen.conversation(50 + i, "harmful") for i in range(3)])
        cls.trainset = root / "trainset"
        cls.patch = patch.multiple(dc, TRAINSET=cls.trainset, DERIVED=cls.trainset / "derived")
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.tmp.cleanup()

    def run_posthoc(self, name, *extra):
        out = Path(self.tmp.name) / name
        ph.main(["--output", str(out), "--dtype", "float32", "--device", "cpu", "--max-length", "100000",
                 "--select-val", self.val, "--select-refusal", self.refusal, "--select-harmful", ""] + list(extra))
        return out

    def test_scale_endpoints(self):
        zero = self.run_posthoc("s0", "--method", "scale", "--adapter", str(self.adapter), "--scale", "0")
        one = self.run_posthoc("s1", "--method", "scale", "--adapter", str(self.adapter), "--scale", "1")
        self.assertLess(max_diff(weights(zero / "merged"), weights(self.base)), 1e-6)
        self.assertLess(max_diff(weights(one / "merged"), weights(self.attacked)), 1e-6)
        self.assertGreater(max_diff(weights(self.attacked), weights(self.base)), 1e-3)
        self.assertTrue((zero / "cost.json").exists() and (zero / "manifest.json").exists())

    def test_scale_select_writes_sweep(self):
        out = self.run_posthoc("ssel", "--method", "scale", "--adapter", str(self.adapter), "--select")
        selection = json.loads((out / "selection.json").read_text())
        self.assertEqual(set(selection["results"]), {"s=1.0", "s=0.25", "s=0.5", "s=0.75"})
        self.assertEqual(json.loads((out / "scale.json").read_text())["scale"], float(selection["chosen"].split("=")[1]))

    def test_safelora_k0_is_identity_and_default_pair_is_gated(self):
        out = self.run_posthoc("sl0", "--method", "safelora", "--adapter", str(self.adapter), "--aligned", self.base,
                               "--unaligned", self.unaligned, "--num-proj", "0")
        self.assertLess(max_diff(weights(out / "merged"), weights(self.attacked)), 1e-6)
        report = json.loads((out / "safelora.json").read_text())
        self.assertEqual((report["modules"], report["n_projected"]), (2 * 7, 0))
        all_k = self.run_posthoc("sl14", "--method", "safelora", "--adapter", str(self.adapter), "--aligned", self.base,
                                 "--unaligned", self.unaligned, "--num-proj", "14")
        self.assertGreater(max_diff(weights(all_k / "merged"), weights(self.attacked)), 1e-4)
        with self.assertRaises(SystemExit):
            self.run_posthoc("sl_default", "--method", "safelora", "--adapter", str(self.adapter))

    def test_antidote_ratio(self):
        zero = self.run_posthoc("a0", "--method", "antidote", "--merged", str(self.attacked), "--calibration",
                                self.calibration, "--ratio", "0")
        self.assertLess(max_diff(weights(zero / "merged"), weights(self.attacked)), 1e-7)
        out = self.run_posthoc("a10", "--method", "antidote", "--merged", str(self.attacked), "--calibration",
                               self.calibration, "--ratio", "0.1")
        report = json.loads((out / "antidote.json").read_text())
        pruned = weights(out / "merged")
        for layer, count in report["zeroed_per_layer"].items():
            w = pruned[f"{layer}.weight"]
            self.assertEqual(count, round(0.1 * w.numel()))
            self.assertEqual(int((w == 0).sum()), count)
        self.assertEqual(len(report["zeroed_per_layer"]), 2 * 7)
        self.assertGreater(report["calibration_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
