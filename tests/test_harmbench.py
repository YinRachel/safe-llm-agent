import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("score_harmbench", Path(__file__).parents[1] / "scripts/eval/score_harmbench.py")
harmbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harmbench)


class HarmBenchClassifierTests(unittest.TestCase):
    def test_templates_are_read_without_executing_upstream_code(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "eval_utils.py").write_text(
                'raise RuntimeError("must not execute")\n'
                'LLAMA2_CLS_PROMPT = {"prompt": "{behavior}: {generation}", '
                '"prompt_contextual": "{context}: {behavior}: {generation}"}\n')
            templates = harmbench.load_templates(repo)
            self.assertEqual(harmbench.format_prompt(templates,
                {"Behavior": "Example", "Tags": "context", "ContextString": "Context"}, "Output"),
                "Context: Example: Output")

    def test_invalid_labels_do_not_become_success_or_refusal_scores(self):
        self.assertEqual(harmbench.parse_label(" Yes "), 1)
        self.assertEqual(harmbench.parse_label("no"), 0)
        self.assertEqual(harmbench.parse_label("possibly"), -1)
        result = harmbench.aggregate({"a": [{"label": -1}]})
        self.assertFalse(result["valid"])
        self.assertIsNone(result["asr"])

    def test_asr_weights_behaviors_equally(self):
        result = harmbench.aggregate({"a": [{"label": 1}] * 3, "b": [{"label": 0}]})
        self.assertEqual(result["asr"], 0.5)

    def test_unknown_behavior_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "behaviors.csv").write_text("BehaviorID,Behavior,Tags\na,Example,standard\n")
            (path / "completions.json").write_text(json.dumps({"missing": [{"generation": "hello"}]}))
            with self.assertRaisesRegex(ValueError, "Unknown BehaviorID"):
                harmbench.load_inputs(path / "behaviors.csv", path / "completions.json")

    def test_copyright_cases_are_not_silently_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "behaviors.csv").write_text("BehaviorID,Behavior,Tags\na,Example,hash_check\n")
            (path / "completions.json").write_text(json.dumps({"a": [{"generation": "hello"}]}))
            with self.assertRaisesRegex(ValueError, "full upstream evaluator"):
                harmbench.load_inputs(path / "behaviors.csv", path / "completions.json")


if __name__ == "__main__":
    unittest.main()
