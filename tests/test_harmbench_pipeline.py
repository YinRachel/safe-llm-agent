"""Exercise generation -> scoring artifacts with fake GPU backends; no downloads."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import MagicMock, patch

from test_harmbench import harmbench

spec = importlib.util.spec_from_file_location("generate_harmbench", Path(__file__).parents[1] / "scripts/eval/generate_harmbench.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def fixture(repo):
    rows = [
        dict(BehaviorID="standard", Behavior="Example request", FunctionalCategory="standard", Tags="", ContextString=""),
        dict(BehaviorID="context", Behavior="Example context request", FunctionalCategory="contextual", Tags="context", ContextString="Some context"),
        dict(BehaviorID="copyright", Behavior="Example text request", FunctionalCategory="copyright", Tags="hash_check, book", ContextString=""),
    ]
    path = repo / "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (repo / "data/copyright_classifier_hashes").mkdir()
    (repo / "data/copyright_classifier_hashes/copyright.pkl").write_bytes(b"fixture; not unpickled")
    (repo / "eval_utils.py").write_text(
        'LLAMA2_CLS_PROMPT={"prompt": "{behavior} {generation}", '
        '"prompt_contextual": "{context} {behavior} {generation}"}\n'
        'def compute_results_hashing(behavior, samples):\n'
        '    from pathlib import Path\n'
        '    assert Path("data/copyright_classifier_hashes/copyright.pkl").is_file()\n'
        '    return [dict(sample, label=0) for sample in samples]\n')
    return rows, path


class HarmBenchPipelineTests(unittest.TestCase):
    def test_direct_request_includes_context_exactly(self):
        self.assertEqual(generator.direct_prompt({"Behavior": "Request", "ContextString": "Context"}),
                         "Context\n\n---\n\nRequest")

    def test_smoke_covers_each_functional_category(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = fixture(Path(directory))
            selected = generator.select_behaviors(rows + rows, smoke=True)
            self.assertEqual([row["BehaviorID"] for row in selected], ["standard", "context", "copyright"])
            self.assertEqual(len(generator.select_behaviors(rows)), 3)

    def test_missing_completion_prevents_full_coverage_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _, path = fixture(repo)
            completions = repo / "partial.json"
            completions.write_text(json.dumps({"standard": [{"generation": "Response"}]}))
            with self.assertRaisesRegex(ValueError, "complete supplied"):
                harmbench.load_inputs(path, completions, require_full_coverage=True)

    def test_gpu_pipeline_preserves_all_cases_and_routes_hashes(self):
        fake_vllm = ModuleType("vllm")
        target = MagicMock()
        target.chat.side_effect = lambda messages, **kwargs: [SimpleNamespace(outputs=[SimpleNamespace(
            text="I cannot help with that.", finish_reason="stop", token_ids=[1, 2])]) for _ in messages]
        classifier = MagicMock()
        classifier.generate.side_effect = lambda prompts, *a, **k: [SimpleNamespace(outputs=[SimpleNamespace(text="no")]) for _ in prompts]
        instances = iter([target, classifier])
        def make_llm(**kwargs):
            self.assertEqual(os.environ["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
            return next(instances)
        fake_vllm.LLM = MagicMock(side_effect=make_llm)
        fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        tokenizer = SimpleNamespace(encode=lambda text, max_length, truncation: text.split()[:max_length],
                                    decode=lambda tokens, skip_special_tokens: " ".join(tokens))
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "upstream"
            fixture(repo)
            output = Path(directory) / "generation"
            scoring = Path(directory) / "scoring"
            previous_cwd = Path.cwd()
            with patch.dict(sys.modules, {"vllm": fake_vllm, "transformers": fake_transformers}), \
                 patch.object(generator, "version", return_value="test"), \
                 patch.object(harmbench, "version", return_value="test"):
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
                generator.main(["--harmbench-repo", str(repo), "--output", str(output)])
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
                harmbench.main(["--harmbench-repo", str(repo), "--behaviors", str(output / "behaviors.csv"),
                    "--completions", str(output / "completions.json"), "--output", str(scoring),
                    "--include-hash-checks", "--require-full-coverage"])
            self.assertEqual(Path.cwd(), previous_cwd)
            completed = json.loads((output / "completions.json").read_text())
            self.assertEqual(len(completed), 3)
            self.assertEqual(completed["context"][0]["test_case"], "Some context\n\n---\n\nExample context request")
            summary = json.loads((scoring / "summary.json").read_text())
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["asr"], 0)
            self.assertTrue(summary["full_coverage_of_input_csv"])
            self.assertEqual(summary["scoring_methods"], {"local_classifier": 2, "copyright_hash": 1})
            self.assertEqual(len(classifier.generate.call_args.args[0]), 2)
            self.assertEqual(target.chat.call_args.args[0][0][0]["role"], "user")

    def test_missing_hash_reference_fails_before_gpu_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            fixture(repo)
            (repo / "data/copyright_classifier_hashes/copyright.pkl").unlink()
            with self.assertRaisesRegex(ValueError, "Missing upstream copyright reference"):
                generator.main(["--harmbench-repo", str(repo), "--output", str(repo / "generation")])
            self.assertFalse((repo / "generation").exists())


if __name__ == "__main__":
    unittest.main()
