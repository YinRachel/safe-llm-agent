"""Installed-package integration without model calls or dataset downloads."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from test_evaluate import evaluate


@unittest.skipUnless(importlib.util.find_spec("inspect_evals"), "Install requirements for Inspect integration")
class InspectIntegrationTests(unittest.TestCase):
    def test_upstream_swallowed_refusal_error_invalidates_results(self):
        from inspect_ai.dataset import MemoryDataset, Sample
        task = SimpleNamespace(dataset=MemoryDataset([Sample(input="Hello", id="sample-1")]))
        log = SimpleNamespace(status="success", results=None, error=None, samples=[
            SimpleNamespace(id="sample-1", metadata={"refusal_explanations": [
                {"explanation": "Error in RefusalJudgeLLM: provider unavailable", "refusal": False}
            ]})])
        args = SimpleNamespace(model=evaluate.MODEL, base_url="http://localhost:8000/v1",
            split="val", limit=1, seed=42, max_tokens=4096, concurrency=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evaluate, "make_judge", return_value=(object(), {})), \
                 patch("inspect_evals.agentharm.agentharm_benign", return_value=task), \
                 patch("inspect_ai.eval", return_value=[log]):
                with self.assertRaisesRegex(RuntimeError, "Refusal judge errors"):
                    evaluate.run_agentharm(args, Path(directory))
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertFalse(summary[0]["valid"])
            self.assertEqual(summary[0]["refusal_judge_error_sample_ids"], ["sample-1"])

    def test_harmful_mode_uses_official_harmful_task(self):
        from inspect_ai.dataset import MemoryDataset, Sample
        task = SimpleNamespace(dataset=MemoryDataset([Sample(input="Hello", id="harmful-test")]))
        args = SimpleNamespace(model=evaluate.MODEL, base_url="http://localhost:8000/v1",
            split="val", limit=1, seed=42, max_tokens=4096, concurrency=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(evaluate, "make_judge", return_value=(object(), {})), \
                 patch("inspect_evals.agentharm.agentharm", return_value=task) as harmful, \
                 patch("inspect_evals.agentharm.agentharm_benign") as benign, \
                 patch("inspect_ai.eval", return_value=[SimpleNamespace(status="success", results=None, error=None)]):
                evaluate.run_agentharm(args, Path(directory), harmful=True)
            harmful.assert_called_once()
            self.assertEqual(harmful.call_args.kwargs["split"], "val")
            benign.assert_not_called()

    def test_harmful_split_cannot_be_test_private(self):
        with self.assertRaises(SystemExit):
            evaluate.main(["agentharm-harmful", "--split", "test_private"])

    def test_official_benign_task_and_runner(self):
        from inspect_ai.dataset import MemoryDataset, Sample
        module = importlib.import_module("inspect_evals.agentharm.agentharm")
        sample = Sample(input="Hello", id="benign-test", metadata={})
        args = SimpleNamespace(model=evaluate.MODEL, base_url="http://localhost:8000/v1",
            judge="openai-api/judge/test-judge", judge_base_url="http://localhost:8001/v1",
            judge_provider="compatible",
            split="val", limit=1, seed=42, max_tokens=4096, concurrency=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, "load_dataset", return_value=MemoryDataset([sample])) as loader:
                with patch("inspect_ai.eval", return_value=[SimpleNamespace(status="success", results=None, error=None)]) as runner:
                    evaluate.run_agentharm(args, Path(directory))
                    loader.assert_called_once_with("benign", "val", False)
                    task = runner.call_args.args[0]
                    self.assertEqual(len(task.dataset), 1)
                    self.assertEqual(runner.call_args.kwargs["model"].api.service_model_name(), evaluate.MODEL)
                    self.assertEqual(runner.call_args.kwargs["max_tokens"], 4096)
            saved = json.loads((Path(directory) / "dataset.jsonl").read_text())
            self.assertEqual(saved["id"], "benign-test")
            self.assertTrue((Path(directory) / "dataset.sha256").exists())


if __name__ == "__main__":
    unittest.main()
