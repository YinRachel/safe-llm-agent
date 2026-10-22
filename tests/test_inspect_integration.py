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
