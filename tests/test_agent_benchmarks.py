"""BFCL v3 multi-turn and tau-bench runners without model calls or dataset downloads."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from test_evaluate import evaluate


def make_args(**overrides):
    values = dict(model=evaluate.MODEL, base_url="http://localhost:8000/v1", limit=2, seed=42,
                  max_tokens=4096, concurrency=1, categories=evaluate.BFCL_MULTI_TURN,
                  domains=["retail", "airline"], message_limit=10,
                  judge="openai-api/judge/test-judge", judge_base_url="http://localhost:8001/v1",
                  judge_provider="compatible")
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_task(name):
    from inspect_ai.dataset import MemoryDataset, Sample
    return SimpleNamespace(name=name, dataset=MemoryDataset([Sample(input="hi", id=f"{name}-1")]))


def fake_log(status="success", samples=()):
    return SimpleNamespace(status=status, results=None, error=None, samples=list(samples),
                           eval=SimpleNamespace(task="task"))


@unittest.skipUnless(importlib.util.find_spec("inspect_evals"), "Install requirements for Inspect integration")
class AgentBenchmarkTests(unittest.TestCase):
    def test_bfcl_multiturn_uses_v3_categories_and_saves_artifacts(self):
        args = make_args()
        with tempfile.TemporaryDirectory() as directory:
            with patch("inspect_evals.bfcl.bfcl", return_value=fake_task("bfcl")) as task, \
                 patch("inspect_ai.eval", return_value=[fake_log()]) as runner:
                evaluate.run_bfcl_multiturn(args, Path(directory))
            task.assert_called_once_with(categories=evaluate.BFCL_MULTI_TURN)
            kwargs = runner.call_args.kwargs
            self.assertEqual(kwargs["model"].api.service_model_name(), evaluate.MODEL)
            self.assertEqual((kwargs["temperature"], kwargs["limit"], kwargs["seed"]), (0, 2, 42))
            self.assertNotIn("model_roles", kwargs)
            saved = json.loads((Path(directory) / "dataset.jsonl").read_text())
            self.assertEqual(saved["task"], "bfcl")
            self.assertTrue((Path(directory) / "dataset.sha256").exists())
            self.assertTrue(json.loads((Path(directory) / "summary.json").read_text())[0]["valid"])

    def test_tau_bench_separates_user_simulator_from_target(self):
        args = make_args()
        with tempfile.TemporaryDirectory() as directory:
            with patch("inspect_evals.tau2.tau2_retail", return_value=fake_task("retail")) as retail, \
                 patch("inspect_evals.tau2.tau2_airline", return_value=fake_task("airline")) as airline, \
                 patch("inspect_ai.eval", return_value=[fake_log(), fake_log()]) as runner:
                evaluate.run_tau_bench(args, Path(directory))
            retail.assert_called_once_with(shuffle=False, message_limit=10)
            airline.assert_called_once_with(shuffle=False, message_limit=10)
            kwargs = runner.call_args.kwargs
            self.assertEqual(len(runner.call_args.args[0]), 2)
            self.assertEqual(kwargs["model"].api.service_model_name(), evaluate.MODEL)
            self.assertEqual(kwargs["model_roles"]["user"].api.service_model_name(), "test-judge")
            tasks = {json.loads(line)["task"] for line in
                     (Path(directory) / "dataset.jsonl").read_text().splitlines()}
            self.assertEqual(tasks, {"retail", "airline"})

    def test_sample_errors_invalidate_results(self):
        log = fake_log(samples=[SimpleNamespace(id="s1", error=object(), output=None)])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "sample errors"):
                evaluate.summarize_logs([log], Path(directory))
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertFalse(summary[0]["valid"])
            self.assertEqual(summary[0]["errored_sample_ids"], ["s1"])

    def test_context_exhaustion_is_reported(self):
        sample = SimpleNamespace(id="s1", error=None, output=SimpleNamespace(stop_reason="model_length"))
        with tempfile.TemporaryDirectory() as directory:
            evaluate.summarize_logs([fake_log(samples=[sample])], Path(directory))
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary[0]["context_exhausted_sample_ids"], ["s1"])

    def test_empty_or_failed_runs_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                evaluate.summarize_logs([], Path(directory))
            with self.assertRaises(RuntimeError):
                evaluate.summarize_logs([fake_log(status="error")], Path(directory))

    def test_tau_bench_requires_user_simulator_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.dict("os.environ", {"RUNWARE_API_KEY": "", "JUDGE_PROVIDER": "runware",
                                           "RUNWARE_JUDGE_MODEL": ""}, clear=False):
                with self.assertRaises(ValueError):
                    evaluate.main(["tau-bench", "--output", str(output)])
            self.assertFalse(output.exists())

    def test_main_dispatch_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(evaluate, "preflight") as preflight, \
                 patch.object(evaluate, "run_bfcl_multiturn") as run:
                evaluate.main(["bfcl-multiturn", "--output", str(output),
                               "--categories", "multi_turn_base"])
            preflight.assert_called_once()
            self.assertEqual(run.call_args.args[0].categories, ["multi_turn_base"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["benchmark"], "bfcl_v3_multi_turn")
            self.assertRegex(manifest["bfcl_data_commit"], "^[0-9a-f]{40}$")

    def test_invalid_category_and_domain_rejected(self):
        for argv in (["bfcl-multiturn", "--categories", "simple_python"],
                     ["tau-bench", "--domains", "nope"]):
            with self.assertRaises(SystemExit):
                evaluate.main(argv)


if __name__ == "__main__":
    unittest.main()
