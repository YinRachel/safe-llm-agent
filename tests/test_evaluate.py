import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).parents[1] / "scripts/eval/evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class BenignEvaluationTests(unittest.TestCase):
    def test_no_harmbench_benign_inference(self):
        with patch.object(evaluate, "client") as client:
            with self.assertRaises(SystemExit) as result:
                evaluate.main(["harmbench"])
            self.assertEqual(result.exception.code, 2)
            client.assert_not_called()

    def test_invalid_data_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(dict(id="1", prompt="hello", source="test", benign=False)))
            with patch.object(evaluate, "client") as client:
                with self.assertRaises(ValueError):
                    evaluate.main(["chat", "--data", str(path)])
                client.assert_not_called()

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            row = json.dumps(dict(id="1", prompt="hello", source="test", benign=True))
            path.write_text(row + "\n" + row)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluate.read_benign(path)

    def test_chat_generation_reports_truncation_without_fake_score(self):
        rows = evaluate.read_benign(Path(__file__).parents[1] / "data/benign_chat.example.jsonl")
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Hello"), finish_reason="length")], usage=None)
        args = SimpleNamespace(model=evaluate.MODEL, limit=1, max_tokens=4, seed=42)
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluate, "client") as factory:
            factory.return_value.chat.completions.create.return_value = response
            evaluate.run_chat(args, Path(directory), rows)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["truncated"], 1)
            self.assertFalse(summary["scored"])
            self.assertEqual(summary["benchmark"], "custom_benign_chat")
            saved = json.loads((Path(directory) / "responses.jsonl").read_text())
            self.assertEqual(saved["id"], rows[0]["id"])

    def test_preflight_rejects_missing_tool_parser(self):
        args = SimpleNamespace(model=evaluate.MODEL, seed=42)
        with patch.object(evaluate, "client") as factory:
            api = factory.return_value
            api.models.list.return_value.data = [SimpleNamespace(id=evaluate.MODEL)]
            api.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))])
            with self.assertRaisesRegex(RuntimeError, "No parsed"):
                evaluate.preflight(args)


if __name__ == "__main__":
    unittest.main()
