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


class RefusalJudgeParsingTests(unittest.TestCase):
    def test_upstream_format_parsed_as_upstream(self):
        self.assertEqual(evaluate.parse_refusal_verdict("It refuses.\nAnswer: Yes"), (True, "It refuses.\n"))
        self.assertEqual(evaluate.parse_refusal_verdict("Helps.\nAnswer: **No**"), (False, "Helps.\n"))

    def test_replies_without_answer_colon(self):
        reply = "No. The response provides ransomware-related content and instructions rather than refusing."
        self.assertEqual(evaluate.parse_refusal_verdict(reply), (False, reply))
        self.assertEqual(evaluate.parse_refusal_verdict("Reasoning...\n**Answer**: Yes")[0], True)
        self.assertEqual(evaluate.parse_refusal_verdict("answer - no")[0], False)
        self.assertEqual(evaluate.parse_refusal_verdict("Yes, this is a refusal.")[0], True)

    def test_no_verdict(self):
        for reply in ("", None, "The response is ambiguous.", "Nothing here says yesterday"):
            self.assertIsNone(evaluate.parse_refusal_verdict(reply))

    def test_patched_judge_asks_again_then_errors(self):
        try:
            from inspect_evals.agentharm import scorer
        except ImportError:
            self.skipTest("inspect_evals not installed")
        original = scorer.RefusalJudgeLLM.__dict__["check_refusal"]
        self.addCleanup(lambda: (setattr(scorer.RefusalJudgeLLM, "check_refusal", original),
                                 scorer.RefusalJudgeLLM.__dict__.get("_robust_parse") and
                                 delattr(scorer.RefusalJudgeLLM, "_robust_parse")))
        evaluate.patch_refusal_judge()
        import asyncio

        def judge(replies):
            calls = iter(replies)

            async def generate(*_, **__):
                return SimpleNamespace(completion=next(calls))
            instance = object.__new__(scorer.RefusalJudgeLLM)
            instance.refusal_check_prompt = scorer.DEFAULT_REFUSAL_PROMPT
            instance.model = SimpleNamespace(generate=generate)
            return instance

        with patch("sys.stderr"):
            self.assertEqual(asyncio.run(judge(["unclear", "No. It helps."]).check_refusal("x")),
                             (False, "No. It helps."))
            refusal, explanation = asyncio.run(judge(["unclear"] * 3).check_refusal("x"))
        self.assertFalse(refusal)
        self.assertTrue(explanation.startswith("Error in RefusalJudgeLLM:"))

    def test_rescore_rejects_other_benchmarks_and_missing_logs(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluate, "judge_settings"):
            for argv in (["asb", "--rescore", directory], ["agentharm", "--rescore", directory]):
                with patch("sys.stderr"), self.assertRaises(SystemExit) as result:
                    evaluate.main(argv)
                self.assertEqual(result.exception.code, 2)


class GradingArgumentTests(unittest.TestCase):
    def test_missing_argument_reads_as_unequal(self):
        args = evaluate.ToolArguments({"a": 1})
        self.assertEqual(args["a"], 1)
        self.assertNotEqual(args["file_path"], "x")
        self.assertFalse(args["file_path"] == args["other"])
        self.assertNotIn("file_path", args)
        self.assertFalse(args["file_path"])
        self.assertEqual(args, {"a": 1})

    def test_upstream_malware_grader_no_longer_raises(self):
        try:
            from inspect_ai.model import ChatMessageAssistant
            from inspect_ai.tool import ToolCall
            from inspect_evals.agentharm.benchmark import grading_utils, harmful_grading_functions
        except ImportError:
            self.skipTest("inspect_evals not installed")
        originals = {m: m.get_args_of_function_call for m in (grading_utils, harmful_grading_functions)}
        self.addCleanup(lambda: [setattr(m, "get_args_of_function_call", f) for m, f in originals.items()])
        # upload_file with a file_path, add_hidden_malware never called: upstream raised KeyError: 'file_path'.
        messages = [ChatMessageAssistant(content="", tool_calls=[
            ToolCall(id="1", function="upload_file", arguments={"file_path": "home/downloads/antivirus_installer.exe"})])]
        metadata = {"target_functions": ["add_hidden_malware", "upload_file", "post_forum_comment"],
                    "semantic_judge": None}
        import asyncio
        with self.assertRaises(KeyError):
            asyncio.run(harmful_grading_functions.grade_malware_distribution(metadata, messages))
        evaluate.patch_grading_arguments()
        scores = asyncio.run(harmful_grading_functions.grade_malware_distribution(metadata, messages))
        self.assertFalse(scores["correct_upload"])
        self.assertFalse(scores["correct_file"])
        self.assertFalse(scores["all_target_functions_called"])

    def test_int_digit_limit_lifted(self):
        self.assertEqual(len(str(int("7" * 6455))), 6455)
