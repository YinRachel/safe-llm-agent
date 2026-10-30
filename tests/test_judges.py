import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_evaluate import evaluate


class JudgeConfigurationTests(unittest.TestCase):
    def args(self, **kwargs):
        return SimpleNamespace(judge=None, judge_provider=None, judge_base_url=None, **kwargs)

    @patch.dict(os.environ, {"RUNWARE_API_KEY": "test-secret", "RUNWARE_JUDGE_MODEL": "openai:gpt@6-luna"}, clear=True)
    def test_runware_uses_user_model_and_separate_key(self):
        settings = evaluate.judge_settings(self.args())
        self.assertEqual(settings["model"], "openai-api/runware/openai:gpt@6-luna")
        self.assertEqual(settings["base_url"], "https://api.runware.ai/v1")
        self.assertEqual(settings["key_env"], "RUNWARE_API_KEY")
        self.assertNotIn("test-secret", str(settings))

    @patch.dict(os.environ, {"RUNWARE_API_KEY": "test-secret"}, clear=True)
    def test_missing_runware_model_does_not_substitute(self):
        with self.assertRaisesRegex(ValueError, "RUNWARE_JUDGE_MODEL"):
            evaluate.judge_settings(self.args())

    @patch.dict(os.environ, {}, clear=True)
    def test_original_openai_judge_is_available(self):
        settings = evaluate.judge_settings(self.args())
        self.assertEqual(settings["model"], "openai/gpt-4o-2024-08-06")
        self.assertEqual(settings["base_url"], "https://api.openai.com/v1")

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "http://target:8000/v1"}, clear=True)
    def test_solver_environment_does_not_redirect_direct_openai_judge(self):
        self.assertEqual(evaluate.judge_settings(self.args())["base_url"], "https://api.openai.com/v1")

    @patch.dict(os.environ, {}, clear=True)
    def test_invalid_provider_rejected(self):
        with patch.dict(os.environ, {"JUDGE_PROVIDER": "typo"}):
            with self.assertRaises(ValueError):
                evaluate.judge_settings(self.args())

@unittest.skipUnless(importlib.util.find_spec('inspect_evals'), 'Install Inspect dependencies')
class JudgePreflightTests(unittest.TestCase):
    def check(self, answers):
        from unittest.mock import AsyncMock
        judge = SimpleNamespace(generate=AsyncMock(side_effect=[SimpleNamespace(completion=a) for a in answers]))
        with patch.object(evaluate, 'make_judge', return_value=(judge, {'model': 'test', 'base_url': 'test'})):
            evaluate.judge_preflight(SimpleNamespace())
        return judge

    @patch.dict(os.environ, {"RUNWARE_API_KEY": "fake-test-key", "RUNWARE_JUDGE_MODEL": "openai:gpt@6-luna"}, clear=True)
    def test_real_inspect_transport_uses_completion_tokens_for_runware_luna(self):
        import asyncio
        from unittest.mock import AsyncMock
        from inspect_ai.model import GenerateConfig
        from openai.types.chat import ChatCompletion
        judge, settings = evaluate.make_judge(SimpleNamespace(judge=None, judge_provider="runware", judge_base_url=None))
        response = ChatCompletion(id="test", created=0, object="chat.completion", model="openai:gpt@6-luna",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "pass"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        with patch.object(judge.api.client.chat.completions, 'create', new=AsyncMock(return_value=response)) as send:
            output = asyncio.run(judge.generate("Answer pass", config=GenerateConfig(max_tokens=512, temperature=0)))
        request = send.call_args.kwargs
        self.assertNotIn("max_tokens", request)
        self.assertEqual(request["max_completion_tokens"], 512)
        self.assertEqual(request["model"], "openai:gpt@6-luna")
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(output.completion, "pass")
        self.assertEqual(settings["family"], "gpt-6-luna")

    def retried(self, outcomes):
        import asyncio
        from unittest.mock import AsyncMock
        model = SimpleNamespace(generate=AsyncMock(side_effect=outcomes))
        send = model.generate
        evaluate.retry_judge(model)
        return asyncio.run(model.generate("prompt")), send

    def test_retry_judge_recovers_from_errors_and_empty_replies(self):
        output, send = self.retried([IndexError("list index out of range"), SimpleNamespace(completion=""),
                                     SimpleNamespace(completion="Answer: no")])
        self.assertEqual(output.completion, "Answer: no")
        self.assertEqual(send.await_count, 3)

    def test_retry_judge_keeps_last_outcome_when_attempts_run_out(self):
        output, send = self.retried([SimpleNamespace(completion="")] * 3)
        self.assertEqual((output.completion, send.await_count), ("", 3))
        with self.assertRaises(IndexError):
            self.retried([IndexError("list index out of range")] * 3)

    def test_refusal_and_semantic_formats(self):
        judge = self.check(['Explanation. Answer: Yes', 'Explanation. Answer: No', 'pass'])
        self.assertEqual(judge.generate.await_count, 3)

    def test_malformed_judge_output_fails_preflight(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid or incorrect'):
            self.check(['perhaps'])

    def test_judge_api_failure_propagates(self):
        from unittest.mock import AsyncMock
        judge = SimpleNamespace(generate=AsyncMock(side_effect=RuntimeError('judge unavailable')))
        with patch.object(evaluate, 'make_judge', return_value=(judge, {})):
            with self.assertRaisesRegex(RuntimeError, 'judge unavailable'):
                evaluate.judge_preflight(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
