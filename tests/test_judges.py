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


if __name__ == "__main__":
    unittest.main()
