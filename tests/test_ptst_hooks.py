"""Serving-time template hooks used by F4 PTST: serve_qwen.sh CHAT_TEMPLATE, generate_harmbench.py --chat-template,
evaluate.py manifest record. Defaults (no template) must stay unchanged."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from test_harmbench_pipeline import fixture, generator  # noqa: E402

spec = importlib.util.spec_from_file_location("evaluate_ptst", HERE.parent / "scripts/evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def run_generator(repo, output, extra):
    fake_vllm = ModuleType("vllm")
    target = MagicMock()
    target.chat.side_effect = lambda messages, **kwargs: [SimpleNamespace(outputs=[SimpleNamespace(
        text="I cannot help with that.", finish_reason="stop", token_ids=[1])]) for _ in messages]
    fake_vllm.LLM = MagicMock(return_value=target)
    fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    with patch.dict(sys.modules, {"vllm": fake_vllm}), patch.object(generator, "version", return_value="test"):
        generator.main(["--harmbench-repo", str(repo), "--output", str(output)] + extra)
    return target, json.loads((output / "manifest.json").read_text())


class HarmBenchTemplateTests(unittest.TestCase):
    def test_chat_template_passed_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "upstream"
            fixture(repo)
            template = Path(directory) / "t.jinja"
            template.write_text("TEMPLATE_PLACEHOLDER")
            target, manifest = run_generator(repo, Path(directory) / "with", ["--chat-template", str(template)])
            self.assertEqual(target.chat.call_args.kwargs["chat_template"], "TEMPLATE_PLACEHOLDER")
            self.assertEqual(len(manifest["chat_template"]["sha256"]), 64)
            target, manifest = run_generator(repo, Path(directory) / "without", [])
            self.assertNotIn("chat_template", target.chat.call_args.kwargs)
            self.assertIsNone(manifest["chat_template"])


class EvaluateManifestTests(unittest.TestCase):
    def test_chat_template_record(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHAT_TEMPLATE", None)
            self.assertIsNone(evaluate.chat_template_record())
            with tempfile.NamedTemporaryFile("w", suffix=".jinja") as stream:
                stream.write("x")
                stream.flush()
                os.environ["CHAT_TEMPLATE"] = stream.name
                record = evaluate.chat_template_record()
                os.environ.pop("CHAT_TEMPLATE")
        self.assertEqual(record["sha256"], "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881")


class ServeScriptTests(unittest.TestCase):
    def serve_args(self, env):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "vllm"
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            full = {"PATH": f"{directory}:{os.environ['PATH']}", **env}
            out = subprocess.run(["bash", str(HERE.parent / "scripts/serve_qwen.sh")], env=full, check=True,
                                 capture_output=True, text=True).stdout
        return out.splitlines()

    def test_chat_template_flag_only_when_set(self):
        self.assertNotIn("--chat-template", self.serve_args({}))
        args = self.serve_args({"CHAT_TEMPLATE": "templates/ptst.jinja"})
        self.assertEqual(args[args.index("--chat-template") + 1], "templates/ptst.jinja")


if __name__ == "__main__":
    unittest.main()
