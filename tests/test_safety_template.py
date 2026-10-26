"""PTST template: the safety text lands in the system block (with and without tools, with and without a system
message) and the rendering is otherwise byte-identical to the checkpoint's own template."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "scripts"), str(HERE)]
import safety_template as st  # noqa: E402

try:
    import transformers  # noqa: F401
    import tokenizers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

SAFETY = "SAFETY_PLACEHOLDER refuse harmful requests."
TOOLS = [{"type": "function", "function": {"name": "tool_a", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
SYSTEM = {"role": "system", "content": "SYSTEM_PLACEHOLDER"}
CHAT = [{"role": "user", "content": "USER_PLACEHOLDER"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "tool_a", "arguments": {}}}]},
        {"role": "tool", "content": "TOOL_PLACEHOLDER"}, {"role": "assistant", "content": "ANSWER_PLACEHOLDER"}]


def qwen_template():
    try:
        return st.load_chat_template()
    except SystemExit:
        return None


@unittest.skipUnless(HAVE_TRANSFORMERS, "needs transformers (.venv-trainset)")
class PtstTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tiny_qwen
        cls.original = qwen_template()
        if cls.original is None:
            raise unittest.SkipTest("Qwen3-4B-Instruct-2507 tokenizer_config.json not in the local HF cache")
        cls.tokenizer = tiny_qwen.make_tokenizer()
        cls.ptst = st.make_template(cls.original, SAFETY)

    def render(self, template, messages, tools=None, generation=False):
        return self.tokenizer.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False,
                                                  add_generation_prompt=generation)

    def check(self, messages, tools, expected_insert, after):
        for generation in (False, True):
            original = self.render(self.original, messages, tools, generation)
            ours = self.render(self.ptst, messages, tools, generation)
            index = original.index(after) + len(after)
            self.assertEqual(ours, original[:index] + expected_insert + original[index:])

    def test_no_system_with_tools(self):
        self.check(CHAT, TOOLS, SAFETY + "\n\n", "<|im_start|>system\n")

    def test_system_with_tools(self):
        self.check([SYSTEM] + CHAT, TOOLS, "\n\n" + SAFETY, "SYSTEM_PLACEHOLDER")

    def test_system_without_tools(self):
        self.check([SYSTEM] + CHAT, None, "\n\n" + SAFETY, "SYSTEM_PLACEHOLDER")

    def test_no_system_no_tools(self):
        original = self.render(self.original, CHAT)
        self.assertEqual(self.render(self.ptst, CHAT), f"<|im_start|>system\n{SAFETY}<|im_end|>\n" + original)

    def test_safety_text_inside_system_block(self):
        text = self.render(self.ptst, CHAT, TOOLS)
        system_block = text[:text.index("<|im_end|>")]
        self.assertTrue(system_block.startswith("<|im_start|>system\n"))
        self.assertIn(SAFETY, system_block)
        self.assertIn("# Tools", system_block)

    def test_committed_template_matches_generator(self):
        committed = HERE.parent / "templates/ptst.jinja"
        if not committed.exists():
            self.skipTest("templates/ptst.jinja not generated yet")
        meta = json.loads(committed.with_suffix(".json").read_text())
        self.assertEqual(committed.read_text(), st.make_template(self.original, meta["safety_text"]))
        self.assertEqual(st.sha256_text(self.original), meta["original_template_sha256"])


class SafetyTextTests(unittest.TestCase):
    def test_rejects_unsafe_literal(self):
        with self.assertRaises(ValueError):
            st.make_template("x", 'has "quote"')

    @unittest.skipUnless(importlib.util.find_spec("inspect_evals"), "needs inspect_evals (.venv)")
    def test_safety_text_is_agentharm_refusal_prompt(self):
        from inspect_evals.agentharm.prompts import prompt_technique_dict
        self.assertEqual(st.load_safety_text(), prompt_technique_dict["refusal"].strip())
        committed = HERE.parent / "templates/ptst.json"
        if committed.exists():
            self.assertEqual(json.loads(committed.read_text())["safety_text"], st.load_safety_text())


if __name__ == "__main__":
    unittest.main()
