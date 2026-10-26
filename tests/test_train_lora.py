"""Offline checks for LoRA data preparation: validation and assistant-only loss masking."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("train_lora", Path(__file__).parents[1] / "scripts/train_lora.py")
train_lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_lora)


class FakeTokenizer:
    """Character-level tokenizer with a Qwen-shaped chat template."""

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        text = ""
        if tools:
            text += "<|im_start|>system\n# Tools\n" + json.dumps(tools) + "<|im_end|>\n"
        for message in messages:
            role = "user" if message["role"] == "tool" else message["role"]
            body = message["content"]
            for call in message.get("tool_calls", []):
                body += "<tool_call>\n" + json.dumps(call["function"]) + "\n</tool_call>"
            text += f"<|im_start|>{role}\n{body}<|im_end|>\n"
        return text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def trained_text(row):
    return "".join(chr(label) for label in row["labels"] if label != train_lora.IGNORE_INDEX)


EXAMPLE = {
    "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
    "messages": [
        {"role": "user", "content": "find X"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": "{\"q\": \"X\"}"}}]},
        {"role": "tool", "content": "X=1"},
        {"role": "assistant", "content": "X is 1."},
    ],
}


class TrainLoraTest(unittest.TestCase):
    def test_only_assistant_turns_are_trained(self):
        row = train_lora.tokenize_example(EXAMPLE, FakeTokenizer(), 10_000)
        text = trained_text(row)
        self.assertIn('{"name": "lookup", "arguments": {"q": "X"}}', text)
        self.assertTrue(text.endswith("X is 1.<|im_end|>"))
        for untrained in ("find X", "X=1", "# Tools", "<|im_start|>"):
            self.assertNotIn(untrained, text)

    def test_overlong_examples_are_dropped(self):
        self.assertIsNone(train_lora.tokenize_example(EXAMPLE, FakeTokenizer(), 10))

    def test_validate(self):
        self.assertIsNone(train_lora.validate(EXAMPLE))
        self.assertEqual(train_lora.validate({"messages": [{"role": "user", "content": "hi"}]}),
                         "does not end with an assistant turn")
        self.assertIn("unknown role", train_lora.validate({"messages": [{"role": "gpt", "content": "x"}]}))


if __name__ == "__main__":
    unittest.main()
