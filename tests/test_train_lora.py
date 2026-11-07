"""Offline checks for LoRA data preparation: validation and assistant-only loss masking."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("train_lora", Path(__file__).parents[1] / "scripts/post-training/train_lora.py")
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


class TemplateTokenizer(FakeTokenizer):
    """Character-level tokenizer that renders a repo template the way Transformers does (sandboxed Jinja,
    trim_blocks/lstrip_blocks, tojson without ASCII escaping); `marker` stands in for the format's special token."""

    def __init__(self, marker, bos=""):
        self.marker, self.bos, self.chat_template, self.pad_token_id = marker, bos, None, 0

    def get_vocab(self):
        return {self.marker: 1}

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(message):
            raise ValueError(message)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = lambda value, indent=None: json.dumps(value, ensure_ascii=False, indent=indent)
        env.globals["raise_exception"] = raise_exception
        return env.from_string(self.chat_template).render(
            messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, bos_token=self.bos)


PARALLEL = {
    "tools": EXAMPLE["tools"],
    "messages": [
        {"role": "system", "content": "Be useful."},
        {"role": "user", "content": "find X and Y"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "lookup", "arguments": "{\"q\": \"X\"}"}},
            {"type": "function", "function": {"name": "lookup", "arguments": {"q": "Y"}}}]},
        {"role": "tool", "content": "X=1"},
        {"role": "tool", "content": "Y=2"},
        {"role": "assistant", "content": "X is 1, Y is 2."},
    ],
}


class ScratchLinkTest(unittest.TestCase):
    def test_links_large_folders_into_scratch(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                Path("results/m_LoRA/p00-1").mkdir(parents=True)
                links = train_lora.link_to_scratch("results/m_LoRA/p00-1", tmp / "scratch", ["checkpoints", "merged"])
            finally:
                os.chdir(cwd)
            run = tmp / "results/m_LoRA/p00-1"
            for name in ("checkpoints", "merged"):
                self.assertTrue((run / name).is_symlink())
                self.assertEqual((run / name).resolve(), (tmp / "scratch/results/m_LoRA/p00-1" / name).resolve())
                self.assertEqual(links[name], str(tmp / "scratch/results/m_LoRA/p00-1" / name))
            (run / "merged/config.json").write_text("{}")  # writes land in scratch
            self.assertTrue((tmp / "scratch/results/m_LoRA/p00-1/merged/config.json").is_file())

    def test_no_root_writes_in_place_and_full_scratch_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            self.assertEqual(train_lora.link_to_scratch(out, None, ["merged"]), {})
            self.assertFalse((out / "merged").exists())
            scratch = Path(tmp) / "scratch"
            busy = train_lora.scratch_dir(out, scratch) / "merged"  # an absolute OUTPUT keeps its full path
            self.assertEqual(busy, scratch / str(out).lstrip("/") / "merged")
            busy.mkdir(parents=True)
            (busy / "old").write_text("x")
            with self.assertRaises(SystemExit):
                train_lora.link_to_scratch(out, scratch, ["merged"])


class ChatFormatTest(unittest.TestCase):
    def prepared(self, marker, bos=""):
        tokenizer = TemplateTokenizer(marker, bos)
        name, path = train_lora.prepare_tokenizer(tokenizer)
        return tokenizer, name, path

    def test_qwen_keeps_its_own_template_and_targets(self):
        tokenizer, name, path = self.prepared("<|im_start|>")
        self.assertEqual((name, path, tokenizer.chat_template), ("chatml", None, None))
        self.assertEqual(train_lora.resolve_target_modules("auto", name), "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    def test_llama3_parallel_calls_and_masking(self):
        tokenizer, name, path = self.prepared("<|start_header_id|>", "<|begin_of_text|>")
        self.assertEqual(name, "llama3")
        self.assertTrue(path.endswith("templates/llama31_tools.jinja"))
        text = tokenizer.apply_chat_template(train_lora.normalize(PARALLEL["messages"]), tools=PARALLEL["tools"])
        self.assertIn("<|start_header_id|>ipython<|end_header_id|>\n\nX=1<|eot_id|>", text)  # tool output not JSON-quoted
        trained = trained_text(train_lora.tokenize_example(PARALLEL, tokenizer, 10_000))
        self.assertEqual(trained, '{"name": "lookup", "parameters": {"q": "X"}}; {"name": "lookup", "parameters": {"q": "Y"}}'
                                  "<|eot_id|>X is 1, Y is 2.<|eot_id|>")
        prompt = tokenizer.apply_chat_template(train_lora.normalize(PARALLEL["messages"][:-1]), tools=PARALLEL["tools"],
                                               add_generation_prompt=True)
        self.assertTrue(prompt.endswith("Y=2<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"))

    def test_glm4_native_tool_format_and_masking(self):
        tokenizer, name, path = self.prepared("<|observation|>")
        self.assertEqual(name, "glm4")
        self.assertEqual(train_lora.resolve_target_modules("auto", name), "q_proj,k_proj,v_proj,o_proj,gate_up_proj,down_proj")
        text = tokenizer.apply_chat_template(train_lora.normalize(PARALLEL["messages"]), tools=PARALLEL["tools"])
        self.assertTrue(text.startswith("[gMASK]<sop><|system|>\n你是一个名为 ChatGLM"))
        self.assertIn("## lookup\n\n{\n    \"name\": \"lookup\"", text)
        self.assertIn('<|system|>\nBe useful.<|user|>\nfind X and Y<|assistant|>lookup\n{"q": "X"}<|assistant|>lookup\n{"q": "Y"}'
                      "<|observation|>\nX=1<|observation|>\nY=2<|assistant|>\nX is 1, Y is 2.<|endoftext|>", text)
        trained = trained_text(train_lora.tokenize_example(PARALLEL, tokenizer, 10_000))
        self.assertEqual(trained, 'lookup\n{"q": "X"}<|assistant|>lookup\n{"q": "Y"}<|observation|>\nX is 1, Y is 2.<|endoftext|>')
        prompt = tokenizer.apply_chat_template(train_lora.normalize(PARALLEL["messages"][:-1]), tools=PARALLEL["tools"],
                                               add_generation_prompt=True)
        self.assertTrue(prompt.endswith("<|observation|>\nY=2<|assistant|>"))

    def test_explicit_template_choices(self):
        tokenizer = TemplateTokenizer("<|observation|>")
        self.assertEqual(train_lora.prepare_tokenizer(tokenizer, "none"), ("glm4", None))
        self.assertIsNone(tokenizer.chat_template)
        self.assertEqual(train_lora.resolve_target_modules("q_proj,v_proj", "glm4"), "q_proj,v_proj")


if __name__ == "__main__":
    unittest.main()
