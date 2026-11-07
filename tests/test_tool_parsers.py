"""scripts/eval/tool_parsers.py: GLM-4 / Llama-3.1 tool-call parsing, offline (vLLM classes are stubbed)."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

from pydantic import BaseModel

ROOT = Path(__file__).parents[1]


def load(with_vllm):
    """Import tool_parsers.py, optionally against minimal stand-ins for the vLLM 0.10.2 classes it uses."""
    names = ["vllm", "vllm.entrypoints", "vllm.entrypoints.openai", "vllm.entrypoints.openai.protocol",
             "vllm.entrypoints.openai.tool_parsers", "vllm.entrypoints.openai.tool_parsers.abstract_tool_parser",
             "vllm.entrypoints.openai.tool_parsers.llama_tool_parser", "vllm.entrypoints.chat_utils"]
    saved = {name: sys.modules.pop(name, None) for name in names}
    if with_vllm:
        modules = {name: types.ModuleType(name) for name in names}

        class FunctionCall(BaseModel):
            name: str
            arguments: str

        class ToolCall(BaseModel):
            type: str = "function"
            function: FunctionCall

        class DeltaFunctionCall(BaseModel):
            name: str | None = None
            arguments: str | None = None

        class DeltaToolCall(BaseModel):
            index: int
            id: str | None = None
            type: str | None = None
            function: dict | None = None

        class DeltaMessage(BaseModel):
            content: str | None = None
            tool_calls: list[DeltaToolCall] = []

        class ExtractedToolCallInformation(BaseModel):
            tools_called: bool
            tool_calls: list[ToolCall]
            content: str | None = None

        class ToolParser:
            def __init__(self, tokenizer):
                self.prev_tool_call_arr, self.current_tool_id, self.streamed_args_for_tool = [], -1, []

        class ToolParserManager:
            tool_parsers = {}

            @classmethod
            def register_module(cls, name):
                def register(module):
                    cls.tool_parsers[name] = module
                    return module
                return register

        class Llama3JsonToolParser(ToolParser):
            pass

        protocol = modules["vllm.entrypoints.openai.protocol"]
        for cls in (FunctionCall, ToolCall, DeltaFunctionCall, DeltaToolCall, DeltaMessage, ExtractedToolCallInformation):
            setattr(protocol, cls.__name__, cls)
        modules["vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"].ToolParser = ToolParser
        modules["vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"].ToolParserManager = ToolParserManager
        modules["vllm.entrypoints.openai.tool_parsers.llama_tool_parser"].Llama3JsonToolParser = Llama3JsonToolParser
        modules["vllm.entrypoints.chat_utils"].make_tool_call_id = lambda: "call_x"
        sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("tool_parsers_under_test", ROOT / "scripts/eval/tool_parsers.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value


tp = load(with_vllm=False)
NESTED = {"user_id": "u1", "flights": [{"flight_number": "HAT001", "date": "2024-05-01"}], "meta": {"a": {"b": [1]}}}


class Glm4ParseTests(unittest.TestCase):
    def test_plain_text(self):
        self.assertEqual(tp.parse_glm4("\nHello there.\nSecond line."), ("Hello there.\nSecond line.", []))

    def test_single_call_with_nested_arguments(self):
        content, calls = tp.parse_glm4("book_reservation\n" + json.dumps(NESTED) + "<|observation|>", {"book_reservation"})
        self.assertIsNone(content)
        self.assertEqual(calls, [("book_reservation", json.dumps(NESTED))])

    def test_parallel_calls_and_text(self):
        text = '\nLooking both up.<|assistant|>lookup\n{"q": "X"}<|assistant|> lookup \n{"q": "Y"}'
        self.assertEqual(tp.parse_glm4(text, {"lookup"}), ("Looking both up.", [("lookup", '{"q": "X"}'), ("lookup", '{"q": "Y"}')]))

    def test_unknown_name_or_bad_json_stays_text(self):
        self.assertEqual(tp.parse_glm4("Sure\nhere it is", {"lookup"}), ("Sure\nhere it is", []))
        self.assertEqual(tp.parse_glm4("lookup\n{not json", {"lookup"}), ("lookup\n{not json", []))

    def test_round_trip_with_training_template_output(self):
        """What templates/glm4_tools.jinja renders after <|assistant|> for a call is what the parser reads."""
        rendered = "lookup\n" + json.dumps({"q": "X", "n": [1, {"k": None}]}, ensure_ascii=False)
        self.assertEqual(tp.parse_glm4(rendered, {"lookup"})[1], [("lookup", '{"q": "X", "n": [1, {"k": null}]}')])


class Llama3ParseTests(unittest.TestCase):
    def test_nested_arguments_that_llama3_json_drops(self):
        text = json.dumps({"name": "book_reservation", "parameters": NESTED})
        self.assertEqual(tp.parse_llama3(text), [("book_reservation", json.dumps(NESTED))])

    def test_parallel_calls_joined_by_semicolon(self):
        text = '{"name": "lookup", "parameters": {"q": "X"}}; {"name": "lookup", "parameters": {"q": "Y"}}<|eot_id|>'
        self.assertEqual(tp.parse_llama3(text), [("lookup", '{"q": "X"}'), ("lookup", '{"q": "Y"}')])

    def test_python_tag_and_arguments_key(self):
        self.assertEqual(tp.parse_llama3('<|python_tag|>{"name": "f", "arguments": "{\\"a\\": 1}"}'), [("f", '{"a": 1}')])

    def test_text_is_not_a_call(self):
        for text in ("The answer is 4.", '{"answer": 4}', '{"name": "f", "parameters": {}} and more text', "{broken"):
            self.assertIsNone(tp.parse_llama3(text), text)


class VllmClassTests(unittest.TestCase):
    """The registered parser classes against stubbed vLLM 0.10.2 types."""

    @classmethod
    def setUpClass(cls):
        cls.tp = load(with_vllm=True)
        tool = types.SimpleNamespace(function=types.SimpleNamespace(name="lookup"))
        cls.request = types.SimpleNamespace(tools=[tool], tool_choice="auto", skip_special_tokens=True)

    def test_registered(self):
        manager = self.tp.ToolParserManager.tool_parsers
        self.assertIs(manager["glm4_native"], self.tp.Glm4NativeToolParser)
        self.assertIs(manager["llama3_json_full"], self.tp.Llama3JsonFullToolParser)

    def test_glm_keeps_special_tokens_and_extracts(self):
        parser = self.tp.Glm4NativeToolParser(None)
        self.assertFalse(parser.adjust_request(self.request).skip_special_tokens)
        info = parser.extract_tool_calls('lookup\n{"q": "X"}<|assistant|>lookup\n{"q": "Y"}', self.request)
        self.assertTrue(info.tools_called)
        self.assertEqual([(c.function.name, c.function.arguments) for c in info.tool_calls],
                         [("lookup", '{"q": "X"}'), ("lookup", '{"q": "Y"}')])
        plain = parser.extract_tool_calls("\nNo tools needed.", self.request)
        self.assertEqual((plain.tools_called, plain.content), (False, "No tools needed."))

    def stream(self, parser, pieces):
        text, out = "", []
        for piece in pieces:
            previous, text = text, text + piece
            delta = parser.extract_tool_calls_streaming(previous, text, piece, [], [], [], self.request)
            if delta is not None:
                out.append(delta)
        return out

    def test_glm_streaming_calls(self):
        deltas = self.stream(self.tp.Glm4NativeToolParser(None),
                             ["look", "up", "\n", '{"q": ', '"X"}', "<|assistant|>", "lookup\n", '{"q": "Y"}'])
        names, args = {}, {}
        for delta in deltas:
            self.assertIsNone(delta.content)
            for call in delta.tool_calls:
                if call.function.get("name"):
                    names[call.index] = call.function["name"]
                args[call.index] = args.get(call.index, "") + call.function.get("arguments", "")
        self.assertEqual(names, {0: "lookup", 1: "lookup"})
        self.assertEqual({i: json.loads(a) for i, a in args.items()}, {0: {"q": "X"}, 1: {"q": "Y"}})

    def test_glm_streaming_text(self):
        deltas = self.stream(self.tp.Glm4NativeToolParser(None), ["\n", "Hello", " world", "<|user|>"])
        self.assertEqual("".join(d.content or "" for d in deltas), "Hello world")
        self.assertFalse(any(d.tool_calls for d in deltas))

    def test_llama_full_extracts_nested(self):
        parser = self.tp.Llama3JsonFullToolParser(None)
        info = parser.extract_tool_calls(json.dumps({"name": "lookup", "parameters": NESTED}), self.request)
        self.assertEqual(json.loads(info.tool_calls[0].function.arguments), NESTED)
        text = parser.extract_tool_calls("Plain answer.", self.request)
        self.assertEqual((text.tools_called, text.content), (False, "Plain answer."))


if __name__ == "__main__":
    unittest.main()
