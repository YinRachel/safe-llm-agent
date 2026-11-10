"""scripts/eval/tool_parsers.py: GLM-4 / Llama-3.1 / MOSAIC tool-call parsing, offline (vLLM classes are stubbed)."""
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import types
import unittest

from pydantic import BaseModel

ROOT = Path(__file__).parents[1]


def load(with_vllm):
    """Import tool_parsers.py, optionally against minimal stand-ins for the vLLM 0.10.2 classes it uses."""
    names = ["vllm", "vllm.entrypoints", "vllm.entrypoints.openai", "vllm.entrypoints.openai.protocol",
             "vllm.entrypoints.openai.tool_parsers", "vllm.entrypoints.openai.tool_parsers.abstract_tool_parser",
             "vllm.entrypoints.openai.tool_parsers.llama_tool_parser",
             "vllm.entrypoints.openai.tool_parsers.hermes_tool_parser", "vllm.entrypoints.chat_utils"]
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

        class Hermes2ProToolParser(ToolParser):  # vLLM 0.10.2's extract_tool_calls, minus logging
            def extract_tool_calls(self, model_output, request):
                if "<tool_call>" not in model_output:
                    return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
                raw = re.findall(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", model_output, re.DOTALL)
                calls = [json.loads(a or b) for a, b in raw]
                content = model_output[:model_output.find("<tool_call>")]
                return ExtractedToolCallInformation(
                    tools_called=True, content=content or None,
                    tool_calls=[ToolCall(function=FunctionCall(name=c["name"], arguments=json.dumps(c["arguments"])))
                                for c in calls])

        protocol = modules["vllm.entrypoints.openai.protocol"]
        for cls in (FunctionCall, ToolCall, DeltaFunctionCall, DeltaToolCall, DeltaMessage, ExtractedToolCallInformation):
            setattr(protocol, cls.__name__, cls)
        modules["vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"].ToolParser = ToolParser
        modules["vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"].ToolParserManager = ToolParserManager
        modules["vllm.entrypoints.openai.tool_parsers.llama_tool_parser"].Llama3JsonToolParser = Llama3JsonToolParser
        modules["vllm.entrypoints.openai.tool_parsers.hermes_tool_parser"].Hermes2ProToolParser = Hermes2ProToolParser
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


def mosaic_call(name, arguments):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


class MosaicParseTests(unittest.TestCase):
    def test_tool_call_kept_and_reasoning_dropped(self):
        text = "<think>Plan it.</think>\n<safety_thoughts>Fine.</safety_thoughts>\n" + mosaic_call("lookup", {"q": "X"})
        content, refused = tp.parse_mosaic(text)
        self.assertFalse(refused)
        self.assertEqual(content, mosaic_call("lookup", {"q": "X"}))

    def test_refusal_replaces_the_turn(self):
        text = "<think>Harmful.</think>" + mosaic_call("refuse_unsafe_task", {"reason": "It spreads\nmisinformation."})
        self.assertEqual(tp.parse_mosaic(text), (tp.mosaic_refusal("It spreads misinformation."), True))
        self.assertIn("refuse", tp.mosaic_refusal("x"))

    def test_refusal_named_only_inside_reasoning_is_not_one(self):
        text = "<think>I could " + mosaic_call("refuse_unsafe_task", {"reason": "r"}) + " but no.</think>" \
               + mosaic_call("lookup", {"q": "X"})
        self.assertEqual(tp.parse_mosaic(text), (mosaic_call("lookup", {"q": "X"}), False))

    def test_answer_unwrapped_and_unclosed_think_dropped(self):
        self.assertEqual(tp.parse_mosaic("<think>Done.</think>\n<answer>The flight is booked.</answer>"),
                         ("The flight is booked.", False))
        self.assertEqual(tp.parse_mosaic("<think>Cut off mid-thought"), ("", False))
        self.assertEqual(tp.parse_mosaic("Plain reply."), ("Plain reply.", False))

    def test_mosaic_calls_reads_like_hermes(self):
        text = "Checking. " + mosaic_call("lookup", {"q": "é"}) + mosaic_call("other", {})
        self.assertEqual(tp.mosaic_calls(text), ("Checking. ", [("lookup", '{"q": "é"}'), ("other", "{}")]))
        self.assertEqual(tp.mosaic_calls(mosaic_call("lookup", {"q": 1})), (None, [("lookup", '{"q": 1}')]))
        unclosed = '<tool_call>\n{"name": "lookup", "arguments": {"q": 1}}'
        self.assertEqual(tp.mosaic_calls(unclosed), (None, [("lookup", '{"q": 1}')]))
        broken = "<tool_call>not json</tool_call>"
        self.assertEqual(tp.mosaic_calls(broken), (broken, []))
        self.assertEqual(tp.mosaic_calls('<tool_call>{"name": "f", "parameters": {"a": 1}}</tool_call>'),
                         (None, [("f", '{"a": 1}')]))
        self.assertEqual(tp.mosaic_calls("Plain."), ("Plain.", []))

    def test_mosaic_calls_reads_native_untagged_calls(self):
        self.assertEqual(tp.mosaic_calls('{"name": "lookup", "parameters": {"q": "x"}}'),
                         (None, [("lookup", '{"q": "x"}')]))
        self.assertEqual(tp.mosaic_calls('\nlookup\n{"q": "x"}'), (None, [("lookup", '{"q": "x"}')]))
        self.assertEqual(tp.mosaic_calls('{"answer": 1}'), ('{"answer": 1}', []))

    def test_mosaic_output_only_with_the_scaffold(self):
        text = "<think>x</think><answer>Hi</answer>"
        saved = os.environ.pop("MOSAIC_SCAFFOLD", None)
        try:
            self.assertEqual(tp.mosaic_output(text), text)
            os.environ["MOSAIC_SCAFFOLD"] = "1"
            self.assertEqual(tp.mosaic_output(text), "Hi")
            self.assertIsNone(tp.mosaic_output(None))
        finally:
            os.environ.pop("MOSAIC_SCAFFOLD", None)
            if saved is not None:
                os.environ["MOSAIC_SCAFFOLD"] = saved


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
        self.assertIs(manager["mosaic"], self.tp.MosaicToolParser)

    def test_mosaic_parser(self):
        parser = self.tp.MosaicToolParser(None)
        info = parser.extract_tool_calls("<think>ok</think>\n" + mosaic_call("lookup", {"q": "X"}), self.request)
        self.assertTrue(info.tools_called)
        self.assertEqual([(c.function.name, json.loads(c.function.arguments)) for c in info.tool_calls],
                         [("lookup", {"q": "X"})])
        self.assertIsNone(info.content)
        refusal = parser.extract_tool_calls("<think>no</think>" + mosaic_call("refuse_unsafe_task", {"reason": "r"}),
                                            self.request)
        self.assertEqual((refusal.tools_called, refusal.tool_calls, refusal.content),
                         (False, [], self.tp.mosaic_refusal("r")))
        answer = parser.extract_tool_calls("<think>done</think><answer>Booked.</answer>", self.request)
        self.assertEqual((answer.tools_called, answer.content), (False, "Booked."))
        native = parser.extract_tool_calls('<think>ok</think>\nlookup\n{"q": "X"}', self.request)
        self.assertEqual([(c.function.name, c.function.arguments) for c in native.tool_calls], [("lookup", '{"q": "X"}')])

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
