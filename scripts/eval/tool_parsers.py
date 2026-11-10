"""vLLM tool-call parsers for the GLM-4 and Llama-3.1 base models and the MOSAIC scaffold (vLLM 0.10.2 API).

Loaded by scripts/eval/serve_qwen.sh with `--tool-parser-plugin scripts/eval/tool_parsers.py`; they read the formats
that templates/glm4_tools.jinja and templates/llama31_tools.jinja train and serve:

  glm4_native       GLM-4's native calls, `name\\n{JSON arguments}`, one per <|assistant|> segment. vLLM ships no
                    parser for GLM-4-9B (glm4_moe reads GLM-4.5's <tool_call> format).
  llama3_json_full  `{"name": ..., "parameters": {...}}` objects joined with "; ", as vLLM's llama3_json, but read with
                    a JSON decoder: llama3_json's regex allows one level of nested braces, so a call whose arguments
                    hold a list of objects comes back as plain text. Streaming is llama3_json's (partial JSON).
  mosaic            the MOSAIC turn format that templates/mosaic*.jinja ask for: <think> and <safety_thoughts> are
                    dropped, <answer> is unwrapped, and a refuse_unsafe_task call becomes a plain-text refusal with no
                    tool call, so the episode ends there as in training and the benchmarks' refusal judges see it.
                    <tool_call> blocks are read as vLLM's hermes parser reads them (mosaic_calls), but without needing
                    <tool_call> to be one token, so Llama-3.1 and GLM-4 tokenizers work too; "parameters" is read as
                    "arguments", and a model's native untagged call (Llama's bare JSON object, GLM-4's
                    `name\n{JSON}`) counts as a call, as scripts/post-training/mosaic.py untagged_call does in training.
                    Non-streaming only (every benchmark here makes plain requests).

Requests without tools never reach a tool parser, so ASB's prompted mode, Agent-SafetyBench's tool-less tasks and
HarmBench's offline generation apply the same rewrite through mosaic_output() when MOSAIC_SCAFFOLD=1.

The pure-text logic (parse_glm4, parse_llama3) has no vLLM dependency and is unit-tested in tests/test_tool_parsers.py.
"""
import json
import os
import re

GLM4_SEPARATOR = "<|assistant|>"
GLM4_STOP_TOKENS = ("<|observation|>", "<|user|>", "<|endoftext|>", "<|system|>")
LLAMA3_PREFIX = "<|python_tag|>"
LLAMA3_STOP_TOKENS = ("<|eot_id|>", "<|eom_id|>")
FUNCTION_NAME = re.compile(r"[\w.\-]+")


def _strip_stop(text, stops):
    text = text.rstrip()
    changed = True
    while changed:
        changed = False
        for stop in stops:
            if text.endswith(stop):
                text, changed = text[: -len(stop)].rstrip(), True
    return text


def _is_function_name(name, tool_names):
    return name in tool_names if tool_names else bool(FUNCTION_NAME.fullmatch(name))


def glm4_segments(text, tool_names=()):
    """Split a (possibly partial) GLM-4 reply into ("text", str) and ("call", name, raw_arguments) segments.
    A segment is a call when its first line is a function name; text segments start with an empty first line."""
    out = []
    for segment in text.split(GLM4_SEPARATOR):
        head, newline, body = segment.partition("\n")
        name = head.strip()
        if newline and name and _is_function_name(name, tool_names):
            out.append(("call", name, body))
        elif newline and not name:
            out.append(("text", body))
        elif segment.strip():
            out.append(("text", segment))
    return out


def parse_glm4(text, tool_names=()):
    """Complete reply -> (content or None, [(name, arguments JSON string)]). A call whose arguments are not a JSON
    object stays in the content, so nothing the model wrote is lost."""
    content, calls = [], []
    for segment in glm4_segments(_strip_stop(text, GLM4_STOP_TOKENS), tool_names):
        if segment[0] == "call":
            try:
                arguments = json.loads(segment[2].strip() or "{}")
            except json.JSONDecodeError:
                arguments = None
            if isinstance(arguments, dict):
                calls.append((segment[1], json.dumps(arguments, ensure_ascii=False)))
                continue
            content.append(segment[1] + "\n" + segment[2])
        else:
            content.append(segment[1])
    joined = "".join(content).strip()
    return (joined or None), calls


def parse_llama3(text):
    """Complete reply -> [(name, arguments JSON string)], or None when it is not wholly tool-call JSON."""
    body = _strip_stop(text, LLAMA3_STOP_TOKENS).strip()
    if body.startswith(LLAMA3_PREFIX):
        body = body[len(LLAMA3_PREFIX):].strip()
    if not body.startswith("{"):
        return None
    decoder, calls, index = json.JSONDecoder(), [], 0
    try:
        while index < len(body):
            obj, end = decoder.raw_decode(body, index)
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                return None
            arguments = obj.get("parameters", obj.get("arguments", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append((obj["name"], json.dumps(arguments, ensure_ascii=False)))
            index = end
            while index < len(body) and (body[index].isspace() or body[index] == ";"):
                index += 1
    except (json.JSONDecodeError, ValueError):
        return None
    return calls or None


MOSAIC_REFUSE_TOOL = "refuse_unsafe_task"  # scripts/post-training/mosaic.py REFUSE_TOOL_NAME
MOSAIC_REASONING = re.compile(r"<(think|safety_thoughts)>.*?(?:</\1>|\Z)", re.S)
MOSAIC_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|\Z)", re.S)
MOSAIC_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)


def mosaic_refusal(reason):
    reason = " ".join(str(reason).split())
    return "I refuse to carry out this task because it is unsafe." + (f" Reason: {reason}" if reason else "")


def parse_mosaic(text):
    """MOSAIC turn -> (content, refused). Reasoning blocks go (an unclosed one runs to the end), a refuse_unsafe_task
    call replaces the whole turn with a refusal, <answer> tags are unwrapped, and other <tool_call> blocks stay in
    place for the hermes parser."""
    text = MOSAIC_REASONING.sub("", text)
    for raw in MOSAIC_CALL.findall(text):
        try:
            call = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and call.get("name") == MOSAIC_REFUSE_TOOL:
            arguments = call.get("arguments")
            return mosaic_refusal(arguments.get("reason", "") if isinstance(arguments, dict) else ""), True
    return MOSAIC_ANSWER.sub(lambda match: match.group(1), text).strip(), False


HERMES_CALL = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.S)
REASONING_BLOCKS = re.compile(r"<(think|safety_thoughts)>.*?</\1>", re.S)
GLM_NATIVE_CALL = re.compile(r"\s*([A-Za-z_][\w.\-]*)\n(\{.*\})\s*", re.S)


def untagged_call(text):
    """Same as scripts/post-training/mosaic.py untagged_call: a native call without tags, as (name, arguments)."""
    body = REASONING_BLOCKS.sub("", text).strip()
    match = GLM_NATIVE_CALL.fullmatch(body)
    try:
        if match:
            return match.group(1), json.loads(match.group(2))
        if body.startswith("{") and body.endswith("}"):
            call = json.loads(body)
            if isinstance(call, dict) and isinstance(call.get("name"), str) and ("arguments" in call
                                                                               or "parameters" in call):
                return call["name"], call.get("arguments", call.get("parameters"))
    except json.JSONDecodeError:
        return None
    return None


def mosaic_calls(text):
    """(content, [(name, JSON arguments)]) as vLLM 0.10.2's hermes parser extracts them: every <tool_call> block (an
    unclosed last one runs to the end), content = the text before the first block or None; any block that is not a
    JSON call leaves the whole text as content with no calls. Without blocks, an untagged native call is read."""
    if "<tool_call>" not in text:
        call = untagged_call(text)
        if call is None:
            return text, []
        return None, [(call[0], json.dumps(call[1] if call[1] is not None else {}, ensure_ascii=False))]
    try:
        calls = []
        for closed, unclosed in HERMES_CALL.findall(text):
            call = json.loads(closed or unclosed)
            arguments = call.get("arguments", call.get("parameters", {}))
            calls.append((call["name"], json.dumps(arguments, ensure_ascii=False)))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return text, []
    content = text[:text.find("<tool_call>")]
    return content or None, calls


def mosaic_output(text):
    """The `mosaic` parser's rewrite for replies that bypass the server's tool parser; identity unless
    MOSAIC_SCAFFOLD=1 (scripts/viking/eval_all.sh sets it with SCAFFOLD=mosaic)."""
    if text is None or os.environ.get("MOSAIC_SCAFFOLD") != "1":
        return text
    return parse_mosaic(text)[0]


def _tool_names(request):
    return {tool.function.name for tool in (getattr(request, "tools", None) or [])}


try:  # the classes below need vLLM; the parsing functions above do not
    from vllm.entrypoints.openai.protocol import (DeltaFunctionCall, DeltaMessage, DeltaToolCall,
                                                  ExtractedToolCallInformation, FunctionCall, ToolCall)
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
    from vllm.entrypoints.openai.tool_parsers.llama_tool_parser import Llama3JsonToolParser
    from vllm.entrypoints.chat_utils import make_tool_call_id
except ImportError:  # pragma: no cover - imported without vLLM (tests)
    ToolParser = None

if ToolParser is not None:

    def _extracted(content, calls):
        return ExtractedToolCallInformation(
            tools_called=bool(calls), content=content,
            tool_calls=[ToolCall(type="function", function=FunctionCall(name=name, arguments=arguments))
                        for name, arguments in calls])

    @ToolParserManager.register_module("glm4_native")
    class Glm4NativeToolParser(ToolParser):
        def __init__(self, tokenizer):
            super().__init__(tokenizer)
            self.sent_text = 0           # characters of text content streamed so far
            self.sent_args = []          # per streamed call: characters of arguments streamed so far

        def adjust_request(self, request):
            # <|assistant|> separates parallel calls and is a special token: keep it in the decoded text.
            if request.tools and request.tool_choice != "none":
                request.skip_special_tokens = False
            return request

        def extract_tool_calls(self, model_output, request):
            content, calls = parse_glm4(model_output, _tool_names(request))
            return _extracted(content, calls)

        def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, previous_token_ids,
                                         current_token_ids, delta_token_ids, request):
            text = current_text
            for stop in GLM4_STOP_TOKENS:
                text = text.split(stop)[0]
            # A trailing segment without a newline yet may still turn out to be a call: wait for its first line.
            pieces = text.split(GLM4_SEPARATOR)
            if "\n" not in pieces[-1]:
                text = GLM4_SEPARATOR.join(pieces[:-1])
            segments = glm4_segments(text, _tool_names(request))
            full_text = "".join(s[1] for s in segments if s[0] == "text").lstrip("\n")
            calls = [s for s in segments if s[0] == "call"]
            deltas, content = [], None
            if len(full_text) > self.sent_text:
                content, self.sent_text = full_text[self.sent_text:], len(full_text)
            for index, (_, name, raw) in enumerate(calls):
                if index == len(self.sent_args):
                    self.sent_args.append(0)
                    self.prev_tool_call_arr.append({"name": name, "arguments": {}})
                    self.streamed_args_for_tool.append("")
                    deltas.append(DeltaToolCall(index=index, type="function", id=make_tool_call_id(),
                                                function=DeltaFunctionCall(name=name).model_dump(exclude_none=True)))
                raw = raw.rstrip() if index < len(calls) - 1 else raw
                if len(raw) > self.sent_args[index]:
                    piece = raw[self.sent_args[index]:]
                    self.sent_args[index] = len(raw)
                    self.streamed_args_for_tool[index] += piece
                    deltas.append(DeltaToolCall(index=index,
                                                function=DeltaFunctionCall(arguments=piece).model_dump(exclude_none=True)))
            self.current_tool_id = len(calls) - 1
            if content is None and not deltas:
                return None
            return DeltaMessage(content=content, tool_calls=deltas) if deltas else DeltaMessage(content=content)

    @ToolParserManager.register_module("llama3_json_full")
    class Llama3JsonFullToolParser(Llama3JsonToolParser):
        def extract_tool_calls(self, model_output, request):
            calls = parse_llama3(model_output)
            return _extracted(None, calls) if calls else _extracted(model_output, [])

    @ToolParserManager.register_module("mosaic")
    class MosaicToolParser(ToolParser):
        def extract_tool_calls(self, model_output, request):
            content, refused = parse_mosaic(model_output)
            if refused:
                return _extracted(content, [])
            remaining, calls = mosaic_calls(content)
            return _extracted(remaining if calls else (content or None), calls)

        def extract_tool_calls_streaming(self, *args, **kwargs):
            raise NotImplementedError("the mosaic parser is non-streaming; send stream=False")
