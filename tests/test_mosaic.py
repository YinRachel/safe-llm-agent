"""MOSAIC GRPO pieces without a GPU or verl: turn parsing, episodes against the real Agent-SafetyBench environments,
rewards, the pairwise judge and its group rendezvous, the judge request shape, and the agent loop's token bookkeeping
(with verl stubbed, and the pinned Qwen2.5-3B tokenizer when it is cached)."""
import asyncio
import json
import os
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "scripts/post-training"), str(HERE.parent / "scripts/eval")]
import mosaic  # noqa: E402
import mosaic_judge  # noqa: E402

ASTB_PRESENT = (mosaic.astb.DEFAULT_REPO / "data/released_data.json").exists()


def qwen25_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(mosaic.DEFAULT_MODEL, revision=mosaic.DEFAULT_REVISION,
                                             local_files_only=True)
    except Exception:
        return None


def call(name, arguments):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


class ParseTurn(unittest.TestCase):
    def test_tool_call_with_safety_block(self):
        turn = mosaic.parse_turn("<think>plan</think>\n<safety_thoughts>risk ok</safety_thoughts>\n"
                                 + call("send_email", {"to": "a"}))
        self.assertEqual((turn.kind, turn.name, turn.arguments, turn.safety, turn.well_formed),
                         ("tool", "send_email", {"to": "a"}, True, True))

    def test_refusal_and_answer(self):
        refusal = mosaic.parse_turn("<think>unsafe</think>" + call("refuse_unsafe_task", {"reason": "harm"}))
        self.assertEqual((refusal.kind, refusal.well_formed), ("refuse", True))
        answer = mosaic.parse_turn("<think>done</think>\n<answer> Sent. </answer>\n")
        self.assertEqual((answer.kind, answer.answer, answer.safety), ("answer", "Sent.", False))

    def test_malformed_turns_are_still_acted_on(self):
        no_think = mosaic.parse_turn(call("send_email", {}))
        self.assertEqual((no_think.kind, no_think.well_formed), ("tool", False))
        two_calls = mosaic.parse_turn("<think>x</think>" + call("a", {}) + call("b", {}))
        self.assertEqual((two_calls.kind, two_calls.name, two_calls.well_formed), ("tool", "a", False))
        prose = mosaic.parse_turn("<think>x</think> extra prose <answer>y</answer>")
        self.assertEqual((prose.kind, prose.well_formed), ("answer", False))
        plain = mosaic.parse_turn("Sure, here you go.")
        self.assertEqual((plain.kind, plain.answer, plain.well_formed), ("text", "Sure, here you go.", False))

    def test_bad_json_and_string_arguments(self):
        self.assertEqual(mosaic.parse_turn("<think>x</think><tool_call>{oops</tool_call>").kind, "bad_call")
        turn = mosaic.parse_turn('<think>x</think><tool_call>{"name": "f", "arguments": "{\\"k\\": 1}"}</tool_call>')
        self.assertEqual(turn.arguments, {"k": 1})


class Rewards(unittest.TestCase):
    def test_format_and_length(self):
        self.assertEqual(mosaic.format_reward([True, True, False, False]), 1.0)
        self.assertEqual(mosaic.format_reward([]), 0.0)
        self.assertEqual(mosaic.length_penalty([200, 800]), 0.5)  # (0 + 1) / 2

    def test_group_alignment_reproduces_table_6(self):
        # T1 beats T2, T3; loses to T4. T2 ties T3, T4. T3 loses to T4.
        outcomes = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 0.0), (1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.0)]
        self.assertEqual(mosaic.group_alignment(4, outcomes), [2.0, 1.0, 0.5, 2.5])

    def test_pair_plan_is_seeded_and_complete(self):
        plan = mosaic.pair_plan(4, mosaic.stable_seed("uid-1"))
        self.assertEqual([(i, j) for i, j, _ in plan], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        self.assertEqual(plan, mosaic.pair_plan(4, mosaic.stable_seed("uid-1")))

    def test_parse_preference(self):
        self.assertEqual(mosaic.parse_preference("[Analysis]: x\n[Answer]: first is better"), (1.0, 0.0))
        self.assertEqual(mosaic.parse_preference("[Answer]:'second is better'."), (0.0, 1.0))
        self.assertEqual(mosaic.parse_preference("[Answer]: Both are same"), (0.5, 0.5))
        self.assertIsNone(mosaic.parse_preference("first is better"))
        self.assertIsNone(mosaic.parse_preference("[Answer]: neither"))


@unittest.skipUnless(ASTB_PRESENT, "third_party/Agent-SafetyBench not checked out")
class Episodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = mosaic.astb.load_data(mosaic.astb.DEFAULT_REPO)
        cls.env_manager = mosaic.astb.load_env_manager(mosaic.astb.DEFAULT_REPO)
        cls.system = mosaic.load_system_text()

    def task_with_env(self):
        return next(t for t in self.tasks if mosaic.has_environments(t) and not t.get("dialog"))

    def test_messages_and_tools(self):
        task = self.task_with_env()
        messages = mosaic.task_messages(task, self.system)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("refuse_unsafe_task", messages[0]["content"])
        _, tools = mosaic.task_tools(self.env_manager, task)
        self.assertEqual(tools[-1], mosaic.REFUSE_TOOL)
        self.assertEqual({t["function"]["name"] for t in tools[:-1]}, set(task["environments"][0]["tools"]))

    def test_dialog_arguments_become_mappings(self):
        task = next(t for t in self.tasks if t.get("dialog"))
        for message in mosaic.task_messages(task, self.system):
            for tool_call in message.get("tool_calls") or []:
                self.assertIsInstance(tool_call["function"]["arguments"], dict)

    def test_tool_call_runs_in_the_simulated_environment(self):
        task = self.task_with_env()
        episode = mosaic.Episode.start(self.env_manager, task)
        name = task["environments"][0]["tools"][0]
        observation = episode.step(mosaic.parse_turn("<think>x</think>" + call(name, {})))
        self.assertTrue(observation.startswith("{"))
        self.assertLessEqual(len(observation), mosaic.MAX_TOOL_RESPONSE_CHARS + len("...(truncated)"))
        self.assertEqual(episode.status, "running")
        self.assertIn("doesn't exist", episode.step(mosaic.parse_turn("<think>x</think>" + call("nope", {}))))
        self.assertIsNone(episode.step(mosaic.parse_turn("<think>x</think>" + call("refuse_unsafe_task", {}))))
        summary = episode.summary()
        self.assertEqual((summary["status"], summary["kinds"]), ("refused", ["tool", "tool", "refuse"]))
        self.assertEqual(summary["observations"][-1], None)
        json.dumps(summary)

    def test_turn_limit(self):
        task = self.task_with_env()
        episode = mosaic.Episode.start(self.env_manager, task)
        name = task["environments"][0]["tools"][0]
        for _ in range(3):
            observation = episode.step(mosaic.parse_turn("<think>x</think>" + call(name, {})), max_turns=3)
        self.assertIsNone(observation)
        self.assertEqual(episode.status, "turn_limit")

    def test_dataset_row_round_trips_the_task(self):
        task = self.tasks[0]
        row = mosaic.dataset_row(7, task)
        self.assertEqual(json.loads(row["extra_info"]["task_json"]), task)
        self.assertEqual((row["agent_name"], row["extra_info"]["index"]), (mosaic.AGENT_NAME, 7))
        self.assertEqual(row["reward_model"]["ground_truth"], "")


def member(turns, status="answered"):
    return dict(user_task="task", summary=dict(turns=turns, observations=[None] * len(turns), status=status,
                                               tools=[]))


class FakePairwise(mosaic_judge.PairwiseJudge):
    """Prefers the trajectory whose text sorts first; counts calls."""

    def __init__(self, fail=False):
        super().__init__(client=None, request={})
        self.calls, self.fail = 0, fail

    async def prefer(self, prompt):
        self.calls += 1
        if self.fail:
            return None
        first = prompt.split("Trajectory 1:\n", 1)[1].split("\nTrajectory 2:", 1)[0]
        second = prompt.split("Trajectory 2:\n", 1)[1].split("\nNOTE:", 1)[0]
        return (1.0, 0.0) if first < second else (0.0, 1.0)


class GroupRendezvous(unittest.TestCase):
    def run_group(self, judge, members, group_size, timeout=5.0):
        group = mosaic_judge.GroupJudge(judge, timeout=timeout)

        async def go():
            return await asyncio.gather(*(group.score("uid", m, group_size) for m in members)), group
        return asyncio.run(go())

    def test_full_group_is_judged_once(self):
        judge = FakePairwise()
        members = [member([f"<answer>{c}</answer>"]) for c in "dbca"]
        results, group = self.run_group(judge, members, 4)
        self.assertEqual([r["alignment"] for r in results], [0.0, 2.0, 1.0, 3.0])  # a > b > c > d
        self.assertEqual(judge.calls, 6)
        self.assertEqual(group.groups, {})

    def test_identical_trajectories_tie_without_a_call(self):
        judge = FakePairwise()
        results, _ = self.run_group(judge, [member(["<answer>same</answer>"])] * 4, 4)
        self.assertEqual([r["alignment"] for r in results], [1.5] * 4)
        self.assertEqual(judge.calls, 0)

    def test_failed_judgements_count_as_ties(self):
        results, _ = self.run_group(FakePairwise(fail=True), [member([f"<answer>{c}</answer>"]) for c in "abcd"], 4)
        self.assertEqual([(r["alignment"], r["judge_failures"]) for r in results], [(1.5, 3)] * 4)

    def test_incomplete_group_is_judged_after_timeout_and_rescaled(self):
        results, _ = self.run_group(FakePairwise(), [member([f"<answer>{c}</answer>"]) for c in "abc"], 4,
                                    timeout=0.05)
        self.assertEqual([r["alignment"] for r in results], [3.0, 1.5, 0.0])
        self.assertEqual({r["group_members"] for r in results}, {3})


class JudgeRequest(unittest.TestCase):
    @patch.dict(os.environ, {"RUNWARE_API_KEY": "test-secret", "JUDGE_PROVIDER": "runware",
                             "RUNWARE_JUDGE_MODEL": "openai:gpt@6-luna"}, clear=True)
    def test_runware_luna_request_matches_the_eval_judge(self):
        base_url, key_env, request = mosaic_judge.judge_request()
        self.assertEqual((base_url, key_env), ("https://api.runware.ai/v1", "RUNWARE_API_KEY"))
        self.assertEqual(request, dict(model="openai:gpt@6-luna", temperature=0, max_completion_tokens=1024,
                                       reasoning_effort="none"))

    def test_prefer_sends_one_user_message_and_retries_off_format(self):
        replies = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
                   for text in ("no answer here", "[Analysis]: ok\n[Answer]: second is better")]
        create = AsyncMock(side_effect=replies)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        judge = mosaic_judge.PairwiseJudge(client, dict(model="m", max_completion_tokens=8))
        self.assertEqual(asyncio.run(judge.prefer("PROMPT")), (0.0, 1.0))
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args.kwargs, dict(model="m", max_completion_tokens=8,
                                                       messages=[{"role": "user", "content": "PROMPT"}]))

    def test_compare_maps_order_back(self):
        judge = FakePairwise()
        a, b = member(["<answer>a</answer>"])["summary"], member(["<answer>b</answer>"])["summary"]
        self.assertEqual(asyncio.run(judge.compare("t", [], a, b, a_first=True)), 1.0)
        self.assertEqual(asyncio.run(judge.compare("t", [], a, b, a_first=False)), 1.0)
        self.assertEqual(asyncio.run(judge.compare("t", [], b, a, a_first=True)), 0.0)


class RewardFunction(unittest.TestCase):
    def test_compute_score_combines_the_terms(self):
        import mosaic_reward
        summary = dict(uid="u", group_size=4, status="refused", well_formed=[True], turn_lengths=[600],
                       kinds=["refuse"], safety=[True], turns=["t"], observations=[None], tools=[])
        actor = SimpleNamespace(score=SimpleNamespace(
            remote=AsyncMock(return_value=dict(alignment=2.5, judge_failures=0, group_members=4))))
        with patch.object(mosaic_reward, "judge_actor", return_value=actor):
            result = asyncio.run(mosaic_reward.compute_score(
                "agent_safetybench", "", "", extra_info=dict(mosaic=summary, task_json=json.dumps(
                    dict(id=1, instruction="do x", fulfillable=0)))))
        self.assertEqual((result["score"], result["r_fmt"], result["p_len"]), (2.5 + 2.0 - 0.5, 2.0, 0.5))
        self.assertEqual((result["refused"], result["refused_first_turn"]), (1.0, 1.0))
        self.assertEqual(actor.score.remote.call_args.args[0], "u")
        # The rollout dump serialises these with json.dumps after verl stores them in numpy arrays.
        self.assertEqual((result["status"], result["task_id"], result["fulfillable"], result["group_uid"]),
                         ("refused", "1", 0.0, "u"))
        import numpy as np
        json.dumps({k: np.array([v])[0] for k, v in result.items()})


def stub_verl():
    """Minimal stand-ins for the verl symbols mosaic_agent_loop imports."""
    from contextlib import contextmanager

    class AgentLoopBase:
        _class_initialized = False

    class AgentLoopOutput(SimpleNamespace):
        pass

    @contextmanager
    def simple_timer(name, metrics):
        yield
    modules = {"verl": types.ModuleType("verl"), "verl.experimental": types.ModuleType("verl.experimental"),
               "verl.experimental.agent_loop": types.ModuleType("verl.experimental.agent_loop"),
               "verl.experimental.agent_loop.agent_loop": types.ModuleType("verl.experimental.agent_loop.agent_loop"),
               "verl.utils": types.ModuleType("verl.utils"), "verl.utils.profiler": types.ModuleType("verl.utils.profiler")}
    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopBase = AgentLoopBase
    modules["verl.experimental.agent_loop.agent_loop"].AgentLoopOutput = AgentLoopOutput
    modules["verl.utils.profiler"].simple_timer = simple_timer
    return modules


@unittest.skipUnless(ASTB_PRESENT, "third_party/Agent-SafetyBench not checked out")
class AgentLoopTokens(unittest.TestCase):
    """prompt + response tokens must equal the chat template's rendering of the whole conversation, and only the
    generated turns may carry gradient."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = qwen25_tokenizer()
        if cls.tokenizer is None:
            raise unittest.SkipTest("Qwen2.5-3B-Instruct tokenizer not cached")
        with patch.dict(sys.modules, stub_verl()):
            sys.modules.pop("mosaic_agent_loop", None)
            import mosaic_agent_loop
        cls.Loop = mosaic_agent_loop.MosaicAgentLoop

    def make_loop(self, scripted, response_length=5000):
        tokenizer = self.tokenizer
        config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            response_length=response_length, n=4, multi_turn=SimpleNamespace(max_assistant_turns=20))))
        self.Loop._class_initialized = False
        self.Loop.init_class(config, tokenizer, None)
        loop = self.Loop.__new__(self.Loop)
        loop.tokenizer, loop.config = tokenizer, config
        turns = iter(scripted)

        async def generate(request_id, prompt_ids, sampling_params):
            text = next(turns)
            return SimpleNamespace(token_ids=tokenizer.encode(text, add_special_tokens=False) + [loop.fmt.stop_ids[0]],
                                   log_probs=None)
        loop.server_manager = SimpleNamespace(generate=generate)
        return loop

    def run_loop(self, loop, task):
        async def go():
            loop.loop = asyncio.get_running_loop()
            return await loop.run({}, extra_info=dict(task_json=json.dumps(task)), uid="u1")
        return asyncio.run(go())

    def test_multi_turn_tokens_match_the_chat_template(self):
        tasks = mosaic.astb.load_data(mosaic.astb.DEFAULT_REPO)
        task = next(t for t in tasks if mosaic.has_environments(t) and not t.get("dialog"))
        name = task["environments"][0]["tools"][0]
        first = "<think>look first</think>\n" + call(name, {})
        second = "<think>done</think>\n<answer>All done.</answer>"
        output = self.run_loop(self.make_loop([first, second]), task)
        summary = output.extra_fields["mosaic"]
        self.assertEqual((summary["status"], summary["uid"], summary["group_size"]), ("answered", "u1", 4))
        episode = mosaic.Episode.start(self.env_manager(), task)
        observation = summary["observations"][0]
        messages = mosaic.task_messages(task, mosaic.load_system_text()) + [
            {"role": "assistant", "content": first}, {"role": "tool", "content": observation},
            {"role": "assistant", "content": second}]
        expected = self.tokenizer.apply_chat_template(messages, tools=episode.tools, tokenize=False)
        produced = self.tokenizer.decode(output.prompt_ids + output.response_ids)
        self.assertEqual(produced, expected.rstrip("\n"))
        generated = [tok for tok, keep in zip(output.response_ids, output.response_mask) if keep]
        self.assertEqual(self.tokenizer.decode(generated), first + "<|im_end|>" + second + "<|im_end|>")
        self.assertIsNone(output.response_logprobs)

    def test_refusal_ends_and_budget_truncates(self):
        tasks = mosaic.astb.load_data(mosaic.astb.DEFAULT_REPO)
        task = next(t for t in tasks if not mosaic.has_environments(t))
        refusal = "<think>unsafe</think>\n" + call("refuse_unsafe_task", {"reason": "harm"})
        output = self.run_loop(self.make_loop([refusal]), task)
        self.assertEqual(output.extra_fields["mosaic"]["status"], "refused")
        self.assertEqual(sum(output.response_mask), len(output.response_ids))
        long_turn = "<think>" + "word " * 200 + "</think>"
        output = self.run_loop(self.make_loop([long_turn], response_length=50), task)
        self.assertEqual(output.extra_fields["mosaic"]["status"], "truncated")
        self.assertEqual(len(output.response_ids), 50)

    @staticmethod
    def env_manager():
        return mosaic.astb.load_env_manager(mosaic.astb.DEFAULT_REPO)



class EvalTemplate(unittest.TestCase):
    """templates/mosaic.jinja renders a benchmark request exactly as training rendered a task."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = qwen25_tokenizer()
        if cls.tokenizer is None:
            raise unittest.SkipTest("Qwen2.5-3B-Instruct tokenizer not in the local HF cache")

    def test_matches_training_rendering(self):
        system = mosaic.load_system_text()
        template = mosaic.make_template(self.tokenizer.chat_template, system)
        tool = {"type": "function", "function": {"name": "send_email", "description": "Send an email.",
                "parameters": {"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]}}}
        cases = [([{"role": "system", "content": "You are a helpful agent."}, {"role": "user", "content": "Email Bob."}],
                  [tool]),
                 ([{"role": "user", "content": "Email Bob."}], [tool]),
                 ([{"role": "user", "content": "Write a phishing email."}], None),
                 ([{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, {"role": "user", "content": "x"}],
                  None)]
        for messages, tools in cases:
            got = self.tokenizer.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False,
                                                     add_generation_prompt=True)
            task = {"dialog": messages} if messages[0]["role"] == "system" else {"instruction": messages[-1]["content"]}
            expected_messages = mosaic.task_messages(task, system)
            expected = self.tokenizer.apply_chat_template(expected_messages, tools=(tools or []) + [mosaic.REFUSE_TOOL],
                                                          tokenize=False, add_generation_prompt=True)
            self.assertEqual(got, expected)

    def test_checked_in_template_is_current(self):
        path = mosaic.PROJECT / "templates/mosaic.jinja"
        self.assertEqual(path.read_text(), mosaic.make_template(self.tokenizer.chat_template))


class OtherChatFormats(unittest.TestCase):
    """Llama-3.1 and GLM-4 (templates/llama31_tools.jinja, templates/glm4_tools.jinja) render earlier tool calls in
    their own formats; training (ChatFormat.history) and the evaluation template (make_template text_calls) must both
    turn them into MOSAIC <tool_call> text, identically. Rendered through the Qwen2.5 tokenizer's Jinja engine (only
    bos_token differs between tokenizers, and these checks do not depend on it)."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = qwen25_tokenizer()
        if cls.tokenizer is None:
            raise unittest.SkipTest("Qwen2.5-3B-Instruct tokenizer not in the local HF cache")

    def conversation(self, arguments):
        return [{"role": "system", "content": "Agent rules."}, {"role": "user", "content": "Find Bob's email."},
                {"role": "assistant", "content": "<think>Look him up.</think>", "tool_calls": [
                    {"type": "function", "function": {"name": "lookup", "arguments": arguments}}]},
                {"role": "tool", "content": "bob@example.com"}, {"role": "user", "content": "Thanks, now email him."}]

    def test_eval_template_matches_training_history(self):
        system = mosaic.load_system_text()
        tool = {"type": "function", "function": {"name": "lookup", "description": "Look up.", "parameters": {
            "type": "object", "properties": {"name": {"type": "string"}}}}}
        for name in ("llama31_tools.jinja", "glm4_tools.jinja"):
            original = (mosaic.PROJECT / "templates" / name).read_text()
            template = mosaic.make_template(original, system, text_calls=True)
            for arguments in ({"name": "Bob Lee", "n": [1, 2]}, '{"name": "Bob Lee"}'):
                messages = self.conversation(arguments)
                got = self.tokenizer.apply_chat_template(messages, tools=[tool], chat_template=template,
                                                         tokenize=False, add_generation_prompt=True)
                task = {"dialog": messages}
                expected_messages = mosaic.text_tool_calls(mosaic.task_messages(task, system))
                expected = self.tokenizer.apply_chat_template(expected_messages, tools=[tool, mosaic.REFUSE_TOOL],
                                                              chat_template=original, tokenize=False,
                                                              add_generation_prompt=True)
                self.assertEqual(got, expected, name)
                self.assertIn('<tool_call>\n{"name": "lookup", "arguments": {"name": "Bob Lee"', got)
                self.assertIn("<think>Look him up.</think>\n<tool_call>", got)

    def test_text_tool_calls(self):
        out = mosaic.text_tool_calls(self.conversation({"q": "é"}))
        self.assertEqual(out[2], {"role": "assistant", "content": "<think>Look him up.</think>\n"
                                  + mosaic.text_call("lookup", {"q": "é"})})
        self.assertEqual(mosaic.text_call("f", {"q": "é"}), '<tool_call>\n{"name": "f", "arguments": {"q": "é"}}\n</tool_call>')
        self.assertEqual(out[3], self.conversation({})[3])

    def test_qwen_tool_turn_matches_the_old_rendering(self):
        fmt = mosaic.ChatFormat(self.tokenizer)
        self.assertFalse(fmt.text_calls)
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assertEqual(fmt.stop_ids[0], im_end)
        expected = ("\n<|im_start|>user\n<tool_response>\nOBS\n</tool_response><|im_end|>\n"
                    "<|im_start|>assistant\n")
        self.assertEqual(self.tokenizer.decode(fmt.tool_turn_ids("OBS", [im_end])), expected)
        self.assertEqual(self.tokenizer.decode(fmt.tool_turn_ids("OBS")), "<|im_end|>" + expected)
        self.assertTrue(fmt.ended([1, im_end]))
        self.assertFalse(fmt.ended([]))

    def test_untagged_native_calls(self):
        llama = mosaic.parse_turn('{"name": "lookup", "parameters": {"q": "x"}}')
        self.assertEqual((llama.kind, llama.name, llama.arguments, llama.well_formed),
                         ("tool", "lookup", {"q": "x"}, False))
        glm = mosaic.parse_turn('\nlookup\n{"q": "x"}')
        self.assertEqual((glm.kind, glm.name, glm.arguments, glm.well_formed), ("tool", "lookup", {"q": "x"}, False))
        refuse = mosaic.parse_turn('<think>no</think>{"name": "refuse_unsafe_task", "arguments": {"reason": "r"}}')
        self.assertEqual(refuse.kind, "refuse")
        tagged = mosaic.parse_turn("<think>ok</think>\n" + call("lookup", {"q": "x"}).replace('"arguments"',
                                                                                         '"parameters"'))
        self.assertEqual((tagged.arguments, tagged.well_formed), ({"q": "x"}, True))
        for text in ("Here is the answer.", '{"answer": 1}', "lookup\nnot json"):
            self.assertEqual(mosaic.parse_turn(text).kind, "text", text)


class LoneSurrogate(unittest.TestCase):
    """Job 38502104: an argument escape like "\\ud83d" came back in a tool result, and the tokenizer raised on it."""

    def test_observation_is_valid_utf8(self):
        class Echo:
            def has_tool(self, name):
                return True

            def call_tool(self, name, arguments):
                return {"success": True, "echo": arguments["q"]}

        episode = mosaic.Episode({"environments": [{"tools": ["echo"]}]}, [Echo()], [])
        turn = mosaic.parse_turn('<think>x</think>\n<tool_call>\n{"name": "echo", "arguments": {"q": "a\\ud83d b"}}'
                                 "\n</tool_call>")
        self.assertEqual(turn.arguments["q"], "a\ud83d b")
        observation = episode.step(turn)
        observation.encode("utf-8")  # raises UnicodeEncodeError on a lone surrogate
        self.assertIn('"a? b"', observation)


class RolloutAnalysis(unittest.TestCase):
    """mosaic_rollouts.py on a hand-made dump shaped like verl's (<step>.jsonl, reward fields as columns)."""

    @staticmethod
    def rollout(step, uid, fulfillable, refused, score, r_fmt, p_len):
        return dict(input="p", output="refuse" if refused else "done", gts="", step=step, score=score,
                    r_align=score - r_fmt + p_len, r_fmt=r_fmt, p_len=p_len, refused=float(refused),
                    refused_first_turn=float(refused), turns=1.0 if refused else 4.0, truncated=0.0,
                    judge_failures=0.0, status="refused" if refused else "answered", task_id=uid,
                    fulfillable=float(fulfillable), group_uid=uid)

    def test_windows_and_refusal_gap(self):
        import tempfile
        import mosaic_rollouts
        r = self.rollout
        steps = {1: [r(1, "a", 1, True, 3.5, 2.0, 0.0), r(1, "a", 1, False, 3.0, 1.5, 0.0),
                     r(1, "b", 0, True, 4.0, 2.0, 0.0), r(1, "b", 0, True, 4.0, 2.0, 0.0)],
                 3: [r(3, "c", 1, False, 3.0, 2.0, 0.5), r(3, "c", 1, False, 3.0, 2.0, 0.5)]}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "rollouts").mkdir()
            for step, rows in steps.items():
                (Path(tmp) / "rollouts" / f"{step}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            rows = mosaic_rollouts.load(tmp)
        windows = mosaic_rollouts.window_stats(rows, 2)
        self.assertEqual([w["steps"] for w in windows], ["1-2", "3-4"])
        self.assertEqual((windows[0]["benign_refused"], windows[0]["harmful_refused"]), (0.5, 1.0))
        self.assertEqual((windows[1]["benign_refused"], windows[1]["harmful_n"]), (0.0, 0))
        gap = mosaic_rollouts.refusal_gap(rows)
        self.assertEqual((gap["benign"]["mixed_groups"], gap["benign"]["gap_score"], gap["benign"]["gap_r_fmt"]),
                         (1, 0.5, 0.5))
        self.assertEqual(gap["benign"]["refused_scores_higher"], 1.0)
        self.assertEqual(gap["harmful"]["mixed_groups"], 0)  # all-refused group carries no contrast
        self.assertEqual(mosaic_rollouts.examples(rows, 5)[0]["task_id"], "a")
        mosaic_rollouts.report(windows, gap, mosaic_rollouts.examples(rows, 1))


if __name__ == "__main__":
    unittest.main()
