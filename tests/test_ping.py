"""PING: the template adds the prefix only after the generation prompt, refusal matching ignores the prefix, and the
search follows Algorithm 1 (threshold-gated seeding, per-criterion top-k, argmax overall)."""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent / "scripts/post-training"), str(HERE)]
import ping  # noqa: E402

try:
    import transformers  # noqa: F401
    import tokenizers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

PREFIX = 'As a "careful" agent {not jinja} 100% safe \\ I refuse harmful requests.'
TOOLS = [{"type": "function", "function": {"name": "tool_a", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
CHAT = [{"role": "system", "content": "SYSTEM_PLACEHOLDER"}, {"role": "user", "content": "USER_PLACEHOLDER"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "tool_a", "arguments": {}}}]},
        {"role": "tool", "content": "TOOL_PLACEHOLDER"}, {"role": "assistant", "content": "ANSWER_PLACEHOLDER"},
        {"role": "user", "content": "USER_2"}]


@unittest.skipUnless(HAVE_TRANSFORMERS, "needs transformers (.venv-trainset)")
class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tiny_qwen
        cls.tokenizer = tiny_qwen.make_tokenizer()
        cls.templates = {"tiny": tiny_qwen.CHAT_TEMPLATE}
        try:
            cls.templates["qwen"] = ping.load_chat_template()
        except SystemExit:
            pass

    def render(self, template, messages, tools, generation):
        return self.tokenizer.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False,
                                                  add_generation_prompt=generation)

    def test_prefix_only_after_generation_prompt(self):
        for name, original in self.templates.items():
            ours = ping.make_template(original, PREFIX)
            for messages in (CHAT, CHAT[1:]):
                for tools in (None, TOOLS):
                    with self.subTest(template=name, system=messages[0]["role"] == "system", tools=bool(tools)):
                        self.assertEqual(self.render(ours, messages, tools, False), self.render(original, messages, tools, False))
                        with_prompt = self.render(original, messages, tools, True)
                        self.assertTrue(with_prompt.endswith("<|im_start|>assistant\n"))
                        self.assertEqual(self.render(ours, messages, tools, True), with_prompt + PREFIX)

    def test_rejects_unembeddable_prefix(self):
        for bad in ("", "x {% endraw %} y"):
            with self.assertRaises(ValueError):
                ping.make_template("t", bad)


class RefusalTests(unittest.TestCase):
    def test_prefix_match_does_not_count(self):
        self.assertFalse(ping.is_refusal("I cannot help with harm. ", "Sure, calling the tool now."))
        self.assertTrue(ping.is_refusal("I cannot help with harm. ", "I won’t do this."))

    def test_match_across_boundary_counts(self):
        self.assertTrue(ping.is_refusal("Safety first. I", " cannot assist with that."))
        self.assertFalse(ping.is_refusal("", "<tool_call>\n{}\n</tool_call>"))


class ParsingTests(unittest.TestCase):
    def test_parse_prefixes(self):
        text = 'Here you go:\nPREFIX: As a safe agent, I help.\n2. PREFIX: "Be careful."\nprefix: As a safe agent, I help.\nnoise'
        self.assertEqual(ping.parse_prefixes(text), ["As a safe agent, I help.", "Be careful."])

    def test_task_prompt_cuts_before_first_assistant(self):
        task = ping.task_prompt({"id": "x", "messages": CHAT, "tools": TOOLS})
        self.assertEqual(task["messages"], CHAT[:2])
        with self.assertRaises(ValueError):
            ping.task_prompt({"id": "y", "messages": [CHAT[0], CHAT[2]]})


class FakeAgent:
    """Refuses a harmful task iff the prefix contains 'refuse'; refuses a benign one iff it contains 'paranoid'."""

    def __init__(self):
        self.calls = 0

    def respond(self, jobs):
        self.calls += 1
        out = []
        for task, prefix in jobs:
            refuse = "paranoid" in prefix if task["id"].startswith("b") else "refuse" in prefix
            out.append(("I cannot do that." if refuse else "<tool_call>{}</tool_call>", "stop"))
        return out


class FakeGenerator:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


BENIGN = [{"id": f"b{i}", "messages": [], "tools": None} for i in range(4)]
HARMFUL = [{"id": f"h{i}", "messages": [], "tools": None} for i in range(4)]


class SearchTests(unittest.TestCase):
    def test_scores(self):
        results = ping.score_prefixes(FakeAgent(), ["refuse harm", "paranoid refuse", "hello"], BENIGN, HARMFUL)
        self.assertEqual([(r["perf"], r["refusal"]) for r in results.values()], [(1, 1), (0, 1), (1, 0)])

    def test_threshold_gates_seeding_and_best_is_argmax(self):
        gen = FakeGenerator(["PREFIX: hello\nPREFIX: paranoid refuse",   # best overall 1.0 < tau -> no seed
                             "PREFIX: hello\nPREFIX: refuse harm",       # 'hello' cached; best 2.0 -> seed
                             "PREFIX: paranoid"])
        agent = FakeAgent()
        best, evaluated = ping.search(agent, gen, BENIGN, HARMFUL, rounds=3, m=5, k=1, tau=1.5)
        self.assertEqual(best["prefix"], "refuse harm")
        self.assertEqual([r["prefix"] for r in evaluated], ["hello", "paranoid refuse", "refuse harm", "paranoid"])
        self.assertIn("(none yet)", gen.prompts[1])
        self.assertNotIn("(none yet)", gen.prompts[2])
        self.assertIn("Prefix: refuse harm\nOverall Score: 2.000", gen.prompts[2])

    def test_tie_prefers_refusal_then_earliest(self):
        gen = FakeGenerator(["PREFIX: hello\nPREFIX: paranoid refuse\nPREFIX: hi"])
        best, _ = ping.search(FakeAgent(), gen, BENIGN, HARMFUL, rounds=1, m=5, k=3, tau=1.5)
        self.assertEqual(best["prefix"], "paranoid refuse")

    def test_empty_replies_retried_then_round_skipped(self):
        gen = FakeGenerator(["no prefixes here", "", "", "PREFIX: refuse"])
        best, evaluated = ping.search(FakeAgent(), gen, BENIGN, HARMFUL, rounds=2, m=5, k=3, tau=1.5)
        self.assertEqual(len(gen.prompts), 4)
        self.assertEqual(best["prefix"], "refuse")
        self.assertEqual(evaluated[0]["round"], 1)

    def test_candidates_capped_at_m(self):
        gen = FakeGenerator(["\n".join(f"PREFIX: p{i}" for i in range(9))])
        _, evaluated = ping.search(FakeAgent(), gen, BENIGN, HARMFUL, rounds=1, m=5, k=3, tau=1.5)
        self.assertEqual(len(evaluated), 5)


if __name__ == "__main__":
    unittest.main()
