"""Offline checks for the training-set builder: conversion, structure/pattern QC, overlap detection."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("build_trainset", Path(__file__).parents[1] / "scripts/build_trainset.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def call(name, **args):
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


TOOLS = [{"type": "function", "function": {"name": "search_bing", "parameters": {"type": "object", "required": ["q"]}}},
         {"type": "function", "function": {"name": "post_weibo", "parameters": {"type": "object", "required": ["text"]}}}]
CANDIDATE = {"tools": TOOLS, "pattern": ["Search.web_search", "Social.create_post"]}
ALTERNATIVES = {"Search.web_search": {"search_bing"}, "Social.create_post": {"post_weibo"}}


def trajectory(second_args=None):
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [call("search_bing", q="x")]},
            {"role": "tool", "content": "{\"ok\": 1}", "tool_call_id": "c1"},
            {"role": "assistant", "content": "", "tool_calls": [dict(call("post_weibo", **(second_args or {"text": "t"})), id="c2")]},
            {"role": "tool", "content": "{\"ok\": 2}", "tool_call_id": "c2"},
            {"role": "assistant", "content": "done"}]


class ConvertTests(unittest.TestCase):
    def test_apigen_mapping(self):
        row = {"system": "policy", "tools": json.dumps([{"name": "get", "description": "d", "parameters": {}}]),
               "conversations": [{"from": "human", "value": "hi"},
                                 {"from": "function_call", "value": json.dumps({"name": "get", "arguments": {"a": 1}})},
                                 {"from": "observation", "value": "{}"}, {"from": "gpt", "value": "ok"}]}
        fields, reason = bt.convert_apigen("apigen-00000", row)
        self.assertIsNone(reason)
        roles = [m["role"] for m in fields["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(fields["messages"][3]["tool_call_id"], fields["messages"][2]["tool_calls"][0]["id"])
        self.assertEqual(fields["tools"][0]["function"]["name"], "get")
        self.assertIsNone(bt.check_structure({"messages": fields["messages"], "tools": fields["tools"]}))

    def test_apigen_ending_on_observation_is_rejected(self):
        row = {"system": "p", "tools": "[]", "conversations": [{"from": "human", "value": "hi"},
               {"from": "function_call", "value": json.dumps({"name": "g", "arguments": {}})}, {"from": "observation", "value": "{}"}]}
        fields, reason = bt.convert_apigen("a", row)
        self.assertIn("assistant", bt.check_structure({"messages": fields["messages"], "tools": []}))

    def test_agentalign_gets_generic_system_prompt_and_drops_extras(self):
        out = bt.convert_agentalign(None, [{"role": "user", "content": "u", "extra": 1}, {"role": "assistant", "content": None}])
        self.assertEqual(out[0], {"role": "system", "content": bt.GENERIC_SYSTEM_PROMPT})
        self.assertEqual(out[2], {"role": "assistant", "content": ""})
        self.assertNotIn("extra", out[1])


class QcTests(unittest.TestCase):
    def test_pattern_pass_and_ordering(self):
        self.assertIsNone(bt.check_pattern(CANDIDATE, trajectory(), ALTERNATIVES))
        reversed_pattern = dict(CANDIDATE, pattern=CANDIDATE["pattern"][::-1])
        self.assertIn("not called in order", bt.check_pattern(reversed_pattern, trajectory(), ALTERNATIVES))

    def test_missing_required_argument(self):
        self.assertIn("missing required", bt.check_pattern(CANDIDATE, trajectory({"wrong": 1}), ALTERNATIVES))

    def test_disclaimer_regex(self):
        self.assertTrue(bt.DISCLAIMER.search("I'm sorry, but I can't help"))
        self.assertTrue(bt.DISCLAIMER.search("Please note this is illegal."))
        self.assertFalse(bt.DISCLAIMER.search("Posted the message and sent the email."))

    def test_structure_requires_final_prose(self):
        messages = trajectory()[:-1]
        self.assertIsNotNone(bt.check_structure({"messages": messages, "tools": TOOLS}))


class OverlapTests(unittest.TestCase):
    def test_ngram_overlap(self):
        a = "please book the cheapest flight from Boston to Denver on May 9 for two adults and email me the receipt today"
        self.assertTrue(bt.ngram_set(a) & bt.ngram_set("Hello! " + a.upper() + " thanks"))
        self.assertFalse(bt.ngram_set(a) & bt.ngram_set("an unrelated sentence about cooking pasta with garlic and olive oil in a pan"))

    def test_shuffle_is_seeded_and_prefix_stable(self):
        ids = [f"id{i}" for i in range(100)]
        self.assertEqual(bt.shuffled(ids, "x"), bt.shuffled(reversed(ids), "x"))
        self.assertNotEqual(bt.shuffled(ids, "x"), bt.shuffled(ids, "y"))


if __name__ == "__main__":
    unittest.main()


class SlotTests(unittest.TestCase):
    ORDER = {"slots": ["a1", "b1", "a2", "b2"], "category_order": ["b", "a"],
             "backups": {"a": ["a3", "a4"], "b": ["b3"]}}
    CAT = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}

    def test_pass_keeps_slot_order(self):
        ids, replaced, cross = bt.select_slots(self.ORDER, {"a1", "b1", "a2", "b2", "a3"}, self.CAT)
        self.assertEqual((ids, replaced, cross), (["a1", "b1", "a2", "b2"], 0, []))

    def test_same_category_backup_replaces_in_slot(self):
        ids, replaced, cross = bt.select_slots(self.ORDER, {"a1", "b1", "b2", "a3", "a4"}, self.CAT)
        self.assertEqual(ids, ["a1", "b1", "a3", "b2"])
        self.assertEqual((replaced, cross), (1, []))

    def test_cross_category_fill_when_backups_exhausted(self):
        ids, replaced, cross = bt.select_slots(self.ORDER, {"a1", "b1", "a2", "a3"}, self.CAT)
        self.assertEqual(ids, ["a1", "b1", "a2", "a3"])
        self.assertEqual(cross, [("b", "a")])

    def test_unfillable_slot_is_skipped_not_fatal(self):
        ids, replaced, cross = bt.select_slots(self.ORDER, {"a1", "b2"}, self.CAT)
        self.assertEqual(ids, ["a1", "b2"])


class JudgeParseTests(unittest.TestCase):
    def test_formats(self):
        self.assertFalse(bt.parse_verdict("Therefore, the answer is: **No**."))
        self.assertTrue(bt.parse_verdict("It refuses.\\n\\nAnswer: Yes"))
        self.assertFalse(bt.parse_verdict("reasoning ... Answer: No"))
        self.assertTrue(bt.parse_verdict("The response is a refusal. Yes"))

    def test_unparseable_raises(self):
        with self.assertRaises(bt.JudgeParseError):
            bt.parse_verdict("I am not sure what to say here at all")
