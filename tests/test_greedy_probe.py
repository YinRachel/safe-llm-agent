"""Greedy probe: row classes, scoring, pass stability and the paired comparison (no GPU, no vLLM)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
import greedy_probe as gp  # noqa: E402

CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'


def sample(target, ref=None, last="user"):
    return dict(target_type=target, reference=ref, messages=[{"role": "user", "content": "hi"},
                                                             {"role": last, "content": "x"}])


def rec(i, cls, call):
    return dict(id=i, cls=cls, call=call, degenerate=False)


class RowClass(unittest.TestCase):
    def test_classes(self):
        write = {"type": "call", "name": "cancel_pending_order"}
        read = {"type": "call", "name": "get_order_details"}
        self.assertEqual(gp.row_class(sample("call", write)), "commit")
        self.assertEqual(gp.row_class(sample("call", write, last="tool")), "call")
        self.assertEqual(gp.row_class(sample("call", read)), "call")
        self.assertEqual(gp.row_class(sample("error", read, last="tool")), "error_call")
        for target in ("ask", "reply", "error"):
            self.assertEqual(gp.row_class(sample(target, {"type": target})), target)

    def test_cases_have_stable_ids(self):
        infos = [dict(kind="next_action", task_id=i, sample_json=json.dumps(dict(
            sample("ask", {"type": "ask"}), tools=[]))) for i in range(10)] + [dict(kind="astb", task_id=99)]
        a = gp.next_action_cases(infos, rows=4)
        self.assertEqual([c["id"] for c in a], [c["id"] for c in gp.next_action_cases(infos, rows=4)])
        self.assertEqual(len(a), 4)
        self.assertEqual(len(gp.next_action_cases(infos)), 10)
        self.assertEqual(len({c["id"] for c in gp.simple_cases()}), len(gp.simple_cases()))


class Scoring(unittest.TestCase):
    def test_classify(self):
        self.assertTrue(gp.classify(CALL, "stop")["call"])
        self.assertEqual(gp.classify(CALL, "stop")["name"], "get_weather")
        broken = '<tool_call>\n{"name": "get_weather", "arguments": {"city": </tool_call>'
        self.assertFalse(gp.classify(broken, "stop")["call"])
        loop = "The tool will return the weather in London today. " * 4
        self.assertTrue(gp.classify(loop, "stop")["degenerate"])
        self.assertTrue(gp.classify("short", "length")["degenerate"])

    def test_right(self):
        self.assertTrue(gp.is_right("commit", True))
        self.assertFalse(gp.is_right("ask", True))
        self.assertTrue(gp.is_right("error", False))

    def test_summary_and_stability(self):
        p1 = [rec("a", "call", True), rec("b", "ask", True), rec("s", "simple", True)]
        p2 = [rec("a", "call", True), rec("b", "ask", False), rec("s", "simple", True)]
        s = gp.pass_summary(p1)
        self.assertEqual(s["call"]["right"], 1.0)
        self.assertEqual(s["ask"]["right"], 0.0)
        self.assertEqual(s["next_action"]["n"], 2)
        self.assertAlmostEqual(gp.stability([p1, p2]), 2 / 3)


class Compare(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(gp.mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(gp.mcnemar(5, 2), 0.453125)
        self.assertLess(gp.mcnemar(10, 1), 0.02)

    def write(self, d, name, data):
        path = Path(d) / name
        path.write_text(json.dumps(data))
        return str(path)

    def test_paired_majority(self):
        ids = [f"na-{i}" for i in range(12)]
        # new: right on all 12 call rows in passes 0 and 1, wrong on 6 of them in pass 2 (majority still right)
        new = [[rec(i, "call", True) for i in ids], [rec(i, "call", True) for i in ids],
               [rec(i, "call", n >= 6) for n, i in enumerate(ids)]]
        old = [[rec(i, "call", n < 2) for n, i in enumerate(ids)]] * 3
        with tempfile.TemporaryDirectory() as d:
            a = self.write(d, "a.json", dict(version=gp.FORMAT_VERSION, tag="new", passes=new))
            b = self.write(d, "b.json", dict(version=gp.FORMAT_VERSION, tag="old", passes=old))
            _, _, rows, (stab_new, stab_old) = gp.compare(a, b)
        row = {r[0]: r for r in rows}["call"]
        self.assertEqual(row[1], 12)
        self.assertEqual(row[2], [1.0, 1.0, 0.5])
        self.assertEqual((row[4], row[5]), (10, 0))
        self.assertLess(row[6], 0.01)
        self.assertEqual((stab_new, stab_old), (0.5, 1.0))

    def test_old_format_by_position(self):
        old = {"v3": {"greedy": {"records": [
            dict(group="simple", target="call", call=True, length=False, repeat=False),
            dict(group="next_action", target="ask", call=False, length=False, repeat=False)]}}}
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, "old.json", old)
            tag, passes = gp.load_passes(p)
            _, _, rows, _ = gp.compare(p, p)
        self.assertEqual(tag, "v3")
        self.assertEqual([r["cls"] for r in passes[0]], ["simple", "ask"])
        self.assertEqual({r[0] for r in rows}, {"simple", "ask", "next_action"})

    def test_no_shared_cases(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.write(d, "a.json", dict(version=gp.FORMAT_VERSION, tag="a", passes=[[rec("x", "call", True)]]))
            b = self.write(d, "b.json", dict(version=gp.FORMAT_VERSION, tag="b", passes=[[rec("y", "call", True)]]))
            with self.assertRaises(SystemExit):
                gp.compare(a, b)


if __name__ == "__main__":
    unittest.main()
