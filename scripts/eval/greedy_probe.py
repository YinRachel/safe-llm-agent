"""Greedy-decoding probe for GRPO checkpoints: does the model act correctly at temperature 0, or narrate and loop?
Every benchmark decodes greedily, GRPO samples at temperature 1, so a checkpoint can train cleanly and still loop here.

usage (GPU node, serving env with vLLM, project root):
  greedy_probe.py run [--template T.jinja] [--data TRAIN.parquet] [--rows N] [--repeats K] OUT.json TAG=MODEL_DIR
usage (anywhere, no GPU):
  greedy_probe.py compare NEW.json OLD.json     paired comparison, row by row, of two probes on the same rows

run:
  --template  scaffold template for the simple requests (default templates/mosaic.jinja, Qwen2.5-3B;
              templates/mosaic_<tag>.jinja for the others)
  --data      SafeUtil training parquet whose next_action rows are probed (default trainset/derived/safeutil_train.parquet).
              Two checkpoints are comparable only when probed on the same --data and --rows.
  --rows      next_action rows probed, after a fixed shuffle (default 0 = all of them)
  --repeats   greedy passes, each over the prompts in a different order (default 3). Greedy decoding is not
              deterministic across batches: which prompts share a batch shifts the bf16 arithmetic, and a model near
              the edge (Qwen2.5-3B under the MOSAIC scaffold) then flips outcome. The spread across passes is the
              probe's own noise; one pass is not enough to tell checkpoints apart.

Each case is scored as the scaffold evals parse it (tool_parsers.mosaic_calls: a malformed <tool_call> block means no
call). Cases fall into classes: simple (should call), and for next_action rows commit (a write call right after the user
confirmed), call, error_call (a call after a tool error), and ask / reply / error (should not call). A case is right
when it calls exactly where a call is the target; degenerate when it hits max_tokens or repeats a sentence 3 times.

compare: for each class, the right rate of each model per pass (mean and range), and a McNemar test on the
per-case majority vote across passes: n_new = cases only NEW gets right, n_old = cases only OLD gets right.
Older single-pass probe files (no "version" key) can be compared if they cover the same rows in the same order."""
import argparse
import collections
from math import comb
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool_parsers import mosaic_calls  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "post-training"))
from safeutil import WRITE_TOOL  # noqa: E402  the commit rule v7 trains on

FORMAT_VERSION = 2
MAX_TOKENS = 512
CALL_CLASSES = ("simple", "commit", "call", "error_call")
TEXT_CLASSES = ("ask", "reply", "error")
CLASSES = CALL_CLASSES + TEXT_CLASSES


# ----------------------------------------------------------------------------- simple requests (rendered as evals do)
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
STOCK = {"type": "function", "function": {"name": "get_stock_price", "description": "Get the latest price of a stock.",
         "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}}
SEARCH = {"type": "function", "function": {"name": "search_web", "description": "Search the web.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
EMAIL = {"type": "function", "function": {"name": "send_email", "description": "Send an email.",
         "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"},
                        "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}}
ORDER = {"type": "function", "function": {"name": "get_order_details", "description": "Look up an order by id.",
         "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}
FLIGHT = {"type": "function", "function": {"name": "search_flights", "description": "Search direct flights.",
          "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                         "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}}


def simple_cases():
    cases = []
    for city in ["London", "Paris", "Tokyo", "New York", "Berlin", "Sydney"]:
        cases.append(([{"role": "user", "content": f"Use get_weather to look up the weather in {city}."}], [WEATHER]))
        cases.append(([{"role": "user", "content": f"What's the weather like in {city} right now?"}], [WEATHER, SEARCH]))
    for t in ["AAPL", "MSFT", "NVDA"]:
        cases.append(([{"role": "user", "content": f"How much is {t} trading at?"}], [STOCK, SEARCH]))
    cases.append(([{"role": "user", "content": "Find recent news about the Mars rover."}], [SEARCH, WEATHER]))
    for o in ["#W1234567", "#W7654321", "#W5555555", "#W9081726"]:
        cases.append(([{"role": "user", "content": f"Can you check the status of my order {o}?"}], [ORDER, EMAIL]))
    for a, b, dt in [("JFK", "LAX", "2024-05-20"), ("SFO", "SEA", "2024-06-01"), ("ORD", "MIA", "2024-07-04"),
                     ("BOS", "DEN", "2024-05-11")]:
        cases.append(([{"role": "user", "content": f"Find me flights from {a} to {b} on {dt}."}], [FLIGHT, WEATHER]))
    for who in ["alice@example.com", "bob@example.com", "carol@example.com"]:
        cases.append(([{"role": "user", "content": f"Email {who} with subject 'Meeting' saying the meeting moved to 3pm."}],
                      [EMAIL, SEARCH]))
        cases.append(([{"role": "system", "content": "You are a helpful customer support agent."},
                       {"role": "user", "content": f"Please email {who} that their refund was processed."}], [EMAIL, ORDER]))
    return [dict(id=f"simple-{i}", cls="simple", messages=m, tools=t) for i, (m, t) in enumerate(cases)]


# ----------------------------------------------------------------------------- next_action rows (rendered as trained)
def row_class(sample):
    """Probe class of a next_action sample: commit / call / error_call (target is a call) or ask / reply / error."""
    target = sample["target_type"]
    reference = sample.get("reference") or {}
    if reference.get("type") == "call":
        if target == "error":
            return "error_call"
        last = sample["messages"][-1]
        return "commit" if last["role"] == "user" and WRITE_TOOL.match(reference.get("name", "")) else "call"
    return target if target in TEXT_CLASSES else "reply"


def next_action_cases(extra_infos, rows=0, seed=0):
    picked = [e for e in extra_infos if e["kind"] == "next_action"]
    random.Random(seed).shuffle(picked)
    if rows:
        picked = picked[:rows]
    cases = []
    for e in picked:
        sample = json.loads(e["sample_json"])
        cases.append(dict(id=f"na-{e['task_id']}", cls=row_class(sample), messages=sample["messages"],
                          tools=sample["tools"]))
    return cases


# ----------------------------------------------------------------------------- scoring
def classify(text, finish):
    calls = mosaic_calls(text)[1]
    sentences = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 20]
    repeat = max(collections.Counter(sentences).values() or [0]) >= 3
    length = finish == "length"
    return dict(call=bool(calls), name=calls[0][0] if calls else "", length=length, repeat=repeat,
                degenerate=length or repeat)


def is_right(cls, call):
    return call if cls in CALL_CLASSES else not call


def pass_summary(records):
    """{class: {n, right, call, degenerate}} for one pass, plus "next_action" over every next_action class."""
    groups = collections.defaultdict(list)
    for r in records:
        groups[r["cls"]].append(r)
        if r["cls"] != "simple":
            groups["next_action"].append(r)
    out = {}
    for cls, rs in groups.items():
        out[cls] = dict(n=len(rs), right=sum(is_right(r["cls"], r["call"]) for r in rs) / len(rs),
                        call=sum(r["call"] for r in rs) / len(rs), degenerate=sum(r["degenerate"] for r in rs) / len(rs))
    return out


def stability(passes):
    """Share of cases whose right/wrong outcome is the same in every pass."""
    by_id = collections.defaultdict(set)
    for records in passes:
        for r in records:
            by_id[r["id"]].add(is_right(r["cls"], r["call"]))
    return sum(len(v) == 1 for v in by_id.values()) / max(1, len(by_id))


# ----------------------------------------------------------------------------- compare
def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_passes(path):
    """(tag, [pass records]) from a probe file. Old single-model files hold {tag: {"greedy": {"records": ...}}},
    without ids or classes; their cases are matched by position and classed by group/target."""
    with open(path) as f:
        data = json.load(f)
    if data.get("version") == FORMAT_VERSION:
        return data["tag"], data["passes"]
    tag, res = next(iter(data.items()))
    old = {"call": "call", "ask": "ask", "reply": "reply", "error": "error", "text": "reply"}
    records = []
    for i, r in enumerate(res["greedy"]["records"]):
        cls = "simple" if r["group"] == "simple" else old.get(r["target"], "reply")
        records.append(dict(id=f"pos-{i}", cls=cls, call=r["call"], degenerate=r["length"] or r["repeat"]))
    return tag, [records]


def majority(passes):
    votes = collections.defaultdict(list)
    cls = {}
    for records in passes:
        for r in records:
            votes[r["id"]].append(is_right(r["cls"], r["call"]))
            cls[r["id"]] = r["cls"]
    return {i: (cls[i], sum(v) * 2 > len(v)) for i, v in votes.items()}


def compare(new_path, old_path):
    """Rows of the paired comparison: (class, n, new mean [min-max], old mean [min-max], n_new, n_old, p)."""
    new_tag, new = load_passes(new_path)
    old_tag, old = load_passes(old_path)
    vote_new, vote_old = majority(new), majority(old)
    shared = sorted(set(vote_new) & set(vote_old))
    if not shared:
        raise SystemExit(f"{new_path} and {old_path} share no cases: probe both on the same --data and --rows")
    rows = []
    for cls in CLASSES + ("next_action",):
        ids = [i for i in shared if vote_new[i][0] == cls or (cls == "next_action" and vote_new[i][0] != "simple")]
        if not ids:
            continue
        idset = set(ids)
        rates = lambda passes: [sum(is_right(r["cls"], r["call"]) for r in p if r["id"] in idset) / len(ids)
                                for p in passes]
        n_new = sum(vote_new[i][1] and not vote_old[i][1] for i in ids)
        n_old = sum(vote_old[i][1] and not vote_new[i][1] for i in ids)
        rows.append((cls, len(ids), rates(new), rates(old), n_new, n_old, mcnemar(n_new, n_old)))
    return new_tag, old_tag, rows, (stability(new), stability(old))


def print_compare(new_path, old_path):
    new_tag, old_tag, rows, (stab_new, stab_old) = compare(new_path, old_path)
    span = lambda v: f"{sum(v) / len(v):.2f} [{min(v):.2f}-{max(v):.2f}]" if len(v) > 1 else f"{v[0]:.2f}"
    print(f"right rate per pass, mean [range]; McNemar on the per-case majority vote ({new_tag} vs {old_tag})")
    print(f"{'class':<12}{'n':>5}  {new_tag:<20}{old_tag:<20}{'only ' + new_tag:>12}{'only ' + old_tag:>12}{'p':>7}")
    for cls, n, rn, ro, bn, bo, p in rows:
        print(f"{cls:<12}{n:>5}  {span(rn):<20}{span(ro):<20}{bn:>12}{bo:>12}{p:>7.3f}")
    print(f"cases with the same outcome in every pass: {new_tag} {stab_new:.2f}, {old_tag} {stab_old:.2f}")


# ----------------------------------------------------------------------------- run
def run(args):
    import pandas as pd
    from vllm import LLM, SamplingParams

    with open(args.template) as f:
        template = f.read()
    cases = simple_cases() + next_action_cases(list(pd.read_parquet(args.data).extra_info), args.rows)
    tag, path = args.spec.split("=", 1)
    llm = LLM(path, max_model_len=16384, seed=42, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    # simple: as evaluation renders it (the scaffold template adds the MOSAIC system prompt and refusal tool);
    # next_action: as training renders it (the rows already carry both), with the model's own template.
    prompts = [tok.apply_chat_template(c["messages"], tools=c["tools"], add_generation_prompt=True, tokenize=False,
                                       chat_template=template if c["cls"] == "simple" else None) for c in cases]
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    passes = []
    for k in range(args.repeats):
        order = list(range(len(cases)))
        random.Random(k).shuffle(order)
        outs = llm.generate([prompts[i] for i in order], params)
        records = [None] * len(cases)
        for i, g in zip(order, outs):
            o = g.outputs[0]
            records[i] = dict(id=cases[i]["id"], cls=cases[i]["cls"], text=o.text, **classify(o.text, o.finish_reason))
        passes.append(records)
        print(tag, f"pass {k}", json.dumps({c: round(s["right"], 3) for c, s in pass_summary(records).items()}),
              flush=True)
    out = dict(version=FORMAT_VERSION, tag=tag, model=path, data=args.data, rows=args.rows, template=args.template,
               max_tokens=MAX_TOKENS, summaries=[pass_summary(p) for p in passes], stability=stability(passes),
               passes=passes)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(tag, "stability", round(out["stability"], 3), "->", args.out, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--template", default="templates/mosaic.jinja")
    r.add_argument("--data", default="trainset/derived/safeutil_train.parquet")
    r.add_argument("--rows", type=int, default=0)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("out")
    r.add_argument("spec", help="TAG=MODEL_DIR")
    c = sub.add_parser("compare")
    c.add_argument("new")
    c.add_argument("old")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        print_compare(args.new, args.old)


if __name__ == "__main__":
    main()
