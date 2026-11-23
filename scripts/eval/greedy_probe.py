"""Greedy-decoding probe for GRPO checkpoints: does the model call tools at temperature 0, or narrate and loop?
Run from the project root with the serving env (vLLM) on a GPU node; docs/safeutil.md (Version 5) has the v2-v4 numbers.
usage: greedy_probe.py OUT.json TAG=MODEL_DIR   (one model per process)"""
import collections, json, re, sys, random
import pandas as pd
from vllm import LLM, SamplingParams

TEMPLATE = open("templates/mosaic.jinja").read()
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
STOCK = {"type": "function", "function": {"name": "get_stock_price", "description": "Get the latest price of a stock.",
         "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}}
SEARCH = {"type": "function", "function": {"name": "search_web", "description": "Search the web.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
simple = []
for city in ["London", "Paris", "Tokyo", "New York", "Berlin", "Sydney"]:
    simple.append(([{"role": "user", "content": f"Use get_weather to look up the weather in {city}."}], [WEATHER]))
    simple.append(([{"role": "user", "content": f"What's the weather like in {city} right now?"}], [WEATHER, SEARCH]))
for t in ["AAPL", "MSFT", "NVDA"]:
    simple.append(([{"role": "user", "content": f"How much is {t} trading at?"}], [STOCK, SEARCH]))
simple.append(([{"role": "user", "content": "Find recent news about the Mars rover."}], [SEARCH, WEATHER]))
EMAIL = {"type": "function", "function": {"name": "send_email", "description": "Send an email.",
         "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"},
                        "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}}
ORDER = {"type": "function", "function": {"name": "get_order_details", "description": "Look up an order by id.",
         "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}
FLIGHT = {"type": "function", "function": {"name": "search_flights", "description": "Search direct flights.",
          "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                         "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}}
for o in ["#W1234567", "#W7654321", "#W5555555", "#W9081726"]:
    simple.append(([{"role": "user", "content": f"Can you check the status of my order {o}?"}], [ORDER, EMAIL]))
for a, b, dt in [("JFK", "LAX", "2024-05-20"), ("SFO", "SEA", "2024-06-01"), ("ORD", "MIA", "2024-07-04"), ("BOS", "DEN", "2024-05-11")]:
    simple.append(([{"role": "user", "content": f"Find me flights from {a} to {b} on {dt}."}], [FLIGHT, WEATHER]))
for who in ["alice@example.com", "bob@example.com", "carol@example.com"]:
    simple.append(([{"role": "user", "content": f"Email {who} with subject 'Meeting' saying the meeting moved to 3pm."}], [EMAIL, SEARCH]))
    simple.append(([{"role": "system", "content": "You are a helpful customer support agent."}, {"role": "user", "content": f"Please email {who} that their refund was processed."}], [EMAIL, ORDER]))

d = pd.read_parquet("trainset/derived/safeutil_train.parquet")
rows = [json.loads(e["sample_json"]) for e in d.extra_info if e["kind"] == "next_action"]
random.Random(0).shuffle(rows)
na = [(r["messages"], r["tools"], r["target_type"]) for r in rows[:120]]
cases = [("simple", m, t, "call") for m, t in simple] + [("next_action", m, t, tt) for m, t, tt in na]

CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def classify(text, finish):
    calls = []
    for c in CALL.findall(text):
        try:
            calls.append(json.loads(c))
        except json.JSONDecodeError:
            pass
    sents = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 20]
    rep = max(collections.Counter(sents).values() or [0])
    pre = text.find("<tool_call>")
    return dict(call=bool(calls), length=finish == "length", repeat=rep >= 3,
                prefix_words=len(text[:pre].split()) if pre >= 0 else None)


out = {}
for spec in sys.argv[2:]:
    tag, path = spec.split("=", 1)
    llm = LLM(path, max_model_len=16384, seed=42, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    # simple: as evaluation renders it (scaffold template adds the MOSAIC system prompt and refusal tool);
    # next_action: as training renders it (the rows already carry both), with the model's own template.
    prompts = [tok.apply_chat_template(m, tools=t, chat_template=TEMPLATE if g == "simple" else None,
                                       add_generation_prompt=True, tokenize=False) for g, m, t, _ in cases]
    res = {}
    for name, sp in [("greedy", SamplingParams(temperature=0, max_tokens=512)),
                     ("t1", SamplingParams(temperature=1.0, max_tokens=512, seed=0))]:
        gens = llm.generate(prompts, sp)
        recs = []
        for (group, _, _, target), g in zip(cases, gens):
            o = g.outputs[0]
            recs.append(dict(group=group, target=target, text=o.text, **classify(o.text, o.finish_reason)))
        summ = {}
        for group in ["simple", "next_action"]:
            for target in ["call", "text"]:
                rs = [r for r in recs if r["group"] == group and (r["target"] == target)]
                if rs:
                    pw = sorted(r["prefix_words"] for r in rs if r["prefix_words"] is not None)
                    summ[f"{group}/{target}"] = dict(n=len(rs), call=sum(r["call"] for r in rs) / len(rs),
                        length=sum(r["length"] for r in rs) / len(rs), repeat=sum(r["repeat"] for r in rs) / len(rs),
                        prefix_median=pw[len(pw) // 2] if pw else None)
        res[name] = dict(summary=summ, records=recs)
        print(tag, name, json.dumps(summ), flush=True)
    out[tag] = res
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    json.dump(out, open(sys.argv[1], "w"), indent=1)
