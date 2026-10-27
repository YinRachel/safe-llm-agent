"""Roll out compliant tool-use trajectories for the attack-pool candidates (build step 6).

A locally served open-weight model (OpenAI-compatible endpoint, e.g. vLLM) acts as the agent inside AgentAlign's
simulated tool environment (third_party/AgentAlign), using each item's own `tools` and AgentAlign's own system
prompt. No external API provider is used. Output is harmful content: keep it under trainset/ (git-ignored).

  python scripts/data/generate_attack_trajectories.py --base-url http://127.0.0.1:8000/v1 --model NAME --model-revision SHA
Resumable: ids already present in the output file are skipped. scripts/data/build_trainset.py qc judges the results.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys
import threading

from build_trainset import GENERIC_SYSTEM_PROMPT, INTERIM, SEED, AGENTALIGN_REPO, read_jsonl

sys.path.insert(0, str(AGENTALIGN_REPO / "multi_step_trajectory_generation"))
from synthetic_tools import AVAILABLE_TOOLS  # noqa: E402



def run_item(client, args, item):
    """One rollout. Returns the OpenAI-style messages after the system prompt."""
    tools = [{"type": "function", "function": t["function"]} for t in item["tools"]]
    messages = [{"role": "system", "content": GENERIC_SYSTEM_PROMPT}, {"role": "user", "content": item["instruction"]}]
    for _ in range(args.max_steps):
        reply = client.chat.completions.create(model=args.model, messages=messages, tools=tools,
                                               temperature=args.temperature, max_tokens=args.max_tokens, seed=SEED)
        message = reply.choices[0].message
        entry = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            entry["content"] = message.content or None  # OpenAI convention; empty "" next to tool calls can 400 on Mistral templates
            entry["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                   for c in message.tool_calls]
        messages.append(entry)
        if not message.tool_calls:
            return messages[1:]
        for call in message.tool_calls:
            try:
                result = AVAILABLE_TOOLS[call.function.name](**json.loads(call.function.arguments))
                content = json.dumps(result, ensure_ascii=False)
            except Exception as error:  # surfaced to the model; QC drops trajectories that end in tool errors
                content = json.dumps({"error": f"Tool execution error: {error}"})
            messages.append({"role": "tool", "content": content, "tool_call_id": call.id})
    raise RuntimeError(f"no final answer within {args.max_steps} steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--model-revision", required=True, help="immutable HF commit of the generator weights (recorded)")
    parser.add_argument("--candidates", default=str(INTERIM / "attack_candidates.jsonl"))
    parser.add_argument("--output", default=str(INTERIM / "attack_generated.jsonl"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=20, help="max model turns per trajectory")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--extra-meta", default="{}", help="JSON with serving settings to record (vLLM version, quantization, ...)")
    parser.add_argument("--retry-errors", action="store_true", help="regenerate ids whose earlier rollout errored")
    parser.add_argument("--limit", type=int, help="smoke test: only the first N candidates")
    args = parser.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local")
    output = Path(args.output)
    if args.retry_errors and output.exists():  # drop failed rollouts so they are generated again
        kept = [r for r in read_jsonl(output) if "error" not in r]
        output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    done = {r["id"] for r in read_jsonl(output)} if output.exists() else set()
    todo = [c for c in read_jsonl(args.candidates) if c["id"] not in done][: args.limit]
    lock = threading.Lock()

    def work(item):
        try:
            record = {"id": item["id"], "messages": run_item(client, args, item)}
        except Exception as error:
            record = {"id": item["id"], "error": repr(error)[:300]}
        with lock, open(output, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return "error" not in record

    with ThreadPoolExecutor(args.workers) as pool:
        ok = sum(pool.map(work, todo))
    output.chmod(0o600)
    meta = {"generator_model": args.model, "generator_revision": args.model_revision, "temperature": args.temperature,
            "max_tokens": args.max_tokens, "seed": SEED, "max_steps": args.max_steps, "system_prompt": GENERIC_SYSTEM_PROMPT,
            "serving": json.loads(args.extra_meta), "environment": "AgentAlign synthetic_tools (third_party/AgentAlign)", "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(output).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"generated {ok}/{len(todo)} new trajectories -> {output}")


if __name__ == "__main__":
    main()
