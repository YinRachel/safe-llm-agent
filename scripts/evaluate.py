#!/usr/bin/env python3
"""Benign baselines (AgentHarm benign, custom chat), the AgentHarm harmful safety task,
and the agentic capability benchmarks BFCL v3 multi-turn and tau-bench (tau2)."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import subprocess
import sys

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARM_BENCH_NOTE = (
    "The linked walledai/HarmBench card documents harmful behavior prompts, "
    "not an official benign split. A HarmBench benign score is unavailable. "
    "Use agentharm for official benign agent evaluation or chat with a reviewed "
    "benign JSONL (reported as custom_benign_chat, never HarmBench). "
    "For harmful direct-request evaluation, use scripts/viking_harmbench.sbatch."
)
# The four BFCL V3 multi-turn leaderboard categories; composite is an extra upstream split.
BFCL_MULTI_TURN = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param",
                   "multi_turn_long_context"]
BFCL_MULTI_TURN_ALL = BFCL_MULTI_TURN + ["multi_turn_composite"]
# retail and airline are the original tau-bench domains, in the tau2-bench release.
TAU_DOMAINS = ["retail", "airline", "telecom", "banking"]
TAU_DEFAULT_DOMAINS = ["retail", "airline"]
BENCHMARK_LABELS = {"agentharm": "agentharm_benign", "agentharm-harmful": "agentharm_harmful",
                    "chat": "custom_benign_chat",
                    "bfcl-multiturn": "bfcl_v3_multi_turn", "tau-bench": "tau2_bench"}


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def judge_settings(args):
    """Return non-secret routing metadata and the credential's environment name."""
    explicit = getattr(args, "judge", None)
    provider = getattr(args, "judge_provider", None) or os.getenv("JUDGE_PROVIDER")
    if not provider:
        provider = ("compatible" if explicit and explicit.startswith("openai-api/")
                    else "openai" if explicit else
                    "runware" if os.getenv("RUNWARE_API_KEY") else "openai")
    endpoint = getattr(args, "judge_base_url", None) or os.getenv("JUDGE_BASE_URL")
    if provider == "runware":
        model_id = explicit or os.getenv("RUNWARE_JUDGE_MODEL")
        if not model_id:
            raise ValueError("Set RUNWARE_JUDGE_MODEL to a verified Runware AIR model ID. "
                             "Runware's documented catalog does not list GPT-4o; no substitute is selected.")
        if model_id.startswith(("openai/", "openai-api/")):
            raise ValueError("Runware requires its AIR model ID, not an Inspect model identifier.")
        settings = dict(provider=provider, model="openai-api/runware/" + model_id,
                        base_url=endpoint or "https://api.runware.ai/v1", key_env="RUNWARE_API_KEY")
        if model_id == "openai:gpt@6-luna":
            settings.update(family="gpt-6-luna", reasoning_effort="none")
        return settings
    if provider == "openai":
        return dict(provider=provider, model=explicit or "openai/gpt-4o-2024-08-06",
                    base_url=endpoint or "https://api.openai.com/v1", key_env="OPENAI_API_KEY")
    if provider == "compatible":
        if not explicit or not endpoint or not explicit.startswith("openai-api/"):
            raise ValueError("Compatible judges require --judge openai-api/SERVICE/MODEL and --judge-base-url.")
        return dict(provider=provider, model=explicit, base_url=endpoint, key_env="JUDGE_API_KEY")
    raise ValueError("JUDGE_PROVIDER must be runware, openai, or compatible")


def make_judge(args):
    from inspect_ai.model import GenerateConfig, ModelInfo, get_model, set_model_info
    settings = judge_settings(args)
    key = os.getenv(settings["key_env"])
    if not key and settings["provider"] != "compatible":
        raise ValueError(f"Set {settings['key_env']} in .env or the environment.")
    config = GenerateConfig(temperature=0, max_tokens=1024, max_retries=2)
    if settings.get("family"):
        # AIR IDs hide the OpenAI family from Inspect's request-shape detection.
        # Register the canonical service ID without changing the model sent on the wire.
        service_id = settings["model"].split("/", 2)[2]
        set_model_info(service_id, ModelInfo(family=settings["family"]))
        config.reasoning_effort = settings["reasoning_effort"]
    return get_model(settings["model"], base_url=settings["base_url"], api_key=key or "local",
                     config=config), settings


def judge_preflight(args):
    """Use the same Inspect transport as scoring; fail on invalid judge outputs."""
    from inspect_ai.model import ChatMessageUser, GenerateConfig
    from inspect_evals.agentharm.scorer import DEFAULT_REFUSAL_PROMPT
    judge, settings = make_judge(args)

    async def check():
        cases = [
            (DEFAULT_REFUSAL_PROMPT.format(generation="I cannot assist with that on ethical grounds."), "yes", True),
            (DEFAULT_REFUSAL_PROMPT.format(generation="Hello! Two plus two equals four."), "no", True),
            ("Does the following text state that two plus two equals four? Answer only pass or fail. 2 + 2 = 4.", "pass", False),
        ]
        for prompt, expected, refusal in cases:
            result = await judge.generate([ChatMessageUser(content=prompt)],
                                          config=GenerateConfig(temperature=0, max_tokens=512))
            answer = (result.completion or "").strip().lower().replace("*", "")
            if refusal:
                answer = answer.rsplit("answer:", 1)[-1].strip()
            if answer != expected:
                raise RuntimeError("Judge preflight returned an invalid or incorrect refusal/semantic answer.")
    asyncio.run(check())
    print(f"Judge preflight passed: {settings['model']} at {settings['base_url']}")


def read_benign(path):
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {number}: expected an object")
        if row.get("benign") is not True:
            raise ValueError(f"line {number}: requires explicit benign: true after human review")
        for key in ("id", "prompt", "source"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"line {number}: requires nonempty string {key}")
        if row["id"] in seen:
            raise ValueError(f"duplicate id: {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("empty benign dataset")
    return rows


def manifest(args, output):
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    info = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), config=config,
        python=sys.version, git_commit=commit,
        packages={d.metadata['Name']: d.version for d in distributions()},
        model_revision=os.getenv("MODEL_REVISION", "unrecorded"),
        benchmark=BENCHMARK_LABELS[args.benchmark],
    )
    if args.benchmark == "chat":
        info["dataset_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.benchmark in ("agentharm", "agentharm-harmful"):
        info["judge"] = judge_settings(args)
    if args.benchmark == "tau-bench":
        info["user_simulator"] = judge_settings(args)
    if args.benchmark == "bfcl-multiturn":
        from inspect_evals.bfcl.bfcl import BFCL_GITHUB_COMMIT
        info["bfcl_data_commit"] = BFCL_GITHUB_COMMIT
    (output / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")


def client(args):
    from openai import OpenAI
    return OpenAI(base_url=args.base_url, api_key=os.getenv("TARGET_API_KEY", "local"),
                  timeout=180, max_retries=2)


def preflight(args):
    """Verify chat and parsed tool-call transport before downloading benchmark data."""
    api = client(args)
    ids = [m.id for m in api.models.list().data]
    if args.model not in ids:
        raise RuntimeError(f"Requested model {args.model!r} absent from server: {ids}")
    api.chat.completions.create(model=args.model,
        messages=[{"role": "user", "content": "Reply with hello."}], max_tokens=32,
        temperature=0, seed=args.seed)
    tool = {"type": "function", "function": {"name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}
    reply = api.chat.completions.create(model=args.model, tools=[tool], tool_choice="auto",
        messages=[{"role": "user", "content": "Use get_weather to look up the weather in London."}],
        max_tokens=256, temperature=0, seed=args.seed)
    calls = reply.choices[0].message.tool_calls
    if not calls or calls[0].function.name != "get_weather":
        raise RuntimeError("No parsed weather tool call. Check vLLM Hermes parser/chat template.")
    json.loads(calls[0].function.arguments)
    print("Preflight passed: model available, chat and automatic tool calls work.")


def run_agentharm(args, output, harmful=False):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model
    from inspect_evals.agentharm import agentharm, agentharm_benign
    # Explicit endpoints keep local solver traffic separate from judge traffic.
    target = get_model("openai-api/local/" + args.model, base_url=args.base_url,
                       api_key=os.getenv("TARGET_API_KEY", "local"))
    judge, _ = make_judge(args)
    task = (agentharm if harmful else agentharm_benign)(
        split=args.split, refusal_judge=judge, semantic_judge=judge,
        agent_kwargs={"tool_choice": "auto", "user_prompt_template": "empty",
                      "prompt_technique": "standard"})
    snapshot = "".join(json.dumps(sample.model_dump(mode="json")) + "\n"
                       for sample in task.dataset)
    (output / "dataset.jsonl").write_text(snapshot)
    (output / "dataset.sha256").write_text(
        hashlib.sha256(snapshot.encode()).hexdigest() + "\n")
    logs = inspect_eval(task, model=target, log_dir=str(output / "inspect"),
                        limit=args.limit, temperature=0, seed=args.seed,
                        max_tokens=args.max_tokens, max_connections=args.concurrency)
    summary = []
    judge_errors = []
    for log in logs:
        log_judge_errors = []
        # Upstream refusal judging catches exceptions and can otherwise count them as non-refusals.
        for sample in getattr(log, "samples", None) or []:
            for decision in (sample.metadata or {}).get("refusal_explanations", []):
                if "Error in RefusalJudgeLLM:" in decision.get("explanation", ""):
                    log_judge_errors.append(str(sample.id))
        judge_errors.extend(log_judge_errors)
        summary.append(dict(status=log.status,
            valid=log.status == "success" and not log_judge_errors,
            refusal_judge_error_sample_ids=log_judge_errors,
            results=log.results.model_dump(mode="json") if log.results else None,
            error=log.error.model_dump(mode="json") if log.error else None))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if judge_errors:
        (output / "judge_errors.json").write_text(json.dumps(dict(sample_ids=judge_errors), indent=2) + "\n")
        raise RuntimeError("Refusal judge errors detected; results are invalid. See judge_errors.json.")
    if not logs or any(log.status != "success" for log in logs):
        raise RuntimeError("Evaluation incomplete; inspect logs and summary before reporting results.")


def target_model(args):
    from inspect_ai.model import get_model
    return get_model("openai-api/local/" + args.model, base_url=args.base_url,
                     api_key=os.getenv("TARGET_API_KEY", "local"))


def snapshot_datasets(tasks, output):
    """Save the exact loaded samples (tagged by task) and their checksum."""
    snapshot = "".join(
        json.dumps(dict(task=task.name, **sample.model_dump(mode="json"))) + "\n"
        for task in tasks for sample in task.dataset)
    (output / "dataset.jsonl").write_text(snapshot)
    (output / "dataset.sha256").write_text(hashlib.sha256(snapshot.encode()).hexdigest() + "\n")


def summarize_logs(logs, output):
    """Write summary.json; raise unless every task finished without sample errors."""
    summary = []
    for log in logs:
        samples = getattr(log, "samples", None) or []
        errored = [str(s.id) for s in samples if getattr(s, "error", None)]
        # Context overflow ends a sample early and scores it as a failure: surface it.
        exhausted = [str(s.id) for s in samples
                     if getattr(getattr(s, "output", None), "stop_reason", None) == "model_length"]
        summary.append(dict(
            task=getattr(getattr(log, "eval", None), "task", None), status=log.status,
            valid=log.status == "success" and not errored,
            samples=len(samples), errored_sample_ids=errored,
            context_exhausted_sample_ids=exhausted,
            results=log.results.model_dump(mode="json") if log.results else None,
            error=log.error.model_dump(mode="json") if log.error else None))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not logs or not all(item["valid"] for item in summary):
        raise RuntimeError("Evaluation incomplete or contains sample errors; inspect logs and summary before reporting results.")


def run_bfcl_multiturn(args, output):
    """BFCL v3 multi-turn: stateful backends scored by final state and execution results (no judge)."""
    from inspect_ai import eval as inspect_eval
    from inspect_evals.bfcl import bfcl
    task = bfcl(categories=args.categories)
    snapshot_datasets([task], output)
    logs = inspect_eval(task, model=target_model(args), log_dir=str(output / "inspect"),
                        limit=args.limit, temperature=0, seed=args.seed,
                        max_tokens=args.max_tokens, max_connections=args.concurrency)
    summarize_logs(logs, output)


def check_user_simulator(args):
    """tau-bench needs a working, separate user-simulator model; fail before spending GPU time."""
    from inspect_ai.model import ChatMessageUser, GenerateConfig
    judge, settings = make_judge(args)
    result = asyncio.run(judge.generate(
        [ChatMessageUser(content="Reply with the single word: ready")],
        config=GenerateConfig(temperature=0, max_tokens=512)))
    if not (result.completion or "").strip():
        raise RuntimeError("User-simulator preflight returned an empty completion.")
    print(f"User-simulator preflight passed: {settings['model']} at {settings['base_url']}")


def run_tau_bench(args, output):
    """tau-bench (inspect_evals tau2): the target model is the agent; the judge-configured model plays the user."""
    from inspect_ai import eval as inspect_eval
    import inspect_evals.tau2 as tau2
    user_model, _ = make_judge(args)
    tasks = [getattr(tau2, f"tau2_{domain}")(shuffle=False, message_limit=args.message_limit)
             for domain in args.domains]
    snapshot_datasets(tasks, output)
    logs = inspect_eval(tasks, model=target_model(args), model_roles={"user": user_model},
                        log_dir=str(output / "inspect"), limit=args.limit, temperature=0,
                        seed=args.seed, max_tokens=args.max_tokens,
                        max_connections=args.concurrency)
    summarize_logs(logs, output)


def run_chat(args, output, rows):
    api = client(args)
    # Exclusive output directory prevents mixing runs. Each response is flushed immediately.
    truncated = 0
    with (output / "responses.jsonl").open("x") as stream:
        for row in rows[:args.limit]:
            response = api.chat.completions.create(model=args.model,
                messages=[{"role": "user", "content": row["prompt"]}],
                temperature=0, max_tokens=args.max_tokens, seed=args.seed)
            choice = response.choices[0]
            truncated += choice.finish_reason == "length"
            stream.write(json.dumps(dict(**row, response=choice.message.content,
                finish_reason=choice.finish_reason,
                usage=response.usage.model_dump() if response.usage else None)) + "\n")
            stream.flush()
    count = len(rows[:args.limit])
    (output / "summary.json").write_text(json.dumps(dict(
        benchmark="custom_benign_chat", completed=count, truncated=truncated,
        scored=False, note="Generation only. Human review or a validated benign utility rubric is required; non-refusal is not task success."
    ), indent=2) + "\n")


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["preflight", "judge-preflight", "agentharm", "agentharm-harmful",
                                                  "bfcl-multiturn",
                                                  "tau-bench", "harmbench", "chat"])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge", help="Inspect identifier, or a Runware AIR ID when using runware")
    parser.add_argument("--judge-provider", choices=["runware", "openai", "compatible"])
    parser.add_argument("--judge-base-url")
    parser.add_argument("--split", choices=["val", "test_public"], default="test_public")
    parser.add_argument("--limit", type=positive)
    parser.add_argument("--max-tokens", type=positive, default=4096)
    parser.add_argument("--concurrency", type=positive, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", choices=BFCL_MULTI_TURN_ALL, default=BFCL_MULTI_TURN,
                        help="bfcl-multiturn categories (default: the four leaderboard V3 categories)")
    parser.add_argument("--domains", nargs="+", choices=TAU_DOMAINS, default=TAU_DEFAULT_DOMAINS,
                        help="tau-bench domains (default: retail airline)")
    parser.add_argument("--message-limit", type=positive,
                        help="tau-bench maximum messages per sample (upstream default 100)")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.benchmark == "harmbench":
        parser.error(HARM_BENCH_NOTE)
    rows = None
    if args.benchmark == "chat":
        if not args.data:
            parser.error("chat requires --data with reviewed benign JSONL")
        rows = read_benign(args.data)
    if args.benchmark == "preflight":
        preflight(args)
        return
    if args.benchmark == "judge-preflight":
        judge_preflight(args)
        return
    if args.benchmark in ("agentharm", "agentharm-harmful", "tau-bench"):
        judge_settings(args)  # Validate routing before creating artifacts or contacting the target.
    output = args.output or Path("results") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest(args, output)
    try:
        if args.benchmark in ("agentharm", "agentharm-harmful"):
            judge_preflight(args)
            preflight(args)
            run_agentharm(args, output, harmful=args.benchmark == "agentharm-harmful")
        elif args.benchmark == "bfcl-multiturn":
            preflight(args)
            run_bfcl_multiturn(args, output)
        elif args.benchmark == "tau-bench":
            check_user_simulator(args)
            preflight(args)
            run_tau_bench(args, output)
        else:
            run_chat(args, output, rows)
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
        raise
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
