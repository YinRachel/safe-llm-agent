#!/usr/bin/env python3
"""Direct-request HarmBench text baseline using local vLLM (no judge API)."""
import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def read_behaviors(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Behavior CSV is empty")
    seen = set()
    for row in rows:
        for key in ("BehaviorID", "Behavior", "FunctionalCategory", "Tags", "ContextString"):
            if key not in row or row[key] is None:
                raise ValueError(f"Missing CSV field: {key}")
        if not row["BehaviorID"].strip() or not row["Behavior"].strip():
            raise ValueError("BehaviorID and Behavior must be nonempty")
        if row["BehaviorID"] in seen:
            raise ValueError("Duplicate BehaviorID")
        seen.add(row["BehaviorID"])
        tags = {tag.strip() for tag in row["Tags"].split(",")}
        if "multimodal" in tags or row["FunctionalCategory"] == "multimodal":
            raise ValueError("This runner supports text behaviors only, not the multimodal benchmark")
        if ("context" in tags or row["FunctionalCategory"] == "contextual") and not row["ContextString"].strip():
            raise ValueError(f"Missing context for {row['BehaviorID']}")
    return rows


def select_behaviors(rows, limit=None, smoke=False):
    if smoke:
        selected, categories = [], set()
        for row in rows:
            if row["FunctionalCategory"] not in categories:
                selected.append(row)
                categories.add(row["FunctionalCategory"])
        return selected
    return rows[:limit]


def direct_prompt(row):
    # Matches upstream baselines/direct_request/direct_request.py, without a jailbreak.
    context = row["ContextString"]
    return f"{context}\n\n---\n\n{row['Behavior']}" if context else row["Behavior"]


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main(argv=None):
    # vLLM workers must not inherit CUDA state initialized by parent libraries.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmbench-repo", type=Path, required=True)
    parser.add_argument("--behaviors", type=Path, help="Default: upstream harmbench_behaviors_text_all.csv")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=positive)
    selection.add_argument("--smoke", action="store_true", help="One behavior per functional category")
    parser.add_argument("--model", default=MODEL, help="HF model or local model path")
    parser.add_argument("--revision", default=os.getenv("MODEL_REVISION"))
    parser.add_argument("--max-tokens", type=positive, default=512)
    parser.add_argument("--batch-size", type=positive, default=8)
    parser.add_argument("--max-model-len", type=positive, default=32768)
    parser.add_argument("--tensor-parallel-size", type=positive, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    source = args.behaviors or args.harmbench_repo / "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    rows = read_behaviors(source)
    selected = select_behaviors(rows, args.limit, args.smoke)
    # Check upstream assets before spending time on GPU inference.
    for row in selected:
        if "hash_check" in {tag.strip() for tag in row["Tags"].split(",")}:
            asset = args.harmbench_repo / "data/copyright_classifier_hashes" / (row["BehaviorID"] + ".pkl")
            if not asset.is_file():
                raise ValueError(f"Missing upstream copyright reference: {asset}")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "behaviors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    try:
        commit = subprocess.check_output(["git", "-C", str(args.harmbench_repo), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = dict(protocol="HarmBench DirectRequest; one greedy completion per behavior; native Qwen chat template",
        source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        upstream_commit=commit, source_rows=len(rows), selected_rows=len(selected),
        selection="smoke" if args.smoke else "limit" if args.limit else "all_rows_of_supplied_text_csv",
        functional_categories=dict(Counter(row["FunctionalCategory"] for row in selected)),
        model=args.model, revision=args.revision, seed=args.seed, temperature=0,
        worker_multiprocessing_method="spawn",
        max_tokens=args.max_tokens, max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        selected_ids=[row["BehaviorID"] for row in selected])
    save_json(args.output / "manifest.json", metadata)
    completions = {}
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from vllm import LLM, SamplingParams
        metadata["vllm"] = version("vllm")
        save_json(args.output / "manifest.json", metadata)
        model = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision,
            tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=args.seed)
        with (args.output / "responses.jsonl").open("x", encoding="utf-8") as stream:
            for offset in range(0, len(selected), args.batch_size):
                batch = selected[offset:offset + args.batch_size]
                prompts = [direct_prompt(row) for row in batch]
                outputs = model.chat([[{"role": "user", "content": prompt}] for prompt in prompts],
                                     sampling_params=params, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise RuntimeError("Unexpected generation count from vLLM")
                for row, prompt, output in zip(batch, prompts, outputs):
                    result = output.outputs[0]
                    completion = dict(test_case=prompt, generation=result.text,
                                      finish_reason=result.finish_reason, generated_tokens=len(result.token_ids))
                    completions[row["BehaviorID"]] = [completion]
                    stream.write(json.dumps(dict(behavior_id=row["BehaviorID"], **completion), ensure_ascii=False) + "\n")
                    stream.flush()
                save_json(args.output / "completions.json", completions)
                print(f"Generated {len(completions)}/{len(selected)} behaviors", flush=True)
        save_json(args.output / "generation_summary.json", dict(complete=True,
            behaviors=len(completions), expected_behaviors=len(selected),
            truncated=sum(sample[0]["finish_reason"] == "length" for sample in completions.values())))
    except Exception as error:
        save_json(args.output / "failure.json", dict(error=str(error), completed_behaviors=len(completions), expected_behaviors=len(selected)))
        raise
    print(f"Generation saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
