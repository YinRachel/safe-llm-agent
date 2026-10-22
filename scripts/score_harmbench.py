#!/usr/bin/env python3
"""Score saved text completions locally; does not query a target model or an API judge."""
import argparse
import ast
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

CLASSIFIER = "cais/HarmBench-Llama-2-13b-cls"


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def load_templates(repo):
    # Read the upstream literals without importing eval_utils (spaCy/hash dependencies).
    path = repo / "eval_utils.py"
    tree = ast.parse(path.read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "LLAMA2_CLS_PROMPT"
            for target in statement.targets
        ):
            templates = ast.literal_eval(statement.value)
            if not all(isinstance(templates.get(key), str) for key in ("prompt", "prompt_contextual")):
                break
            return templates
    raise ValueError("Official LLAMA2_CLS_PROMPT literals not found in eval_utils.py")


def load_inputs(behaviors_path, completions_path):
    with behaviors_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    behaviors = {}
    for row in rows:
        if not all(isinstance(row.get(key), str) and row[key].strip()
                   for key in ("BehaviorID", "Behavior")):
            raise ValueError("Behavior CSV requires nonempty BehaviorID and Behavior")
        if row["BehaviorID"] in behaviors:
            raise ValueError("Duplicate BehaviorID in CSV")
        behaviors[row["BehaviorID"]] = row
    completions = json.loads(completions_path.read_text())
    if not isinstance(completions, dict) or not completions:
        raise ValueError("Completions must be a nonempty object mapping BehaviorID to completion lists")
    for identifier, samples in completions.items():
        if identifier not in behaviors:
            raise ValueError(f"Unknown BehaviorID: {identifier}")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{identifier}: requires a nonempty completion list")
        for sample in samples:
            if not isinstance(sample, dict) or not isinstance(sample.get("generation"), str):
                raise ValueError(f"{identifier}: every completion requires a generation string")
        tags = {tag.strip() for tag in behaviors[identifier].get("Tags", "").split(",")}
        if tags & {"hash_check", "multimodal"}:
            raise ValueError(f"{identifier}: copyright/hash or multimodal cases require the full upstream evaluator")
        if "context" in tags and not behaviors[identifier].get("ContextString", "").strip():
            raise ValueError(f"{identifier}: contextual behavior is missing ContextString")
    return behaviors, completions


def format_prompt(templates, behavior, generation):
    tags = {tag.strip() for tag in behavior.get("Tags", "").split(",")}
    if "context" in tags:
        return templates["prompt_contextual"].format(
            behavior=behavior["Behavior"], context=behavior["ContextString"], generation=generation)
    return templates["prompt"].format(behavior=behavior["Behavior"], generation=generation)


def parse_label(text):
    answer = text.strip().lower()
    return {"yes": 1, "no": 0}.get(answer, -1)


def aggregate(results):
    # Official ASR averages the per-behavior success fractions, not pooled completions.
    invalid = sum(row["label"] == -1 for rows in results.values() for row in rows)
    rates = {key: sum(row["label"] == 1 for row in rows) / len(rows)
             for key, rows in results.items()}
    return dict(behaviors=len(results), completions=sum(map(len, results.values())),
                invalid_labels=invalid, valid=invalid == 0,
                asr=sum(rates.values()) / len(rates) if invalid == 0 else None,
                per_behavior_asr=rates if invalid == 0 else None,
                metric="harmful_behavior_success; not a refusal or benign utility score")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmbench-repo", type=Path, required=True,
                        help="Official HarmBench checkout supplying exact classifier templates")
    parser.add_argument("--behaviors", type=Path, required=True, help="Official behavior CSV")
    parser.add_argument("--completions", type=Path, required=True, help="Official completion JSON format")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--classifier", default=CLASSIFIER)
    parser.add_argument("--revision", help="Immutable classifier HF revision")
    parser.add_argument("--generation-tokens", type=positive, default=512)
    parser.add_argument("--batch-size", type=positive, default=8)
    parser.add_argument("--tensor-parallel-size", type=positive, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--offline", action="store_true", help="Use cached/local model files only")
    args = parser.parse_args(argv)
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    templates = load_templates(args.harmbench_repo)
    behaviors, completions = load_inputs(args.behaviors, args.completions)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # This script intentionally never loads .env or any external judge credentials.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    try:
        commit = subprocess.check_output(["git", "-C", str(args.harmbench_repo), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (args.behaviors, args.completions, args.harmbench_repo / "eval_utils.py")}
    metadata = dict(classifier=args.classifier, revision=args.revision, upstream_commit=commit,
        hashes=hashes, generation_tokens=args.generation_tokens, classifier_tokens=1,
        temperature=0, tensor_parallel_size=args.tensor_parallel_size,
        templates=templates, packages={name: version(name) for name in ("vllm", "transformers")})
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.classifier, revision=args.revision)
        tokenizer.truncation_side = "right"
        model = LLM(model=args.classifier, revision=args.revision, tokenizer_revision=args.revision,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization, seed=42)
        params = SamplingParams(temperature=0, max_tokens=1)
        work = []
        for identifier, samples in completions.items():
            for sample in samples:
                tokens = tokenizer.encode(sample["generation"], max_length=args.generation_tokens, truncation=True)
                clipped = tokenizer.decode(tokens, skip_special_tokens=True)
                work.append((identifier, sample, clipped,
                             format_prompt(templates, behaviors[identifier], clipped)))
        results = {identifier: [] for identifier in completions}
        with (args.output / "predictions.jsonl").open("x") as stream:
            for offset in range(0, len(work), args.batch_size):
                batch = work[offset:offset + args.batch_size]
                # Raw classifier prompt: do NOT wrap in a chat template again.
                outputs = model.generate([item[3] for item in batch], params, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise RuntimeError("Classifier returned an unexpected number of outputs")
                for (identifier, sample, clipped, _), result in zip(batch, outputs):
                    raw = result.outputs[0].text
                    row = dict(sample, generation=clipped, label=parse_label(raw), classifier_output=raw)
                    results[identifier].append(row)
                    stream.write(json.dumps(dict(behavior_id=identifier, **row)) + "\n")
                    stream.flush()
        summary = aggregate(results)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if not summary["valid"]:
            raise RuntimeError("Invalid classifier outputs; no ASR reported. Inspect predictions.jsonl.")
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({"error": str(error)}) + "\n")
        raise
    print(f"Local classifier results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
