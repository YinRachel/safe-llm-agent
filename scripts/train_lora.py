"""LoRA supervised fine-tuning of Qwen3-4B-Instruct-2507 on agent conversations.

Input: JSONL, one conversation per line, in OpenAI chat format:
  {"id", "source", "label", "tools": [...], "messages": [{"role", "content", "tool_calls"?}, ...]}
The checkpoint's own chat template renders each conversation (with its tools), and
loss is computed on assistant turns only (content, <tool_call> blocks and <|im_end|>).
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from datetime import datetime, timezone

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Revision the training set was rendered and length-checked against (trainset/manifest.json).
DEFAULT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
CONDITIONS = ("p00", "p05", "p10", "p20")  # harmful proportion of the training mix
ROLES = {"system", "user", "assistant", "tool"}
# Qwen chat templates open each assistant turn with this header and close it with <|im_end|>.
ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.DOTALL)
IGNORE_INDEX = -100


def load_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(example):
    """Return a reason string if the example is unusable, else None."""
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return "no messages"
    for message in messages:
        if message.get("role") not in ROLES:
            return f"unknown role {message.get('role')!r}"
    if messages[-1]["role"] != "assistant":
        return "does not end with an assistant turn"
    if not isinstance(example.get("tools", []), list):
        return "tools is not a list"
    return None


def normalize(messages):
    """Make tool-call arguments dicts, as the Qwen template renders them with tojson."""
    out = []
    for message in messages:
        message = dict(message)
        if message.get("content") is None:
            message["content"] = ""
        calls = []
        for call in message.get("tool_calls") or []:
            function = dict(call.get("function", call))
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    pass
            calls.append({"type": "function", "function": function})
        if calls:
            message["tool_calls"] = calls
        else:
            message.pop("tool_calls", None)
        out.append(message)
    return out


def assistant_char_spans(text):
    return [match.span(1) for match in ASSISTANT_TURN.finditer(text)]


def tokenize_example(example, tokenizer, max_length):
    """Render with the chat template and mask everything outside assistant turns."""
    text = tokenizer.apply_chat_template(
        normalize(example["messages"]), tools=example.get("tools") or None, tokenize=False
    )
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = assistant_char_spans(text)
    labels, span_index = [], 0
    for token_id, (start, _) in zip(encoding["input_ids"], encoding["offset_mapping"]):
        while span_index < len(spans) and start >= spans[span_index][1]:
            span_index += 1
        inside = span_index < len(spans) and spans[span_index][0] <= start
        labels.append(token_id if inside else IGNORE_INDEX)
    if len(encoding["input_ids"]) > max_length:
        return None
    if all(label == IGNORE_INDEX for label in labels):
        return None
    return {"input_ids": encoding["input_ids"], "labels": labels}


def build_dataset(examples, tokenizer, max_length, name):
    rows, skipped = [], {}
    for example in examples:
        reason = validate(example)
        row = None if reason else tokenize_example(example, tokenizer, max_length)
        if row is None:
            reason = reason or f"longer than {max_length} tokens or no assistant tokens"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        rows.append(row)
    lengths = sorted(len(row["input_ids"]) for row in rows)
    stats = {
        "kept": len(rows),
        "skipped": skipped,
        "tokens": sum(lengths),
        "trained_tokens": sum(sum(label != IGNORE_INDEX for label in row["labels"]) for row in rows),
        "max_len": lengths[-1] if lengths else 0,
        "p50_len": lengths[len(lengths) // 2] if lengths else 0,
    }
    print(f"[{name}] {json.dumps(stats)}", flush=True)
    return rows, stats


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch

        width = max(len(row["input_ids"]) for row in batch)
        def pad(key, value):
            return torch.tensor([row[key] + [value] * (width - len(row[key])) for row in batch])
        input_ids = pad("input_ids", self.pad_token_id)
        attention = torch.tensor([[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in batch])
        return {"input_ids": input_ids, "attention_mask": attention, "labels": pad("labels", IGNORE_INDEX)}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--revision", default=os.environ.get("MODEL_REVISION", DEFAULT_REVISION))
    parser.add_argument("--condition", choices=CONDITIONS, help="Train on trainset/data/train_<condition>.jsonl.")
    parser.add_argument("--train", help="Explicit training JSONL; overrides --condition.")
    parser.add_argument("--val", default="trainset/data/val_benign.jsonl", help="Optional; skipped if missing.")
    parser.add_argument("--output", required=True, help="New directory for the adapter and logs.")
    parser.add_argument("--max-length", type=int, default=8192, help="Longer conversations are dropped, not truncated.")
    parser.add_argument("--limit", type=int, help="Use only the first N training examples (smoke tests).")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device micro-batch size.")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge", action="store_true", help="Also write a merged full model to OUTPUT/merged for vLLM.")
    parser.add_argument("--dry-run", action="store_true", help="Tokenize and report statistics only (no GPU needed).")
    args = parser.parse_args(argv)
    if not args.train:
        if not args.condition:
            parser.error("give --condition or --train")
        args.train = f"trainset/data/train_{args.condition}.jsonl"
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.dry_run:
        sys.exit(f"Output directory {output} is not empty; choose a new one.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples = load_jsonl(args.train)
    random.Random(args.seed).shuffle(train_examples)
    if args.limit:
        train_examples = train_examples[: args.limit]
    train_rows, train_stats = build_dataset(train_examples, tokenizer, args.max_length, "train")
    val_rows, val_stats = [], None
    if args.val and Path(args.val).exists():
        val_rows, val_stats = build_dataset(load_jsonl(args.val), tokenizer, args.max_length, "val")
    if not train_rows:
        sys.exit("No usable training examples.")
    if args.dry_run:
        return

    import torch
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    transformers.set_seed(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "commit": git_commit(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","), task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    has_val = bool(val_rows)
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=10,
        eval_strategy="steps" if has_val else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        group_by_length=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_rows,
        eval_dataset=val_rows or None, data_collator=Collator(tokenizer.pad_token_id),
    )
    result = trainer.train()

    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {"train": result.metrics}
    if has_val:
        summary["eval"] = trainer.evaluate()
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output / "merged", safe_serialization=True)
        tokenizer.save_pretrained(output / "merged")
    print(f"Saved adapter to {adapter_dir}" + (f" and merged model to {output / 'merged'}" if args.merge else ""))


if __name__ == "__main__":
    main()
