"""Shared training utilities for the life-cycle defence baselines (HANDOFF_lifecycle_baselines.md §4.0.2).

Everything data-related (validation, chat-template rendering, assistant-only masking, collation) is imported from
train_lora.py so every defence trains on exactly the tokens the undefended reference run trains on.

Loss normalisation: the reference run uses the stock HF Trainer, whose causal-LM loss is
sum(token NLL) / (assistant tokens in the whole accumulation window), passed down as `num_items_in_batch`. Token-level
defence losses (Lisa, constrained SFT, Booster's alignment term) keep that convention, so they reduce *exactly* to
`--method sft` in their degenerate settings. Only DPO, whose loss is per pair, sets `model_accepts_loss_kwargs = False`
(HF then divides each micro-batch loss by the accumulation steps).
"""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from train_lora import (  # noqa: E402,F401  (re-exported for the defence scripts)
    CONDITIONS, DEFAULT_MODEL, DEFAULT_REVISION, IGNORE_INDEX, MAX_LENGTH, Collator, assistant_char_spans,
    build_dataset, git_commit, load_jsonl, normalize, tokenize_example, validate,
)

PROJECT = Path(__file__).resolve().parents[1]
TRAINSET = Path(os.environ.get("TRAINSET_ROOT", PROJECT / "trainset"))
DERIVED = TRAINSET / "derived"
DEFAULT_VAL = "trainset/data/val_benign.jsonl"
SELECT_REFUSAL = "trainset/derived/select_refusal.jsonl"
HARMFUL_SELECT = "trainset/derived/defender_harmful_select.jsonl"
TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


# ----------------------------------------------------------------------------- arguments
def add_shared_args(parser, defaults=None):
    """train_lora.py's flags with the same names and defaults (tests compare them). `defaults` overrides a few
    per-method defaults (DPO's recipe) without changing the flag set."""
    d = {"lr": 1e-4, "warmup_ratio": 0.03, "epochs": 2, "eval_steps": 50, "save_steps": 50, "val": DEFAULT_VAL}
    d.update(defaults or {})
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--revision", default=os.environ.get("MODEL_REVISION", DEFAULT_REVISION),
                        help="Hub revision; ignored (not passed) when --model is a local directory.")
    parser.add_argument("--condition", choices=CONDITIONS, help="Train on trainset/data/train_<condition>.jsonl.")
    parser.add_argument("--train", help="Explicit training JSONL; overrides --condition.")
    parser.add_argument("--val", default=d["val"], help="Optional; skipped if missing.")
    parser.add_argument("--output", required=True, help="New directory for the adapter and logs.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Longer conversations are dropped, not truncated.")
    parser.add_argument("--limit", type=int, help="Use only the first N training examples (smoke tests).")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=TARGET_MODULES)
    parser.add_argument("--epochs", type=float, default=d["epochs"])
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=d["lr"])
    parser.add_argument("--warmup-ratio", type=float, default=d["warmup_ratio"])
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device micro-batch size.")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=d["eval_steps"])
    parser.add_argument("--save-steps", type=int, default=d["save_steps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge", action="store_true", help="Also write a merged full model to OUTPUT/merged for vLLM.")
    parser.add_argument("--dry-run", action="store_true", help="Tokenize and report statistics only (no GPU needed).")


def resolve_train(args, parser):
    if not args.train:
        if not args.condition:
            parser.error("give --condition or --train")
        args.train = f"trainset/data/train_{args.condition}.jsonl"
    return args


def model_revision(args):
    """The default revision belongs to the hub id; a local (merged) directory has no revisions."""
    return None if Path(args.model).is_dir() else args.revision


def require(path, hint=""):
    if not path or not Path(path).exists():
        sys.exit(f"Missing input {path}" + (f": {hint}" if hint else ""))
    return Path(path)


# ----------------------------------------------------------------------------- model
def load_tokenizer(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_revision(args))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def lora_config(args):
    from peft import LoraConfig
    return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=args.target_modules.split(","), task_type="CAUSAL_LM")


def load_model_and_lora(args, dtype=None):
    """Same dtype / attention / LoRA config as train_lora.py."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=model_revision(args),
                                                 torch_dtype=dtype or torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, lora_config(args))
    model.print_trainable_parameters()
    return model


def training_arguments(args, output, has_val, group_by_length=True, metric="eval_loss"):
    """train_lora.py's TrainingArguments, verbatim."""
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(Path(output) / "checkpoints"),
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
        metric_for_best_model=metric if has_val else None,
        group_by_length=group_by_length,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )


# ----------------------------------------------------------------------------- provenance and cost
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def input_record(path):
    """sha256 of a data file and of its id list (JSONL with an `id` per line)."""
    path = Path(path)
    info = {"path": str(path), "sha256": sha256_file(path)}
    if path.suffix == ".jsonl":
        ids = [row.get("id", "") for row in load_jsonl(path)]
        info.update(n=len(ids), ids_sha256=ids_sha256(ids))
    return info


def versions():
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def write_manifest(output, args, extra=None, inputs=()):
    """train_lora.py's manifest schema plus input sha256s and method-specific fields."""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "commit": git_commit(),
        "versions": versions(),
        "inputs": {name: input_record(path) for name, path in inputs if path and Path(path).exists()},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    manifest.update(extra or {})
    (Path(output) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[args] {json.dumps(manifest['args'], sort_keys=True)}", flush=True)
    return manifest


class Cost:
    """Counters for cost.json. Trainers increment forward/backward explicitly; CostCallback adds steps and time."""

    def __init__(self):
        self.start = time.time()
        self.counts = {"optimizer_steps": 0, "micro_steps": 0, "forward_passes": 0, "backward_passes": 0,
                       "trained_tokens": 0, "calibration_tokens": 0}
        self.extra = {}

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.counts[key] = self.counts.get(key, 0) + int(value)

    def write(self, output):
        import torch
        cuda = torch.cuda.is_available()
        info = {"wall_clock_s": round(time.time() - self.start, 2), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None, **self.counts, **self.extra}
        (Path(output) / "cost.json").write_text(json.dumps(info, indent=2))
        return info


def cost_callback(cost):
    from transformers import TrainerCallback

    class CostCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            cost.counts["optimizer_steps"] = state.global_step

    return CostCallback()


def counting_trainer_class():
    """Trainer that only counts (numerically identical to the stock Trainer used by train_lora.py)."""
    from transformers import Trainer

    class CountingTrainer(Trainer):
        cost = None
        passes_per_micro_step = (1, 1)  # (forward, backward)

        def training_step(self, model, inputs, num_items_in_batch=None):
            if self.cost is not None:
                forward, backward = self.passes_per_micro_step
                tokens = sum(int((v != IGNORE_INDEX).sum()) for k, v in inputs.items() if k.endswith("labels"))
                self.cost.add(micro_steps=1, forward_passes=forward, backward_passes=backward, trained_tokens=tokens)
            return super().training_step(model, inputs, num_items_in_batch)

        def accumulation_steps(self):
            return getattr(self, "current_gradient_accumulation_steps", None) or self.args.gradient_accumulation_steps

    return CountingTrainer


class SeededCycle:
    """Endless seeded reshuffling over rows (one micro-batch per draw), independent of torch's RNG."""

    def __init__(self, rows, label, seed):
        import random
        self.rows, self.rng, self.order, self.drawn = rows, random.Random(f"{seed}:{label}"), [], 0

    def next(self):
        if not self.order:
            self.order = list(range(len(self.rows)))
            self.rng.shuffle(self.order)
        self.drawn += 1
        return self.rows[self.order.pop()]


# ----------------------------------------------------------------------------- token log-probs
def token_logps(logits, labels, chunk=2048):
    """Per-token log p(label) for a causal LM (logits[t] predicts labels[t+1]); returns (logps, mask) of shape
    [B, L-1], logps = 0 where masked. Float32 log-softmax in sequence chunks to bound memory."""
    import torch
    labels = labels[:, 1:]
    mask = labels != IGNORE_INDEX
    safe = labels.masked_fill(~mask, 0)
    pieces = []
    for start in range(0, labels.shape[1], chunk):
        part = logits[:, start:start + chunk].float().log_softmax(-1)
        pieces.append(part.gather(-1, safe[:, start:start + chunk, None]).squeeze(-1))
    return torch.cat(pieces, 1) * mask, mask


def turn_positions(mask, scope="turn"):
    """0-based position of each trained token within its assistant turn (scope='turn': restarts at each contiguous run
    of trained tokens, i.e. each <|im_start|>assistant span) or within the whole conversation ('conversation')."""
    import torch
    mask = mask.long()
    count = mask.cumsum(1)
    if scope == "conversation":
        return (count - 1).clamp(min=0) * mask
    if scope != "turn":
        raise ValueError(scope)
    starts = mask * (1 - torch.nn.functional.pad(mask, (1, 0))[:, :-1])  # first token of each run
    run_base = torch.cummax(torch.where(starts.bool(), count, torch.zeros_like(count)), 1).values
    return (count - run_base) * mask


@contextmanager
def reference_mode(model):
    """Adapter off, dropout off, no grad: the reference policy for constrained SFT and DPO."""
    import torch
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), model.disable_adapter():
            yield
    finally:
        model.train(was_training)


# ----------------------------------------------------------------------------- selection metrics (§4.0.4)
def mean_nll(model, rows, device, autocast=None):
    """Token-weighted mean assistant-token NLL over pre-tokenised rows (training masking), and the per-example mean."""
    import torch
    total, tokens, per_example = 0.0, 0, []
    model.eval()
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([row["input_ids"]], device=device)
            labels = torch.tensor([row["labels"]], device=device)
            with (autocast or nullcontext)():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            logps, mask = token_logps(logits, labels)
            nll, n = float(-logps.sum()), int(mask.sum())
            total, tokens = total + nll, tokens + n
            per_example.append(nll / max(n, 1))
    return {"nll": total / max(tokens, 1), "nll_per_example": sum(per_example) / max(len(per_example), 1),
            "n": len(rows), "tokens": tokens}


def selection_rows(tokenizer, max_length, n_val=200, val=DEFAULT_VAL, refusal=SELECT_REFUSAL, harmful=HARMFUL_SELECT):
    """The fixed held-out slices. Missing val/refusal files are an error; the harmful slice is optional."""
    out = {}
    out["val_benign_loss"], _ = build_dataset(load_jsonl(require(val))[:n_val], tokenizer, max_length, "select:val")
    out["refusal_nll"], _ = build_dataset(load_jsonl(require(refusal, "run defense_data.py build")), tokenizer, max_length,
                                          "select:refusal")
    if harmful and Path(harmful).exists():
        out["harmful_nll"], _ = build_dataset(load_jsonl(harmful), tokenizer, max_length, "select:harmful")
    return out


def selection_metrics(model, rows_by_metric, device, autocast=None):
    metrics, detail = {}, {}
    for name, rows in rows_by_metric.items():
        result = mean_nll(model, rows, device, autocast)
        metrics[name], detail[name] = result["nll"], result
    metrics["detail"] = detail
    return metrics


def pick_setting(reference, candidates, tolerance=1.05):
    """§4.0.4 rule: among settings whose val_benign_loss <= tolerance x the undefended model's, the lowest refusal_nll
    (ties -> highest harmful_nll). `candidates` maps setting -> metrics. Returns (chosen setting or None, eligible)."""
    bound = tolerance * reference["val_benign_loss"]
    eligible = {k: m for k, m in candidates.items() if m["val_benign_loss"] <= bound}
    if not eligible:
        return None, []
    key = lambda k: (round(eligible[k]["refusal_nll"], 6), -eligible[k].get("harmful_nll", 0.0))
    return min(eligible, key=key), sorted(eligible, key=key)


def finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None
