"""Forward-pass-only selection metrics (HANDOFF_lifecycle_baselines.md §4.0.4); never touches a test benchmark.

  measure  --model DIR (merged model, or a PEFT adapter dir) --out metrics.json [--n 200]
           val_benign_loss  mean assistant-token NLL on the first N of trainset/data/val_benign.jsonl
           refusal_nll      same on trainset/derived/select_refusal.jsonl (lower = still prefers refusing)
           harmful_nll      same on trainset/derived/defender_harmful_select.jsonl if it exists (higher = less compliant)
  pick     --reference undefended.json --candidate NAME=metrics.json ... --out selection.json
           among candidates with val_benign_loss <= 1.05 x reference, the lowest refusal_nll (ties -> highest harmful_nll)

Used for settings that need separate training runs (SafeInstr m, DPO beta); posthoc_defense.py --select runs the same
rule in-process for Safe LoRA K, Antidote ratio and LoRA scale.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import defense_common as dc  # noqa: E402


def load_model(path, revision, dtype, device):
    from transformers import AutoModelForCausalLM
    path = Path(path)
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        base = json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base, revision=None if Path(base).is_dir() else revision, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, str(path))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype)
    return model.to(device).eval()


def measure(args):
    import torch
    from transformers import AutoTokenizer
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = dc.selection_rows(tokenizer, args.max_length, args.n, args.val, args.refusal, args.harmful)
    model = load_model(args.model, args.revision, getattr(torch, args.dtype), device)
    metrics = dc.selection_metrics(model, rows, device)
    metrics["model"] = str(Path(args.model).resolve())
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "detail"}))


def pick(args):
    reference = json.loads(Path(args.reference).read_text())
    candidates = {}
    for item in args.candidate:
        name, path = item.split("=", 1)
        candidates[name] = json.loads(Path(path).read_text())
    chosen, ranked = dc.pick_setting(reference, candidates, args.tolerance)
    result = {"reference": args.reference, "chosen": chosen, "eligible_ranked": ranked, "tolerance": args.tolerance,
              "rule": "val_benign_loss <= tolerance x undefended; then lowest refusal_nll; ties -> highest harmful_nll",
              "results": {"reference": reference, **candidates}}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"chosen: {chosen} (eligible: {ranked})")
    if chosen is None:
        sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--model", required=True)
    m.add_argument("--revision", default=dc.DEFAULT_REVISION, help="hub revision of an adapter's base model")
    m.add_argument("--out", required=True)
    m.add_argument("--n", type=int, default=200)
    m.add_argument("--max-length", type=int, default=8192)
    m.add_argument("--val", default=dc.DEFAULT_VAL)
    m.add_argument("--refusal", default=dc.SELECT_REFUSAL)
    m.add_argument("--harmful", default=dc.HARMFUL_SELECT)
    m.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    m.add_argument("--device")
    p = sub.add_parser("pick")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", action="append", required=True, help="NAME=metrics.json (repeatable)")
    p.add_argument("--tolerance", type=float, default=1.05)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"measure": measure, "pick": pick}[args.command](args)


if __name__ == "__main__":
    main()
