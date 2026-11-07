"""Rebuild RUN/merged from RUN/adapter, e.g. after scratch was wiped and the merged/ symlink points at nothing.

  python scripts/post-training/merge_adapter.py results/glm4-9B_LoRA/p00-123 [RUN ...]

The base model and revision come from RUN/manifest.json (train_lora.py, train_defense.py, posthoc_defense.py runs),
else from the adapter's base_model_name_or_path. The merge is the one the training scripts do (bf16 base,
PeftModel.merge_and_unload, safetensors), and the tokenizer, with its chat template, is the one saved in adapter/.
merged/ goes back where it was: into the target of a dangling symlink, else under --scratch-root (SCRATCH_OUTPUT_ROOT)
with a symlink as in training, else in place. A record of the rebuild is written to RUN/merged_rebuild.json.
Antidote runs have no adapter (they prune the merged weights) and cannot be rebuilt this way.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_lora import link_to_scratch  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_model(run, adapter):
    """(base model, revision or None) for RUN, as its training run loaded it."""
    args = {}
    if (run / "manifest.json").is_file():
        args = json.loads((run / "manifest.json").read_text()).get("args", {})
    config = json.loads((adapter / "adapter_config.json").read_text())
    model = args.get("base_model") or args.get("model") or config["base_model_name_or_path"]
    if args.get("method") == "antidote":
        sys.exit(f"{run} is an Antidote run: its merged weights were pruned directly and there is no adapter to merge.")
    local = Path(model)
    if local.is_symlink() and not local.exists():
        sys.exit(f"The base model {model} of {run} is itself a symlink into scratch that no longer exists; rebuild it first.")
    return model, (None if local.is_dir() else args.get("revision"))


def merged_target(run, scratch_root):
    """Where merged/ is written: a dangling symlink's target, a new scratch folder, or RUN/merged."""
    link = run / "merged"
    if link.is_symlink():
        target = Path(os.readlink(link))
        target = target if target.is_absolute() else (run / target)
        if target.exists() and any(target.iterdir()):
            sys.exit(f"{link} -> {target} is not empty; nothing to rebuild.")
        target.mkdir(parents=True, exist_ok=True)
        return target, {"merged": str(target)}
    if link.exists():
        if any(link.iterdir()):
            sys.exit(f"{link} is not empty; nothing to rebuild.")
        return link, {}
    links = link_to_scratch(run, scratch_root, ["merged"])
    return (Path(links["merged"]) if links else link), links


def rebuild(run, scratch_root=None, device=None):
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run = Path(run)
    adapter = run / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        sys.exit(f"No adapter in {run}: nothing to rebuild merged/ from.")
    model_id, revision = base_model(run, adapter)
    target, links = merged_target(run, scratch_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run}] merging {adapter} into {model_id}" + (f" @ {revision}" if revision else "") + f" on {device} -> {target}",
          flush=True)
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16).to(device)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(target)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": model_id, "revision": revision, "device": device, "merged": str(target), "scratch_dirs": links,
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (run / "merged_rebuild.json").write_text(json.dumps(record, indent=2))
    print(f"[{run}] rebuilt merged/ ({target})", flush=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run folders holding adapter/ (and usually manifest.json).")
    parser.add_argument("--scratch-root", default=os.environ.get("SCRATCH_OUTPUT_ROOT") or None,
                        help="Put a missing merged/ under this folder and symlink it into the run, as in training.")
    parser.add_argument("--device", help="cuda or cpu (default: cuda when available).")
    args = parser.parse_args(argv)
    for run in args.runs:
        rebuild(run, args.scratch_root, args.device)


if __name__ == "__main__":
    main()
