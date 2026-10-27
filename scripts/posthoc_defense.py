"""Post-fine-tuning defences (HANDOFF_lifecycle_baselines.md §4.P1/P2/P4, §4.x).

  python scripts/posthoc_defense.py --method safelora --adapter results/lora-p10-JOB/adapter --output DIR [--select]
  python scripts/posthoc_defense.py --method scale    --adapter results/lora-p10-JOB/adapter --output DIR [--select]
  python scripts/posthoc_defense.py --method antidote --merged  results/lora-p10-JOB/merged  --output DIR [--select]

safelora  P1 Safe LoRA: Hsu et al., NeurIPS 2024, arXiv:2405.16833; follows IBM/SafeLoRA@dc756cf model.py
          (get_aligned_matrix, projected_weighted). Per LoRA module, V = W_aligned - W_unaligned and C = V V^T / ||V||_F
          (upstream's Frobenius-normalised, non-orthonormal projector); B' = C B, cos = cos(C B A, B A). The K modules
          with the lowest cosine (or every module with cos <= threshold) get B'. Modules are matched by *name*, and
          C B is computed as V (V^T B) / ||V||_F (same quantity, no out x out matrix is ever formed).
antidote  P2 Antidote: Huang et al., ICML 2025, arXiv:2408.09600 (re-implemented from the paper; the authors' group's
          git-disl/Antidote@ca92dc5 has no licence file and was not copied). Wanda score S_ij = |W_ij| sqrt(sum_t x_tj^2)
          on harmful calibration trajectories; per linear layer the top-ratio fraction of entries is zeroed.
scale     P4 LoRA down-scaling: theta = theta_base + s * Delta_W_LoRA (WiSE-FT / RESTA-style interpolation), by scaling
          lora_B before merging.

--select sweeps the method's knob on this model and picks a value with the §4.0.4 rule (val_benign_loss within 5 % of
the undefended model, then lowest refusal_nll, ties -> highest harmful_nll); without --select the given value is used.
Outputs: merged/ (vLLM-ready, tokenizer copied), adapter/ (safelora, scale), manifest.json, cost.json, <method>.json,
selection.json.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import defense_common as dc  # noqa: E402

ALIGNED = ("Qwen/Qwen3-4B-Instruct-2507", dc.DEFAULT_REVISION)
UNALIGNED = ("Qwen/Qwen3-4B-Base", "906bfd4b4dc7f14ee4320094d8b41684abff8539")
SAFELORA_K = (13, 25, 50, 126)  # ~5/10/20/50 % of 36 layers x 7 modules
ANTIDOTE_RATIOS = (0.01, 0.02, 0.05, 0.1)
SCALES = (0.25, 0.5, 0.75)
LINEAGE_NOTE = (
    "Neither model card declares base_model. Qwen3-4B-Instruct-2507 lists 'Pretraining & Post-training', native 262,144 "
    "context and rope_theta 5e6; Qwen3-4B-Base lists 'Pretraining', 32,768 context and rope_theta 1e6. All 252 target "
    "module shapes match (checked 2026-10-26 from hub safetensors headers), but V = W_instruct - W_base may contain an "
    "unreleased long-context continued-pretraining delta as well as alignment.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("safelora", "antidote", "scale"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", help="PEFT adapter dir of the fine-tuned model (safelora, scale)")
    source.add_argument("--merged", help="merged fine-tuned model dir (antidote)")
    parser.add_argument("--base-model", help="model the adapter was trained on (default: adapter_config.json)")
    parser.add_argument("--revision", default=dc.DEFAULT_REVISION, help="hub revision of --base-model (ignored for dirs)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--select", action="store_true", help="sweep the knob and pick with the selection rule")
    parser.add_argument("--select-n", type=int, default=200)
    parser.add_argument("--select-val", default=dc.DEFAULT_VAL)
    parser.add_argument("--select-refusal", default=dc.SELECT_REFUSAL)
    parser.add_argument("--select-harmful", default=dc.HARMFUL_SELECT, help="optional; skipped if missing")
    parser.add_argument("--max-length", type=int, default=dc.MAX_LENGTH, help="selection-metric rows longer than this are dropped")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--device", default=None, help="default: cuda if available")
    g = parser.add_argument_group("safelora")
    g.add_argument("--aligned", default=ALIGNED[0])
    g.add_argument("--aligned-revision", default=ALIGNED[1])
    g.add_argument("--unaligned", default=UNALIGNED[0])
    g.add_argument("--unaligned-revision", default=UNALIGNED[1])
    g.add_argument("--accept-lineage", action="store_true",
                   help="run with the default Instruct-2507 / Qwen3-4B-Base pair despite the undeclared lineage")
    g.add_argument("--report-only", action="store_true", help="write per-module cosines and ||V||/||W|| only")
    g.add_argument("--select-type", choices=("number", "threshold"), default="number")
    g.add_argument("--num-proj", type=int, default=25)
    g.add_argument("--threshold", type=float, default=0.5)
    g.add_argument("--sweep-k", default=",".join(map(str, SAFELORA_K)))
    g = parser.add_argument_group("antidote")
    g.add_argument("--ratio", type=float, default=0.05)
    g.add_argument("--sweep-ratios", default=",".join(map(str, ANTIDOTE_RATIOS)))
    g.add_argument("--calibration", default="trainset/derived/defender_harmful.jsonl")
    g.add_argument("--calibration-n", type=int, default=256)
    g.add_argument("--calibration-max-tokens", type=int, default=2048)
    g.add_argument("--target-modules", default=dc.TARGET_MODULES)
    g = parser.add_argument_group("scale")
    g.add_argument("--scale", type=float, default=0.5)
    g.add_argument("--sweep-scales", default=",".join(map(str, SCALES)))
    args = parser.parse_args(argv)
    if args.method in ("safelora", "scale") and not args.adapter:
        parser.error(f"--method {args.method} needs --adapter")
    if args.method == "antidote" and not args.merged:
        parser.error("--method antidote needs --merged")
    return args


# ----------------------------------------------------------------------------- Safe LoRA core (unit-tested)
def lora_modules(named_params):
    """{module path: (lora_A param, lora_B param)} from a PeftModel's named_parameters(), matched by name."""
    out = {}
    for name, param in named_params:
        for side in ("lora_A", "lora_B"):
            marker = f".{side}."
            if marker in name:
                module = name.split(marker)[0]
                module = module[len("base_model.model."):] if module.startswith("base_model.model.") else module
                out.setdefault(module, {})[side] = param
    return {m: (p["lora_A"], p["lora_B"]) for m, p in sorted(out.items())}


def project_module(A, B, V):
    """Upstream projected_weighted for one module (fp32): B' = C B with C = V V^T / ||V||_F; cos(C B A, B A) and
    upstream's distance 1 / (1 + ||C B A - B A||)."""
    import torch
    A, B, V = A.float(), B.float(), V.float()
    projected_B = V @ (V.T @ B) / torch.linalg.norm(V)
    new, old = projected_B @ A, B @ A
    cos = torch.nn.functional.cosine_similarity(new.reshape(1, -1), old.reshape(1, -1)).item()
    dist = 1.0 / (1.0 + torch.linalg.norm(new - old).item())
    return projected_B, cos, dist


def choose_modules(cosines, select_type, k=None, threshold=None):
    """Lowest-cosine modules: exactly K (ties by name) or all with cos <= threshold."""
    if select_type == "number":
        return sorted(sorted(cosines, key=lambda m: (cosines[m], m))[:k])
    return sorted(m for m, c in cosines.items() if c <= threshold)


def safelora_scores(modules, weight_pair, device):
    """For each LoRA module: projected B (CPU), cosine, distance, ||V||/||W_aligned||. `weight_pair(module)` returns
    (W_aligned, W_unaligned) for `<module>.weight`."""
    import torch
    out = {}
    for module, (A, B) in modules.items():
        aligned, unaligned = weight_pair(module)
        if aligned.shape != unaligned.shape or aligned.shape != (B.shape[0], A.shape[1]):
            sys.exit(f"[safelora] shape mismatch at {module}: aligned {tuple(aligned.shape)}, unaligned "
                     f"{tuple(unaligned.shape)}, LoRA {(B.shape[0], A.shape[1])}; stop and report (do not substitute)")
        aligned = aligned.to(device, torch.float32)
        V = aligned - unaligned.to(device, torch.float32)
        projected, cos, dist = project_module(A.detach().to(device), B.detach().to(device), V)
        out[module] = {"projected_B": projected.cpu(), "cos": cos, "dist": dist,
                       "rel_norm_V": (torch.linalg.norm(V) / torch.linalg.norm(aligned)).item()}
        del V, aligned
    return out


class SafetensorWeights:
    """Lazy `<name>.weight` reader over a model's safetensors shards (local dir or HF cache snapshot, no download)."""

    def __init__(self, repo, revision):
        from safetensors import safe_open
        path = Path(repo)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo, revision=revision, local_files_only=True,
                                          allow_patterns=["*.safetensors", "*.json"]))
        index = path / "model.safetensors.index.json"
        files = sorted({f for f in json.loads(index.read_text())["weight_map"].values()}) if index.exists() else ["model.safetensors"]
        self.handles = {f: safe_open(str(path / f), framework="pt") for f in files}
        self.where = {key: f for f, h in self.handles.items() for key in h.keys()}
        self.source = {"repo": repo, "revision": revision, "path": str(path)}

    def __call__(self, name):
        if name not in self.where:
            sys.exit(f"[safelora] {name} not found in {self.source['repo']}")
        return self.handles[self.where[name]].get_tensor(name)


# ----------------------------------------------------------------------------- Antidote core (unit-tested)
def wanda_prune_(weight, act_sqnorm, ratio):
    """Zero (in place) the top-`ratio` fraction of entries of `weight` [out, in] by |W_ij| * sqrt(act_sqnorm_j).
    Returns the number of zeroed entries."""
    import torch
    k = int(round(ratio * weight.numel()))
    if k == 0:
        return 0
    score = weight.detach().abs().float() * act_sqnorm.to(weight.device, torch.float32).sqrt()[None, :]
    index = torch.topk(score.flatten(), k, sorted=False).indices
    weight.data.view(-1)[index] = 0
    return k


class ActivationStats:
    """Context manager: forward pre-hooks accumulating sum_t x_tj^2 (fp32) for every matching nn.Linear; hooks are
    removed on exit."""

    def __init__(self, model, suffixes):
        import torch
        self.model, self.suffixes, self.stats, self.handles, self.tokens = model, tuple(suffixes), {}, [], 0
        self.linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in self.suffixes}

    def __enter__(self):
        import torch
        for name, module in self.linears.items():
            self.stats[name] = torch.zeros(module.in_features, dtype=torch.float32, device=module.weight.device)

            def hook(mod, inputs, name=name):
                x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
                self.stats[name] += (x * x).sum(0)
            self.handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


# ----------------------------------------------------------------------------- model helpers
def device_and_dtype(args):
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device), getattr(torch, args.dtype)


def load_causal_lm(path, revision, dtype, device):
    from transformers import AutoModelForCausalLM
    rev = None if Path(path).is_dir() else revision
    model = AutoModelForCausalLM.from_pretrained(path, revision=rev, torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval()


def load_tokenizer(path, revision):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, revision=None if Path(path).is_dir() else revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_adapter_model(args, device, dtype):
    from peft import PeftModel
    config = json.loads((dc.require(Path(args.adapter) / "adapter_config.json")).read_text())
    base = args.base_model or config["base_model_name_or_path"]
    model = PeftModel.from_pretrained(load_causal_lm(base, args.revision, dtype, device), args.adapter, is_trainable=False)
    return model.to(device).eval(), base, config


def save_merged(model, tokenizer, output):
    model.save_pretrained(Path(output) / "merged", safe_serialization=True)
    tokenizer.save_pretrained(Path(output) / "merged")


class Selector:
    """Selection metrics for the undefended model and each candidate setting (§4.0.4)."""

    def __init__(self, args, tokenizer, device, cost):
        self.rows = dc.selection_rows(tokenizer, args.max_length, args.select_n, args.select_val, args.select_refusal,
                                       args.select_harmful)
        self.device, self.cost, self.results = device, cost, {}

    def measure(self, name, model):
        metrics = dc.selection_metrics(model, self.rows, self.device)
        self.cost.add(selection_tokens=sum(len(r["input_ids"]) for rows in self.rows.values() for r in rows))
        self.results[name] = metrics
        print(f"[select] {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "detail"), flush=True)
        return metrics

    def pick(self, reference, output, knob):
        candidates = {k: v for k, v in self.results.items() if k != reference}
        chosen, ranked = dc.pick_setting(self.results[reference], candidates)
        if chosen is None:
            sys.exit(f"[select] no setting keeps val_benign_loss within 5% of the undefended model; see {output}/selection.json")
        (Path(output) / "selection.json").write_text(json.dumps({
            "knob": knob, "reference": reference, "chosen": chosen, "eligible_ranked": ranked,
            "rule": "val_benign_loss <= 1.05 x undefended; then lowest refusal_nll; ties -> highest harmful_nll",
            "results": self.results}, indent=2))
        return chosen


# ----------------------------------------------------------------------------- methods
def run_safelora(args, output, cost):
    import torch
    from safetensors.torch import save_file  # noqa: F401  (import check before the long pass)
    if (args.aligned, args.unaligned) == (ALIGNED[0], UNALIGNED[0]) and not (args.accept_lineage or args.report_only):
        sys.exit("[safelora] stopped: " + LINEAGE_NOTE + " Re-run with --report-only to inspect ||V||/||W|| per module, "
                 "and pass --accept-lineage once the reference pair is approved (see docs/baselines_status.md).")
    device, dtype = device_and_dtype(args)
    model, base, config = load_adapter_model(args, device, dtype)
    tokenizer = load_tokenizer(args.adapter, args.revision)
    modules = lora_modules(model.named_parameters())
    targets = set(config["target_modules"])
    assert all(m.split(".")[-1] in targets for m in modules), "projected modules must be LoRA target modules"
    aligned, unaligned = SafetensorWeights(args.aligned, args.aligned_revision), SafetensorWeights(args.unaligned, args.unaligned_revision)
    scores = safelora_scores(modules, lambda m: (aligned(f"{m}.weight"), unaligned(f"{m}.weight")), device)
    cosines = {m: s["cos"] for m, s in scores.items()}
    report = {"aligned": aligned.source, "unaligned": unaligned.source, "lineage_note": LINEAGE_NOTE,
              "modules": len(modules), "pdst_upstream": sum(s["dist"] for s in scores.values()) / len(scores),
              "per_module": {m: {k: v for k, v in s.items() if k != "projected_B"} for m, s in scores.items()}}
    if args.report_only:
        (output / "safelora.json").write_text(json.dumps(report, indent=2))
        print(f"[safelora] report only: {len(modules)} modules -> {output / 'safelora.json'}")
        return None
    original = {m: B.detach().clone() for m, (_, B) in modules.items()}

    def apply(chosen):
        with torch.no_grad():
            for m, (_, B) in modules.items():
                B.copy_((scores[m]["projected_B"] if m in chosen else original[m]).to(B.device, B.dtype))

    if args.select:
        selector = Selector(args, tokenizer, device, cost)
        selector.measure("undefended", model)
        for k in [int(x) for x in args.sweep_k.split(",")]:
            apply(choose_modules(cosines, "number", k=k))
            selector.measure(f"K={k}", model)
        chosen_name = selector.pick("undefended", output, "num_proj")
        chosen = choose_modules(cosines, "number", k=int(chosen_name.split("=")[1]))
    else:
        chosen = choose_modules(cosines, args.select_type, args.num_proj, args.threshold)
    apply(chosen)
    report.update(select_type="number" if args.select else args.select_type, projected=chosen, n_projected=len(chosen),
                  pdst_projected=(sum(scores[m]["dist"] for m in chosen) / len(chosen)) if chosen else None)
    (output / "safelora.json").write_text(json.dumps(report, indent=2))
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    save_merged(model.merge_and_unload(), tokenizer, output)
    print(f"[safelora] projected {len(chosen)}/{len(modules)} modules; Pdst (upstream, all modules) {report['pdst_upstream']:.4f}")
    return {"base_model": base}


def run_scale(args, output, cost):
    import torch
    device, dtype = device_and_dtype(args)
    model, base, _ = load_adapter_model(args, device, dtype)
    tokenizer = load_tokenizer(args.adapter, args.revision)
    modules = lora_modules(model.named_parameters())
    original = {m: B.detach().clone() for m, (_, B) in modules.items()}

    def apply(s):
        with torch.no_grad():
            for m, (_, B) in modules.items():
                B.copy_(original[m] * s)

    scale = args.scale
    if args.select:
        selector = Selector(args, tokenizer, device, cost)
        selector.measure("s=1.0", model)
        for s in [float(x) for x in args.sweep_scales.split(",")]:
            apply(s)
            selector.measure(f"s={s}", model)
        scale = float(selector.pick("s=1.0", output, "scale").split("=")[1])
    apply(scale)
    (output / "scale.json").write_text(json.dumps({"scale": scale, "modules": len(modules), "base_model": base}, indent=2))
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    save_merged(model.merge_and_unload(), tokenizer, output)
    print(f"[scale] s={scale} applied to {len(modules)} LoRA modules")
    return {"base_model": base}


def calibration_batches(args, tokenizer):
    rows = dc.load_jsonl(dc.require(args.calibration, "run defense_data.py qc-defender (needs the generation job)"))
    held = dc.DERIVED / "defender_harmful_select.jsonl"
    if held.exists():
        clash = {r["id"] for r in rows} & {r["id"] for r in dc.load_jsonl(held)}
        assert not clash, f"calibration overlaps the selection subset ({len(clash)} ids)"
    attack = dc.TRAINSET / "interim/attack_candidates.jsonl"
    if attack.exists():
        clash = {r["id"] for r in rows} & {r["id"] for r in dc.load_jsonl(attack)}
        assert not clash, f"calibration contains attack candidates ({len(clash)} ids)"
    rows = rows[: args.calibration_n]
    out = []
    for row in rows:
        text = tokenizer.apply_chat_template(dc.normalize(row["messages"]), tools=row.get("tools") or None, tokenize=False)
        out.append(tokenizer(text, add_special_tokens=False)["input_ids"][: args.calibration_max_tokens])
    return [r["id"] for r in rows], out


def run_antidote(args, output, cost):
    import torch
    device, dtype = device_and_dtype(args)
    model = load_causal_lm(args.merged, args.revision, dtype, device)
    tokenizer = load_tokenizer(args.merged, args.revision)
    ids, batches = calibration_batches(args, tokenizer)
    suffixes = args.target_modules.split(",")
    with ActivationStats(model, suffixes) as stats, torch.no_grad():
        for input_ids in batches:
            tensor = torch.tensor([input_ids], device=device)
            model(input_ids=tensor, attention_mask=torch.ones_like(tensor))
            cost.add(forward_passes=1, calibration_tokens=len(input_ids))
    assert not any(m._forward_pre_hooks for m in stats.linears.values()), "activation hooks were not removed"
    linears = stats.linears
    backup = {n: m.weight.detach().to("cpu", copy=True) for n, m in linears.items()}

    def apply(ratio):
        counts = {}
        with torch.no_grad():
            for name, module in linears.items():
                module.weight.copy_(backup[name].to(module.weight.device))
                counts[name] = wanda_prune_(module.weight, stats.stats[name], ratio)
        return counts

    ratio = args.ratio
    if args.select:
        selector = Selector(args, tokenizer, device, cost)
        selector.measure("ratio=0", model)
        for r in [float(x) for x in args.sweep_ratios.split(",")]:
            apply(r)
            selector.measure(f"ratio={r}", model)
        ratio = float(selector.pick("ratio=0", output, "ratio").split("=")[1])
    counts = apply(ratio)
    (output / "antidote.json").write_text(json.dumps({
        "ratio": ratio, "layers": len(linears), "zeroed_total": sum(counts.values()),
        "zeroed_per_layer": counts, "calibration": dc.input_record(args.calibration),
        "calibration_ids_sha256": dc.ids_sha256(ids), "calibration_n": len(ids),
        "calibration_tokens": sum(len(b) for b in batches), "max_tokens_per_trajectory": args.calibration_max_tokens,
        "score": "|W_ij| * sqrt(sum_t x_tj^2), all tokens of the rendered (prompt + harmful answer) trajectory",
        "granularity": "unstructured, top-ratio per linear layer"}, indent=2))
    save_merged(model, tokenizer, output)
    print(f"[antidote] ratio {ratio}: zeroed {sum(counts.values())} weights in {len(linears)} linear layers")
    return {"base_model": args.merged}


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        sys.exit(f"Output directory {output} is not empty; choose a new one.")
    output.mkdir(parents=True)
    cost = dc.Cost()
    inputs = [("adapter", Path(args.adapter) / "adapter_model.safetensors" if args.adapter else None),
              ("calibration", args.calibration if args.method == "antidote" else None)]
    dc.write_manifest(output, args, {"method": args.method, "stage": "post-fine-tuning"}, inputs=inputs)
    runner = {"safelora": run_safelora, "antidote": run_antidote, "scale": run_scale}[args.method]
    extra = runner(args, output, cost)
    if extra:
        manifest = json.loads((output / "manifest.json").read_text())
        manifest.update(extra)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cost.write(output)


if __name__ == "__main__":
    main()
