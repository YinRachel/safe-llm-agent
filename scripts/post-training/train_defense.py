"""One entry point for the training-based defence baselines (HANDOFF_lifecycle_baselines.md §4.0.3).

  python scripts/post-training/train_defense.py --method {sft,lisa,booster,csft,dpo} --train JSONL --output DIR [--model M] [--merge]

  sft      plain LoRA SFT; reproduces train_lora.py exactly (anchor). A0 = sft on derived/align_sft.jsonl.
  lisa     F2 Lisa: bi-state optimisation (guide / user phases at optimizer-step granularity) + proximal term.
           Huang et al., NeurIPS 2024, arXiv:2405.18641; follows git-disl/Lisa@68e3659 trainer.py::LisaTrainer.
  booster  A1 Booster: attenuate the harmful-loss reduction rate of a simulated harmful step.
           Huang et al., ICLR 2025, arXiv:2409.01586; follows git-disl/Booster@8f14566 trainer.py::BoosterAlignmentTrainer.
  csft     F3 token-wise constrained SFT ("deep alignment"), reference = the model with the adapter disabled.
           Qi et al., ICLR 2025, arXiv:2406.05946; follows Unispac/shallow-vs-deep-alignment@fb1749e
           finetuning_buckets/trainer/trainer.py::soft_sft_loss / get_beta_list.
  dpo      A2 / P3 safety DPO (Rafailov et al., NeurIPS 2023, arXiv:2305.18290) on per-record tools, assistant-only
           log-probs; reference = the model with the adapter disabled.

Shared flags have train_lora.py's names and defaults (DPO changes lr/warmup/epochs/eval cadence, §4.A2). Outputs:
adapter/, merged/ (--merge), manifest.json, summary.json, log_history.json, cost.json, method_state.json,
selection.json (--select).
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).parent))
import defense_common as dc  # noqa: E402
from defense_common import IGNORE_INDEX  # noqa: E402

METHODS = ("sft", "lisa", "booster", "csft", "dpo")
DPO_DEFAULTS = {"lr": 5e-6, "warmup_ratio": 0.1, "epochs": 1, "eval_steps": 20, "save_steps": 20,
                "val": "trainset/derived/dpo_pairs_val.jsonl"}
PASSES = {"sft": (1, 1), "lisa": (1, 1), "booster": (3, 3), "csft": (2, 1), "dpo": (2, 1)}  # (forward, backward) per micro-step


def parse_args(argv=None):
    method = argparse.ArgumentParser(add_help=False)
    method.add_argument("--method", choices=METHODS, required=True)
    known, _ = method.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     parents=[method])
    dc.add_shared_args(parser, DPO_DEFAULTS if known.method == "dpo" else None)
    parser.add_argument("--select", action="store_true", help="Compute the §4.0.4 selection metrics of the result.")
    parser.add_argument("--select-n", type=int, default=200, help="val_benign items used by the selection metrics")
    parser.add_argument("--select-val", default=dc.DEFAULT_VAL)
    parser.add_argument("--select-refusal", default=dc.SELECT_REFUSAL)
    parser.add_argument("--select-harmful", default=dc.HARMFUL_SELECT, help="optional; skipped if missing")
    g = parser.add_argument_group("lisa")
    g.add_argument("--lisa-guide", default="trainset/derived/lisa_guide.jsonl")
    g.add_argument("--lisa-align-steps", type=int, default=5, help="optimizer steps per alignment (guide) phase")
    g.add_argument("--lisa-finetune-steps", type=int, default=45, help="optimizer steps per fine-tune (user) phase")
    g.add_argument("--lisa-rho", type=float, default=1.0)
    g.add_argument("--lisa-prox-warmup", type=float, default=0.1, help="no proximal term before this fraction of steps")
    g = parser.add_argument_group("booster")
    g.add_argument("--booster-harmful", default="trainset/derived/defender_harmful.jsonl")
    g.add_argument("--booster-lambda", type=float, default=5.0)
    g.add_argument("--booster-alpha", type=float, default=0.1)
    g = parser.add_argument_group("csft")
    g.add_argument("--csft-beta", type=float, default=0.1)
    g.add_argument("--csft-bias-factor", type=float, default=20.0)
    g.add_argument("--csft-first-token-bias-factor", type=float, default=5.0)
    g.add_argument("--csft-bias-length", type=int, default=5)
    g.add_argument("--csft-position-scope", choices=("turn", "conversation"), default="turn",
                   help="turn: beta schedule restarts at every assistant turn (agent adaptation); conversation: upstream")
    g = parser.add_argument_group("dpo")
    g.add_argument("--dpo-beta", type=float, default=0.1)
    g.add_argument("--no-precompute-ref", dest="precompute_ref", action="store_false",
                   help="compute reference log-probs on the fly instead of once at start")
    g.add_argument("--sft-weight", type=float, default=0.0, help="RPO-style NLL on chosen (ablation only)")
    args = parser.parse_args(argv)
    if args.method == "dpo" and not args.train:
        args.train = "trainset/derived/dpo_pairs.jsonl"
    return dc.resolve_train(args, parser)


# ----------------------------------------------------------------------------- method trainers
def lisa_phase(step, align_steps, finetune_steps):
    """'alignment' for the first `align_steps` optimizer steps of every (align + finetune) cycle, else 'finetune'."""
    if align_steps <= 0:
        return "finetune"
    if finetune_steps <= 0:
        return "alignment"
    return "alignment" if step % (align_steps + finetune_steps) < align_steps else "finetune"


def csft_betas(positions, beta, bias_factor, first_factor, bias_length):
    """Upstream get_beta_list applied per position: beta*first_factor at 0, beta*bias_factor at 1..bias_length-1,
    beta afterwards."""
    import torch
    betas = torch.full(positions.shape, beta, dtype=torch.float32, device=positions.device)
    if bias_length > 0:
        betas = torch.where(positions < bias_length, torch.full_like(betas, beta * bias_factor), betas)
        betas = torch.where(positions == 0, torch.full_like(betas, beta * first_factor), betas)
    return betas


def csft_weights(policy_logps, ref_logps, betas):
    import torch
    return (2 * (1 - torch.sigmoid(betas * (policy_logps - ref_logps)))).detach().clamp(min=1e-3)


def dpo_loss(policy_c, policy_r, ref_c, ref_r, beta):
    import torch.nn.functional as F
    margin = (policy_c - ref_c) - (policy_r - ref_r)
    return -F.logsigmoid(beta * margin), beta * (policy_c - ref_c), beta * (policy_r - ref_r)


def make_trainer_class(method, args, collator, extra):
    import torch
    Base = dc.counting_trainer_class()

    class LisaTrainer(Base):
        def setup(self):
            self.guide = dc.SeededCycle(extra["guide"], "lisa_guide", args.seed)
            self.status = lisa_phase(0, args.lisa_align_steps, args.lisa_finetune_steps)
            params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            self.alignment_weights = {n: p.detach().clone() for n, p in params.items()}
            self.finetune_weights = {n: p.detach().clone() for n, p in params.items()}
            self.window_step, self.window = None, []
            self.state_log = {"phases": [{"step": 0, "status": self.status}], "user_micro_batches_seen": 0,
                              "user_micro_batches_discarded": 0, "guide_micro_batches": 0}

        def switch(self, status):
            # Anchors exactly as upstream switch_model: entering fine-tune stores the end-of-alignment weights, entering
            # alignment stores the end-of-fine-tune weights; drift = ||w_ft - w_align||^2 over LoRA params.
            self.status = status
            target = self.finetune_weights if status == "alignment" else self.alignment_weights
            drift = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    target[name] = param.detach().clone()
                    drift += float(torch.sum((self.finetune_weights[name] - self.alignment_weights[name]).float() ** 2))
            self.state_log["phases"].append({"step": self.state.global_step, "status": status, "drift_sq": drift})

        def training_step(self, model, inputs, num_items_in_batch=None):
            step = self.state.global_step
            status = lisa_phase(step, args.lisa_align_steps, args.lisa_finetune_steps)
            if status != self.status:
                self.switch(status)
            if status == "alignment":
                if self.window_step != step:  # first micro-step of this optimizer step: draw the whole window
                    n = self.accumulation_steps()
                    self.window_step = step
                    self.window = [collator([self.guide.next() for _ in range(args.batch_size)]) for _ in range(n)]
                    self.window_items = sum(int((b["labels"] != IGNORE_INDEX).sum()) for b in self.window)
                self.state_log["user_micro_batches_discarded"] += 1
                self.state_log["guide_micro_batches"] += 1
                inputs = self.window.pop(0)
                num_items_in_batch = torch.tensor(self.window_items, device=self.args.device)
            else:
                self.state_log["user_micro_batches_seen"] += 1
            return super().training_step(model, inputs, num_items_in_batch)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            if not model.training or args.lisa_rho <= 0 or self.state.global_step <= args.lisa_prox_warmup * self.state.max_steps:
                return result
            anchors = self.finetune_weights if self.status == "alignment" else self.alignment_weights
            prox = sum(torch.sum((p - anchors[n]) ** 2) for n, p in model.named_parameters() if p.requires_grad)
            # once per optimizer step in total: the micro-step losses are summed by gradient accumulation
            prox = args.lisa_rho / 2 * prox / self.accumulation_steps()
            if return_outputs:
                return result[0] + prox, result[1]
            return result + prox

    class BoosterTrainer(Base):
        def setup(self):
            self.harmful = dc.SeededCycle(extra["harmful"], "booster_harmful", args.seed)
            self.state_log = {"harmful_micro_batches": 0, "harmful_loss_sum": 0.0, "perturbed_loss_sum": 0.0}

        def training_step(self, model, inputs, num_items_in_batch=None):
            model.train()
            harmful = self._prepare_inputs(collator([self.harmful.next() for _ in range(args.batch_size)]))
            params = [p for p in model.parameters() if p.requires_grad]
            devices = [self.args.device.index or 0] if self.args.device.type == "cuda" else []
            # The two harmful passes run on a forked RNG so dropout in the alignment pass sees the same random stream
            # as A0 (lambda = 0 then reproduces A0 exactly).
            with torch.random.fork_rng(devices=devices):
                with self.compute_loss_context_manager():
                    loss_h = self.compute_loss(model, harmful)
                grads_h = torch.autograd.grad(loss_h, params)
                with torch.no_grad():
                    norm = torch.sqrt(sum(torch.sum(g.float() ** 2) for g in grads_h)) + 1e-7
                    backup = [p.detach().clone() for p in params]
                    for p, g in zip(params, grads_h):
                        p.sub_(args.booster_alpha * g / norm)
                with self.compute_loss_context_manager():
                    loss_p = self.compute_loss(model, harmful)
                grads_p = torch.autograd.grad(loss_p, params)
                with torch.no_grad():
                    for p, b in zip(params, backup):
                        p.copy_(b)
            scale = args.booster_lambda / self.accumulation_steps()
            with torch.no_grad():
                for p, gh, gp in zip(params, grads_h, grads_p):
                    delta = scale * (gh - gp)
                    p.grad = delta.to(p.dtype) if p.grad is None else p.grad.add_(delta.to(p.grad.dtype))
            self.state_log["harmful_micro_batches"] += 1
            self.state_log["harmful_loss_sum"] += float(loss_h)
            self.state_log["perturbed_loss_sum"] += float(loss_p)
            return super().training_step(model, inputs, num_items_in_batch)

    class CsftTrainer(Base):
        def setup(self):
            self.state_log = {"mean_weight_sum": 0.0, "micro_batches": 0}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if not model.training:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            labels = inputs["labels"]
            with dc.reference_mode(model):
                ref_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
                ref_logps, _ = dc.token_logps(ref_logits, labels)
                del ref_logits
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logps, mask = dc.token_logps(outputs.logits, labels)
            positions = dc.turn_positions(mask, args.csft_position_scope)
            betas = csft_betas(positions, args.csft_beta, args.csft_bias_factor, args.csft_first_token_bias_factor,
                               args.csft_bias_length)
            weights = csft_weights(logps, ref_logps, betas) * mask
            denominator = num_items_in_batch if num_items_in_batch is not None else mask.sum()
            loss = -(weights * logps).sum() / denominator
            self.state_log["mean_weight_sum"] += float(weights.sum() / mask.sum().clamp(min=1))
            self.state_log["micro_batches"] += 1
            return (loss, outputs) if return_outputs else loss

    class DpoTrainer(Base):
        def setup(self):
            self.model_accepts_loss_kwargs = False  # per-pair loss: HF divides each micro-batch by accumulation steps
            self.stats = {"loss": [], "rewards/chosen": [], "rewards/rejected": [], "reward_accuracy": [], "margin": []}
            self.state_log = {"ref_precomputed": args.precompute_ref, "eval": []}

        def pair_logps(self, model, inputs):
            """Sequence log-probs (sum over assistant tokens) of chosen and rejected; two forwards so the short
            refusal is not padded to the long trajectory."""
            out = []
            for side in ("chosen", "rejected"):
                logits = model(input_ids=inputs[f"{side}_input_ids"], attention_mask=inputs[f"{side}_attention_mask"]).logits
                logps, mask = dc.token_logps(logits, inputs[f"{side}_labels"])
                out += [logps.sum(1), logps, mask]
            return out[0], out[3], out[1], out[2]

        def reference(self, model, inputs):
            if args.precompute_ref:
                return inputs["ref"][:, 0], inputs["ref"][:, 1]
            with dc.reference_mode(model):
                ref_c, ref_r, _, _ = self.pair_logps(model, inputs)
            return ref_c, ref_r

        def pair_loss(self, model, inputs, record):
            ref_c, ref_r = self.reference(model, inputs)
            pol_c, pol_r, logps_c, mask_c = self.pair_logps(model, inputs)
            losses, reward_c, reward_r = dpo_loss(pol_c, pol_r, ref_c.to(pol_c.dtype), ref_r.to(pol_r.dtype), args.dpo_beta)
            loss = losses.mean()
            if args.sft_weight:
                loss = loss + args.sft_weight * (-(logps_c.sum(1) / mask_c.sum(1).clamp(min=1))).mean()
            if record is not None:
                record["loss"].append(float(losses.mean()))
                record["rewards/chosen"].append(float(reward_c.mean()))
                record["rewards/rejected"].append(float(reward_r.mean()))
                record["reward_accuracy"].append(float((reward_c > reward_r).float().mean()))
                record["margin"].append(float((reward_c - reward_r).mean()))
            return loss

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = self.pair_loss(model, inputs, self.stats if model.training else None)
            return (loss, None) if return_outputs else loss

        def log(self, logs, *rest, **kwargs):
            if self.stats["loss"] and "loss" in logs:
                for key, values in self.stats.items():
                    logs[f"dpo/{key}"] = sum(values) / len(values)
                    values.clear()
            return super().log(logs, *rest, **kwargs)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            rows = eval_dataset if eval_dataset is not None else self.eval_dataset
            record = {k: [] for k in self.stats}
            self.model.eval()
            with torch.no_grad():
                for row in rows:
                    batch = self._prepare_inputs(collator([row]))
                    with self.compute_loss_context_manager():
                        self.pair_loss(self.model, batch, record)
            self.model.train()
            metrics = {f"{metric_key_prefix}_{k.replace('/', '_')}": sum(v) / max(len(v), 1) for k, v in record.items()}
            self.state_log["eval"].append({"step": self.state.global_step, **metrics})
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    classes = {"sft": Base, "lisa": LisaTrainer, "booster": BoosterTrainer, "csft": CsftTrainer, "dpo": DpoTrainer}
    return classes[method]


# ----------------------------------------------------------------------------- DPO data
class PairCollator:
    """One micro-batch = B pairs: chosen_* and rejected_* padded separately; `ref` holds reference log-probs [B, 2]."""

    def __init__(self, pad_token_id):
        self.inner = dc.Collator(pad_token_id)

    def __call__(self, batch):
        import torch
        out = {}
        for side in ("chosen", "rejected"):
            part = self.inner([{"input_ids": r[f"{side}_input_ids"], "labels": r[f"{side}_labels"]} for r in batch])
            out.update({f"{side}_{k}": v for k, v in part.items()})
        out["ref"] = torch.tensor([[r.get("ref_chosen", 0.0), r.get("ref_rejected", 0.0)] for r in batch], dtype=torch.float32)
        return out


def build_pairs(pairs, tokenizer, max_length, name):
    rows, skipped = [], {}
    for pair in pairs:
        sides = {}
        for side in ("chosen", "rejected"):
            example = {"tools": pair.get("tools"), "messages": pair["prompt"] + pair[side]}
            reason = dc.validate(example)
            row = None if reason else dc.tokenize_example(example, tokenizer, max_length)
            if row is None:
                reason = reason or f"{side} longer than {max_length} tokens or no assistant tokens"
                skipped[reason] = skipped.get(reason, 0) + 1
                break
            sides[side] = row
        if len(sides) == 2:
            rows.append({"id": pair["id"], "chosen_input_ids": sides["chosen"]["input_ids"], "chosen_labels": sides["chosen"]["labels"],
                         "rejected_input_ids": sides["rejected"]["input_ids"], "rejected_labels": sides["rejected"]["labels"]})
    stats = {"kept": len(rows), "skipped": skipped,
             "trained_tokens": sum(sum(l != IGNORE_INDEX for l in r[f"{s}_labels"]) for r in rows for s in ("chosen", "rejected"))}
    print(f"[{name}] {json.dumps(stats)}", flush=True)
    return rows, stats


def precompute_reference(trainer, model, rows, collator, cost):
    import torch
    with dc.reference_mode(model):
        for row in rows:
            batch = trainer._prepare_inputs(collator([row]))
            with trainer.accelerator.autocast():
                ref_c, ref_r, _, _ = trainer.pair_logps(model, batch)
            row["ref_chosen"], row["ref_rejected"] = float(ref_c[0]), float(ref_r[0])
            cost.add(reference_forward_passes=1)
    return {r["id"]: (r["ref_chosen"], r["ref_rejected"]) for r in rows}


# ----------------------------------------------------------------------------- disjointness
def assert_disjoint(rows, forbidden, what):
    """Defence-side streams must not contain attack candidates or the held-out selection items."""
    ids = {r["id"] for r in rows}
    for path in forbidden:
        if Path(path).exists():
            clash = ids & {r["id"] for r in dc.load_jsonl(path)}
            assert not clash, f"{what}: {len(clash)} ids overlap {path}"


def forbidden_for(name):
    interim = dc.TRAINSET / "interim/attack_candidates.jsonl"
    extra = {"harmful": [dc.DERIVED / "defender_harmful_select.jsonl"], "guide": [dc.DERIVED / "select_refusal.jsonl"],
             "pairs": [dc.DERIVED / "defender_harmful_select.jsonl"]}
    return [interim] + extra.get(name, [])


# ----------------------------------------------------------------------------- main
def load_side_streams(args, tokenizer):
    """Method-specific data, tokenised with the training masking."""
    extra, stats, inputs = {}, {}, []
    if args.method == "lisa":
        guide = dc.load_jsonl(dc.require(args.lisa_guide, "run defense_data.py build"))
        assert_disjoint(guide, forbidden_for("guide"), "lisa guide")
        extra["guide"], stats["guide"] = dc.build_dataset(guide, tokenizer, args.max_length, "guide")
        inputs.append(("lisa_guide", args.lisa_guide))
        if not extra["guide"] and args.lisa_align_steps > 0:
            sys.exit("No usable guide examples.")
    if args.method == "booster":
        harmful = dc.load_jsonl(dc.require(args.booster_harmful, "run defense_data.py qc-defender (needs the generation job)"))
        assert_disjoint(harmful, forbidden_for("harmful"), "booster harmful")
        extra["harmful"], stats["harmful"] = dc.build_dataset(harmful, tokenizer, args.max_length, "harmful")
        inputs.append(("booster_harmful", args.booster_harmful))
        if not extra["harmful"]:
            sys.exit("No usable harmful examples.")
    return extra, stats, inputs


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.dry_run:
        sys.exit(f"Output directory {output} is not empty; choose a new one.")
    tokenizer = dc.load_tokenizer(args)

    # Same order of operations as train_lora.py: load, seeded shuffle, limit, tokenise.
    train_examples = dc.load_jsonl(dc.require(args.train))
    random.Random(args.seed).shuffle(train_examples)
    if args.limit:
        train_examples = train_examples[: args.limit]
    if args.method == "dpo":
        assert_disjoint(train_examples, forbidden_for("pairs"), "dpo pairs")
        train_rows, train_stats = build_pairs(train_examples, tokenizer, args.max_length, "train")
    else:
        train_rows, train_stats = dc.build_dataset(train_examples, tokenizer, args.max_length, "train")
    val_rows, val_stats = [], None
    if args.val and Path(args.val).exists():
        val = dc.load_jsonl(args.val)
        if args.method == "dpo":
            val_rows, val_stats = build_pairs(val, tokenizer, args.max_length, "val")
        else:
            val_rows, val_stats = dc.build_dataset(val, tokenizer, args.max_length, "val")
    if not train_rows:
        sys.exit("No usable training examples.")
    extra, side_stats, side_inputs = load_side_streams(args, tokenizer)
    if args.dry_run:
        return

    import torch
    import transformers

    cost = dc.Cost()
    transformers.set_seed(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    dc.write_manifest(output, args, {
        "method": args.method, "train_stats": train_stats, "val_stats": val_stats, "side_stats": side_stats,
        "loss_normalisation": "per-pair mean / grad_accum" if args.method == "dpo" else
                              "sum(token loss) / assistant tokens per accumulation window (as train_lora.py)",
        "csft_position_scope": args.csft_position_scope if args.method == "csft" else None,
    }, inputs=[("train", args.train), ("val", args.val)] + side_inputs)

    model = dc.load_model_and_lora(args)
    collator = PairCollator(tokenizer.pad_token_id) if args.method == "dpo" else dc.Collator(tokenizer.pad_token_id)
    has_val = bool(val_rows)
    training_args = dc.training_arguments(args, output, has_val, group_by_length=args.method != "dpo")
    Trainer = make_trainer_class(args.method, args, collator, extra)
    Trainer.passes_per_micro_step = PASSES[args.method]
    trainer = Trainer(model=model, args=training_args, train_dataset=train_rows, eval_dataset=val_rows or None,
                      data_collator=collator, callbacks=[dc.cost_callback(cost)])
    trainer.cost = cost
    if hasattr(trainer, "setup"):
        trainer.setup()
    if args.method == "dpo" and args.precompute_ref:
        refs = precompute_reference(trainer, model, train_rows + val_rows, collator, cost)
        torch.save(refs, output / "ref_logps.pt")
    result = trainer.train()

    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {"train": result.metrics}
    if has_val:
        summary["eval"] = trainer.evaluate()
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    state = getattr(trainer, "state_log", {})
    state.update(method=args.method, optimizer_steps=trainer.state.global_step)
    if args.method == "booster" and state["harmful_micro_batches"]:
        state["mean_harmful_loss"] = state["harmful_loss_sum"] / state["harmful_micro_batches"]
        state["mean_perturbed_harmful_loss"] = state["perturbed_loss_sum"] / state["harmful_micro_batches"]
    (output / "method_state.json").write_text(json.dumps(state, indent=2))
    if args.select:
        rows = dc.selection_rows(tokenizer, args.max_length, args.select_n, args.select_val, args.select_refusal,
                                 args.select_harmful)
        cost.add(calibration_tokens=sum(len(r["input_ids"]) for rs in rows.values() for r in rs))
        metrics = dc.selection_metrics(trainer.model, rows, trainer.args.device, trainer.accelerator.autocast)
        (output / "selection.json").write_text(json.dumps({"setting": vars(args), "metrics": metrics}, indent=2))
    cost.write(output)

    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output / "merged", safe_serialization=True)
        tokenizer.save_pretrained(output / "merged")
    print(f"Saved adapter to {adapter_dir}" + (f" and merged model to {output / 'merged'}" if args.merge else ""))


if __name__ == "__main__":
    main()
