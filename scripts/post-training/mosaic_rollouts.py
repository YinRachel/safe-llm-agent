"""Read a MOSAIC run's rollout dump (<run>/rollouts/<step>.jsonl, written by trainer.rollout_data_dir) and answer two
questions the console score curve cannot, because r_align always averages 1.5 within a group:

  1. Does the policy drift into refusing benign tasks? Refusal rates per step window, split by Agent-SafetyBench's
     `fulfillable` label (1 = the task can be done safely, 0 = it should be refused).
  2. Is refusing the cheap option? In groups where some rollouts refused and some did not, the reward gap
     (refused minus not refused) and which reward term makes it, again split by `fulfillable`.

  python scripts/post-training/mosaic_rollouts.py results/qwen2.5-3B_MOSAIC/<run> [--window 50] [--json out.json]
                                                  [--examples 5]

<run> is the run folder or its rollouts/ folder. Standard library only, so it runs in any env. Runs before f1a7815 have
no rollout files.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

LABELS = {1.0: "benign", 0.0: "harmful"}
TERMS = ("score", "r_align", "r_fmt", "p_len")
REQUIRED = ("score", "r_align", "r_fmt", "p_len", "refused", "fulfillable", "group_uid")


def load(path):
    folder = Path(path)
    if (folder / "rollouts").is_dir():
        folder = folder / "rollouts"
    files = sorted(folder.glob("*.jsonl"), key=lambda f: int(f.stem) if f.stem.isdigit() else -1)
    if not files:
        sys.exit(f"no <step>.jsonl files in {folder}")
    rows = []
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED if k not in row]
            if missing:
                sys.exit(f"{f}:{n} lacks {missing}; this run predates the rollout fields of f1a7815")
            row.setdefault("step", int(f.stem) if f.stem.isdigit() else 0)
            rows.append(row)
    return rows


def label(row):
    return LABELS.get(float(row["fulfillable"]), "unknown")


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def window_stats(rows, window):
    """One entry per step window: reward terms, and refusal behaviour per label."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[(int(row["step"]) - 1) // window].append(row)
    out = []
    for b in sorted(buckets):
        part = buckets[b]
        entry = dict(steps=f"{b * window + 1}-{(b + 1) * window}", n=len(part),
                     **{t: mean(float(r[t]) for r in part) for t in TERMS},
                     turns=mean(float(r.get("turns", 0)) for r in part),
                     truncated=mean(float(r.get("truncated", 0)) for r in part),
                     judge_failures=sum(float(r.get("judge_failures", 0)) for r in part))
        for name in ("benign", "harmful"):
            sub = [r for r in part if label(r) == name]
            entry[f"{name}_n"] = len(sub)
            entry[f"{name}_refused"] = mean(float(r["refused"]) for r in sub)
            entry[f"{name}_refused_first_turn"] = mean(float(r.get("refused_first_turn", 0)) for r in sub)
        out.append(entry)
    return out


def refusal_gap(rows):
    """Within groups that mix refused and non-refused rollouts: mean reward of refused minus non-refused, per term."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["step"], row["group_uid"])].append(row)
    result = {}
    for name in ("benign", "harmful"):
        gaps = {t: [] for t in TERMS}
        wins = 0
        mixed = 0
        for members in groups.values():
            if label(members[0]) != name:
                continue
            refused = [r for r in members if float(r["refused"]) == 1.0]
            done = [r for r in members if float(r["refused"]) != 1.0]
            if not refused or not done:
                continue
            mixed += 1
            for t in TERMS:
                gaps[t].append(mean(float(r[t]) for r in refused) - mean(float(r[t]) for r in done))
            wins += max(float(r["score"]) for r in refused) > max(float(r["score"]) for r in done)
        result[name] = dict(mixed_groups=mixed, refused_scores_higher=wins / mixed if mixed else None,
                            **{f"gap_{t}": mean(gaps[t]) for t in TERMS})
    return result


def examples(rows, k, chars=600):
    """The last k benign tasks the policy refused: the over-refusals to read by hand."""
    hits = [r for r in rows if label(r) == "benign" and float(r["refused"]) == 1.0]
    return [dict(step=r["step"], task_id=r.get("task_id"), score=r["score"], status=r.get("status"),
                 output=str(r.get("output", ""))[-chars:]) for r in (hits[-k:] if k > 0 else [])]


def fmt(value, width=7):
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.3f}" if isinstance(value, float) else str(value).rjust(width)


def report(windows, gap, samples):
    cols = [("steps", "steps", 9), ("n", "n", 5), ("score", "score", 7), ("r_align", "align", 7),
            ("r_fmt", "fmt", 7), ("p_len", "len", 7), ("turns", "turns", 7),
            ("benign_refused", "ben.ref", 7), ("benign_refused_first_turn", "ben.ref1", 8),
            ("harmful_refused", "harm.ref", 8), ("harmful_refused_first_turn", "harm.ref1", 9),
            ("truncated", "trunc", 7), ("judge_failures", "j.fail", 6)]
    lines = ["Per step window (refusal rates split by fulfillable: ben = 1, harm = 0)",
             " ".join(h.rjust(w) for _, h, w in cols)]
    lines += [" ".join(fmt(e[k], w) for k, _, w in cols) for e in windows]
    lines += ["", "Refused minus not-refused, within groups that have both (positive = refusing pays more)",
              " ".join(h.rjust(w) for h, w in (("label", 8), ("groups", 7), ("score", 7), ("align", 7),
                                               ("fmt", 7), ("len", 7), ("ref.best", 8)))]
    for name, g in gap.items():
        lines.append(" ".join([name.rjust(8), fmt(g["mixed_groups"]), fmt(g["gap_score"]), fmt(g["gap_r_align"]),
                               fmt(g["gap_r_fmt"]), fmt(g["gap_p_len"]), fmt(g["refused_scores_higher"], 8)]))
    lines.append("ref.best: share of those groups whose best rollout refused. A positive len gap means refusals "
                 "carry a larger length penalty.")
    for s in samples:
        lines += ["", f"--- refused benign task {s['task_id']} at step {s['step']} (score {s['score']:.3f}, "
                      f"{s['status']}) ---", s["output"]]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run folder or its rollouts/ folder")
    p.add_argument("--window", type=int, default=50, help="steps per row (default 50)")
    p.add_argument("--examples", type=int, default=0, help="print the last K refused benign rollouts")
    p.add_argument("--json", help="also write the numbers here")
    args = p.parse_args(argv)
    rows = load(args.run)
    windows, gap, samples = window_stats(rows, args.window), refusal_gap(rows), examples(rows, args.examples)
    print(report(windows, gap, samples))
    if args.json:
        Path(args.json).write_text(json.dumps(dict(windows=windows, refusal_gap=gap, examples=samples), indent=1))


if __name__ == "__main__":
    main()
