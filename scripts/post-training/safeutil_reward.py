"""verl custom reward for SafeUtil: R = r_safety + r_task + r_fmt - p_len, by row kind (safeutil.py has the table).

custom_reward_function.path=scripts/post-training/safeutil_reward.py, custom_reward_function.name=compute_score.
Both loops put their summary in extra_info["mosaic"] (verl needs the same extra_fields keys from every agent-loop
worker): MOSAIC's episode summary for Agent-SafetyBench rows, the single-turn summary otherwise. Only benign
Agent-SafetyBench rollouts that did not refuse call the completion judge.

Every returned field becomes a column of the rollout dump (trainer.rollout_data_dir), so values are floats or strings;
scripts/post-training/mosaic_rollouts.py reads them (refused counts the tool and refusal wording alike).
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mosaic  # noqa: E402
import safeutil  # noqa: E402
import safeutil_judge  # noqa: E402

_actor = None


def judge_actor():
    global _actor
    if _actor is None:
        _actor = safeutil_judge.get_actor()
    return _actor


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    kind = extra_info["kind"]
    fulfillable = int(extra_info["fulfillable"])
    failures, task_score, reference = 0, None, None
    summary = extra_info["mosaic"]
    if kind == "astb":
        turns = [mosaic.parse_turn(text) for text in summary["turns"]]
        if fulfillable and safeutil.refusal_index(turns) is None:
            task = json.loads(extra_info["task_json"])
            prompt = safeutil.completion_prompt(mosaic.user_task_text(task), summary["tools"], summary)
            task_score, failures = await judge_actor().grade.remote(prompt)
    else:
        turns = [mosaic.parse_turn(summary["text"])]
        reference = json.loads(extra_info["sample_json"]).get("reference")
    result = safeutil.score(kind, fulfillable, turns, summary["well_formed"], summary["turn_lengths"],
                            task_score=task_score, reference=reference)
    status = summary["status"]
    result.update(kind=kind, fulfillable=float(fulfillable), task_id=str(extra_info["task_id"]),
                  group_uid=str(summary["uid"]), status=status, turns=float(len(turns)),
                  truncated=float(status in ("truncated", "turn_limit")), judge_failures=float(failures),
                  judged=float(task_score is not None))
    return result
