"""verl custom reward for MOSAIC: R(t) = r_align + r_fmt - p_len (Sec. 2.2.2, Algorithm 1).

custom_reward_function.path=scripts/post-training/mosaic_reward.py, custom_reward_function.name=compute_score.
verl loads this file by path as `custom_module`, so everything a Ray actor must unpickle lives in importable modules
(mosaic, mosaic_judge). The agent loop's episode summary arrives as extra_info["mosaic"].
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mosaic  # noqa: E402
import mosaic_judge  # noqa: E402

_actor = None


def judge_actor():
    global _actor
    if _actor is None:
        _actor = mosaic_judge.get_actor()
    return _actor


def components(summary, judged):
    r_fmt = mosaic.format_reward(summary["well_formed"])
    p_len = mosaic.length_penalty(summary["turn_lengths"])
    r_align = judged["alignment"]
    kinds = summary["kinds"]
    return dict(score=r_align + r_fmt - p_len, r_align=r_align, r_fmt=r_fmt, p_len=p_len,
                refused=float(summary["status"] == "refused"),
                refused_first_turn=float(bool(kinds) and kinds[0] == "refuse"),
                safety_thoughts_rate=sum(summary["safety"]) / max(1, len(summary["safety"])),
                turns=float(len(kinds)), truncated=float(summary["status"] in ("truncated", "turn_limit")),
                judge_failures=float(judged["judge_failures"]), group_members=float(judged["group_members"]))


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    summary = extra_info["mosaic"]
    task = json.loads(extra_info["task_json"])
    member = dict(summary=summary, user_task=mosaic.user_task_text(task))
    judged = await judge_actor().score.remote(summary["uid"], member, summary["group_size"])
    return components(summary, judged)
