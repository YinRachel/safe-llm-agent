"""verl (0.6.1) agent loop for MOSAIC rollouts on Agent-SafetyBench: generate a turn, parse it (mosaic.parse_turn), run
the tool call in the benchmark's simulated environment, append the tool response, until refusal, an answer, the turn
limit or the response-length budget.

Token bookkeeping follows verl's ToolAgentLoop: prompt + generated tokens stay as produced (no re-tokenisation), tool
turns are appended as the chat template renders them, and response_mask is 1 only on generated tokens, so tool
responses carry no gradient (Sec. 2.2). The episode summary goes to the reward through extra_fields.

Registered through mosaic_agent_loop.yaml (actor_rollout_ref.rollout.agent.agent_loop_config_path); needs
scripts/post-training on PYTHONPATH (scripts/viking/mosaic_grpo.sbatch sets it).
"""
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer

import mosaic
from mosaic import astb


class MosaicAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        repo = Path(os.environ.get("ASTB_REPO", astb.DEFAULT_REPO))
        astb.check_repo(repo)
        cls.env_manager = astb.load_env_manager(repo)
        cls.system_text = mosaic.load_system_text()
        rollout = config.actor_rollout_ref.rollout
        cls.response_length = rollout.response_length
        cls.max_turns = rollout.multi_turn.max_assistant_turns or mosaic.MAX_TURNS
        cls.group_size = rollout.n
        cls.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        task = json.loads(kwargs["extra_info"]["task_json"])
        episode = mosaic.Episode.start(self.env_manager, task)
        messages = mosaic.task_messages(task, self.system_text)
        prompt_ids = await self.loop.run_in_executor(None, lambda: self.tokenizer.apply_chat_template(
            messages, tools=episode.tools, add_generation_prompt=True, tokenize=True))
        request_id = uuid4().hex
        ids, mask, logprobs, turn_lengths, metrics = list(prompt_ids), [], [], [], {}
        while episode.status == "running":
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(request_id=request_id, prompt_ids=ids,
                                                            sampling_params=sampling_params)
            generated = list(output.token_ids)
            turn_lengths.append(len(generated))
            ids += generated
            mask += [1] * len(generated)
            if output.log_probs is None:
                logprobs = None
            elif logprobs is not None:
                logprobs += output.log_probs
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if len(mask) >= self.response_length or not generated or generated[-1] != self.im_end:
                # Cut off by the budget (or vLLM's own length limit): act on nothing, the judge sees the partial turn.
                episode.turns.append(mosaic.parse_turn(text))
                episode.observations.append(None)
                episode.status = "truncated"
                break
            observation = episode.step(mosaic.parse_turn(text), self.max_turns)
            if observation is None:
                break
            tool_ids = mosaic.tool_turn_ids(self.tokenizer, observation)
            if len(mask) + len(tool_ids) >= self.response_length:
                episode.status = "truncated"
                break
            ids += tool_ids
            mask += [0] * len(tool_ids)
            if logprobs is not None:
                logprobs += [0.0] * len(tool_ids)
        summary = episode.summary()
        summary.update(turn_lengths=turn_lengths, uid=kwargs.get("uid"), group_size=self.group_size)
        response_ids = ids[len(prompt_ids):]
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=mask[:self.response_length],
            response_logprobs=logprobs[:self.response_length] if logprobs is not None else None,
            num_turns=1 + 2 * len(episode.turns),
            metrics=metrics,
            extra_fields={"mosaic": summary},
        )
