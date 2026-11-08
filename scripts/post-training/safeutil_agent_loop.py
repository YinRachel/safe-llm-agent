"""verl (0.6.1) agent loop for SafeUtil's single-turn rows (next_action, harmful_prompt): render the stored prompt
(MOSAIC system text, the record's messages, its tools plus refuse_unsafe_task), generate one assistant turn, and pass
what the reward needs through extra_fields["mosaic"], the key MOSAIC's loop uses. verl concatenates the agent-loop
workers' outputs only when every worker returns the same extra_fields keys; a worker that drew only Agent-SafetyBench
rows would otherwise lack a separate single-turn key (smoke job 38513179 failed on that). Agent-SafetyBench rows keep
MOSAIC's multi-turn loop; the dataset's agent_name column picks the loop (safeutil_agent_loop.yaml registers both).
"""
import json
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer

import mosaic


class SafeUtilSingleTurnLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        rollout = config.actor_rollout_ref.rollout
        cls.response_length = rollout.response_length
        cls.group_size = rollout.n
        cls.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        sample = json.loads(kwargs["extra_info"]["sample_json"])
        prompt_ids = await self.loop.run_in_executor(None, lambda: self.tokenizer.apply_chat_template(
            sample["messages"], tools=sample["tools"], add_generation_prompt=True, tokenize=True))
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=prompt_ids,
                                                        sampling_params=sampling_params)
        generated = list(output.token_ids)[:self.response_length]
        complete = bool(generated) and generated[-1] == self.im_end
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        turn = mosaic.parse_turn(text)
        status = "truncated" if not complete else {"refuse": "refused", "tool": "tool_call"}.get(turn.kind, "answered")
        summary = dict(text=text, kind=turn.kind, status=status, well_formed=[turn.well_formed and complete],
                       turn_lengths=[len(generated)], uid=kwargs.get("uid"), group_size=self.group_size)
        logprobs = list(output.log_probs)[:len(generated)] if output.log_probs is not None else None
        return AgentLoopOutput(prompt_ids=prompt_ids, response_ids=generated, response_mask=[1] * len(generated),
                               response_logprobs=logprobs, num_turns=2, metrics=metrics,
                               extra_fields={"mosaic": summary})
