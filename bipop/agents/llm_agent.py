"""LLM-agent: the LLM generates the full bimanual plan directly."""

from __future__ import annotations

import time

from ..llm.llm_direct import llm_direct
from ..plan_format import compute_plan_stats
from .base import AgentRunContext, BasePlanningAgent, GenerationResult


class LLMAgent(BasePlanningAgent):
    agent_id = "llm-agent"
    display_name = "LLM-agent"
    is_pddl = False

    def __init__(self, model_name: str, task_name: str, temperature: float = 0.0):
        super().__init__(model_name, task_name)
        self.temperature = temperature

    def generate(self, context: AgentRunContext) -> GenerationResult:
        if context.direct_plan_dir is None:
            raise ValueError("direct_plan_dir must be provided for llm-agent")

        context.direct_plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = context.direct_plan_dir / f"direct_plan_{self.model_name}_{context.index}.json"
        plan_path.unlink(missing_ok=True)  # never report the plan of an earlier run

        start = time.time()
        plan, llm_time_cost = llm_direct(
            self.model_name, self.task_name, context.task_description, str(plan_path), temperature=self.temperature
        )
        elapsed = time.time() - start
        plan_steps, _ = compute_plan_stats(plan)

        return GenerationResult(
            agent_id=self.agent_id,
            plan_path=plan_path if plan_path.exists() else None,
            plan=plan,
            success=plan is not None and plan_steps > 0,
            llm_time_cost=llm_time_cost,
            generation_time=elapsed,
            plan_length=plan_steps,
        )
