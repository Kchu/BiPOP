"""Planning agents compared in the experiments.

* ``bipop-agent``  BiPOP: the LLM generates the PDDL problem, TFD plans;
* ``llm-agent``    baseline: the LLM generates the bimanual plan directly;
* ``react-agent``  baseline: the LLM acts step by step with feedback from NICOL.
"""

from __future__ import annotations

from ..llm.llm_pddl import DEFAULT_MAX_TRIALS
from .base import (
    AgentRunContext,
    BasePlanningAgent,
    EvaluationResult,
    GenerationResult,
    default_failed_evaluation,
    evaluate_saved_plan,
)
from .bipop_agent import BiPOPAgent
from .llm_agent import LLMAgent
from .react_agent import ReActExecutionAgent

VALID_TASKS = ("ServeWater", "ServeFruit", "StackBlock")
VALID_MODELS = ("gpt-4.1", "o3", "deepseek-r1")
AGENT_IDS = ("bipop-agent", "llm-agent", "react-agent")

AGENT_ALIASES = {
    "bipop-agent": "bipop-agent",
    "bipop": "bipop-agent",
    "pddl": "bipop-agent",
    "llm-agent": "llm-agent",
    "llm": "llm-agent",
    "direct": "llm-agent",
    "react-agent": "react-agent",
    "react": "react-agent",
}


def normalize_agent_name(name: str) -> str:
    key = name.strip().lower()
    if key not in AGENT_ALIASES:
        raise ValueError(f"Unknown agent '{name}'. Valid agents: {', '.join(AGENT_IDS)}")
    return AGENT_ALIASES[key]


def create_agent(
    agent_name: str,
    model_name: str,
    task_name: str,
    temperature: float = 0.0,
    max_trials: int = DEFAULT_MAX_TRIALS,
    max_steps: int = 20,
    divide_goal: bool = True,
) -> BasePlanningAgent:
    """Agent by id or alias (see ``AGENT_ALIASES``)."""
    normalized = normalize_agent_name(agent_name)
    if normalized == "llm-agent":
        return LLMAgent(model_name=model_name, task_name=task_name, temperature=temperature)
    if normalized == "bipop-agent":
        return BiPOPAgent(
            model_name=model_name,
            task_name=task_name,
            temperature=temperature,
            max_trials=max_trials,
            divide_goal=divide_goal,
        )
    return ReActExecutionAgent(model_name=model_name, task_name=task_name, temperature=temperature, max_steps=max_steps)


__all__ = [
    "AGENT_ALIASES",
    "AGENT_IDS",
    "DEFAULT_MAX_TRIALS",
    "AgentRunContext",
    "BasePlanningAgent",
    "BiPOPAgent",
    "EvaluationResult",
    "GenerationResult",
    "LLMAgent",
    "ReActExecutionAgent",
    "VALID_MODELS",
    "VALID_TASKS",
    "create_agent",
    "default_failed_evaluation",
    "evaluate_saved_plan",
    "normalize_agent_name",
]
