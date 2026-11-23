"""Data types shared by the planning agents and the evaluation of saved plans on NICOL."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from NICOL.nicol_controller import nicol_control

from ..plan_format import compute_plan_stats


@dataclass
class AgentRunContext:
    """Inputs of one agent run on one task instance."""

    index: int
    task_description: str
    direct_plan_dir: Optional[Path] = None
    pddl_plan_dir: Optional[Path] = None
    react_plan_dir: Optional[Path] = None
    task: Any = None                  # NICOL task (needed by the ReAct-agent)
    validate_with_val: bool = False
    val_bin: Optional[Path] = None
    divide_goal: Optional[bool] = None


@dataclass
class EvaluationResult:
    """Outcome of executing a plan on NICOL."""

    success: bool
    finished_steps: int
    execution_time: float
    plan_steps: int
    number_of_waits: int
    task_achieved: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Outcome of generating a plan."""

    agent_id: str
    plan_path: Optional[Path]
    plan: Optional[Dict[str, List[str]]]
    success: bool
    llm_time_cost: float
    planner_time_cost: float = 0.0
    generation_time: float = 0.0
    plan_length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Set by agents that execute while planning (ReAct-agent).
    embedded_evaluation: Optional[EvaluationResult] = None


class BasePlanningAgent(ABC):
    agent_id: str = "base-agent"
    display_name: str = "Base agent"
    is_pddl: bool = False  # plans use PDDL action signatures (BiPOP-agent)

    def __init__(self, model_name: str, task_name: str):
        self.model_name = model_name
        self.task_name = task_name

    @abstractmethod
    def generate(self, context: AgentRunContext) -> GenerationResult:
        raise NotImplementedError


def default_failed_evaluation() -> EvaluationResult:
    return EvaluationResult(
        success=False, finished_steps=0, execution_time=0.0, plan_steps=0, number_of_waits=0, task_achieved=False
    )


def evaluate_saved_plan(task: Any, plan_path: Path, is_pddl: bool) -> EvaluationResult:
    """Execute a saved plan on NICOL in CoppeliaSim and check the task goal."""
    start = time.time()
    success, finished_steps = nicol_control(str(plan_path), is_pddl, task)
    duration = time.time() - start

    plan = None
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_steps, number_of_waits = compute_plan_stats(plan)

    return EvaluationResult(
        success=success,
        finished_steps=finished_steps,
        execution_time=duration,
        task_achieved=success,
        plan_steps=plan_steps,
        number_of_waits=number_of_waits,
    )
