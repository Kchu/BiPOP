"""Artifacts of earlier runs must not leak into a new run, and cleanup must not touch other instances."""

import subprocess
from pathlib import Path

import pytest

from bipop.agents import bipop_agent
from bipop.agents.base import AgentRunContext
from bipop.agents.bipop_agent import BiPOPAgent
from bipop.llm import planner


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old", encoding="utf-8")
    return path


def test_bipop_agent_removes_only_its_own_stale_artifacts(monkeypatch, tmp_path: Path) -> None:
    task, plan_dir = "StackBlock", tmp_path / "pddl_plans"
    idir = plan_dir / "intermediate"
    own = [
        _touch(idir / f"{task}_plan_1.txt"),
        _touch(idir / f"{task}_plan_1_2.txt"),
        _touch(idir / f"{task}_plan_1.json"),
        _touch(idir / f"{task}_domain_1.pddl"),
        _touch(idir / f"{task}_problem_1_stage2.pddl"),
        _touch(plan_dir / f"{task}_pddl_plan_1.json"),
        _touch(plan_dir / f"{task}_pddl_plan_1_1.json"),
    ]
    others = [
        _touch(idir / f"{task}_plan_10.txt"),
        _touch(idir / f"{task}_plan_12_1.txt"),
        _touch(idir / f"{task}_domain_11.pddl"),
        _touch(plan_dir / f"{task}_pddl_plan_10.json"),
    ]
    seen_before_planning = {}

    def fake_llm_pddl(*args, **kwargs):
        seen_before_planning.update({path: path.exists() for path in own + others})
        return "code", "No valid plan generated.", 0.0, 0.0

    monkeypatch.setattr(bipop_agent, "llm_pddl", fake_llm_pddl)
    result = BiPOPAgent("gpt-4.1", task, divide_goal=True).generate(
        AgentRunContext(index=1, task_description="", pddl_plan_dir=plan_dir)
    )

    assert not any(seen_before_planning[path] for path in own)
    assert all(seen_before_planning[path] for path in others)
    assert result.success is False
    assert result.plan_path is None  # no stale plan is reported


def test_solve_code_string_removes_stale_pddl_before_generation(monkeypatch, tmp_path: Path) -> None:
    domain = _touch(tmp_path / "ServeWater_domain_0.pddl")
    problem = _touch(tmp_path / "ServeWater_problem_0.pddl")

    def crashing_generator(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output="", stderr="NameError: name 'cup' is not defined")

    monkeypatch.setattr(planner.subprocess, "run", crashing_generator)
    with pytest.raises(subprocess.CalledProcessError):
        planner.solve_code_string("problem.add_goal(cup)", str(tmp_path), "ServeWater", 0, "single", "", trial=0)

    assert not domain.exists() and not problem.exists()
