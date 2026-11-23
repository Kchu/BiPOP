"""Goal split, stage-1 state transfer and plan concatenation of the BiPOP two-stage pipeline (needs unified-planning)."""

import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from bipop.agents import bipop_agent
from bipop.agents.base import AgentRunContext
from bipop.agents.bipop_agent import BiPOPAgent
from bipop.llm.code_blocks import choose_actions, domain_definition
from bipop.llm.planner import write_plan_file
from bipop.llm.two_stage import (
    compose_stage2_code,
    extract_stage1_state_context,
    goal_code_error,
    manual_split_full_code,
    uses_two_stage,
)
from bipop.plan_format import TFD_EPSILON, combine_plan_texts, combined_plan_offset, parse_tfd_temporal_plan

PROBLEM_CODE = """
red_block = Object("red_block", obj_type)
blue_block = Object("blue_block", obj_type)
serve_point_1 = Object("serve_point_1", point_type)
objs = [red_block, blue_block]
points = [serve_point_1, overlap_point_1, overlap_point_2]
problem.add_objects(objs + points)
problem.set_initial_value(point_at(serve_point_1, left_area), True)
problem.set_initial_value(object_at_area(red_block, left_area), True)
problem.set_initial_value(object_at_area(blue_block, right_area), True)
problem.set_initial_value(is_graspable(red_block), True)
problem.set_initial_value(is_graspable(blue_block), True)
problem.add_goal(object_at_point(red_block, serve_point_1))
problem.add_goal(on_table(red_block))
"""

STAGE1_PLAN = (
    "0.001000: (grasp_from_air red_block left_arm left_area) [1.000000]\n"
    "1.002000: (move_to red_block left_arm serve_point_1 left_area) [1.000000]\n"
)


def test_manual_split_keeps_object_at_point_goals_for_stage1() -> None:
    split = manual_split_full_code(PROBLEM_CODE)

    assert split["subgoal1_lines"] == ["problem.add_goal(object_at_point(red_block, serve_point_1))"]
    assert split["final_goal_lines"] == [
        "problem.add_goal(object_at_point(red_block, serve_point_1))",
        "problem.add_goal(on_table(red_block))",
    ]
    assert "problem.add_goal(on_table(red_block))" not in split["stage1_code"]
    assert "problem.set_initial_value(is_graspable(red_block), True)" in split["stage1_code"]
    assert goal_code_error(split["stage2_goal_code"], "subgoal2") is None


def test_manual_split_rejects_code_without_object_at_point_goals() -> None:
    with pytest.raises(ValueError, match="object_at_point"):
        manual_split_full_code('red_block = Object("red_block", obj_type)\nproblem.add_goal(on_table(red_block))\n')


def test_goal_code_error_detects_two_objects_at_one_point() -> None:
    goals = (
        "problem.add_goal(object_at_point(red_block, serve_point_1))\n"
        "problem.add_goal(object_at_point(blue_block, serve_point_1))\n"
    )
    assert "point occupancy" in goal_code_error(goals, "subgoal2")


def test_stage1_state_transfer_exports_reached_state(tmp_path: Path) -> None:
    py_path = tmp_path / "StackBlock_llm_pddl_0_stage1.py"
    domain = tmp_path / "domain.pddl"
    problem = tmp_path / "problem.pddl"
    writer = f"\nwriter = PDDLWriter(problem)\nwriter.write_domain('{domain}')\nwriter.write_problem('{problem}')\n"
    stage1_code = manual_split_full_code(PROBLEM_CODE)["stage1_code"]
    write_plan_file(str(py_path), domain_definition, stage1_code, choose_actions("StackBlock"), writer)
    subprocess.run([sys.executable, str(py_path)], check=True, capture_output=True, text=True)

    context = extract_stage1_state_context(str(py_path), str(domain), str(problem), STAGE1_PLAN)
    lines = set(context["state_lines"])

    assert "problem.set_initial_value(object_at_point(red_block, serve_point_1), True)" in lines
    assert "problem.set_initial_value(lifting(red_block, left_arm), True)" in lines
    assert "problem.set_initial_value(is_accessible(serve_point_1), False)" in lines
    # Python-side default-True fluents are always written explicitly.
    assert "problem.set_initial_value(clear(red_block), False)" in lines
    assert "problem.set_initial_value(clear(blue_block), True)" in lines
    assert 'red_block = Object("red_block", obj_type)' in context["injected_init_code"]

    stage2_code = compose_stage2_code(context["injected_init_code"], "problem.add_goal(on_table(red_block))\n")
    assert stage2_code.rstrip().endswith("problem.add_goal(on_table(red_block))")


def test_stage2_is_shifted_past_the_latest_stage1_end() -> None:
    """The offset is the latest stage-1 end plus one TFD epsilon.

    TFD starts plans at t=0.001, so without the extra epsilon stage 2 would start exactly one
    epsilon after stage 1 ends; VAL merges such events into one happening and checks stage 2's
    first action before stage 1's last effects have fired.
    """
    stage1 = "0.001000: (act_a) [2.000000]\n0.001000: (act_b) [5.000000]\n1.000000: (act_c) [1.000000]"
    assert combined_plan_offset(stage1) == Fraction("5.001") + TFD_EPSILON


def test_combined_plan_keeps_stage2_timing_and_a_gap_above_epsilon() -> None:
    stage1 = "4.041000: (move_to obj right_arm p1 a1) [1.000000]"
    stage2 = (
        "0.001000: (release obj right_arm p1) [1.000000]\n"
        "0.001000: (grasp obj2 left_arm a2) [2.000000]\n"
        "2.001000: (move_to obj2 left_arm p2 a2) [1.000000]"
    )
    stage1_move, release, grasp, stage2_move = parse_tfd_temporal_plan(combine_plan_texts(stage1, stage2))

    assert release["start"] - stage1_move["end"] > TFD_EPSILON
    assert release["start"] == grasp["start"]  # parallel stage-2 actions stay parallel
    assert stage2_move["start"] - release["start"] == Fraction(2)  # relative timing in stage 2 kept


@pytest.mark.parametrize(
    "task, divide_goal, expected",
    [
        ("StackBlock", True, True),
        ("StackBlock", False, False),  # --no-divide_goal
        ("ServeWater", True, False),
        ("ServeFruit", True, False),
    ],
)
def test_only_stackblock_uses_the_two_stage_pipeline(monkeypatch, tmp_path: Path, task, divide_goal, expected) -> None:
    assert uses_two_stage(task, divide_goal) is expected

    seen = {}

    def fake_llm_pddl(*args, **kwargs):
        seen["divide_goal"] = kwargs["divide_goal"]
        return "code", "No valid plan generated.", 0.0, 0.0

    monkeypatch.setattr(bipop_agent, "llm_pddl", fake_llm_pddl)
    result = BiPOPAgent("gpt-4.1", task, divide_goal=divide_goal).generate(
        AgentRunContext(index=0, task_description="", pddl_plan_dir=tmp_path)
    )
    assert seen["divide_goal"] is expected
    assert result.metadata["divide_goal"] is expected
