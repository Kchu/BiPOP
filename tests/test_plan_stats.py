"""Plan steps and waits are counted the same way for every agent: on the plan as executed on NICOL."""

import pytest

from bipop.plan_format import align_step_plan, compute_plan_stats, parse_temporal_plan_text


@pytest.mark.parametrize(
    "plan, expected",
    [
        (None, (0, 0)),
        ({"Left": [], "Right": []}, (0, 0)),
        # equal lengths, explicit waits (BiPOP / ReAct style)
        ({"Left": ["grasp(a)", "wait"], "Right": ["wait()", "grasp(b)"]}, (2, 2)),
        # a shorter hand waits in the missing steps (LLM-agent plans may have unequal lengths)
        ({"Left": ["grasp(a)", "move_to(a, p1)", "release(a, p1)"], "Right": ["grasp(b)"]}, (3, 2)),
        # ``a || b`` runs as two steps; the other hand waits in the second one
        ({"Left": ["grasp(a) || move_to(a, p1)"], "Right": ["grasp(b)"]}, (2, 1)),
        # a co-action uses both hands: the other hand is not waiting
        ({"Left": ["co_hold(tray)", "co_move_to(p1)"], "Right": ["co_hold(tray)", "wait"]}, (2, 0)),
    ],
)
def test_compute_plan_stats(plan, expected) -> None:
    assert compute_plan_stats(plan) == expected


def test_stats_are_independent_of_how_the_plan_is_written() -> None:
    """The same execution written with explicit waits or with a shorter hand gives the same counts."""
    explicit = {"Left": ["grasp(a)", "move_to(a, p1)", "release(a, p1)"], "Right": ["grasp(b)", "wait()", "wait()"]}
    shorter = {"Left": explicit["Left"], "Right": ["grasp(b)"]}
    assert compute_plan_stats(explicit) == compute_plan_stats(shorter) == (3, 2)
    assert align_step_plan(shorter["Left"], shorter["Right"])[1] == ["grasp(b)", "wait", "wait"]


def test_bipop_plan_steps_are_time_steps() -> None:
    """A BiPOP plan (all actions last 1 time unit) has one step per time unit, like the other agents' plans."""
    temporal = (
        "0.000: (grasp apple left_arm left_area) [1.000]\n"
        "0.000: (grasp banana right_arm right_area) [1.000]\n"
        "1.001: (move_to apple left_arm overlap_point_1 overlap_area) [1.000]\n"
        "2.002: (co_hold tray) [1.000]\n"
    )
    plan = parse_temporal_plan_text(temporal)
    assert plan["Left"][0].startswith("grasp(") and plan["Right"][1] == "wait"
    assert compute_plan_stats(plan) == (3, 1)
