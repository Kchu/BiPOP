"""Plan execution on NICOL: hand alignment, unreadable steps and same-target moves (no CoppeliaSim needed)."""

import json
from pathlib import Path
from typing import List, Tuple

import pytest

from NICOL import nicol_controller as ctrl
from bipop.plan_format import align_step_plan, compute_plan_stats


class _FakeProcess:
    def __init__(self, stdout: str = "") -> None:
        self._stdout = stdout

    def communicate(self) -> Tuple[str, str]:
        return self._stdout, ""


class _Task:
    def __init__(self, success: bool = False) -> None:
        self.success = success

    def check_success(self) -> bool:
        return self.success


@pytest.fixture
def executed(monkeypatch) -> List[Tuple[str, str, dict]]:
    """Replace the executor processes; record every (side, command, params) that is run."""
    calls: List[Tuple[str, str, dict]] = []

    def fake_executor(side, command, params, state, other_holding=None, other_command=None):
        calls.append((side, command, json.loads(params)))
        prefix = side.upper()
        return _FakeProcess(f"{prefix}_RESULT: The {side} hand did {command}. {prefix}_HAND: Vertical {prefix}_FINGER: Open")

    monkeypatch.setattr(ctrl, "run_executor", fake_executor)
    monkeypatch.setattr(ctrl, "run_head_command", lambda look_at: _FakeProcess())
    monkeypatch.setattr(ctrl.time, "sleep", lambda seconds: None)
    return calls


def _run(tmp_path: Path, left, right, is_pddl: bool = False, success: bool = True):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"Left": left, "Right": right}), encoding="utf-8")
    return ctrl.nicol_control(str(plan_path), is_pddl, _Task(success))


def _steps(calls, side: str) -> List[str]:
    return [command for call_side, command, _ in calls if call_side == side]


def test_align_step_plan_pads_and_splits_combined_steps() -> None:
    left, right = align_step_plan(["grasp(cup) || move_to(cup, point_1)", "pour()"], ["co_hold(tray)"])
    assert left == ["grasp(cup)", "move_to(cup, point_1)", "pour()"]
    assert right == ["co_hold(tray)", "wait", "wait"]


def test_longer_left_hand_runs_all_steps_while_right_waits(tmp_path: Path, executed) -> None:
    assert _run(tmp_path, ["grasp(apple)", "grasp(banana)", "pour(blue_cup, yellow_cup)"], ["wait()"]) == (True, 3)
    assert _steps(executed, "left") == ["top_grasp", "top_grasp", "pour"]
    assert _steps(executed, "right") == ["wait", "wait", "wait"]


def test_longer_right_hand_runs_its_extra_steps(tmp_path: Path, executed) -> None:
    assert _run(tmp_path, ["grasp(apple)"], ["wait", "grasp(banana)"]) == (True, 2)
    assert _steps(executed, "right") == ["wait", "top_grasp"]


def test_unreadable_step_fails_the_plan_at_that_step(tmp_path: Path, executed) -> None:
    assert _run(tmp_path, ["grasp(apple)", "None", "pour()"], ["wait", "grasp(banana)", "wait"]) == (False, 2)
    assert len(executed) == 2  # only step 1 ran (one process per hand)


def test_surrounding_spaces_and_wait_spellings_are_accepted(tmp_path: Path, executed) -> None:
    assert _run(tmp_path, [" grasp(apple) ", "WAIT", ""], ["wait()", "grasp(banana)", "pour(blue_cup, yellow_cup)"]) == (True, 3)
    assert _steps(executed, "left") == ["top_grasp", "wait", "wait"]


def test_combined_steps_run_one_after_the_other(tmp_path: Path, executed) -> None:
    left = ["grasp(apple, left_arm, left_area) || move_to(apple, left_arm, overlap_point_1)"]
    assert _run(tmp_path, left, ["wait"], is_pddl=True) == (True, 2)
    assert _steps(executed, "left") == ["top_grasp", "move_single_to_pose"]
    assert _steps(executed, "right") == ["wait", "wait"]


def test_both_hands_to_the_same_point_fail_without_execution(tmp_path: Path, executed) -> None:
    result = _run(tmp_path, ["move_to(apple, overlap_point_1)"], ["move_to(banana, overlap_point_1)"], success=False)
    assert result == (False, 1)
    assert executed == []


def test_both_hands_to_different_points_are_executed(tmp_path: Path, executed) -> None:
    assert _run(tmp_path, ["move_to(apple, overlap_point_1)"], ["move_to(banana, overlap_point_2)"]) == (True, 1)
    assert _steps(executed, "left") == _steps(executed, "right") == ["move_single_to_pose"]


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (("move_single_to_pose", '{"target": "p1"}'), ("move_single_to_pose", '{"target": "p1"}'), True),
        (("move_single_to_pose", '{"obj_name": "a", "target": "p1"}'),
         ("move_single_to_pose", '{"obj_name": "b", "target": "p1"}'), True),
        (("move_above", '{"target": "bowl", "off_set": "up"}'), ("move_above", '{"target": "bowl", "off_set": "up"}'), True),
        (("move_single_to_pose", '{"target": "p1"}'), ("move_single_to_pose", '{"target": "p2"}'), False),
        (("move_single_to_pose", '{"target": "p1"}'), ("move_above", '{"target": "p1"}'), False),
        (("move_single_to_pose", '{}'), ("move_single_to_pose", '{}'), False),
        (("top_grasp", '{"obj_name": "a"}'), ("top_grasp", '{"obj_name": "a"}'), False),
    ],
)
def test_same_move_target(left, right, expected) -> None:
    assert ctrl.same_move_target(left[0], left[1], right[0], right[1]) is expected


def test_replay_without_a_task_reports_failed_steps(tmp_path: Path, executed, monkeypatch) -> None:
    """Standalone replay (task=None) must not report a plan with a failed step as successful."""
    def failing_executor(side, command, params, state, other_holding=None, other_command=None):
        prefix = side.upper()
        outcome = "failed to grasp" if command == "top_grasp" and side == "right" else f"did {command}"
        return _FakeProcess(f"{prefix}_RESULT: The {side} hand {outcome}. {prefix}_HAND: Vertical {prefix}_FINGER: Open")

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"Left": ["grasp(apple)", "wait"], "Right": ["wait", "grasp(banana)"]}))
    assert ctrl.nicol_control(str(plan_path), False, task=None) == (True, 2)

    monkeypatch.setattr(ctrl, "run_executor", failing_executor)
    assert ctrl.nicol_control(str(plan_path), False, task=None) == (False, 2)


def test_executed_steps_match_the_counted_plan_steps(tmp_path: Path, executed) -> None:
    left = ["grasp(apple, left_arm, left_area) || move_to(apple, left_arm, overlap_point_1)", "wait"]
    right = ["wait"]
    success, finished_steps = _run(tmp_path, left, right, is_pddl=True)
    assert success and finished_steps == compute_plan_stats({"Left": left, "Right": right})[0] == 3
