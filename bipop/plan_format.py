"""Parsing and conversion of plans; no planner or LLM dependencies.

Two plan formats are used throughout the code:

* TFD temporal plans, one action per line: ``0.001000: (grasp cup left_arm left_area) [1.000000]``
* step-aligned JSON plans for the robot: ``{"Left": [...], "Right": [...]}``, where
  step k of both hands is executed at the same time and ``wait`` keeps a hand still.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

GRASP_ALIASES = {"grasp_from_air", "grasp_from_block", "grasp_a", "grasp_b"}

_TEMPORAL_LINE_RE = re.compile(r"^\s*([\d.]+):\s*\(([^)]+)\)\s*\[[^\]]*\]\s*$")
_TEMPORAL_STEP_RE = re.compile(r"^\s*([\d.]+):\s*\(([^)]+)\)\s*\[([\d.]+)\]")
_TEMPORAL_SHIFT_RE = re.compile(r"^(\s*)([\d.]+)(:\s*\([^)]+\)\s*\[[^\]]*\].*)$")

# TFD plans start their first action at t=0.001 (one epsilon after t=0). When
# stage 1 and stage 2 are concatenated, stage 2 is shifted by the end time of
# stage 1's last action, which leaves a gap of exactly one epsilon. VAL merges
# events that are <= epsilon apart into one happening, so stage 2's first
# action (which depends on stage 1's final effects) could be checked before
# those effects fire. One extra epsilon in the offset gives a 2*epsilon gap,
# which VAL keeps as distinct time points.
TFD_EPSILON = Fraction(1, 1000)


# ---------------------------------------------------------------------------
# Step-aligned JSON plans
# ---------------------------------------------------------------------------

def is_wait_action(action: Any) -> bool:
    value = str(action).strip().lower()
    return value in ("wait", "wait()", "")


def is_co_action(action: Any) -> bool:
    """``co_hold`` / ``co_move_to``: one action that uses both hands."""
    return str(action).strip().lower().startswith("co_")


def align_step_plan(left: List[Any], right: List[Any]) -> Tuple[List[str], List[str]]:
    """Both hands' steps as they are executed on NICOL (NICOL/nicol_controller.py).

    The shorter hand is padded with ``wait``; an entry ``a || b`` (two actions
    of one hand starting in the same step) becomes consecutive steps while the
    other hand waits. A ``co_*`` action written for both hands stays in one step.
    """
    steps = max(len(left), len(right))
    left = [str(entry) for entry in left] + ["wait"] * (steps - len(left))
    right = [str(entry) for entry in right] + ["wait"] * (steps - len(right))

    aligned_left: List[str] = []
    aligned_right: List[str] = []
    for left_entry, right_entry in zip(left, right):
        left_parts = [part.strip() for part in left_entry.split("||")]
        right_parts = [part.strip() for part in right_entry.split("||")]
        sub_steps = max(len(left_parts), len(right_parts))
        aligned_left += left_parts + ["wait"] * (sub_steps - len(left_parts))
        aligned_right += right_parts + ["wait"] * (sub_steps - len(right_parts))
    return aligned_left, aligned_right


def compute_plan_stats(plan: Optional[Dict[str, List[str]]]) -> Tuple[int, int]:
    """Return (number of steps, number of wait actions) of a step-aligned plan.

    Counted the same way for every agent, on the plan as it is executed
    (``align_step_plan``): a step is one time step of both hands, and a wait is
    a hand that stays still in a step. The other hand of a ``co_*`` action is
    not waiting, as the co-action uses both hands.
    """
    if not plan:
        return 0, 0
    left, right = align_step_plan(plan.get("Left") or [], plan.get("Right") or [])
    waits = 0
    for left_action, right_action in zip(left, right):
        if is_co_action(left_action) or is_co_action(right_action):
            continue
        waits += int(is_wait_action(left_action)) + int(is_wait_action(right_action))
    return len(left), waits


def format_temporal_action(action_str: str) -> str:
    """``grasp_from_air cup left_arm left_area`` -> ``grasp(cup, left_arm, left_area)``."""
    tokens = action_str.split()
    if not tokens:
        return action_str
    name, args = tokens[0], tokens[1:]
    if name.lower().replace("-", "_") in GRASP_ALIASES:
        name = "grasp"
    return f"{name}(" + ", ".join(args) + ")"


def parse_temporal_plan_text(plan_text: str) -> Dict[str, List[str]]:
    """Discretize a TFD temporal plan into a step-aligned ``{"Left", "Right"}`` plan.

    Each action is put at the integer part of its start time; ``co_*`` actions
    occupy both hands. Actions of the same hand starting in the same step are
    joined with ``||``.
    """
    steps: Dict[int, Dict[str, str]] = {}
    min_step: Optional[int] = None
    max_step: Optional[int] = None

    def ensure_step(ts: int) -> None:
        if ts not in steps:
            steps[ts] = {"Left": "wait", "Right": "wait"}

    for raw in plan_text.splitlines():
        if not raw.strip():
            continue
        matched = _TEMPORAL_LINE_RE.match(raw)
        if not matched:
            continue

        ts = int(float(matched.group(1)))
        action_raw = matched.group(2).strip()
        tokens = action_raw.split()
        formatted = format_temporal_action(action_raw)

        if tokens and tokens[0].startswith("co_"):
            ensure_step(ts)
            steps[ts]["Left"] = formatted
            steps[ts]["Right"] = formatted
            min_step = ts if min_step is None else min(min_step, ts)
            max_step = ts if max_step is None else max(max_step, ts)
            continue

        arm = None
        for token in tokens:
            if token == "left_arm":
                arm = "Left"
                break
            if token == "right_arm":
                arm = "Right"
                break
        if arm is None:
            for token in reversed(tokens):
                if token.endswith("_arm"):
                    arm = "Left" if "left" in token else "Right"
                    break
        if arm is None:
            continue

        ensure_step(ts)
        if steps[ts][arm] != "wait":
            steps[ts][arm] += " || " + formatted
        else:
            steps[ts][arm] = formatted

        min_step = ts if min_step is None else min(min_step, ts)
        max_step = ts if max_step is None else max(max_step, ts)

    if min_step is None:
        return {"Left": [], "Right": []}

    left, right = [], []
    for ts in range(min_step, max_step + 1):
        slot = steps.get(ts, {"Left": "wait", "Right": "wait"})
        left.append(slot["Left"])
        right.append(slot["Right"])
    return {"Left": left, "Right": right}


# ---------------------------------------------------------------------------
# TFD temporal plans
# ---------------------------------------------------------------------------

def parse_tfd_temporal_plan(plan_text: str) -> List[Dict[str, Any]]:
    """Parse TFD plan lines into dicts with exact (Fraction) start/end/duration."""
    plan_steps: List[Dict[str, Any]] = []
    for raw_line in plan_text.splitlines():
        matched = _TEMPORAL_STEP_RE.match(raw_line)
        if not matched:
            continue
        tokens = matched.group(2).strip().split()
        if not tokens:
            continue
        start = Fraction(matched.group(1))
        duration = Fraction(matched.group(3))
        plan_steps.append(
            {
                "start": start,
                "end": start + duration,
                "duration": duration,
                "name": tokens[0],
                "args": tokens[1:],
            }
        )
    return plan_steps


def parse_makespan(plan_text: str) -> float | None:
    """Latest end time (start + duration) of all actions, or None."""
    best = None
    for line in plan_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        try:
            colon = line.index(":")
            lb = line.rindex("[")
            rb = line.rindex("]")
            start = float(line[:colon])
            dur = float(line[lb + 1 : rb])
            makespan = start + dur
            if best is None or makespan > best:
                best = makespan
        except Exception:
            continue
    return best


def parse_plan_length(plan_text: str) -> int:
    """Number of action lines in a TFD plan."""
    count = 0
    for line in plan_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if ":" not in stripped:
            continue
        count += 1
    return count


def combined_plan_offset(plan_text: str) -> Fraction:
    """Time shift for stage 2: end of stage 1's last action plus one TFD epsilon."""
    plan_steps = parse_tfd_temporal_plan(plan_text)
    if plan_steps:
        return max(step["end"] for step in plan_steps) + TFD_EPSILON
    return Fraction(parse_plan_length(plan_text))


def shift_plan_text(plan_text: str, offset: Fraction) -> str:
    shifted_lines: List[str] = []
    for line in plan_text.splitlines():
        matched = _TEMPORAL_SHIFT_RE.match(line)
        if not matched:
            shifted_lines.append(line)
            continue
        shifted_time = Fraction(matched.group(2)) + offset
        shifted_lines.append(f"{matched.group(1)}{float(shifted_time):.6f}{matched.group(3)}")
    return "\n".join(shifted_lines)


def combine_plan_texts(plan_1: str, plan_2: str) -> str:
    """Concatenate two temporal plans, stage 2 shifted to start after stage 1."""
    offset = combined_plan_offset(plan_1)
    first = plan_1.strip()
    second = shift_plan_text(plan_2.strip(), offset)
    return "\n".join(part for part in (first, second) if part) + "\n"
