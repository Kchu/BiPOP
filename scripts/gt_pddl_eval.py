#!/usr/bin/env python3
"""Rule-based ground-truth PDDL generation and plan evaluation.

For every task configuration under ``task_configs/<task>/task_config_<i>.json``
this script deterministically (no LLM) builds the ground-truth PDDL
domain/problem pair, reusing the exact same domain blocks the BiPOP
pipeline uses (``bipop/llm/code_blocks.py``). The task-specific part (objects,
initial areas/points/stacks, goals) is derived rule-based from the config
fields only.

Each saved plan (bipop / direct / react methods) is then grounded into a
temporal PDDL plan and checked with VAL against the ground truth. Results are
written to a CSV, including a concise failure reason for every invalid plan.

Usage (from the repository root):
    python scripts/gt_pddl_eval.py                            # everything
    python scripts/gt_pddl_eval.py --tasks ServeWater --methods react direct
    python scripts/gt_pddl_eval.py --indices 0-9 --verbose

Outputs (default under results/gt_eval/):
    <task>/gt/domain_<i>.pddl, problem_<i>.pddl   ground-truth PDDL
    <task>/plans/<method>_<model>_<i>.tplan       grounded temporal plans
    <task>/val_logs/<method>_<model>_<i>.log      full VAL output
    gt_evaluation.csv                             one row per plan
    gt_evaluation_summary.csv                     success rates per method
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bipop.llm.code_blocks import choose_actions, domain_definition, problem_initialization  # noqa: E402
from bipop.llm.planner import val_output_is_valid  # noqa: E402
from bipop.paths import RESULTS_DIR, TASK_CONFIGS_DIR, default_val_bin  # noqa: E402
from bipop.plan_format import (  # noqa: E402
    align_step_plan,
    combine_plan_texts,
    compute_plan_stats,
    is_wait_action,
    parse_temporal_plan_text,
)

# Same rationale as bipop/llm/planner.py: keep VAL's tolerance well below the
# smallest timestamp separation we emit (0.001) so distinct happenings never collapse.
VAL_TOLERANCE = "0.0001"
ACTION_DURATION = 1.0
# One JSON step maps to a 1.002s slot: the action lasts 1.0s, the right arm is
# offset +0.001 from the left arm, and 0.001 separates consecutive steps.
STEP_PERIOD = 1.002
RIGHT_ARM_OFFSET = 0.001

AREAS = {"left": "left_area", "right": "right_area", "overlap": "overlap_area"}
STACK_COLORS = ["red", "yellow", "green", "blue", "orange", "black"]

CALL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\(\s*(.*?)\s*\))?\s*$", re.DOTALL)


class GroundingError(Exception):
    """A plan step cannot be mapped onto the ground-truth domain."""


def normalize_area(token: str) -> str:
    token = token.strip().lower()
    if token in AREAS:
        return AREAS[token]
    if token in AREAS.values():
        return token
    raise GroundingError(f"unknown area '{token}'")


# ---------------------------------------------------------------------------
# Task specs: everything the generator/grounder needs, derived from one config
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    task_name: str
    objects: List[str]                      # obj_type objects (order stable)
    points: List[str]                       # task points excl. overlap points
    point_areas: Dict[str, str]             # every point (incl. overlap) -> area
    obj_areas: Dict[str, str]               # object -> initial area
    graspable: Dict[str, bool]
    water: List[str] = field(default_factory=list)
    stacks: List[List[str]] = field(default_factory=list)  # top -> bottom
    non_releasable_points: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)         # python fluent calls


def _config_area(config: Dict[str, Any], *keys: str) -> str:
    area_map = config.get("area", {})
    for key in keys:
        if key in area_map:
            return normalize_area(str(area_map[key]))
    raise ValueError(f"config has no area entry for any of {keys}")


def build_serve_water_spec(config: Dict[str, Any]) -> TaskSpec:
    serve_area = _config_area(config, "human")
    store_area = _config_area(config, "brown box", "box")
    return TaskSpec(
        task_name="ServeWater",
        objects=["blue_cup", "yellow_cup"],
        points=["store_point", "serve_point"],
        point_areas={
            "store_point": store_area,
            "serve_point": serve_area,
            "overlap_point_1": "overlap_area",
            "overlap_point_2": "overlap_area",
        },
        obj_areas={
            "blue_cup": _config_area(config, "blue cup", "blue_cup"),
            "yellow_cup": _config_area(config, "yellow cup", "yellow_cup"),
        },
        graspable={"blue_cup": True, "yellow_cup": True},
        water=["blue_cup"],
        non_releasable_points=["serve_point"],
        goals=[
            "object_at_point(yellow_cup, serve_point)",
            "water_in_cup(yellow_cup)",
            "object_at_point(blue_cup, store_point)",
            "on_table(blue_cup)",
        ],
    )


def build_serve_fruit_spec(config: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_name="ServeFruit",
        objects=["apple", "banana", "bowl"],
        points=["serve_point"],
        point_areas={
            "serve_point": "overlap_area",
            "overlap_point_1": "overlap_area",
            "overlap_point_2": "overlap_area",
        },
        obj_areas={
            "apple": _config_area(config, "apple"),
            "banana": _config_area(config, "banana"),
            "bowl": _config_area(config, "bowl"),
        },
        graspable={"apple": True, "banana": True, "bowl": False},
        non_releasable_points=["serve_point"],
        goals=[
            "object_on_object(apple, bowl)",
            "object_on_object(banana, bowl)",
            "object_at_point(bowl, serve_point)",
        ],
    )


def build_stackblock_spec(task_name: str, config: Dict[str, Any]) -> TaskSpec:
    blocks = [f"{color}_block" for color in STACK_COLORS]
    obj_areas = {
        f"{color}_block": _config_area(config, color) for color in STACK_COLORS
    }
    stacks = [
        [f"{color}_block" for color in stack]
        for stack in config.get("initial_stacks", []) or []
    ]

    # Blocks stacked on other blocks share the supporting block's area.
    for stack in stacks:
        base_area = obj_areas[stack[-1]]
        for block in stack:
            obj_areas[block] = base_area

    goals: List[str] = []
    pile_desc = config.get("pile_desc") or config.get("pile_description") or []
    if not pile_desc:
        # Fall back to config["goal"]: bottom -> top order.
        pile_desc = [
            {"serve_point": f"serve_point_{k + 1}", "top_to_bottom": list(reversed(pile))}
            for k, pile in enumerate(config.get("goal", []))
        ]
    for pile in pile_desc:
        serve_point = pile["serve_point"]
        top_to_bottom = [f"{color}_block" for color in pile["top_to_bottom"]]
        bottom, top = top_to_bottom[-1], top_to_bottom[0]
        goals.append(f"object_at_point({bottom}, {serve_point})")
        goals.append(f"on_table({bottom})")
        for upper, lower in zip(top_to_bottom, top_to_bottom[1:]):
            goals.append(f"object_on_object({upper}, {lower})")
        goals.append(f"clear({top})")

    return TaskSpec(
        task_name=task_name,
        objects=blocks,
        points=["serve_point_1", "serve_point_2"],
        point_areas={
            "serve_point_1": _config_area(config, "tray"),
            "serve_point_2": _config_area(config, "human"),
            "overlap_point_1": "overlap_area",
            "overlap_point_2": "overlap_area",
        },
        obj_areas=obj_areas,
        graspable={block: True for block in blocks},
        stacks=stacks,
        goals=goals,
    )


def build_task_spec(task_name: str, config: Dict[str, Any]) -> TaskSpec:
    if task_name == "ServeWater":
        return build_serve_water_spec(config)
    if task_name == "ServeFruit":
        return build_serve_fruit_spec(config)
    if task_name.startswith("StackBlock"):
        return build_stackblock_spec(task_name, config)
    raise ValueError(f"Unsupported task: {task_name}")


# ---------------------------------------------------------------------------
# Ground-truth problem code generation + PDDL writing
# ---------------------------------------------------------------------------


def spec_to_init_code(spec: TaskSpec) -> str:
    lines: List[str] = ["", "## RULE-BASED GROUND-TRUTH INIT AND GOALS", ""]
    for obj in spec.objects:
        lines.append(f'{obj} = Object("{obj}", obj_type)')
    for point in spec.points:
        lines.append(f'{point} = Object("{point}", point_type)')
    lines.append(f"objs = [{', '.join(spec.objects)}]")
    all_points = spec.points + ["overlap_point_1", "overlap_point_2"]
    lines.append(f"points = [{', '.join(all_points)}]")
    lines.append("problem.add_objects(objs + points)")
    lines.append("")

    for point in spec.points:
        lines.append(
            f"problem.set_initial_value(point_at({point}, {spec.point_areas[point]}), True)"
        )
    for point in spec.non_releasable_points:
        lines.append(f"problem.set_initial_value(is_releasable({point}), False)")
    lines.append("")

    for obj in spec.objects:
        lines.append(
            f"problem.set_initial_value(object_at_area({obj}, {spec.obj_areas[obj]}), True)"
        )
        lines.append(
            f"problem.set_initial_value(is_graspable({obj}), {spec.graspable[obj]})"
        )
    for cup in spec.water:
        lines.append(f"problem.set_initial_value(water_in_cup({cup}), True)")

    for stack in spec.stacks:
        lines.append(f"# initial stack (top->bottom): {', '.join(stack)}")
        for upper, lower in zip(stack, stack[1:]):
            lines.append(
                f"problem.set_initial_value(object_on_object({upper}, {lower}), True)"
            )
            lines.append(f"problem.set_initial_value(on_table({upper}), False)")
            lines.append(f"problem.set_initial_value(clear({lower}), False)")
    lines.append("")

    for goal in spec.goals:
        lines.append(f"problem.add_goal({goal})")
    return "\n".join(lines)


def generate_ground_truth_pddl(
    spec: TaskSpec, domain_path: Path, problem_path: Path, py_path: Optional[Path]
) -> None:
    action_definition = choose_actions(spec.task_name)
    if not action_definition:
        raise ValueError(f"No action definition for task {spec.task_name}")

    writer_code = f"""
writer = PDDLWriter(problem)
writer.write_domain({str(domain_path)!r})
writer.write_problem({str(problem_path)!r})
"""
    content = "\n".join(
        [
            domain_definition,
            problem_initialization,
            spec_to_init_code(spec),
            action_definition,
            writer_code,
        ]
    )
    if py_path is not None:
        py_path.write_text(content, encoding="utf-8")

    namespace: Dict[str, Any] = {"__name__": "__gt_pddl__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(content, str(py_path or "<gt>"), "exec"), namespace)


# ---------------------------------------------------------------------------
# Plan grounding: {"Left": [...], "Right": [...]} -> temporal PDDL plan
# ---------------------------------------------------------------------------


def parse_call(entry: str) -> Tuple[str, List[str]]:
    matched = CALL_RE.match(str(entry))
    if not matched:
        raise GroundingError(f"cannot parse action '{entry}'")
    name = matched.group(1).lower()
    raw_args = matched.group(2)
    args = [a.strip().lower() for a in raw_args.split(",") if a.strip()] if raw_args else []
    return name, args


@dataclass
class GroundAction:
    start: float
    name: str
    args: List[str]
    duration: float = ACTION_DURATION

    def to_line(self) -> str:
        return f"{self.start:.4f}: ({' '.join([self.name] + self.args)}) [{self.duration:.3f}]"


class PlanGrounder:
    """Grounds short skill calls (react/direct) and passes through already
    grounded calls (bipop method) using a light rule-based state tracker."""

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.is_stack = spec.task_name.startswith("StackBlock")
        self.objects = set(spec.objects)
        self.points = set(spec.point_areas)
        self.arms = {"left_arm", "right_arm"}
        self.known_areas = set(AREAS.values())
        # tracked state (only used to fill in omitted arguments)
        self.area = dict(spec.obj_areas)
        self.at_point: Dict[str, Optional[str]] = {o: None for o in spec.objects}
        self.on: Dict[str, Optional[str]] = {o: None for o in spec.objects}
        for stack in spec.stacks:
            for upper, lower in zip(stack, stack[1:]):
                self.on[upper] = lower
        self.co_held: Optional[str] = None

    # -- helpers ------------------------------------------------------------
    def _check_obj(self, token: str) -> str:
        if token not in self.objects:
            raise GroundingError(f"unknown object '{token}'")
        return token

    def _check_point(self, token: str) -> str:
        if token not in self.points:
            raise GroundingError(f"unknown point '{token}'")
        return token

    def _obj_area(self, obj: str) -> str:
        area = self.area.get(obj)
        if area is None:
            raise GroundingError(f"no known area for object '{obj}'")
        return area

    # -- grounding of one call ----------------------------------------------
    def ground_call(self, name: str, args: List[str], arm: str) -> Tuple[str, List[str]]:
        """Returns (action_name, grounded_args); updates the tracker."""
        if name.startswith("co_"):
            return self._ground_co(name, args)
        if any(a in self.arms for a in args):
            return self._passthrough(name, args)
        return self._ground_short(name, args, arm)

    def _ground_co(self, name: str, args: List[str]) -> Tuple[str, List[str]]:
        if name == "co_hold":
            if len(args) != 1:
                raise GroundingError(f"co_hold expects 1 argument, got {args}")
            obj = self._check_obj(args[0])
            self.co_held = obj
            return "co_hold", [obj]
        if name == "co_move_to":
            if len(args) == 2:
                obj, point = self._check_obj(args[0]), self._check_point(args[1])
            elif len(args) == 1:
                if self.co_held is None:
                    raise GroundingError("co_move_to before any co_hold")
                obj, point = self.co_held, self._check_point(args[0])
            else:
                raise GroundingError(f"co_move_to expects 1-2 arguments, got {args}")
            self.at_point[obj] = point
            return "co_move_to", [obj, point]
        raise GroundingError(f"unknown skill '{name}'")

    def _passthrough(self, name: str, args: List[str]) -> Tuple[str, List[str]]:
        """Already grounded call from the bipop method's saved plans."""
        if name == "grasp":
            if len(args) == 4 and self.is_stack:
                name = "grasp_from_block"
            elif len(args) == 3 and self.is_stack:
                name = "grasp_from_air"
            elif len(args) != 3:
                raise GroundingError(f"grasp with unexpected arguments {args}")
        known = self.objects | self.points | self.arms | self.known_areas
        for token in args:
            if token not in known:
                raise GroundingError(f"unknown symbol '{token}' in {name}({', '.join(args)})")
        self._track(name, args)
        return name, args

    def _track(self, name: str, args: List[str]) -> None:
        """Best-effort state tracking shared by both plan styles."""
        if name in ("grasp", "grasp_from_air", "grasp_from_block"):
            obj = args[0]
            if name != "grasp":
                self.at_point[obj] = None
            if self.on.get(obj):
                self.on[obj] = None
        elif name in ("move_to", "move_to_from"):
            obj, point = args[0], args[2]
            self.at_point[obj] = point
            self.area[obj] = self.spec.point_areas.get(point, self.area.get(obj))
        elif name == "move_above":
            obj, target = args[0], args[2]
            self.at_point[obj] = None
            self.area[obj] = self.area.get(target, self.area.get(obj))
        elif name == "place":
            obj, target = args[0], args[2]
            self.on[obj] = target
            self.area[obj] = self.area.get(target, self.area.get(obj))
        elif name == "push":
            obj, to_area = args[0], args[3]
            self.area[obj] = to_area

    def _ground_short(self, name: str, args: List[str], arm: str) -> Tuple[str, List[str]]:
        if name == "grasp":
            if len(args) != 1:
                raise GroundingError(f"grasp expects 1 argument, got {args}")
            obj = self._check_obj(args[0])
            area = self._obj_area(obj)
            if self.is_stack:
                support = self.on.get(obj)
                grounded = (
                    ("grasp_from_block", [obj, arm, area, support])
                    if support
                    else ("grasp_from_air", [obj, arm, area])
                )
            else:
                grounded = ("grasp", [obj, arm, area])
        elif name == "move_to":
            if len(args) != 2:
                raise GroundingError(f"move_to expects 2 arguments, got {args}")
            obj, point = self._check_obj(args[0]), self._check_point(args[1])
            point_area = self.spec.point_areas[point]
            origin = self.at_point.get(obj)
            grounded = (
                ("move_to_from", [obj, arm, point, point_area, origin])
                if origin
                else ("move_to", [obj, arm, point, point_area])
            )
        elif name == "move_above":
            if len(args) != 2:
                raise GroundingError(f"move_above expects 2 arguments, got {args}")
            obj = self._check_obj(args[0])
            if args[1] in self.points:
                raise GroundingError(
                    f"move_above target must be an object, got point '{args[1]}'"
                )
            target = self._check_obj(args[1])
            grounded = ("move_above", [obj, arm, target, self._obj_area(target)])
        elif name == "place":
            if len(args) != 2:
                raise GroundingError(f"place expects 2 arguments, got {args}")
            obj, target = self._check_obj(args[0]), self._check_obj(args[1])
            grounded = ("place", [obj, arm, target, self._obj_area(target)])
        elif name == "release":
            if len(args) != 2:
                raise GroundingError(f"release expects 2 arguments, got {args}")
            obj, point = self._check_obj(args[0]), self._check_point(args[1])
            grounded = ("release", [obj, arm, point])
        elif name == "pour":
            if len(args) != 2:
                raise GroundingError(f"pour expects 2 arguments, got {args}")
            obj, target = self._check_obj(args[0]), self._check_obj(args[1])
            grounded = ("pour", [obj, arm, target])
        elif name in ("push", "push_to"):
            if len(args) != 2:
                raise GroundingError(f"{name} expects 2 arguments, got {args}")
            obj = self._check_obj(args[0])
            to_area = normalize_area(args[1])
            grounded = ("push", [obj, arm, self._obj_area(obj), to_area])
        else:
            raise GroundingError(f"unknown skill '{name}'")

        self._track(*grounded)
        return grounded

    # -- whole plan ----------------------------------------------------------
    def ground_plan(self, plan: Dict[str, Any]) -> List[GroundAction]:
        # Same steps as executed on NICOL: the shorter hand waits, ``a || b`` runs as two steps.
        left, right = align_step_plan(plan.get("Left") or [], plan.get("Right") or [])

        actions: List[GroundAction] = []
        for step in range(len(left)):
            base_time = step * STEP_PERIOD + RIGHT_ARM_OFFSET
            emitted_co: set[str] = set()
            for arm, entry, offset in (
                ("left_arm", left[step], 0.0),
                ("right_arm", right[step], RIGHT_ARM_OFFSET),
            ):
                if is_wait_action(entry):
                    continue
                try:
                    name, args = parse_call(entry)
                    gname, gargs = self.ground_call(name, args, arm)
                except GroundingError as exc:
                    raise GroundingError(
                        f"step {step} ({arm.split('_')[0].capitalize()}): "
                        f"'{entry}': {exc}"
                    ) from exc
                if gname.startswith("co_"):
                    key = f"{gname}({','.join(gargs)})"
                    if key in emitted_co:
                        continue  # same co-action written for both arms
                    emitted_co.add(key)
                    actions.append(GroundAction(base_time, gname, gargs))
                else:
                    actions.append(GroundAction(base_time + offset, gname, gargs))
        if not actions:
            raise GroundingError("plan contains no actions (only waits)")
        return actions


# ---------------------------------------------------------------------------
# Original temporal plan recovery for the bipop method
#
# The saved <task>_pddl_plan_<i>.json is an integer-step discretization of the
# TFD temporal plan; that discretization loses the sub-second overlaps the
# planner exploited (e.g. move_above starting during place). To judge the bipop
# method faithfully we recover the raw temporal plan from the intermediates:
# it is the candidate whose discretization equals the saved JSON.
# ---------------------------------------------------------------------------

TEMPORAL_LINE_RE = re.compile(r"^\s*([\d.]+)\s*:\s*\(([^)]+)\)\s*\[([\d.]+)\]")


def parse_temporal_lines(plan_text: str) -> List[GroundAction]:
    from fractions import Fraction

    actions: List[GroundAction] = []
    for raw in plan_text.splitlines():
        matched = TEMPORAL_LINE_RE.match(raw)
        if not matched:
            continue
        tokens = matched.group(2).strip().split()
        actions.append(
            GroundAction(
                start=float(Fraction(matched.group(1))),
                name=tokens[0].lower(),
                args=[t.lower() for t in tokens[1:]],
                duration=float(Fraction(matched.group(3))),
            )
        )
    return actions


def _plan_candidates(task_dir: Path) -> List[str]:
    if not task_dir.is_dir():
        return []
    texts = []
    for path in sorted(task_dir.iterdir()):
        if (
            path.is_file()
            and "_plan_" in path.name
            and ".txt" in path.name
            and not path.name.endswith(".log")
        ):
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return texts


def _plans_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    def norm(plan: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        return (
            tuple(str(s).strip() for s in plan.get("Left", [])),
            tuple(str(s).strip() for s in plan.get("Right", [])),
        )

    return norm(a) == norm(b)


def find_original_temporal_plan(
    results_root: Path, task: str, index: int, saved_plan: Dict[str, Any]
) -> Optional[str]:
    intermediate = results_root / "pddl_plans" / task / "intermediate"
    single = _plan_candidates(intermediate / f"task_{index}")
    for text in single:
        if _plans_equal(parse_temporal_plan_text(text), saved_plan):
            return text
    stage1 = _plan_candidates(intermediate / f"task_{index}_stage1")
    stage2 = _plan_candidates(intermediate / f"task_{index}_stage2")
    for text1 in stage1:
        for text2 in stage2:
            combined = combine_plan_texts(text1, text2)
            if _plans_equal(parse_temporal_plan_text(combined), saved_plan):
                return combined
    return None


# ---------------------------------------------------------------------------
# VAL execution and failure-reason extraction
# ---------------------------------------------------------------------------

ADVICE_ACTION_RE = re.compile(
    r"^(\(.*\))(?:\s*-\s*\w+)?\s+has an unsatisfied (precondition|condition)\s+at time\s+([\d.]+)"
)
ADVICE_SET_RE = re.compile(r"^\(Set\s+(\(.*\))\s+to\s+(true|false)\)")


def run_val(domain: Path, problem: Path, plan: Path, val_bin: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [val_bin, "-t", VAL_TOLERANCE, "-v", str(domain), str(problem), str(plan)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:  # missing or not executable
        return {"available": False, "is_valid": False, "output": f"VAL not found at {val_bin} ({type(exc).__name__})"}
    output = result.stdout or ""
    is_valid = val_output_is_valid(output)
    return {"available": True, "is_valid": is_valid, "output": output}


def _repair_advice_reason(lines: List[str]) -> Optional[str]:
    """Builds a reason like:
    '(release yellow_cup right_arm serve_point) fails at t=7.016: requires
    (is_releasable serve_point)=true' from VAL's Plan Repair Advice block."""
    action, time_point, conditions = None, None, []
    for line in lines:
        line = line.strip()
        matched = ADVICE_ACTION_RE.match(line)
        if matched:
            if action is not None:
                break  # only report the first failing action
            action, time_point = matched.group(1), matched.group(3)
            continue
        if action is not None:
            set_match = ADVICE_SET_RE.match(line)
            if set_match:
                conditions.append(f"requires {set_match.group(1)}={set_match.group(2)}")
            elif not line or line.startswith("Failed plans"):
                break
    if action is None:
        return None
    detail = "; ".join(conditions) if conditions else "unsatisfied precondition"
    return f"{action} fails at t={time_point}: {detail}"


def extract_val_failure(output: str) -> Tuple[str, str]:
    """Maps VAL's verbose output to (failure_stage, concise_reason)."""
    lines = output.splitlines()

    def context(idx: int, before: int = 1, after: int = 3, limit: int = 4) -> str:
        chunk = [l.strip() for l in lines[max(0, idx - before): idx + after] if l.strip()]
        return " | ".join(chunk[:limit])

    if "unsatisfied precondition" in output or "unsatisfied condition" in output:
        reason = _repair_advice_reason(lines)
        if reason is None:
            idx = next(
                i for i, l in enumerate(lines)
                if "unsatisfied precondition" in l or "unsatisfied condition" in l
            )
            reason = context(idx)
        return "precondition", reason
    for idx, line in enumerate(lines):
        low = line.lower()
        if "mutex violation" in low:
            return "mutex", context(idx)
        if "invariant" in low and ("fail" in low or "violat" in low):
            return "invariant", context(idx)
    for idx, line in enumerate(lines):
        low = line.lower()
        if "goal not satisfied" in low:
            return "goal", "plan executed but the goal is not satisfied"
        if "bad plan description" in low or "error in type-checking" in low:
            return "plan_syntax", context(idx)
        if "unknown" in low and ("object" in low or "action" in low or "type" in low):
            return "plan_syntax", context(idx)
    for idx, line in enumerate(lines):
        if "failed" in line.lower() or "error" in line.lower():
            return "invalid_other", context(idx)
    return "invalid_other", " | ".join(l.strip() for l in lines[-4:] if l.strip())


# -- unmet-goal diagnosis via unified_planning ------------------------------


def diagnose_unmet_goals(
    domain_path: Path, problem_path: Path, actions: List[GroundAction]
) -> Optional[str]:
    """Simulates the grounded plan with unified_planning and lists which goal
    fluents are not satisfied in the final state. Returns None on any failure
    (the generic VAL message is used instead)."""
    try:
        from fractions import Fraction

        from unified_planning.engines import TimeTriggeredPlanValidator
        from unified_planning.engines.results import ValidationResultStatus
        from unified_planning.io import PDDLReader
        from unified_planning.plans import ActionInstance, TimeTriggeredPlan

        with contextlib.redirect_stdout(io.StringIO()):
            problem = PDDLReader().parse_problem(str(domain_path), str(problem_path))
        up_actions = {a.name.lower(): a for a in problem.actions}
        up_objects = {o.name.lower(): o for o in problem.all_objects}
        timed = []
        for action in actions:
            up_action = up_actions[action.name.lower()]
            params = [up_objects[arg.lower()] for arg in action.args]
            timed.append(
                (
                    Fraction(action.start).limit_denominator(10**6),
                    ActionInstance(up_action, params),
                    Fraction(1),
                )
            )
        plan = TimeTriggeredPlan(timed, environment=problem.environment)
        with contextlib.redirect_stdout(io.StringIO()):
            validation = TimeTriggeredPlanValidator().validate(problem, plan)
        if validation.status != ValidationResultStatus.VALID or not validation.trace:
            return None
        trace = validation.trace
        final_state = trace[max(trace)] if isinstance(trace, dict) else trace[-1]
        unmet = []
        for goal in problem.goals:
            value = final_state.get_value(goal)
            if not (value.is_bool_constant() and value.bool_constant_value()):
                unmet.append(str(goal))
        return "; ".join(unmet) if unmet else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plan discovery
# ---------------------------------------------------------------------------

PLAN_FILE_PATTERNS = {
    "react": re.compile(r"^react_plan_(?P<model>.+)_(?P<index>\d+)\.json$"),
    "direct": re.compile(r"^direct_plan_(?P<model>.+)_(?P<index>\d+)\.json$"),
}


def discover_models(plans_dir: Path, method: str) -> List[str]:
    if not plans_dir.is_dir():
        return []
    pattern = PLAN_FILE_PATTERNS[method]
    models = {
        m.group("model")
        for f in plans_dir.iterdir()
        if (m := pattern.match(f.name)) is not None
    }
    return sorted(models)


def plan_path_for(
    results_root: Path, task: str, method: str, model: str, index: int
) -> Path:
    if method == "react":
        return results_root / "react_plans" / task / f"react_plan_{model}_{index}.json"
    if method == "direct":
        return results_root / "direct_plans" / task / f"direct_plan_{model}_{index}.json"
    if method == "bipop":
        return results_root / "pddl_plans" / task / f"{task}_pddl_plan_{index}.json"
    raise ValueError(method)


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "task_name",
    "task_index",
    "task_type",
    "method",
    "model_name",
    "plan_path",
    "plan_source",
    "plan_steps",
    "number_of_waits",
    "num_actions",
    "makespan",
    "valid",
    "failure_stage",
    "failure_reason",
    "gt_domain",
    "gt_problem",
    "grounded_plan",
    "val_log",
]


def evaluate_plan(
    spec: TaskSpec,
    plan_path: Path,
    domain_path: Path,
    problem_path: Path,
    tplan_path: Path,
    val_log_path: Path,
    val_bin: str,
    original_plan_text: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "plan_path": str(plan_path),
        "plan_source": "reconstructed_steps",
        "plan_steps": "",
        "number_of_waits": "",
        "num_actions": "",
        "makespan": "",
        "valid": False,
        "failure_stage": "",
        "failure_reason": "",
        "grounded_plan": "",
        "val_log": "",
    }

    if not plan_path.is_file():
        row.update(failure_stage="missing_plan", failure_reason="plan file not found")
        return row
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError("plan JSON is not an object with Left/Right lists")
    except (json.JSONDecodeError, ValueError) as exc:
        row.update(failure_stage="invalid_json", failure_reason=str(exc))
        return row

    steps, waits = compute_plan_stats(plan)
    row.update(plan_steps=steps, number_of_waits=waits)

    if original_plan_text is not None:
        actions = parse_temporal_lines(original_plan_text)
        row["plan_source"] = "original_temporal"
        if not actions:
            row.update(
                failure_stage="grounding",
                failure_reason="recovered temporal plan contains no actions",
            )
            return row
    else:
        grounder = PlanGrounder(spec)
        try:
            actions = grounder.ground_plan(plan)
        except GroundingError as exc:
            row.update(failure_stage="grounding", failure_reason=str(exc))
            return row

    tplan_path.write_text(
        "\n".join(action.to_line() for action in actions) + "\n", encoding="utf-8"
    )
    row.update(
        num_actions=len(actions),
        makespan=f"{max(a.start + a.duration for a in actions):.4f}",
        grounded_plan=str(tplan_path),
    )

    val = run_val(domain_path, problem_path, tplan_path, val_bin)
    if not val["available"]:
        row.update(failure_stage="val_unavailable", failure_reason=val["output"])
        return row

    val_log_path.write_text(val["output"], encoding="utf-8")
    row["val_log"] = str(val_log_path)
    if val["is_valid"]:
        row.update(valid=True)
    else:
        stage, reason = extract_val_failure(val["output"])
        if stage == "goal":
            unmet = diagnose_unmet_goals(domain_path, problem_path, actions)
            if unmet:
                reason = f"goal not satisfied: {unmet}"
        row.update(failure_stage=stage, failure_reason=reason)
    return row


def parse_indices(text: str) -> Optional[List[int]]:
    if not text.strip():
        return None
    indices: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        elif part:
            indices.append(int(part))
    return sorted(set(indices))


def config_indices(config_dir: Path) -> List[int]:
    indices = []
    for f in config_dir.glob("task_config_*.json"):
        matched = re.match(r"task_config_(\d+)\.json$", f.name)
        if matched:
            indices.append(int(matched.group(1)))
    return sorted(indices)


# Mirrors the agent naming used by results/val_results/*_val.csv files.
AGENT_IDS = {"bipop": "bipop-agent", "direct": "llm-agent", "react": "react-agent"}

PLAN_QUALITY_COLUMNS = [
    "task_name",
    "task_index",
    "task_type",
    "agent_id",
    "model_name",
    "plan_path",
    "success",
    "plan_steps",
    "number_of_waits",
    "num_actions",
    "makespan",
    "plan_source",
    "failure_stage",
    "failure_reason",
]


def write_plan_quality_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Per-plan step/length/makespan table in the same shape as the existing
    results/val_results/<task>_<model>_<agent>_val.csv files, so plan quality
    can be compared across methods."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_QUALITY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task_name": row["task_name"],
                    "task_index": row["task_index"],
                    "task_type": row.get("task_type", ""),
                    "agent_id": AGENT_IDS.get(row["method"], row["method"]),
                    "model_name": row["model_name"],
                    "plan_path": row["plan_path"],
                    "success": row["valid"],
                    "plan_steps": row.get("plan_steps", ""),
                    "number_of_waits": row.get("number_of_waits", ""),
                    "num_actions": row.get("num_actions", ""),
                    "makespan": row.get("makespan", ""),
                    "plan_source": row.get("plan_source", ""),
                    "failure_stage": row.get("failure_stage", ""),
                    "failure_reason": row.get("failure_reason", ""),
                }
            )


# File-name agent labels used by plan_validation.py (e.g.
# StackBlock_gpt-4.1_bipop_agent_val.csv).
AGENT_FILE_LABELS = {
    "direct": "llm_agent",
    "bipop": "bipop_agent",
    "react": "react_agent",
}


def write_per_task_method_csvs(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    """One quality CSV per (task, model, method), named like the existing
    val_results files: <task>_<model>_<agent>_gt_val.csv. `task_achieved`
    mirrors `success` so these are drop-in for tools expecting the val.csv
    schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["task_name"], row["method"], row["model_name"]), []
        ).append(row)

    columns = PLAN_QUALITY_COLUMNS + ["task_achieved"]
    written: List[Path] = []
    for (task, method, model), group in sorted(groups.items()):
        agent_label = AGENT_FILE_LABELS.get(method, method)
        path = out_dir / f"{task}_{model}_{agent_label}_gt_val.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in sorted(group, key=lambda r: int(r["task_index"])):
                writer.writerow(
                    {
                        "task_name": row["task_name"],
                        "task_index": row["task_index"],
                        "task_type": row.get("task_type", ""),
                        "agent_id": AGENT_IDS.get(row["method"], row["method"]),
                        "model_name": row["model_name"],
                        "plan_path": row["plan_path"],
                        "success": row["valid"],
                        "plan_steps": row.get("plan_steps", ""),
                        "number_of_waits": row.get("number_of_waits", ""),
                        "num_actions": row.get("num_actions", ""),
                        "makespan": row.get("makespan", ""),
                        "plan_source": row.get("plan_source", ""),
                        "failure_stage": row.get("failure_stage", ""),
                        "failure_reason": row.get("failure_reason", ""),
                        "task_achieved": row["valid"],
                    }
                )
        written.append(path)
    return written


def write_plan_quality_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    """Aggregated plan quality per (task, method, model): success rate and
    mean/median steps and makespan over the GT-valid plans only."""
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["task_name"], row["method"], row["model_name"]), []
        ).append(row)

    def mean(values: List[float]) -> str:
        return f"{sum(values) / len(values):.3f}" if values else ""

    def median(values: List[float]) -> str:
        if not values:
            return ""
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return f"{ordered[mid]:.3f}"
        return f"{(ordered[mid - 1] + ordered[mid]) / 2:.3f}"

    columns = [
        "task_name", "method", "agent_id", "model_name", "n_plans", "n_valid",
        "success_rate", "mean_plan_steps_valid", "mean_waits_valid",
        "mean_num_actions_valid", "mean_makespan_valid", "median_makespan_valid",
        "min_makespan_valid", "max_makespan_valid",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for (task, method, model), group in sorted(groups.items()):
            valid = [r for r in group if r["valid"]]
            steps = [float(r["plan_steps"]) for r in valid if r.get("plan_steps") != ""]
            waits = [
                float(r["number_of_waits"]) for r in valid if r.get("number_of_waits") != ""
            ]
            n_actions = [
                float(r["num_actions"]) for r in valid if r.get("num_actions") != ""
            ]
            makespans = [float(r["makespan"]) for r in valid if r.get("makespan") != ""]
            writer.writerow(
                {
                    "task_name": task,
                    "method": method,
                    "agent_id": AGENT_IDS.get(method, method),
                    "model_name": model,
                    "n_plans": len(group),
                    "n_valid": len(valid),
                    "success_rate": f"{len(valid) / len(group):.3f}" if group else "",
                    "mean_plan_steps_valid": mean(steps),
                    "mean_waits_valid": mean(waits),
                    "mean_num_actions_valid": mean(n_actions),
                    "mean_makespan_valid": mean(makespans),
                    "median_makespan_valid": median(makespans),
                    "min_makespan_valid": f"{min(makespans):.3f}" if makespans else "",
                    "max_makespan_valid": f"{max(makespans):.3f}" if makespans else "",
                }
            )


def write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["task_name"], row["method"], row["model_name"]), []
        ).append(row)

    columns = [
        "task_name", "method", "model_name", "n_plans", "n_valid", "success_rate",
        "top_failure_stages",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for (task, method, model), group in sorted(groups.items()):
            n = len(group)
            n_valid = sum(1 for r in group if r["valid"])
            stages = Counter(r["failure_stage"] for r in group if not r["valid"])
            writer.writerow(
                {
                    "task_name": task,
                    "method": method,
                    "model_name": model,
                    "n_plans": n,
                    "n_valid": n_valid,
                    "success_rate": f"{n_valid / n:.3f}" if n else "",
                    "top_failure_stages": "; ".join(
                        f"{stage}:{count}" for stage, count in stages.most_common()
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["ServeWater", "ServeFruit", "StackBlock"],
        help="Tasks with configurations in --task-configs (default: the three shipped tasks).",
    )
    parser.add_argument("--methods", nargs="+", default=["bipop", "direct", "react"],
                        choices=["bipop", "direct", "react"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Restrict react/direct evaluation to these models.")
    parser.add_argument("--indices", type=str, default="",
                        help="Task indices, e.g. '0-9,15'. Default: all configs.")
    parser.add_argument("--task-configs", type=Path, default=TASK_CONFIGS_DIR)
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <results-root>/gt_eval")
    parser.add_argument("--val-bin", type=str, default=default_val_bin() or "Validate")
    parser.add_argument("--bipop-model-label", type=str, default="gpt-4.1",
                        help="model_name column value for the bipop method.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_root / "gt_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_indices = parse_indices(args.indices)

    rows: List[Dict[str, Any]] = []
    for task in args.tasks:
        config_dir = args.task_configs / task
        if not config_dir.is_dir():
            print(f"[skip] no task_configs for {task}", file=sys.stderr)
            continue
        indices = config_indices(config_dir)
        if requested_indices is not None:
            indices = [i for i in indices if i in requested_indices]

        gt_dir = output_dir / task / "gt"
        plans_dir = output_dir / task / "plans"
        logs_dir = output_dir / task / "val_logs"
        for directory in (gt_dir, plans_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # methods/models available for this task
        method_models: List[Tuple[str, str]] = []
        if "react" in args.methods:
            for model in discover_models(args.results_root / "react_plans" / task, "react"):
                method_models.append(("react", model))
        if "direct" in args.methods:
            for model in discover_models(args.results_root / "direct_plans" / task, "direct"):
                method_models.append(("direct", model))
        if "bipop" in args.methods and (args.results_root / "pddl_plans" / task).is_dir():
            method_models.append(("bipop", args.bipop_model_label))
        if args.models:
            method_models = [
                (method, model)
                for method, model in method_models
                if method == "bipop" or model in args.models
            ]

        print(
            f"[task] {task}: {len(indices)} configs | evaluating "
            + ", ".join(f"{m}:{mod}" for m, mod in method_models)
        )

        for index in indices:
            config = json.loads(
                (config_dir / f"task_config_{index}.json").read_text(encoding="utf-8")
            )
            spec = build_task_spec(task, config)
            domain_path = gt_dir / f"domain_{index}.pddl"
            problem_path = gt_dir / f"problem_{index}.pddl"
            try:
                generate_ground_truth_pddl(
                    spec, domain_path, problem_path, gt_dir / f"gt_gen_{index}.py"
                )
            except Exception as exc:
                print(f"[error] GT generation failed for {task} #{index}: {exc}",
                      file=sys.stderr)
                continue

            for method, model in method_models:
                plan_path = plan_path_for(args.results_root, task, method, model, index)
                stem = f"{method}_{model}_{index}"
                original_plan_text = None
                if method == "bipop" and plan_path.is_file():
                    try:
                        saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                        original_plan_text = find_original_temporal_plan(
                            args.results_root, task, index, saved_plan
                        )
                    except (json.JSONDecodeError, OSError):
                        original_plan_text = None
                row = evaluate_plan(
                    spec,
                    plan_path,
                    domain_path,
                    problem_path,
                    plans_dir / f"{stem}.tplan",
                    logs_dir / f"{stem}.log",
                    args.val_bin,
                    original_plan_text=original_plan_text,
                )
                row.update(
                    task_name=task,
                    task_index=index,
                    task_type=json.dumps(config.get("type", "")),
                    method=method,
                    model_name=model,
                    gt_domain=str(domain_path),
                    gt_problem=str(problem_path),
                )
                rows.append(row)
                if args.verbose:
                    status = "VALID" if row["valid"] else f"FAIL[{row['failure_stage']}]"
                    print(f"  #{index} {method}/{model}: {status} {row['failure_reason'][:100]}")

    csv_path = output_dir / "gt_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})

    summary_path = output_dir / "gt_evaluation_summary.csv"
    write_summary(rows, summary_path)

    quality_path = output_dir / "gt_plan_quality.csv"
    write_plan_quality_csv(rows, quality_path)
    quality_summary_path = output_dir / "gt_plan_quality_summary.csv"
    write_plan_quality_summary(rows, quality_summary_path)
    per_file_paths = write_per_task_method_csvs(rows, output_dir / "quality_csv")

    n_valid = sum(1 for r in rows if r["valid"])
    print(f"\n[done] {len(rows)} plans evaluated | {n_valid} valid")
    print(f"[out]  {csv_path}")
    print(f"[out]  {summary_path}")
    print(f"[out]  {quality_path}")
    print(f"[out]  {quality_summary_path}")
    print(f"[out]  {len(per_file_paths)} per-task/method CSVs in {output_dir / 'quality_csv'}")


if __name__ == "__main__":
    main()
