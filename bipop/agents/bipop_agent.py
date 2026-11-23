"""BiPOP-agent: the LLM writes the PDDL problem, TFD finds a temporal partial-order plan."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from ..llm.llm_pddl import DEFAULT_MAX_TRIALS, llm_pddl
from ..llm.planner import run_val
from ..llm.two_stage import uses_two_stage
from ..plan_format import compute_plan_stats, parse_temporal_plan_text
from .base import AgentRunContext, BasePlanningAgent, GenerationResult

BEST_VALID_PRESET_RE = re.compile(r"^Best valid planner preset:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
STAGE_BEST_VALID_PRESET_RE = re.compile(
    r"^Stage\s+([12])\s+best valid planner preset:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
SINGLE_PIPELINE_PRESET_RE = re.compile(
    r"^Single pipeline best valid planner preset:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
PRESET_NOT_AVAILABLE = "NA"


# ---------------------------------------------------------------------------
# Which TFD preset produced the selected plan (for the logs)
# ---------------------------------------------------------------------------

def _normalize_best_valid_preset(value: Any) -> str:
    if not isinstance(value, str):
        return PRESET_NOT_AVAILABLE
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null", "na", "n/a"}:
        return PRESET_NOT_AVAILABLE
    return normalized


def _read_combined_plan_stage_presets(combined_plan_path: Path) -> Dict[str, str]:
    if not combined_plan_path.exists():
        return {}
    try:
        payload = json.loads(combined_plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    presets: Dict[str, str] = {}
    for stage_key in ("stage1", "stage2"):
        stage_payload = payload.get(stage_key)
        if not isinstance(stage_payload, dict):
            continue
        preset = _normalize_best_valid_preset(stage_payload.get("best_valid_preset", stage_payload.get("best_preset")))
        if preset != PRESET_NOT_AVAILABLE:
            presets[stage_key] = preset
    return presets


def _parse_stage_presets_from_output(planner_output: str | None) -> Dict[str, str]:
    presets: Dict[str, str] = {}
    for match in STAGE_BEST_VALID_PRESET_RE.finditer(planner_output or ""):
        preset = _normalize_best_valid_preset(match.group(2))
        if preset != PRESET_NOT_AVAILABLE:
            presets[f"stage{match.group(1)}"] = preset
    return presets


def _parse_single_preset_from_output(planner_output: str | None) -> str:
    matches = BEST_VALID_PRESET_RE.findall(planner_output or "")
    if not matches:
        return PRESET_NOT_AVAILABLE
    return _normalize_best_valid_preset(matches[-1])


def extract_best_valid_planner_preset_metadata(
    intermediate_dir: Path,
    task_name: str,
    index: int,
    planner_output: str | None,
    divide_goal: bool,
) -> Dict[str, Any]:
    """Best valid TFD preset of the selected plan.

    With ``divide_goal`` it is one preset per stage if the two-stage plan won,
    or ``single=<preset>`` if the single pipeline won.
    """
    if not divide_goal:
        return {
            "best_valid_planner_preset": _parse_single_preset_from_output(planner_output),
            "stage_best_valid_planner_presets": {},
        }

    stage_presets = _read_combined_plan_stage_presets(intermediate_dir / f"{task_name}_plan_{index}.json")
    for stage_key, preset in _parse_stage_presets_from_output(planner_output).items():
        stage_presets.setdefault(stage_key, preset)

    stage1 = stage_presets.get("stage1", PRESET_NOT_AVAILABLE)
    stage2 = stage_presets.get("stage2", PRESET_NOT_AVAILABLE)
    if stage1 == PRESET_NOT_AVAILABLE and stage2 == PRESET_NOT_AVAILABLE:
        single_matches = SINGLE_PIPELINE_PRESET_RE.findall(planner_output or "")
        single = _normalize_best_valid_preset(single_matches[-1]) if single_matches else PRESET_NOT_AVAILABLE
        summary = PRESET_NOT_AVAILABLE if single == PRESET_NOT_AVAILABLE else f"single={single}"
    else:
        summary = f"stage1={stage1};stage2={stage2}"
    return {
        "best_valid_planner_preset": summary,
        "stage_best_valid_planner_presets": {"stage1": stage1, "stage2": stage2},
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _stale_artifacts(plan_dir: Path, intermediate_dir: Path, task: str, index: int) -> List[Path]:
    """Final plans and PDDL files a previous run of this instance may have left behind.

    Exact names only: a wildcard after the index (``_plan_1*``) would also match instance 10, 11, ...
    """
    stems = [f"{task}_plan_{index}", f"{task}_plan_{index}_1", f"{task}_plan_{index}_2"]
    paths = [path for stem in stems for path in intermediate_dir.glob(f"{stem}.txt*")]
    paths.append(intermediate_dir / f"{task}_plan_{index}.json")
    for kind in ("domain", "problem"):
        paths += [intermediate_dir / f"{task}_{kind}_{index}{suffix}.pddl" for suffix in ("", "_stage1", "_stage2")]
    paths += [plan_dir / f"{task}_pddl_plan_{index}{suffix}.json" for suffix in ("", "_1", "_2")]
    return paths


class BiPOPAgent(BasePlanningAgent):
    agent_id = "bipop-agent"
    display_name = "BiPOP-agent"
    is_pddl = True

    def __init__(
        self,
        model_name: str,
        task_name: str,
        temperature: float = 0.0,
        max_trials: int = DEFAULT_MAX_TRIALS,
        divide_goal: bool = True,
    ):
        super().__init__(model_name, task_name)
        self.temperature = temperature
        self.max_trials = max_trials
        self.divide_goal = divide_goal

    def generate(self, context: AgentRunContext) -> GenerationResult:
        """Generate a plan and save it as ``<pddl_plan_dir>/<task>_pddl_plan_<index>.json``.

        All planner artifacts go to ``<pddl_plan_dir>/intermediate``.
        """
        if context.pddl_plan_dir is None:
            raise ValueError("pddl_plan_dir must be provided for bipop-agent")

        plan_dir = context.pddl_plan_dir
        intermediate_dir = plan_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        context_divide_goal = context.divide_goal if context.divide_goal is not None else True
        divide_goal = uses_two_stage(self.task_name, context_divide_goal and self.divide_goal)
        task, index = self.task_name, context.index

        # Without this a failed run would report (and execute) the plan of an earlier run.
        for old_path in _stale_artifacts(plan_dir, intermediate_dir, task, index):
            if old_path.is_file():
                old_path.unlink()

        start = time.time()
        code_string, planner_output, llm_time_cost, planner_time_cost = llm_pddl(
            self.model_name,
            task,
            context.task_description,
            str(plan_dir),
            index,
            temperature=self.temperature,
            max_trials=self.max_trials,
            divide_goal=divide_goal,
            val_bin=context.val_bin,
            intermediate_dir=str(intermediate_dir),
        )
        elapsed = time.time() - start

        (intermediate_dir / f"code_string_{index}.json").write_text(code_string or "", encoding="utf-8")

        # Discretize the temporal plan into the step-aligned plan the robot executes.
        plan_text_path = intermediate_dir / f"{task}_plan_{index}.txt"
        plan_json_path = plan_dir / f"{task}_pddl_plan_{index}.json"
        plan = None
        if plan_text_path.exists():
            plan = parse_temporal_plan_text(plan_text_path.read_text(encoding="utf-8"))
            plan_json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        stage_plan_lengths: Dict[str, int] = {}
        if divide_goal:
            for stage in (1, 2):
                stage_text_path = intermediate_dir / f"{task}_plan_{index}_{stage}.txt"
                if stage_text_path.exists():
                    stage_plan = parse_temporal_plan_text(stage_text_path.read_text(encoding="utf-8"))
                    stage_json_path = plan_dir / f"{task}_pddl_plan_{index}_{stage}.json"
                    stage_json_path.write_text(json.dumps(stage_plan, indent=2), encoding="utf-8")
                    stage_plan_lengths[f"stage{stage}"] = compute_plan_stats(stage_plan)[0]

        # End-to-end check of the final plan against the full (single) problem.
        # It has stage 1's initial state and stage 2's final goals, so it also
        # confirms that a combined two-stage plan is consistent: stage 2
        # really starts from the state stage 1 reaches.
        val_report = None
        if context.validate_with_val and context.val_bin is not None and plan_text_path.exists():
            domain_path = intermediate_dir / f"{task}_domain_{index}.pddl"
            problem_path = intermediate_dir / f"{task}_problem_{index}.pddl"
            if domain_path.exists() and problem_path.exists():
                val_report = run_val(domain_path, problem_path, plan_text_path, context.val_bin, verbose=False)

        plan_steps, _ = compute_plan_stats(plan)
        planner_success = bool(planner_output and "New solution has been found" in planner_output)
        final_success = planner_success and plan_steps > 0
        if val_report and val_report.get("available"):
            final_success = final_success and bool(val_report.get("is_valid"))

        return GenerationResult(
            agent_id=self.agent_id,
            plan_path=plan_json_path if plan_json_path.exists() else None,
            plan=plan,
            success=final_success,
            llm_time_cost=llm_time_cost,
            planner_time_cost=planner_time_cost,
            generation_time=elapsed,
            plan_length=plan_steps,
            metadata={
                "planner_output": planner_output,
                "val": val_report,
                "divide_goal": divide_goal,
                # divide_goal only means the split was tried; this says it also won.
                "two_stage_winner": divide_goal and bool(planner_output and "Winner: two-stage" in planner_output),
                "stage_plan_lengths": stage_plan_lengths if divide_goal else {},
                **extract_best_valid_planner_preset_metadata(
                    intermediate_dir, task, index, planner_output, divide_goal
                ),
            },
        )
