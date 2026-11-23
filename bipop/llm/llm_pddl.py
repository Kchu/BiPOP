"""BiPOP planning loop: the LLM writes the PDDL problem as Unified Planning code, TFD plans it.

Each trial asks the LLM for one full code block (objects, initial state,
goals). The code is solved and VAL-validated; on failure the error report is
fed back to the LLM and the next trial starts.

With ``divide_goal`` two pipelines run in parallel on the same code:

* single:    the full problem in one TFD run;
* two-stage: stage 1 (``object_at_point`` goals) and stage 2 (all goals, from
  stage 1's reached state), see :mod:`bipop.llm.two_stage`.

The valid plan with the smaller makespan wins (ties go to two-stage).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

from ..plan_format import combine_plan_texts, combined_plan_offset
from .client import build_client, chat
from .code_blocks import choose_actions
from .log import debug_print
from .planner import (
    cleanup_files,
    emit_tfd_val_report,
    format_best_valid_preset,
    format_optional_float,
    plan_quality_key,
    solve_code_string,
)
from .prompts import choose_pddl_rule_description, pddl_system_message
from .two_stage import (
    combined_plan_json_payload,
    compose_stage2_code,
    extract_stage1_state_context,
    goal_code_error,
    manual_split_full_code,
)

# LLM trials per task instance; the default of --max_trials in plan_generation.py and main.py.
DEFAULT_MAX_TRIALS = 3

NO_CODE_BLOCK_MESSAGE = (
    "No valid Python code block output was returned. "
    "Return only runnable Python in one fenced ```python``` block."
)
CRASH_RETRY_INSTRUCTION = "The code crashed while generating PDDL. Fix and regenerate based on the traceback above."
NO_PLAN_RETRY_INSTRUCTION = (
    "The code you generated is incorrect. There is no valid plan generated! "
    "Please check your definitions, initial values, object_at_point final "
    "goals, and final goals, then regenerate one full Python code block."
)
STAGE1_VALID_RETRY_INSTRUCTION = (
    "The first manually derived stage is already VAL-valid. "
    "Regenerate one full Python code block, keeping object/point definitions "
    "and initial values equivalent if possible, and fix the final goal lines. "
    "The pipeline will reuse or recompute stage1 as needed and will derive "
    "stage2 from all final problem.add_goal(...) lines."
)


def extract_code_string(response_content: str) -> str | None:
    """Content of the first fenced ```python``` block, or None."""
    matched = re.search(r"```python\s*(.*?)\s*```", response_content, re.DOTALL)
    return matched.group(1).strip() if matched else None


@dataclass
class _PlanningJob:
    """Everything needed to solve generated code for one task instance."""

    plan_dir: str            # generated planning scripts
    intermediate_dir: str    # PDDL files, planner runs, logs
    task_name: str
    index: int
    action_definition: str
    val_bin: Any

    def solve(self, code: str, stage_label: str, trial: int) -> Dict[str, Any]:
        return solve_code_string(
            code,
            self.plan_dir,
            self.task_name,
            self.index,
            stage_label,
            self.action_definition,
            trial=trial,
            val_bin=self.val_bin,
            intermediate_dir=self.intermediate_dir,
        )


def _crash_output(exc: subprocess.CalledProcessError) -> str:
    """stdout + stderr of a crashed generated script."""
    return ((exc.stdout or "").strip() + "\n" + (exc.stderr or "").strip()).strip()


def _failed_result(output: str) -> Dict[str, Any]:
    return {
        "success": False,
        "output": output,
        "plan_path": None,
        "plan_text": None,
        "makespan": None,
        "plan_length": None,
        "planner_time_cost": 0.0,
    }


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def _save_dialog(messages: List[Dict[str, str]], log_dir: str, task_name: str, index: int) -> None:
    dialog_path = Path(log_dir) / f"{task_name}_dialog_{index}.json"
    dialog_path.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_code_string(stage_label: str, trial: int, code_string: str, note: str = "") -> None:
    note_text = f" ({note})" if note else ""
    debug_print(f"\n=== {stage_label} trial {trial}{note_text} ===", flush=True)
    debug_print(code_string.rstrip(), flush=True)
    debug_print(f"=== END {stage_label} trial {trial} ===\n", flush=True)


def _write_code_string_log(
    log_dir: str,
    index: int,
    stage_label: str,
    trial: int,
    code_string: str,
    response_content: str,
) -> Dict[str, str]:
    """Save ``code_string_<index>_<stage>_trial_<n>.py`` and a JSON log with the raw LLM answer."""
    log_base = Path(log_dir) / f"code_string_{index}_{stage_label}_trial_{trial}"
    code_path = log_base.with_suffix(".py")
    log_path = log_base.with_name(log_base.name + "_log.json")

    code_path.write_text(code_string, encoding="utf-8")
    payload = {
        "index": index,
        "stage": stage_label,
        "trial": trial,
        "code_path": str(code_path),
        "raw_response": response_content or "",
        "extracted_code": code_string,
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    debug_print(f"[CODE LOG] {stage_label} trial {trial}: {code_path} | {log_path}", flush=True)
    return {"code_path": str(code_path), "log_path": str(log_path)}


def _write_trial_summary(
    log_dir: str,
    index: int,
    trials_attempted: int,
    max_trials: int,
    success: bool,
    last_error_message: str,
    trial_errors: List[Dict[str, Any]],
) -> None:
    summary_path = Path(log_dir) / f"llm_trial_summary_{index}.json"
    if not trial_errors and success:
        last_error_message = ""
    payload = {
        "index": index,
        "trials_attempted": trials_attempted,
        "max_trials": max_trials,
        "success": success,
        "last_error_message": last_error_message or "",
        "trial_errors": trial_errors,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    debug_print(f"[LLM] trial summary: {summary_path}", flush=True)


# ---------------------------------------------------------------------------
# The two pipelines of one trial (run in parallel; they never raise)
# ---------------------------------------------------------------------------

def _run_single_pipeline(job: _PlanningJob, code_string: str, trial: int) -> Dict[str, Any]:
    """Solve the full problem in one planner run."""
    try:
        return job.solve(code_string, "single", trial)
    except subprocess.CalledProcessError as exc:
        message = _crash_output(exc) or f"Single PDDL script crashed (rc={exc.returncode})."
        debug_print(f"[SINGLE] trial {trial} PDDL crashed: {message[:120]}", flush=True)
        return _failed_result(message)
    except Exception as exc:
        message = f"Single pipeline error: {type(exc).__name__}: {exc}"
        debug_print(f"[SINGLE] trial {trial} exception: {message[:120]}", flush=True)
        return _failed_result(message)


def _run_two_stage_pipeline(
    job: _PlanningJob,
    trial: int,
    stage1_code: str,
    stage2_goal_code: str,
    stage1_cache: Optional[Dict[str, Any]],
    stage2_log_paths: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Stage 1, state transfer, stage 2. A valid cached stage 1 is reused."""
    idir = job.intermediate_dir
    result_1 = None
    result_2 = None
    cache_update = None
    planner_time_1 = 0.0
    planner_time_2 = 0.0
    stage2_code = None
    goal_error = None
    error = None

    try:
        if stage1_cache is not None:
            result_1 = stage1_cache["result_1"]
            injected_init_code = stage1_cache["injected_init_code"]
            emit_tfd_val_report(
                idir, job.index, "stage1", trial,
                result_1.get("output") or "Reusing cached VAL-valid stage-1 result.",
                cached=True,
            )
        else:
            result_1 = job.solve(stage1_code, "stage1", trial)
            planner_time_1 = result_1.get("planner_time_cost", 0.0)

        if result_1["success"]:
            if stage1_cache is None:
                state_context = extract_stage1_state_context(
                    result_1["py_path"], result_1["domain"], result_1["problem"], result_1["plan_text"]
                )
                injected_init_code = state_context["injected_init_code"]
                achieved_state_path = Path(idir) / f"achieved_state_{job.index}_stage1.txt"
                achieved_state_path.write_text("\n".join(state_context["state_lines"]), encoding="utf-8")
                (Path(idir) / f"stage2_injected_init_{job.index}.py").write_text(injected_init_code, encoding="utf-8")
                cache_update = {
                    "stage1_code": stage1_code,
                    "result_1": result_1,
                    "injected_init_code": injected_init_code,
                }
                debug_print(
                    f"[STATE] stage1 achieved state: {len(state_context['state_lines'])} facts -> {achieved_state_path}",
                    flush=True,
                )

            goal_error = goal_code_error(stage2_goal_code, "subgoal2")
            if goal_error is None:
                stage2_code = compose_stage2_code(injected_init_code, stage2_goal_code)
                _print_code_string("MANUAL_STAGE2_EXECUTABLE", trial, stage2_code, "injected init + goals")
                executable_path = Path(idir) / f"manual_stage2_executable_{job.index}_trial_{trial}.py"
                executable_path.write_text(stage2_code, encoding="utf-8")
                if stage2_log_paths:
                    log_path = Path(stage2_log_paths["log_path"])
                    log_payload = json.loads(log_path.read_text(encoding="utf-8"))
                    log_payload["executable_code_path"] = str(executable_path)
                    log_payload["executable_code"] = stage2_code
                    log_path.write_text(json.dumps(log_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                debug_print(f"[CODE LOG] manual stage2 trial {trial} executable: {executable_path}", flush=True)
                result_2 = job.solve(stage2_code, "stage2", trial)
                planner_time_2 = result_2.get("planner_time_cost", 0.0)

    except subprocess.CalledProcessError as exc:
        error = ("pddl_crash", exc)
    except Exception as exc:
        error = ("state_error", exc)

    return {
        "result_1": result_1,
        "result_2": result_2,
        "cache_update": cache_update,
        "planner_time_1": planner_time_1,
        "planner_time_2": planner_time_2,
        "stage2_executable_code": stage2_code,
        "goal_error": goal_error,
        "error": error,
    }


def _two_stage_makespan(result_1: Dict[str, Any], result_2: Dict[str, Any]) -> float | None:
    """Makespan of the combined plan: stage-1 end + epsilon + stage-2 makespan."""
    stage2_makespan = result_2.get("makespan")
    if stage2_makespan is None:
        return None
    return float(combined_plan_offset(result_1["plan_text"])) + stage2_makespan


# ---------------------------------------------------------------------------
# Final plan
# ---------------------------------------------------------------------------

def _write_two_stage_artifacts(
    job: _PlanningJob,
    canonical_plan: str,
    result_1: Dict[str, Any],
    result_2: Dict[str, Any],
    stage1_code: str,
    stage2_goal_code: Optional[str],
    stage2_executable_code: Optional[str],
    manual_split: Optional[Dict[str, Any]],
) -> str:
    """Write stage plans, the combined plan and the split metadata. Returns the combined JSON path."""
    idir, task_name, index = job.intermediate_dir, job.task_name, job.index
    Path(f"{idir}/{task_name}_plan_{index}_1.txt").write_text(result_1["plan_text"], encoding="utf-8")
    Path(f"{idir}/{task_name}_plan_{index}_2.txt").write_text(result_2["plan_text"], encoding="utf-8")
    final_plan_text = combine_plan_texts(result_1["plan_text"], result_2["plan_text"])
    Path(canonical_plan).write_text(final_plan_text, encoding="utf-8")

    combined_plan_json_path = f"{idir}/{task_name}_plan_{index}.json"
    Path(combined_plan_json_path).write_text(
        json.dumps(
            combined_plan_json_payload(task_name, index, result_1, result_2, final_plan_text),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    debug_print(f"[PLAN LOG] combined plan json: {combined_plan_json_path}", flush=True)

    manual_stage1_path = Path(idir) / f"manual_stage1_code_{index}.py"
    manual_stage2_goal_path = Path(idir) / f"manual_stage2_goal_code_{index}.py"
    manual_stage2_executable_path = Path(idir) / f"manual_stage2_executable_code_{index}.py"
    manual_split_path = Path(idir) / f"manual_split_{index}.json"
    manual_stage1_path.write_text(stage1_code, encoding="utf-8")
    manual_stage2_goal_path.write_text(stage2_goal_code or "", encoding="utf-8")
    manual_stage2_executable_path.write_text(stage2_executable_code or "", encoding="utf-8")
    manual_split_payload = {
        "source": "single_llm_code_string",
        "stage1_principle": "subgoal1 is the object_at_point(...) final goals",
        "stage2_principle": "subgoal2 is all final goals from the full code string",
        "stage1_code_path": str(manual_stage1_path),
        "stage2_goal_code_path": str(manual_stage2_goal_path),
        "stage2_executable_code_path": str(manual_stage2_executable_path),
        "subgoal1_lines": manual_split.get("subgoal1_lines", []) if manual_split else [],
        "final_goal_lines": manual_split.get("final_goal_lines", []) if manual_split else [],
    }
    manual_split_path.write_text(json.dumps(manual_split_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    debug_print(f"[PLAN LOG] manual split metadata: {manual_split_path}", flush=True)
    return combined_plan_json_path


def _two_stage_winner_output(
    job: _PlanningJob,
    canonical_plan: str,
    combined_plan_json_path: Optional[str],
    result_1: Dict[str, Any],
    result_2: Dict[str, Any],
    result_single: Optional[Dict[str, Any]],
    single_pipeline_success: bool,
    stage_times: Dict[str, float],
    planner_time_cost: float,
) -> str:
    idir, task_name, index = job.intermediate_dir, job.task_name, job.index
    two_stage_length = (result_1.get("plan_length") or 0) + (result_2.get("plan_length") or 0)
    two_stage_makespan = _two_stage_makespan(result_1, result_2)
    single_makespan = result_single.get("makespan") if result_single else None
    winner_note = (
        f"combined makespan {format_optional_float(two_stage_makespan)} <= "
        f"single makespan {format_optional_float(single_makespan)}"
        if single_pipeline_success
        else "single pipeline did not find a valid plan"
    )
    effort = stage_times["stage1"] + stage_times["stage2"] + stage_times["single"]
    return (
        "New solution has been found.\n"
        f"Full code string: {idir}/code_string_{index}.json\n"
        f"Manual split metadata: {idir}/manual_split_{index}.json\n"
        f"Stage 1 best valid planner preset: {format_best_valid_preset(result_1.get('best_valid_preset'))}\n"
        f"Stage 2 best valid planner preset: {format_best_valid_preset(result_2.get('best_valid_preset'))}\n"
        f"Stage 1 plan length: {result_1.get('plan_length')}\n"
        f"Stage 2 plan length: {result_2.get('plan_length')}\n"
        f"Combined plan length: {two_stage_length}\n"
        f"Single pipeline plan length: {result_single.get('plan_length') if result_single else 'NA'}\n"
        f"Stage 1 planner time (effort): {stage_times['stage1']:.6f}\n"
        f"Stage 2 planner time (effort): {stage_times['stage2']:.6f}\n"
        f"Single pipeline planner time (effort): {stage_times['single']:.6f}\n"
        f"Total planner effort: {effort:.6f}\n"
        f"Total planner time (TFD wall-clock): {planner_time_cost:.6f}\n"
        f"Plan 1: {idir}/{task_name}_plan_{index}_1.txt\n"
        f"Plan 2: {idir}/{task_name}_plan_{index}_2.txt\n"
        f"Final plan: {canonical_plan}\n"
        f"Final plan JSON: {combined_plan_json_path}\n"
        f"Combined makespan: {format_optional_float(two_stage_makespan)}\n"
        f"Winner: two-stage ({winner_note})\n"
        "\n=== PLAN 1 SOLVER OUTPUT ===\n"
        + result_1["output"]
        + "\n=== PLAN 2 SOLVER OUTPUT ===\n"
        + result_2["output"]
    )


def _single_winner_output(
    job: _PlanningJob,
    canonical_plan: str,
    result_1: Dict[str, Any],
    result_2: Optional[Dict[str, Any]],
    result_single: Optional[Dict[str, Any]],
    two_stage_success: bool,
    stage_times: Dict[str, float],
    planner_time_cost: float,
) -> str:
    idir, index = job.intermediate_dir, job.index
    single_makespan = result_single.get("makespan") if result_single else None
    if two_stage_success:
        single_note = (
            f" (single makespan {format_optional_float(single_makespan)}"
            f" < two-stage combined makespan {format_optional_float(_two_stage_makespan(result_1, result_2))})"
        )
    else:
        single_note = " (two-stage pipeline did not find a valid plan)"
    effort = stage_times["single"] + stage_times["stage1"] + stage_times["stage2"]
    single_preset = result_single.get("best_valid_preset") if result_single else None
    return (
        "New solution has been found.\n"
        f"Full code string: {idir}/code_string_{index}.json\n"
        f"Single pipeline best valid planner preset: {format_best_valid_preset(single_preset)}\n"
        f"Single pipeline plan length: {result_single.get('plan_length') if result_single else 'NA'}\n"
        f"Stage 1 plan length: {result_1.get('plan_length') if result_1.get('success') else 'NA'}\n"
        f"Stage 2 plan length: {result_2.get('plan_length') if result_2 and result_2.get('success') else 'NA'}\n"
        f"Single pipeline planner time (effort): {stage_times['single']:.6f}\n"
        f"Stage 1 planner time (effort): {stage_times['stage1']:.6f}\n"
        f"Stage 2 planner time (effort): {stage_times['stage2']:.6f}\n"
        f"Total planner effort: {effort:.6f}\n"
        f"Total planner time (TFD wall-clock): {planner_time_cost:.6f}\n"
        f"Final plan: {canonical_plan}\n"
        f"Winner: single{single_note}\n"
        "\n=== SINGLE PLAN SOLVER OUTPUT ===\n"
        + ((result_single.get("output") or "") if result_single else "")
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def llm_pddl(
    model_name: str,
    task_name: str,
    task_description: str,
    plan_file_path: str,
    index: int,
    temperature: float | None = 0,
    max_trials: int = DEFAULT_MAX_TRIALS,
    divide_goal: bool = True,
    val_bin: str | os.PathLike[str] | None = None,
    intermediate_dir: str | None = None,
) -> Tuple[str | None, str, float, float]:
    """Generate a plan with BiPOP.

    Generated planning scripts go to ``plan_file_path``; PDDL files, planner
    runs, LLM logs and the final temporal plan
    (``<task>_plan_<index>.txt``) go to ``intermediate_dir``.

    Returns ``(code_string, planner_output, llm_time_cost, planner_time_cost)``;
    ``planner_output`` starts with "New solution has been found." on success
    and holds the last error otherwise.
    """
    idir = intermediate_dir if intermediate_dir is not None else plan_file_path
    Path(idir).mkdir(parents=True, exist_ok=True)
    client = build_client(model_name)
    llm_time_cost = 0.0
    planner_time_cost = 0.0
    stage_times = {"single": 0.0, "stage1": 0.0, "stage2": 0.0}

    action_definition = choose_actions(task_name)
    if not action_definition:
        message = f"Unsupported task for PDDL generation: {task_name}"
        _write_trial_summary(idir, index, 0, max_trials, False, message, [])
        return None, message, llm_time_cost, planner_time_cost

    job = _PlanningJob(plan_file_path, idir, task_name, index, action_definition, val_bin)
    messages = [
        {"role": "system", "content": pddl_system_message},
        {"role": "system", "content": task_description},
        {"role": "system", "content": choose_pddl_rule_description(task_name)},
        {"role": "system", "content": action_definition},
    ]
    if "gpt-5" in model_name:
        temperature = None

    code_string = None
    last_error_message = "No valid plan found after max trials."
    stage1_cache: Dict[str, Any] | None = None
    trial_errors: List[Dict[str, Any]] = []
    trials_attempted = 0

    def finish(success: bool, output: str) -> Tuple[str | None, str, float, float]:
        _save_dialog(messages, idir, task_name, index)
        _write_trial_summary(idir, index, trials_attempted, max_trials, success, last_error_message, trial_errors)
        return code_string, output, llm_time_cost, planner_time_cost

    for trial in range(max_trials):
        trials_attempted = trial + 1
        manual_split: Dict[str, Any] | None = None
        stage2_log_paths: Dict[str, str] | None = None

        # ---- Ask the LLM for one full code block ----
        llm_start = time.time()
        debug_print(f"[LLM] trial {trial} requesting one full code string", flush=True)
        response_content = chat(client, model_name, messages, temperature)
        llm_time_cost += round(time.time() - llm_start, 2)
        debug_print(f"[LLM] trial {trial} response received", flush=True)

        code_string = extract_code_string(response_content)
        stage1_code = code_string
        stage2_goal_code = None
        if not code_string:
            last_error_message = NO_CODE_BLOCK_MESSAGE
            messages.extend(
                [
                    {"role": "assistant", "content": response_content or ""},
                    {"role": "user", "content": NO_CODE_BLOCK_MESSAGE},
                ]
            )
            trial_errors.append({"trial": trial, "error": last_error_message})
            continue

        _print_code_string("CODE_STRING", trial, code_string, "full LLM output")
        _write_code_string_log(idir, index, "full" if divide_goal else "single", trial, code_string, response_content)

        # ---- Split the code into stage 1 and stage 2 ----
        if divide_goal:
            try:
                manual_split = manual_split_full_code(code_string)
            except ValueError as exc:
                last_error_message = f"Could not manually split the one full code string: {exc}"
                messages.extend(
                    [
                        {"role": "assistant", "content": response_content or ""},
                        {
                            "role": "user",
                            "content": (
                                f"Could not manually split the one full code string: {exc}\n"
                                "Regenerate one full Python code block with explicit top-level "
                                "problem.add_goal(...) final goal lines, including object_at_point(...) "
                                "goals for objects that must end at required points."
                            ),
                        },
                    ]
                )
                trial_errors.append({"trial": trial, "error": last_error_message})
                continue

            stage1_code = manual_split["stage1_code"]
            stage2_goal_code = manual_split["stage2_goal_code"]
            _print_code_string("MANUAL_STAGE1_CODE", trial, stage1_code)
            _write_code_string_log(idir, index, "manual_stage1", trial, stage1_code, response_content)
            _print_code_string("MANUAL_STAGE2_GOALS", trial, stage2_goal_code, "final goals")
            stage2_log_paths = _write_code_string_log(
                idir, index, "manual_stage2_goal", trial, stage2_goal_code, response_content
            )

            # Reuse a VAL-valid stage 1 only if the stage-1 code did not change.
            if stage1_cache is not None and stage1_cache["stage1_code"].strip() != stage1_code.strip():
                debug_print("[CODE LOG] manually derived stage1 changed; recomputing stage1.", flush=True)
                stage1_cache = None

        stage2_executable_code = None
        result_single = None
        result_2 = None
        two_stage_success = False
        single_pipeline_success = False
        use_two_stage = False

        if divide_goal:
            # ---- Run the single and the two-stage pipeline in parallel ----
            debug_print(f"[LLM] trial {trial} launching single + two-stage pipelines in parallel", flush=True)
            wall_start = time.time()
            with ThreadPoolExecutor(max_workers=2) as pool:
                single_future = pool.submit(_run_single_pipeline, job, code_string, trial)
                two_stage_future = pool.submit(
                    _run_two_stage_pipeline, job, trial, stage1_code, stage2_goal_code, stage1_cache, stage2_log_paths
                )
                result_single = single_future.result()
                two_stage = two_stage_future.result()
            wall_elapsed = time.time() - wall_start

            result_1 = two_stage["result_1"] if two_stage["result_1"] is not None else {"success": False, "output": ""}
            result_2 = two_stage["result_2"]
            stage2_executable_code = two_stage["stage2_executable_code"]
            two_stage_goal_error = two_stage["goal_error"]
            two_stage_error = two_stage["error"]  # ("pddl_crash" | "state_error", exception) or None
            if two_stage["cache_update"] is not None:
                stage1_cache = two_stage["cache_update"]

            # planner_time_cost counts TFD only (translate + preprocess + search),
            # not VAL, PDDL generation or the stage-1 simulation. The branches run
            # concurrently, so the wall-clock contribution is
            # max(single, stage1 + stage2); per-branch effort is in stage_times.
            single_time = result_single.get("planner_time_cost", 0.0)
            tfd_wall = max(single_time, two_stage["planner_time_1"] + two_stage["planner_time_2"])
            planner_time_cost += tfd_wall
            stage_times["single"] += single_time
            stage_times["stage1"] += two_stage["planner_time_1"]
            stage_times["stage2"] += two_stage["planner_time_2"]
            debug_print(
                f"[TIMING] trial {trial} TFD parallel wall={tfd_wall:.3f}s (section wall={wall_elapsed:.3f}s) "
                f"single={single_time:.3f}s stage1={two_stage['planner_time_1']:.3f}s "
                f"stage2={two_stage['planner_time_2']:.3f}s",
                flush=True,
            )

            if two_stage_error is not None:
                error_type, error = two_stage_error
                if error_type == "pddl_crash":
                    error_message = _crash_output(error) or f"Two-stage PDDL script crashed (rc={error.returncode})."
                else:
                    error_message = (
                        "Failed while extracting or using the achieved state after stage 1: "
                        f"{type(error).__name__}: {error}"
                    )
                last_error_message = error_message
                trial_errors.append({"trial": trial, "error": last_error_message})
                result_1 = {"success": False, "output": error_message}
            elif two_stage_goal_error:
                last_error_message = two_stage_goal_error
                trial_errors.append({"trial": trial, "error": last_error_message})

            two_stage_success = (
                result_1.get("success", False) and result_2 is not None and result_2.get("success", False)
            )
            single_pipeline_success = result_single.get("success", False)

            if not (two_stage_success or single_pipeline_success):
                # ---- Feed the failure back to the LLM ----
                if two_stage_error is not None:
                    error_type, error = two_stage_error
                    if error_type == "pddl_crash":
                        no_plan_msg = _crash_output(error) or last_error_message
                        messages.extend(
                            [
                                {"role": "assistant", "content": code_string},
                                {"role": "user", "content": no_plan_msg},
                                {"role": "user", "content": CRASH_RETRY_INSTRUCTION},
                            ]
                        )
                    else:
                        no_plan_msg = last_error_message
                        messages.extend(
                            [
                                {"role": "assistant", "content": code_string or ""},
                                {"role": "user", "content": no_plan_msg},
                            ]
                        )
                elif two_stage_goal_error:
                    no_plan_msg = two_stage_goal_error
                    messages.extend(
                        [
                            {"role": "assistant", "content": response_content or ""},
                            {
                                "role": "user",
                                "content": (
                                    f"{two_stage_goal_error}\n"
                                    "Regenerate one full Python code block. The pipeline will "
                                    "derive stage1 from object_at_point(...) final goals and "
                                    "derive stage2 from all final goals."
                                ),
                            },
                        ]
                    )
                else:
                    if not result_1.get("success", False):
                        no_plan_msg = "No valid plan generated for stage 1.\n" + (result_1.get("output") or "")
                    else:
                        no_plan_msg = "No valid plan generated for stage 2.\n" + (
                            (result_2.get("output") or "") if result_2 else ""
                        )
                    no_plan_msg += "\nSingle pipeline also failed.\n" + (result_single.get("output") or "")
                    instruction = (
                        STAGE1_VALID_RETRY_INSTRUCTION if result_1.get("success", False) else NO_PLAN_RETRY_INSTRUCTION
                    )
                    messages.extend(
                        [
                            {"role": "assistant", "content": code_string},
                            {"role": "user", "content": no_plan_msg},
                            {"role": "user", "content": instruction},
                        ]
                    )
                    # Errors and goal errors were recorded above; record plain planning failures too.
                    trial_errors.append({"trial": trial, "error": no_plan_msg})
                last_error_message = no_plan_msg
                if trial == max_trials - 1:
                    return finish(False, no_plan_msg)
                continue

            # ---- Pick the plan with the smaller makespan ----
            # Plans are temporal and dual-arm, so the action count is a poor
            # quality measure (parallel actions overlap). The action count only
            # breaks ties or replaces a missing makespan; ties prefer two-stage.
            if two_stage_success and single_pipeline_success:
                two_stage_length = (result_1.get("plan_length") or 0) + (result_2.get("plan_length") or 0)
                single_length = result_single.get("plan_length") or 0
                two_stage_makespan = _two_stage_makespan(result_1, result_2)
                single_makespan = result_single.get("makespan")
                use_two_stage = plan_quality_key(two_stage_makespan, two_stage_length) <= plan_quality_key(
                    single_makespan, single_length
                )
                print(
                    f"[BEST PLAN] trial={trial} "
                    f"two-stage makespan={format_optional_float(two_stage_makespan)} "
                    f"actions={two_stage_length} (stage1={result_1.get('plan_length')} + "
                    f"stage2={result_2.get('plan_length')}) "
                    f"| single makespan={format_optional_float(single_makespan)} actions={single_length} "
                    f"| winner={'two-stage' if use_two_stage else 'single'}",
                    flush=True,
                )
            elif two_stage_success:
                use_two_stage = True
                print(f"[BEST PLAN] trial={trial} only two-stage succeeded; using two-stage plan.", flush=True)
            else:
                use_two_stage = False
                print(f"[BEST PLAN] trial={trial} only single-pipeline succeeded; using single plan.", flush=True)

        else:
            # ---- Single pipeline only ----
            try:
                result_1 = job.solve(stage1_code, "single", trial)
                planner_time_cost += result_1.get("planner_time_cost", 0.0)
                stage_times["single"] += result_1.get("planner_time_cost", 0.0)
            except subprocess.CalledProcessError as exc:
                error_message = _crash_output(exc)
                last_error_message = error_message or f"Procedure failed with return code {exc.returncode}."
                messages.extend(
                    [
                        {"role": "assistant", "content": code_string},
                        {"role": "user", "content": error_message},
                        {"role": "user", "content": CRASH_RETRY_INSTRUCTION},
                    ]
                )
                trial_errors.append({"trial": trial, "error": last_error_message})
                continue
            except Exception as exc:
                # There is no stage 1 here; same message as _run_single_pipeline.
                last_error_message = f"Single pipeline error: {type(exc).__name__}: {exc}"
                messages.extend(
                    [
                        {"role": "assistant", "content": code_string or ""},
                        {"role": "user", "content": last_error_message},
                    ]
                )
                trial_errors.append({"trial": trial, "error": last_error_message})
                continue

            if not result_1["success"]:
                no_plan_msg = "No valid plan generated for stage 1.\n" + (result_1.get("output") or "")
                last_error_message = no_plan_msg
                trial_errors.append({"trial": trial, "error": last_error_message})
                if trial == max_trials - 1:
                    return finish(False, no_plan_msg)
                messages.extend(
                    [
                        {"role": "assistant", "content": code_string},
                        {"role": "user", "content": no_plan_msg},
                        {"role": "user", "content": NO_PLAN_RETRY_INSTRUCTION},
                    ]
                )
                continue
            single_pipeline_success = True

        # ---- Success: write the final plan <task>_plan_<index>.txt ----
        canonical_plan = f"{idir}/{task_name}_plan_{index}.txt"
        combined_plan_json_path = None
        try:
            cleanup_files(f"{idir}/{task_name}_plan_{index}.txt*")
            if divide_goal and use_two_stage:
                combined_plan_json_path = _write_two_stage_artifacts(
                    job, canonical_plan, result_1, result_2, stage1_code, stage2_goal_code,
                    stage2_executable_code, manual_split,
                )
            elif divide_goal:
                single_source = result_single.get("plan_path") if result_single else None
                if single_source and Path(single_source).exists():
                    copyfile(single_source, canonical_plan)
                else:
                    debug_print("[PLAN LOG] single pipeline plan_path missing; cannot write canonical plan.", flush=True)
            else:
                copyfile(result_1["plan_path"], canonical_plan)
        except Exception as exc:
            debug_print(f"[PLAN LOG] failed to write final plan artifacts: {exc}", flush=True)

        if divide_goal and use_two_stage:
            output = _two_stage_winner_output(
                job, canonical_plan, combined_plan_json_path, result_1, result_2, result_single,
                single_pipeline_success, stage_times, planner_time_cost,
            )
        elif divide_goal:
            output = _single_winner_output(
                job, canonical_plan, result_1, result_2, result_single, two_stage_success,
                stage_times, planner_time_cost,
            )
        else:
            output = result_1["output"]
        return finish(True, output)

    return finish(False, last_error_message)
