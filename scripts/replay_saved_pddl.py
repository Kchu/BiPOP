#!/usr/bin/env python
"""Re-plan saved BiPOP code strings with TFD + VAL, without calling an LLM.

Useful to re-run the planners (e.g. with other TFD presets) on the exact code
the LLM generated in an earlier run. The saved code is read from
``<source_dir>/<task>/intermediate`` and the new plans are written below
``<output_dir>`` with the same layout as plan_generation.py.

Example:
    python scripts/replay_saved_pddl.py --task_name StackBlock --task_indices 3,7
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from argparse import BooleanOptionalAction
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bipop.agents import VALID_TASKS  # noqa: E402
from bipop.llm.code_blocks import choose_actions  # noqa: E402
from bipop.llm.planner import check_planner_setup, solve_code_string  # noqa: E402
from bipop.llm.two_stage import (  # noqa: E402
    combined_plan_json_payload,
    compose_stage2_code,
    extract_stage1_state_context,
    manual_split_full_code,
    uses_two_stage,
)
from bipop.paths import RESULTS_DIR, default_val_bin, result_paths  # noqa: E402
from bipop.plan_format import combine_plan_texts, compute_plan_stats, parse_temporal_plan_text  # noqa: E402
from bipop.task_selection import describe_task_selection, select_task_indices  # noqa: E402

AGENT_ID = "bipop-replay-saved"
MODEL_NAME = "saved_code"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_plain_or_json_string(path: Path) -> str:
    raw = _read_text(path)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return parsed if isinstance(parsed, str) else raw


def _goal_lines(code_string: str) -> List[str]:
    return [
        line.strip()
        for line in code_string.splitlines()
        if line.strip().startswith("problem.add_goal")
    ]


def _resolve_code_source(source_dir: Path, index: int) -> Path:
    """Return the subdirectory that contains the saved code for this index."""
    for candidate in (source_dir / "intermediate", source_dir):
        if (
            (candidate / f"code_string_{index}.json").exists()
            or any(candidate.glob(f"code_string_{index}_full_trial_*.py"))
            or any(candidate.glob(f"code_string_{index}_single_trial_*.py"))
            or (
                (candidate / f"manual_stage1_code_{index}.py").exists()
                and (candidate / f"manual_stage2_goal_code_{index}.py").exists()
            )
        ):
            return candidate
    return source_dir


def _load_saved_full_code(source_dir: Path, index: int) -> str:
    """Load the full code string for the single-goal workflow."""
    candidates: List[Path] = [source_dir / f"code_string_{index}.json"]
    candidates.extend(sorted(source_dir.glob(f"code_string_{index}_single_trial_*.py")))
    candidates.extend(sorted(source_dir.glob(f"code_string_{index}_full_trial_*.py")))

    for candidate in candidates:
        if not candidate.exists():
            continue
        return (
            _load_plain_or_json_string(candidate)
            if candidate.suffix == ".json"
            else _read_text(candidate)
        )

    raise FileNotFoundError(
        f"No saved code found in {source_dir}. Expected "
        f"code_string_{index}.json or code_string_{index}_*_trial_*.py."
    )


def _load_saved_stage_code(source_dir: Path, index: int) -> Dict[str, Any]:
    """Load stage1/stage2 code from saved files, splitting a full code_string if needed."""
    stage1_path = source_dir / f"manual_stage1_code_{index}.py"
    stage2_goal_path = source_dir / f"manual_stage2_goal_code_{index}.py"
    split_path = source_dir / f"manual_split_{index}.json"

    if stage1_path.exists() and stage2_goal_path.exists():
        stage1_code = _read_text(stage1_path)
        stage2_goal_code = _read_text(stage2_goal_path)
        split_payload: Dict[str, Any] = {}
        if split_path.exists():
            split_payload = json.loads(_read_text(split_path))
        return {
            "source": split_payload.get("source", "manual_stage_files"),
            "stage1_code": stage1_code,
            "stage2_goal_code": stage2_goal_code,
            "subgoal1_lines": split_payload.get("subgoal1_lines", _goal_lines(stage1_code)),
            "final_goal_lines": split_payload.get("final_goal_lines", _goal_lines(stage2_goal_code)),
        }

    full_candidates: List[Path] = [
        source_dir / f"code_string_{index}.json",
        source_dir / f"code_string_{index}_full_trial_0.py",
    ]
    full_candidates.extend(sorted(source_dir.glob(f"code_string_{index}_full_trial_*.py")))
    for candidate in full_candidates:
        if not candidate.exists():
            continue
        full_code = (
            _load_plain_or_json_string(candidate)
            if candidate.suffix == ".json"
            else _read_text(candidate)
        )
        split = manual_split_full_code(full_code)
        return {
            "source": str(candidate),
            "stage1_code": split["stage1_code"],
            "stage2_goal_code": split["stage2_goal_code"],
            "subgoal1_lines": split.get("subgoal1_lines", []),
            "final_goal_lines": split.get("final_goal_lines", []),
        }

    raise FileNotFoundError(
        f"No saved code found in {source_dir}. Expected "
        f"manual_stage1_code_{index}.py / manual_stage2_goal_code_{index}.py "
        f"or code_string_{index}.json."
    )


def _write_parsed_plan_json(plan_path: Path, json_path: Path) -> int:
    text = plan_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_temporal_plan_text(text)
    json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    steps, _ = compute_plan_stats(parsed)
    return steps


def _write_summary_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "task_name", "task_index", "agent_id", "model_name", "plan_path",
        "plan_generation_success", "plan_length", "llm_time_cost", "planner_time_cost",
        "generation_time", "divide_goal", "val_checked", "val_available", "val_valid",
        "val_value", "stage1_plan_length", "stage2_plan_length",
        "stage1_val_value", "stage2_val_value", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def _failure_record(
    task_name: str, index: int, error: str, divide_goal: bool = True
) -> Dict[str, Any]:
    return {
        "task_name": task_name, "task_index": index, "agent_id": AGENT_ID,
        "model_name": MODEL_NAME, "plan_path": "", "plan_generation_success": False,
        "plan_length": 0, "llm_time_cost": 0.0, "planner_time_cost": 0.0,
        "generation_time": 0.0, "divide_goal": divide_goal, "val_checked": False,
        "val_available": False, "val_valid": False, "val_value": None,
        "stage1_plan_length": 0, "stage2_plan_length": 0,
        "stage1_val_value": None, "stage2_val_value": None, "error": error,
    }


def _print_task_report(index: int, record: Dict[str, Any]) -> None:
    val_status = "N/A"
    if record.get("val_checked"):
        val_status = "valid" if record.get("val_valid") else "invalid"
    if record.get("divide_goal"):
        s1 = record.get("stage1_plan_length")
        s2 = record.get("stage2_plan_length")
        plan_length_text = (
            f"| stage1_plan_len={s1} | stage2_plan_len={s2} "
            f"| final_plan_len={record['plan_length']} "
        )
    else:
        plan_length_text = f"| plan_length={record['plan_length']} "
    print(
        f"[done] task {index} | success={record['plan_generation_success']} "
        f"{plan_length_text}"
        f"| planner={record['planner_time_cost']:.2f}s "
        f"| total={record['generation_time']:.2f}s "
        f"| val={val_status}"
    )


def _run_single_goal(
    full_code: str,
    output_pddl_dir: Path,
    idir: Path,
    task_name: str,
    index: int,
    action_definition: str,
    val_bin: Path,
    start_time: float,
) -> Dict[str, Any]:
    """Run the single-goal (non-divide) workflow for one task index."""
    try:
        result = solve_code_string(
            full_code,
            str(output_pddl_dir),
            task_name,
            index,
            "single",
            action_definition,
            trial=None,
            val_bin=val_bin,
            intermediate_dir=str(idir),
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        record = _failure_record(task_name, index, f"Script error: {err[:400]}", divide_goal=False)
        record["generation_time"] = time.time() - start_time
        return record

    generation_time = time.time() - start_time
    planner_time_cost = result.get("planner_time_cost", 0.0)
    val = result.get("val") or {}
    val_available = bool(val.get("val_available"))

    if not result["success"]:
        record = _failure_record(
            task_name, index, (result.get("output") or "")[:400], divide_goal=False
        )
        record["planner_time_cost"] = planner_time_cost
        record["generation_time"] = generation_time
        record["val_checked"] = val_available
        record["val_available"] = val_available
        record["val_value"] = val.get("best_val_value")
        return record

    canonical = output_pddl_dir / f"{task_name}_plan_{index}.txt"
    copyfile(result["plan_path"], str(canonical))
    plan_length = _write_parsed_plan_json(
        canonical, output_pddl_dir / f"{task_name}_pddl_plan_{index}.json"
    )

    return {
        "task_name": task_name,
        "task_index": index,
        "agent_id": AGENT_ID,
        "model_name": MODEL_NAME,
        "plan_path": str(canonical),
        "plan_generation_success": True,
        "plan_length": plan_length,
        "llm_time_cost": 0.0,
        "planner_time_cost": planner_time_cost,
        "generation_time": generation_time,
        "divide_goal": False,
        "val_checked": val_available,
        "val_available": val_available,
        "val_valid": bool(val_available and val.get("best_plan_path")),
        "val_value": val.get("best_val_value"),
        "stage1_plan_length": 0,
        "stage2_plan_length": 0,
        "stage1_val_value": None,
        "stage2_val_value": None,
        "error": "",
    }


def _run_divide_goal(
    source_payload: Dict[str, Any],
    output_pddl_dir: Path,
    idir: Path,
    task_name: str,
    index: int,
    action_definition: str,
    val_bin: Path,
    start_time: float,
) -> Dict[str, Any]:
    """Run the divide-goal (two-stage) workflow for one task index."""
    # Stage 1
    try:
        result_1 = solve_code_string(
            source_payload["stage1_code"],
            str(output_pddl_dir),
            task_name,
            index,
            "stage1",
            action_definition,
            trial=None,
            val_bin=val_bin,
            intermediate_dir=str(idir),
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        record = _failure_record(
            task_name, index, f"Stage1 script error: {err[:400]}", divide_goal=True
        )
        record["generation_time"] = time.time() - start_time
        return record

    if not result_1["success"]:
        val1 = result_1.get("val") or {}
        record = _failure_record(
            task_name, index,
            f"Stage1 no plan: {(result_1.get('output') or '')[:400]}",
            divide_goal=True,
        )
        record["planner_time_cost"] = result_1.get("planner_time_cost", 0.0)
        record["generation_time"] = time.time() - start_time
        record["val_checked"] = bool(val1.get("val_available"))
        record["val_available"] = bool(val1.get("val_available"))
        record["stage1_val_value"] = val1.get("best_val_value")
        return record

    # Extract stage1 achieved state to seed stage2 initial conditions
    try:
        stage1_state_context = extract_stage1_state_context(
            result_1["py_path"],
            result_1["domain"],
            result_1["problem"],
            result_1["plan_text"],
        )
    except Exception as exc:
        record = _failure_record(
            task_name, index, f"Stage1 state extraction failed: {exc}", divide_goal=True
        )
        record["planner_time_cost"] = result_1.get("planner_time_cost", 0.0)
        record["generation_time"] = time.time() - start_time
        return record

    injected_init_code = stage1_state_context["injected_init_code"]
    (idir / f"achieved_state_{index}_stage1.txt").write_text(
        "\n".join(stage1_state_context["state_lines"]), encoding="utf-8"
    )
    (idir / f"stage2_injected_init_{index}.py").write_text(injected_init_code, encoding="utf-8")

    stage2_executable_code = compose_stage2_code(
        injected_init_code,
        source_payload["stage2_goal_code"],
    )
    (idir / f"manual_stage2_executable_code_{index}.py").write_text(
        stage2_executable_code, encoding="utf-8"
    )

    # Stage 2
    try:
        result_2 = solve_code_string(
            stage2_executable_code,
            str(output_pddl_dir),
            task_name,
            index,
            "stage2",
            action_definition,
            trial=None,
            val_bin=val_bin,
            intermediate_dir=str(idir),
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        record = _failure_record(
            task_name, index, f"Stage2 script error: {err[:400]}", divide_goal=True
        )
        record["planner_time_cost"] = result_1.get("planner_time_cost", 0.0)
        record["generation_time"] = time.time() - start_time
        return record

    generation_time = time.time() - start_time
    planner_time_cost = (
        result_1.get("planner_time_cost", 0.0) + result_2.get("planner_time_cost", 0.0)
    )
    success = result_1["success"] and result_2["success"]

    val1 = result_1.get("val") or {}
    val2 = result_2.get("val") or {}
    both_val_available = bool(val1.get("val_available") and val2.get("val_available"))
    both_val_valid = bool(success and val1.get("best_plan_path") and val2.get("best_plan_path"))

    plan_length = 0
    plan_path = ""
    stage1_plan_length = result_1.get("plan_length") or 0
    stage2_plan_length = result_2.get("plan_length") or 0

    if success:
        final_text = combine_plan_texts(result_1["plan_text"], result_2["plan_text"])
        plan_1_path = output_pddl_dir / f"{task_name}_plan_{index}_1.txt"
        plan_2_path = output_pddl_dir / f"{task_name}_plan_{index}_2.txt"
        canonical = output_pddl_dir / f"{task_name}_plan_{index}.txt"
        plan_1_path.write_text(result_1["plan_text"], encoding="utf-8")
        plan_2_path.write_text(result_2["plan_text"], encoding="utf-8")
        canonical.write_text(final_text, encoding="utf-8")
        plan_path = str(canonical)

        combined_json_path = output_pddl_dir / f"{task_name}_plan_{index}.json"
        combined_json_path.write_text(
            json.dumps(
                combined_plan_json_payload(task_name, index, result_1, result_2, final_text),
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        plan_length = _write_parsed_plan_json(
            canonical, output_pddl_dir / f"{task_name}_pddl_plan_{index}.json"
        )
        stage1_plan_length = _write_parsed_plan_json(
            plan_1_path, output_pddl_dir / f"{task_name}_pddl_plan_{index}_1.json"
        )
        stage2_plan_length = _write_parsed_plan_json(
            plan_2_path, output_pddl_dir / f"{task_name}_pddl_plan_{index}_2.json"
        )

    error = "" if success else (result_2.get("output") or result_1.get("output") or "")

    return {
        "task_name": task_name,
        "task_index": index,
        "agent_id": AGENT_ID,
        "model_name": MODEL_NAME,
        "plan_path": plan_path,
        "plan_generation_success": success,
        "plan_length": plan_length,
        "llm_time_cost": 0.0,
        "planner_time_cost": planner_time_cost,
        "generation_time": generation_time,
        "divide_goal": True,
        "val_checked": both_val_available,
        "val_available": both_val_available,
        "val_valid": both_val_valid,
        "val_value": val2.get("best_val_value"),
        "stage1_plan_length": stage1_plan_length,
        "stage2_plan_length": stage2_plan_length,
        "stage1_val_value": val1.get("best_val_value"),
        "stage2_val_value": val2.get("best_val_value"),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run saved code strings through the BiPOP planning pipeline without calling an LLM. "
            "Supports single-goal (--no-divide_goal) and divide-goal (--divide_goal) workflows."
        )
    )
    parser.add_argument("--task_name", choices=VALID_TASKS, required=True)
    parser.add_argument("--task_nums", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--task_indices", type=str, default="",
        help="Comma-separated explicit task indices, overrides --start_index/--task_nums.",
    )
    parser.add_argument(
        "--source_dir", type=str, default=str(RESULTS_DIR / "pddl_plans"),
        help="Parent directory containing the {task_name}/ subfolder with saved code files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(RESULTS_DIR / "replay"),
        help="Results root of the replayed plans (kept apart from the original plans).",
    )
    parser.add_argument(
        "--divide_goal",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "StackBlock only: when true (default), run the two-stage divide-goal workflow. "
            "When false (--no-divide_goal), run the single-goal workflow directly."
        ),
    )
    parser.add_argument("--val_bin", type=str, default=default_val_bin() or "Validate")
    args = parser.parse_args()
    args.divide_goal = uses_two_stage(args.task_name, args.divide_goal)  # StackBlock only

    explicit_task_indices = bool(args.task_indices.strip())
    task_indices = select_task_indices(args.start_index, args.task_nums, args.task_indices)
    print(describe_task_selection(task_indices, explicit_task_indices))

    source_dir = (Path(args.source_dir) / args.task_name).resolve()
    val_bin = Path(args.val_bin)
    check_planner_setup(args.val_bin)

    action_definition = choose_actions(args.task_name)
    if not action_definition:
        raise ValueError(f"Unsupported task for PDDL generation: {args.task_name}")

    paths = result_paths(Path(args.output_dir), args.task_name)
    output_pddl_dir = paths["pddl_plans"]
    output_pddl_dir.mkdir(parents=True, exist_ok=True)
    idir = output_pddl_dir / "intermediate"
    idir.mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    workflow_tag = "divide_goal" if args.divide_goal else "single_goal"
    summary_csv = paths["logs"] / f"{args.task_name}_saved_stage_pddl_{workflow_tag}.csv"

    print(f"source_dir  = {source_dir}")
    print(f"output_dir  = {output_pddl_dir}")
    print(f"workflow    = {workflow_tag}")

    records: List[Dict[str, Any]] = []

    for index in task_indices:
        start_time = time.time()
        print(f"[run] task {index}")

        code_source = _resolve_code_source(source_dir, index)

        if args.divide_goal:
            try:
                source_payload = _load_saved_stage_code(code_source, index)
            except (FileNotFoundError, ValueError) as exc:
                record = _failure_record(args.task_name, index, str(exc), divide_goal=True)
                records.append(record)
                _write_summary_csv(summary_csv, records)
                print(f"  [skip] {record['error']}")
                continue

            record = _run_divide_goal(
                source_payload, output_pddl_dir, idir,
                args.task_name, index, action_definition, val_bin, start_time,
            )
        else:
            try:
                full_code = _load_saved_full_code(code_source, index)
            except (FileNotFoundError, ValueError) as exc:
                record = _failure_record(args.task_name, index, str(exc), divide_goal=False)
                records.append(record)
                _write_summary_csv(summary_csv, records)
                print(f"  [skip] {record['error']}")
                continue

            record = _run_single_goal(
                full_code, output_pddl_dir, idir,
                args.task_name, index, action_definition, val_bin, start_time,
            )

        records.append(record)
        _write_summary_csv(summary_csv, records)
        _print_task_report(index, record)

    success_count = sum(1 for r in records if r["plan_generation_success"])
    print(f"\nSummary CSV: {summary_csv}")
    print(f"Finished. success={success_count}/{len(records)}")


if __name__ == "__main__":
    main()
