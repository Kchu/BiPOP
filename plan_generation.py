#!/usr/bin/env python
"""Generate plans for the saved task configs.

The BiPOP-agent and the LLM-agent run without a simulator. The ReAct-agent
executes every step on NICOL, so it needs CoppeliaSim with NICOL/nicol.ttt.

Example:
    python plan_generation.py --task_name StackBlock --model_name gpt-4.1 --agent bipop-agent --task_nums 10
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from bipop.agents import (
    AGENT_IDS,
    DEFAULT_MAX_TRIALS,
    VALID_MODELS,
    VALID_TASKS,
    AgentRunContext,
    create_agent,
    normalize_agent_name,
)
from bipop.experiment import (
    add_ik_arg,
    add_io_args,
    add_task_selection_args,
    init_wandb,
    load_task_config,
    select_ik_solver,
    task_description,
)
from bipop.llm.planner import check_planner_setup, get_presets_for_task
from bipop.llm.two_stage import uses_two_stage
from bipop.paths import default_val_bin, result_paths
from bipop.task_selection import describe_task_selection, select_task_indices

LOG_SUFFIX = {"bipop-agent": "bipop", "llm-agent": "direct", "react-agent": "react"}


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_presets(task_name: str, divide_goal: bool) -> None:
    """TFD presets of every planning stage that will run (only StackBlock has per-stage presets)."""
    stages = ("single", "stage1", "stage2") if uses_two_stage(task_name, divide_goal) else ("single",)
    for stage_label in stages:
        presets = get_presets_for_task(task_name, stage_label)
        print(f"TFD presets ({stage_label}): " + ", ".join(f"{name}={config}" for name, config in presets.items()))


def print_block(title: str, index: int, text: str) -> None:
    print(f"\n=== {title} task {index} ===")
    print(text)
    print(f"=== END {title} ===")


def print_file_block(title: str, index: int, path: Path | None, missing: str) -> None:
    if path is not None and path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore").rstrip()
        print_block(title, index, text if text else "<empty file>")
    else:
        print_block(title, index, missing)


def print_pddl_details(index: int, record: Dict[str, Any], intermediate_dir: Path) -> None:
    """The generated code string and, after retries, the last error that was fed back to the LLM."""
    code_path = intermediate_dir / f"code_string_{index}.json"
    if code_path.exists():
        code = code_path.read_text(encoding="utf-8", errors="ignore")
        print_block("CODE STRING", index, code.rstrip() if code.strip() else "<empty code string>")

    summary_path = intermediate_dir / f"llm_trial_summary_{index}.json"
    if not record.get("plan_generation_success") or not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if summary.get("trials_attempted", 0) > 1 and summary.get("last_error_message"):
        print_block("RETRY ERROR", index, str(summary["last_error_message"]).rstrip())


def print_task_report(index: int, record: Dict[str, Any]) -> None:
    val_status = "N/A"
    if record.get("val_checked"):
        val_status = "valid" if record.get("val_valid") else "invalid"
    length_text = f"| final_plan_len={record['plan_length']} "
    if record.get("divide_goal") and record.get("stage1_plan_length") is not None:
        length_text = (
            f"| stage1_plan_len={record.get('stage1_plan_length')} "
            f"| stage2_plan_len={record.get('stage2_plan_length')} " + length_text
        )
    print(
        f"[done] task {index} | success={record['plan_generation_success']} {length_text}"
        f"| llm={record['llm_time_cost']:.2f}s | planner={record['planner_time_cost']:.2f}s "
        f"| total={record['generation_time']:.2f}s | val={val_status} "
        f"| best_valid_preset={record.get('best_valid_planner_preset') or 'NA'}"
    )


def print_react_summary(df: pd.DataFrame) -> None:
    """Success rate (goal checked by the environment), LLM claims and failure reasons of a ReAct run."""
    print(f"\n{'=' * 60}\nSummary Statistics:")
    print(f"  - Success Rate (goal reached, checked by the environment): {df['success'].mean() * 100:.1f}%")
    if "llm_claimed_complete" in df.columns:
        print(f"  - LLM Claimed COMPLETE Rate: {df['llm_claimed_complete'].astype(bool).mean() * 100:.1f}%")
    if "failure_reason" in df.columns:
        reasons = df["failure_reason"].fillna("").astype(str)
        counts = reasons[reasons != ""].value_counts()
        for reason in ("Incomplete", "Misjudged", "Aborted", "Error"):
            print(f"  - Failed ({reason}): {int(counts.get(reason, 0))}")
    print(f"  - Average Plan Length: {df['plan_length'].mean():.1f}")
    print(f"  - Average LLM Time: {df['llm_time_cost'].mean():.2f}s\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def empty_record(task_name: str, index: int, task_type: Any, agent_id: str, model_name: str,
                 divide_goal: bool, error: str) -> Dict[str, Any]:
    return {
        "task_name": task_name,
        "task_index": index,
        "task_type": task_type,
        "agent_id": agent_id,
        "model_name": model_name,
        "plan_path": "",
        "plan_generation_success": False,
        "plan_length": 0,
        "llm_time_cost": 0.0,
        "planner_time_cost": 0.0,
        "generation_time": 0.0,
        "divide_goal": divide_goal,
        "val_checked": False,
        "val_available": False,
        "val_valid": False,
        "val_value": None,
        "best_valid_planner_preset": "NA",
        "stage1_best_valid_planner_preset": "NA",
        "stage2_best_valid_planner_preset": "NA",
        "error": error,
    }


def generation_record(task_name: str, index: int, config: Dict[str, Any], model_name: str, generation) -> Dict[str, Any]:
    metadata = generation.metadata or {}
    val_meta = metadata.get("val")
    stage_presets = metadata.get("stage_best_valid_planner_presets") or {}
    record = {
        "task_name": task_name,
        "task_index": index,
        "task_type": config.get("type", ""),
        "agent_id": generation.agent_id,
        "model_name": model_name,
        "plan_path": str(generation.plan_path) if generation.plan_path else "",
        "plan_generation_success": generation.success,
        "plan_length": generation.plan_length,
        "llm_time_cost": generation.llm_time_cost,
        "planner_time_cost": generation.planner_time_cost,
        "generation_time": generation.generation_time,
        "divide_goal": metadata.get("divide_goal", False),
        "val_checked": bool(val_meta),
        "val_available": bool(val_meta and val_meta.get("available")),
        "val_valid": bool(val_meta and val_meta.get("is_valid")),
        "val_value": val_meta.get("value") if val_meta else None,
        "best_valid_planner_preset": metadata.get("best_valid_planner_preset") or "NA",
        "stage1_best_valid_planner_preset": stage_presets.get("stage1") or "NA",
        "stage2_best_valid_planner_preset": stage_presets.get("stage2") or "NA",
    }
    if metadata.get("divide_goal", False):
        stage_lengths = metadata.get("stage_plan_lengths", {})
        record["stage1_plan_length"] = stage_lengths.get("stage1")
        record["stage2_plan_length"] = stage_lengths.get("stage2")
    if not generation.success:
        if val_meta and val_meta.get("available") and not val_meta.get("is_valid"):
            record["error"] = val_meta.get("output") or "VAL rejected the generated plan."
        else:
            record["error"] = metadata.get("planner_output") or "No successful plan was generated."

    evaluation = generation.embedded_evaluation  # ReAct-agent: executed while planning
    if evaluation is not None:
        record.update(
            {
                "success": evaluation.success,
                "task_achieved": evaluation.task_achieved,
                "execution_time": evaluation.execution_time,
                "finished_steps": evaluation.finished_steps,
                "total_time": metadata.get("total_time", 0.0),
                "final_status": evaluation.metadata.get("final_status", ""),
                "llm_claimed_complete": evaluation.metadata.get("llm_claimed_complete", False),
                "stop_reason": evaluation.metadata.get("stop_reason", ""),
                "failure_reason": evaluation.metadata.get("failure_reason", ""),
            }
        )
    return record


def collect_generated_plan_files(
    agent_name: str, index: int, model_name: str, task_name: str, paths: Dict[str, Path]
) -> List[Path]:
    """Final plan files of one task instance (uploaded to W&B)."""
    if agent_name == "bipop-agent":
        pddl_dir = paths["pddl_plans"]
        candidates = [pddl_dir / f"{task_name}_llm_pddl_{index}.py"]
        candidates += sorted(pddl_dir.glob(f"{task_name}_llm_pddl_{index}_*.py"))
        candidates += [pddl_dir / f"{task_name}_pddl_plan_{index}{suffix}.json" for suffix in ("", "_1", "_2")]
    elif agent_name == "react-agent":
        candidates = [
            paths["react_plans"] / f"react_{kind}_{model_name}_{index}.json" for kind in ("plan", "result", "step_log")
        ]
    else:
        candidates = [paths["direct_plans"] / f"direct_plan_{model_name}_{index}.json"]

    unique: List[Path] = []
    for path in candidates:
        if path.is_file() and path not in unique:
            unique.append(path)
    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan generation for the BiPOP-, LLM- and ReAct-agent.")
    parser.add_argument("--task_name", choices=VALID_TASKS, required=True)
    parser.add_argument("--model_name", choices=VALID_MODELS, required=True)
    parser.add_argument("--agent", type=str, default="bipop-agent", help=f"One of {', '.join(AGENT_IDS)}.")
    add_task_selection_args(parser)
    add_io_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_trials", type=int, default=DEFAULT_MAX_TRIALS, help="LLM trials of the BiPOP-agent.")
    parser.add_argument(
        "--divide_goal",
        action=BooleanOptionalAction,
        default=True,
        help="BiPOP, StackBlock only: also plan with the two-stage goal split and keep the better plan.",
    )
    parser.add_argument("--val_bin", type=str, default=default_val_bin(), help="VAL Validate binary.")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps of the ReAct-agent.")
    add_ik_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.divide_goal = uses_two_stage(args.task_name, args.divide_goal)  # StackBlock only
    agent_name = normalize_agent_name(args.agent)
    explicit_indices = bool(args.task_indices.strip())
    task_indices = select_task_indices(args.start_index, args.task_nums, args.task_indices)
    validate_with_val = agent_name == "bipop-agent"
    if agent_name == "bipop-agent":
        check_planner_setup(args.val_bin)

    paths = result_paths(args.results_dir, args.task_name)
    for key in ("direct_plans", "pddl_plans", "react_plans", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    log_path = paths["logs"] / f"{args.task_name}_{args.model_name}_{LOG_SUFFIX[agent_name]}_plan_gen.csv"

    agent = create_agent(
        agent_name=agent_name,
        model_name=args.model_name,
        task_name=args.task_name,
        temperature=args.temperature,
        max_trials=args.max_trials,
        max_steps=args.max_steps,
        divide_goal=args.divide_goal,
    )

    # The ReAct-agent acts in the simulator while it plans.
    task = sim = None
    if agent_name == "react-agent":
        select_ik_solver(args.ik)
        from NICOL.tasks import create_task

        task = create_task(args.task_name)
        sim = task.sim
        print(f"NICOL environment initialized for {args.task_name}")

    wandb = init_wandb(
        args.log_wandb,
        f"BiPOP-{args.task_name}",
        {
            "task_name": args.task_name,
            "task_indices": task_indices,
            "model_name": args.model_name,
            "agent": agent_name,
            "divide_goal": args.divide_goal,
            "max_steps": args.max_steps if agent_name == "react-agent" else None,
        },
        args.results_dir,
    )

    if agent_name == "bipop-agent":
        print_presets(args.task_name, args.divide_goal)
    print(describe_task_selection(task_indices, explicit_indices))

    records: List[Dict[str, Any]] = []
    for index in task_indices:
        config = load_task_config(args.task_configs_dir, args.task_name, index)
        if config is None:
            missing = args.task_configs_dir / args.task_name / f"task_config_{index}.json"
            record = empty_record(args.task_name, index, "", agent_name, args.model_name, args.divide_goal,
                                  f"missing config: {missing}")
            records.append(record)
            print_block("ERROR", index, record["error"])
            print_task_report(index, record)
            continue

        if args.task_name.startswith("StackBlock") and config.get("pile_desc"):
            print_block("PILE DESCRIPTION", index, json.dumps(config["pile_desc"], indent=2))

        if sim is not None:
            sim.startSimulation()
            task.reset(config)
            time.sleep(2)  # let the scene settle

        context = AgentRunContext(
            index=index,
            task_description=task_description(config),
            direct_plan_dir=paths["direct_plans"],
            pddl_plan_dir=paths["pddl_plans"],
            react_plan_dir=paths["react_plans"],
            task=task,
            validate_with_val=validate_with_val,
            val_bin=Path(args.val_bin) if args.val_bin else None,
            divide_goal=args.divide_goal,
        )
        try:
            record = generation_record(args.task_name, index, config, args.model_name, agent.generate(context))
        except Exception as error:
            record = empty_record(args.task_name, index, config.get("type", ""), agent_name, args.model_name,
                                  args.divide_goal, str(error))
            if agent_name == "react-agent":
                record.update(success=False, task_achieved=False, execution_time=0.0, finished_steps=0,
                              total_time=0.0, final_status="ERROR", llm_claimed_complete=False,
                              stop_reason="error", failure_reason="Error")

        if sim is not None:
            sim.stopSimulation()
            time.sleep(4)  # wait until the simulation has stopped

        records.append(record)
        pd.DataFrame(records).to_csv(log_path, index=False)

        if wandb is not None:
            metrics = {
                "task": index,
                "plan_generation_success": int(record["plan_generation_success"]),
                "plan_length": record["plan_length"],
                "generation_time": record["generation_time"],
                "best_valid_planner_preset": record.get("best_valid_planner_preset", "NA"),
                "stage1_best_valid_planner_preset": record.get("stage1_best_valid_planner_preset", "NA"),
                "stage2_best_valid_planner_preset": record.get("stage2_best_valid_planner_preset", "NA"),
            }
            plan_files = collect_generated_plan_files(agent_name, index, args.model_name, args.task_name, paths)
            for plan_file in plan_files:
                wandb.save(str(plan_file), base_path=str(plan_file.parent))
            metrics["uploaded_plan_files"] = len(plan_files)
            metrics["uploaded_plan_json_files"] = sum(1 for path in plan_files if path.suffix == ".json")
            if agent_name == "react-agent":
                metrics.update(
                    success=int(record.get("success", False)),
                    task_achieved=int(record.get("task_achieved", False)),
                    llm_claimed_complete=int(record.get("llm_claimed_complete", False)),
                    execution_time=record.get("execution_time", 0.0),
                    total_time=record.get("total_time", 0.0),
                )
            wandb.log(metrics)
            wandb.save(str(log_path), base_path=str(log_path.parent))

        if agent_name == "bipop-agent":
            print_pddl_details(index, record, paths["pddl_plans"] / "intermediate")
        else:
            plan_path = Path(record["plan_path"]) if record.get("plan_path") else None
            print_file_block("PLAN", index, plan_path, "<no plan generated>")
        if record.get("error"):
            print_block("ERROR", index, record["error"].rstrip())
        print_task_report(index, record)
        if agent_name == "react-agent":
            print(
                f"\nTask {index} Result: success={record.get('success', False)} "
                f"| llm_claimed_complete={record.get('llm_claimed_complete', False)} "
                f"| failure_reason={record.get('failure_reason') or '-'} | plan_length={record['plan_length']} "
                f"| llm={record['llm_time_cost']:.2f}s | execution={record.get('execution_time', 0):.2f}s"
            )

    df = pd.DataFrame(records)
    if agent_name == "react-agent" and "success" in df.columns:
        print_react_summary(df)

    print(f"\nLog: {log_path}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
