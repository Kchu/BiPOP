#!/usr/bin/env python
"""Execute saved plans on NICOL in CoppeliaSim and measure task success.

Needs CoppeliaSim running with NICOL/nicol.ttt loaded. Plans are read from the
results folder written by plan_generation.py.

Example:
    python plan_validation.py --task_name ServeWater --model_name gpt-4.1 --agent bipop-agent --task_nums 10
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from bipop.agents import AGENT_IDS, VALID_MODELS, VALID_TASKS, evaluate_saved_plan, normalize_agent_name
from bipop.experiment import (
    add_ik_arg,
    add_io_args,
    add_task_selection_args,
    init_wandb,
    load_task_config,
    select_ik_solver,
)
from bipop.paths import result_paths
from bipop.task_selection import describe_task_selection, select_task_indices


def plan_path_for(paths: Dict[str, Path], task_name: str, model_name: str, index: int, agent_name: str) -> Path:
    if agent_name == "llm-agent":
        return paths["direct_plans"] / f"direct_plan_{model_name}_{index}.json"
    if agent_name == "bipop-agent":
        return paths["pddl_plans"] / f"{task_name}_pddl_plan_{index}.json"
    return paths["react_plans"] / f"react_plan_{model_name}_{index}.json"


class _Tee:
    """Write to the terminal and to a buffer at the same time."""

    def __init__(self, buffer: io.StringIO, original):
        self.buffer = buffer
        self.original = original

    def write(self, text: str) -> None:
        self.buffer.write(text)
        self.original.write(text)

    def flush(self) -> None:
        self.buffer.flush()
        self.original.flush()

    def __getattr__(self, name: str):  # isatty, fileno, encoding, ... of the real stream
        return getattr(self.original, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute saved plans on NICOL in CoppeliaSim.")
    parser.add_argument("--task_name", choices=VALID_TASKS, required=True)
    parser.add_argument("--model_name", choices=VALID_MODELS, required=True)
    parser.add_argument("--agent", type=str, required=True, help=f"Agent that generated the plans: {', '.join(AGENT_IDS)}.")
    add_task_selection_args(parser)
    add_io_args(parser)
    add_ik_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_name = normalize_agent_name(args.agent)
    explicit_indices = bool(args.task_indices.strip())
    task_indices = select_task_indices(args.start_index, args.task_nums, args.task_indices)
    is_pddl = agent_name == "bipop-agent"

    paths = result_paths(args.results_dir, args.task_name)
    execution_logs_dir = paths["logs"] / "execution_logs"
    execution_logs_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = agent_name.replace("-", "_")
    log_file = paths["logs"] / f"{args.task_name}_{args.model_name}_{file_suffix}_val.csv"

    select_ik_solver(args.ik)
    from NICOL.tasks import create_task

    task = create_task(args.task_name)
    if task is None:
        raise ValueError(f"Unsupported task: {args.task_name}")
    sim = task.sim

    wandb = init_wandb(
        args.log_wandb,
        f"BiPOP-VAL-{args.task_name}",
        {"task_name": args.task_name, "task_indices": task_indices, "model_name": args.model_name, "agent": agent_name},
        args.results_dir,
    )

    print(f"\n{'=' * 60}")
    print("Plan Validation Configuration:")
    print(f"  - Task: {args.task_name}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Agent: {agent_name}")
    print(f"  - {describe_task_selection(task_indices, explicit_indices)}")
    print(f"  - Results: {paths['root']}")
    print(f"{'=' * 60}\n")

    results: List[Dict[str, Any]] = []
    for loop_num, index in enumerate(task_indices, start=1):
        print(f"\n[{loop_num}/{len(task_indices)} | task_index={index}] Model: {args.model_name} | Agent: {agent_name}")
        print("-" * 40)
        config = load_task_config(args.task_configs_dir, args.task_name, index)
        if config is None:
            print(f"[skip] missing config for task {index} in {args.task_configs_dir / args.task_name}")
            continue

        plan_path = plan_path_for(paths, args.task_name, args.model_name, index, agent_name)
        exec_log_file = execution_logs_dir / f"exec_{args.task_name}_{args.model_name}_{file_suffix}_{index}.log"
        log: Dict[str, Any] = {
            "task_name": args.task_name,
            "task_index": index,
            "task_type": config.get("type", ""),
            "agent_id": agent_name,
            "model_name": args.model_name,
            "plan_path": str(plan_path),
            "success": False,
            "task_achieved": False,
            "finished_steps": 0,
            "execution_time": 0.0,
            "plan_steps": 0,
            "number_of_waits": 0,
            "execution_log": None,
        }

        if not plan_path.exists():
            print(f"[skip] missing plan: {plan_path}")
            log["error"] = "plan_missing"
        else:
            sim.startSimulation()
            time.sleep(1)
            task.reset(config)
            time.sleep(2)

            # Keep a copy of the execution output in the execution log.
            stdout_buffer = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = _Tee(stdout_buffer, original_stdout)
            try:
                evaluation = evaluate_saved_plan(task=task, plan_path=plan_path, is_pddl=is_pddl)
                log.update(
                    success=evaluation.success,
                    task_achieved=evaluation.task_achieved,
                    finished_steps=evaluation.finished_steps,
                    execution_time=evaluation.execution_time,
                    plan_steps=evaluation.plan_steps,
                    number_of_waits=evaluation.number_of_waits,
                )
            except Exception as error:
                log["error"] = str(error)
                print(f"[error] task {index}: {error}")
            finally:
                sys.stdout = original_stdout
            exec_log_file.write_text(stdout_buffer.getvalue(), encoding="utf-8")
            log["execution_log"] = str(exec_log_file)

            sim.stopSimulation()
            time.sleep(3)

        print(f"\nTask {index} Result:")
        print(f"  - Success: {log['success']}")
        print(f"  - Task Achieved: {log['task_achieved']}")
        print(f"  - Finished Steps: {log['finished_steps']}")
        print(f"  - Execution Time: {log['execution_time']:.2f}s")

        results.append(log)
        pd.DataFrame(results).to_csv(log_file, index=False)
        if wandb is not None:
            wandb.log(
                {
                    "task": index,
                    "success": int(log["success"]),
                    "finished_steps": log["finished_steps"],
                    "execution_time": log["execution_time"],
                    "number_of_waits": log["number_of_waits"],
                }
            )
            wandb.save(str(log_file), base_path=str(log_file.parent))

    if results:
        df = pd.DataFrame(results)
        print(f"\n{'=' * 60}")
        print("Validation Summary Statistics:")
        print(f"  - Total Tasks: {len(results)}")
        print(f"  - Success Rate: {df['success'].mean() * 100:.1f}%")
        print(f"  - Task Achieved Rate: {df['task_achieved'].mean() * 100:.1f}%")
        print(f"  - Average Finished Steps: {df['finished_steps'].mean():.1f}")
        print(f"  - Average Execution Time: {df['execution_time'].mean():.2f}s")
        print(f"{'=' * 60}")
    print(f"\nLog: {log_file}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
