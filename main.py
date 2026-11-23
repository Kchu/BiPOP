#!/usr/bin/env python
"""End-to-end experiment on NICOL in CoppeliaSim: (perception ->) planning -> execution.

For every task instance each selected agent generates a plan, which is then
executed on NICOL and checked against the task goal. Needs CoppeliaSim with
NICOL/nicol.ttt loaded.

Example:
    python main.py --task_name ServeFruit --model_name gpt-4.1 --task_nums 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from PIL import Image

from bipop.agents import (
    AGENT_IDS,
    DEFAULT_MAX_TRIALS,
    VALID_MODELS,
    VALID_TASKS,
    AgentRunContext,
    create_agent,
    default_failed_evaluation,
    evaluate_saved_plan,
    normalize_agent_name,
)
from bipop.experiment import (
    add_ik_arg,
    add_io_args,
    add_task_selection_args,
    init_wandb,
    load_task_config,
    select_ik_solver,
)
from bipop.llm.planner import check_planner_setup
from bipop.llm.two_stage import uses_two_stage
from bipop.paths import default_val_bin, result_paths
from bipop.task_selection import describe_task_selection, select_task_indices

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_or_sample_config(task: Any, args: argparse.Namespace, index: int) -> Dict[str, Any]:
    """Reset the scene from the saved config, or sample a new one (saved below results/task_configs)."""
    if not args.sample_configs:
        config = load_task_config(args.task_configs_dir, args.task_name, index)
        task.reset(config)
        return config

    task.reset()
    config = task.config
    config_path = args.results_dir / "task_configs" / args.task_name / f"task_config_{index}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def build_task_description(task: Any, config: Dict[str, Any], index: int, args: argparse.Namespace) -> str:
    """Task text plus the scene description, from the config or from OWLv2 on NICOL's eye camera."""
    visual_description = config.get("task_des_visual", "")
    if args.use_owlv2:
        from NICOL.tasks import generate_visual_description
        from bipop.perception.owlv2 import owlv2_solve

        image_dir = args.results_dir / "images" / args.task_name
        image_dir.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(task.get_image_from_eyes(index=index))
        image.save(image_dir / f"{args.task_name}_{index}.png")
        directions, _ = owlv2_solve(
            image, task.text_queries, task.owl_threshold, str(image_dir / f"{args.task_name}_{index}_owlv2.png")
        )
        visual_description = generate_visual_description(task, directions)
    return config["task_des"] + visual_description


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planning and execution with the BiPOP-, LLM- and ReAct-agent.")
    parser.add_argument("--task_name", choices=VALID_TASKS, required=True)
    parser.add_argument("--model_name", choices=VALID_MODELS, required=True)
    parser.add_argument("--agents", type=str, default=",".join(AGENT_IDS), help="Comma-separated agent ids/aliases.")
    add_task_selection_args(parser, default_task_nums=1)
    add_io_args(parser)
    parser.add_argument(
        "--sample_configs",
        action=BooleanOptionalAction,
        default=False,
        help="Sample new random task configs instead of loading the saved ones.",
    )
    parser.add_argument("--use_owlv2", action=BooleanOptionalAction, default=False,
                        help="Describe the scene with OWLv2 detections instead of the config.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_trials", type=int, default=DEFAULT_MAX_TRIALS, help="LLM trials of the BiPOP-agent.")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps of the ReAct-agent.")
    parser.add_argument("--divide_goal", action=BooleanOptionalAction, default=True,
                        help="BiPOP, StackBlock only: also plan with the two-stage goal split and keep the better plan.")
    parser.add_argument("--validate_with_val", action=BooleanOptionalAction, default=False,
                        help="BiPOP: validate the final plan once more with VAL.")
    parser.add_argument("--val_bin", type=str, default=default_val_bin(), help="VAL Validate binary.")
    add_ik_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.divide_goal = uses_two_stage(args.task_name, args.divide_goal)  # StackBlock only
    explicit_indices = bool(args.task_indices.strip())
    task_indices = select_task_indices(args.start_index, args.task_nums, args.task_indices)
    selected_agents = [normalize_agent_name(name) for name in args.agents.split(",") if name.strip()]
    if not selected_agents:
        raise ValueError("No agents selected. Provide --agents with at least one valid agent.")
    if "bipop-agent" in selected_agents:
        check_planner_setup(args.val_bin)

    paths = result_paths(args.results_dir, args.task_name)
    for key in ("direct_plans", "pddl_plans", "react_plans", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    log_path = paths["logs"] / f"{args.task_name}_{args.model_name}_pipeline_eval.csv"

    agents = [
        create_agent(
            agent_name=agent_name,
            model_name=args.model_name,
            task_name=args.task_name,
            temperature=args.temperature,
            max_trials=args.max_trials,
            max_steps=args.max_steps,
            divide_goal=args.divide_goal,
        )
        for agent_name in selected_agents
    ]

    select_ik_solver(args.ik)
    from NICOL.tasks import create_task

    task = create_task(args.task_name)
    if task is None:
        raise ValueError(f"Unsupported task: {args.task_name}")
    sim = task.sim

    wandb = init_wandb(
        args.log_wandb,
        f"BiPOP-{args.task_name}",
        {
            "task_name": args.task_name,
            "model_name": args.model_name,
            "task_indices": task_indices,
            "agents": selected_agents,
            "use_owlv2": args.use_owlv2,
            "validate_with_val": args.validate_with_val,
            "divide_goal": args.divide_goal,
        },
        args.results_dir,
    )

    records: List[Dict[str, Any]] = []
    print(describe_task_selection(task_indices, explicit_indices))
    for index in task_indices:
        if not args.sample_configs and load_task_config(args.task_configs_dir, args.task_name, index) is None:
            print(f"[skip] missing config for task {index} in {args.task_configs_dir / args.task_name}")
            continue
        sim.startSimulation()
        config = load_or_sample_config(task, args, index)
        sim.stopSimulation()
        time.sleep(1)

        for agent in agents:
            print("=" * 90)
            print(f"Task {index} ({config.get('type', 'unknown')}) | Agent: {agent.display_name}")
            sim.startSimulation()
            task.reset(config)
            time.sleep(2)

            record: Dict[str, Any] = {
                "task_name": args.task_name,
                "task_index": index,
                "task_type": config.get("type", ""),
                "agent_id": agent.agent_id,
                "model_name": args.model_name,
            }
            try:
                context = AgentRunContext(
                    index=index,
                    task_description=build_task_description(task, config, index, args),
                    direct_plan_dir=paths["direct_plans"],
                    pddl_plan_dir=paths["pddl_plans"],
                    react_plan_dir=paths["react_plans"],
                    task=task,
                    validate_with_val=args.validate_with_val,
                    val_bin=Path(args.val_bin) if args.val_bin else None,
                    divide_goal=args.divide_goal,
                )
                generation = agent.generate(context)

                if generation.embedded_evaluation is not None:  # ReAct-agent executed while planning
                    evaluation = generation.embedded_evaluation
                elif generation.plan_path is not None and generation.plan_path.exists():
                    evaluation = evaluate_saved_plan(task, generation.plan_path, agent.is_pddl)
                else:
                    evaluation = default_failed_evaluation()

                val_meta = (generation.metadata or {}).get("val")
                record.update(
                    {
                        "agent_id": generation.agent_id,
                        "plan_path": str(generation.plan_path) if generation.plan_path else "",
                        "plan_generation_success": generation.success,
                        "plan_length": generation.plan_length,
                        "llm_time_cost": generation.llm_time_cost,
                        "planner_time_cost": generation.planner_time_cost,
                        "generation_time": generation.generation_time,
                        "divide_goal": generation.metadata.get("divide_goal", False),
                        "eval_success": evaluation.success,
                        "task_achieved": evaluation.task_achieved,
                        "finished_steps": evaluation.finished_steps,
                        "execution_time": evaluation.execution_time,
                        "plan_steps": evaluation.plan_steps,
                        "number_of_waits": evaluation.number_of_waits,
                        "final_status": evaluation.metadata.get("final_status", ""),
                        "agent_status": evaluation.metadata.get("agent_status", ""),
                        # ReAct-agent only: the LLM's own claim and why a run failed
                        "llm_claimed_complete": evaluation.metadata.get("llm_claimed_complete", ""),
                        "stop_reason": evaluation.metadata.get("stop_reason", ""),
                        "failure_reason": evaluation.metadata.get("failure_reason", ""),
                        "val_checked": bool(val_meta),
                        "val_available": bool(val_meta and val_meta.get("available")),
                        "val_valid": bool(val_meta and val_meta.get("is_valid")),
                        "val_value": val_meta.get("value") if val_meta else None,
                    }
                )
            except Exception as error:
                record.update(
                    {
                        "plan_path": "",
                        "plan_generation_success": False,
                        "plan_length": 0,
                        "llm_time_cost": 0.0,
                        "planner_time_cost": 0.0,
                        "generation_time": 0.0,
                        "divide_goal": args.divide_goal,
                        "eval_success": False,
                        "task_achieved": False,
                        "finished_steps": 0,
                        "execution_time": 0.0,
                        "plan_steps": 0,
                        "number_of_waits": 0,
                        "final_status": "ERROR",
                        "agent_status": "",
                        "llm_claimed_complete": "",
                        "stop_reason": "error",
                        "failure_reason": "Error",
                        "val_checked": False,
                        "val_available": False,
                        "val_valid": False,
                        "val_value": None,
                        "error": str(error),
                    }
                )
                print(f"Error while running {agent.display_name} on task {index}: {error}")

            records.append(record)
            pd.DataFrame(records).to_csv(log_path, index=False)
            if wandb is not None:
                wandb.log(
                    {
                        "task": index,
                        "agent": record["agent_id"],
                        "plan_generation_success": int(record["plan_generation_success"]),
                        "eval_success": int(record["eval_success"]),
                        "plan_length": record["plan_length"],
                        "execution_time": record["execution_time"],
                    }
                )
                wandb.save(str(log_path), base_path=str(log_path.parent))

            sim.stopSimulation()
            time.sleep(3)

    print(f"\nLog: {log_path}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
