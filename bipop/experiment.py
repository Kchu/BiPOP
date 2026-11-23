"""Helpers shared by the experiment scripts (main.py, plan_generation.py, plan_validation.py)."""

from __future__ import annotations

import argparse
import json
import os
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import RESULTS_DIR, TASK_CONFIGS_DIR

IK_SOLVERS = ("cycleik", "coppelia", "dls")


def add_task_selection_args(parser: argparse.ArgumentParser, default_task_nums: int = 100) -> None:
    parser.add_argument("--task_nums", type=int, default=default_task_nums, help="Number of task instances.")
    parser.add_argument("--start_index", type=int, default=0, help="First task index.")
    parser.add_argument(
        "--task_indices",
        type=str,
        default="",
        help="Comma-separated task indices; overrides --start_index and --task_nums.",
    )


def add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR, help="Root folder of all outputs.")
    parser.add_argument(
        "--task_configs_dir",
        type=Path,
        default=TASK_CONFIGS_DIR,
        help="Folder with <task>/task_config_<i>.json.",
    )
    parser.add_argument("--log_wandb", action=BooleanOptionalAction, default=False, help="Log to Weights & Biases.")


def add_ik_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ik",
        choices=IK_SOLVERS,
        default=os.environ.get("NICOL_IK", "cycleik"),
        help="Arm IK solver of the simulated NICOL (see NICOL/nicol_sim.py).",
    )


def select_ik_solver(ik: str) -> None:
    """Select the IK solver; set before NICOL is created so the executor processes inherit it."""
    os.environ["NICOL_IK"] = ik


def load_task_config(task_configs_dir: Path, task_name: str, index: int) -> Optional[Dict[str, Any]]:
    config_path = Path(task_configs_dir) / task_name / f"task_config_{index}.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def task_description(config: Dict[str, Any]) -> str:
    """Task description given to the agents: the task text and the visual scene description."""
    return config["task_des"] + config.get("task_des_visual", "")


def init_wandb(enabled: bool, project: str, run_config: Dict[str, Any], results_dir: Path):
    """Start a W&B run (files go to <results_dir>/wandb) and return the module, or None if disabled."""
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ModuleNotFoundError("wandb is not installed. Use --no-log_wandb or install wandb.") from exc
    wandb_dir = Path(results_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project=project, config=run_config, dir=str(wandb_dir))
    return wandb
