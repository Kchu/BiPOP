#!/usr/bin/env python
"""Sample new random task configs in CoppeliaSim.

Needs CoppeliaSim running with NICOL/nicol.ttt loaded. Configs are written to
``<output_dir>/<task>/task_config_<i>.json`` (default: results/task_configs, so
the shipped configs in task_configs/ are not overwritten).

Example:
    python scripts/generate_task_configs.py --task_name StackBlock --task_nums 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bipop.experiment import add_ik_arg, select_ik_solver  # noqa: E402
from bipop.paths import RESULTS_DIR  # noqa: E402

TASKS = ("ServeWater", "ServeFruit", "StackBlock-simple", "StackBlock-complex", "StackBlock-super", "StackBlock")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample random task configs in CoppeliaSim.")
    parser.add_argument("--task_name", choices=TASKS, required=True)
    parser.add_argument("--task_nums", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR / "task_configs")
    add_ik_arg(parser)
    args = parser.parse_args()

    select_ik_solver(args.ik)
    from NICOL.tasks import create_task

    task = create_task(args.task_name)
    sim = task.sim
    config_dir = args.output_dir / args.task_name
    config_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.start_index, args.start_index + args.task_nums):
        print("*" * 80)
        sim.startSimulation()
        time.sleep(2)
        task.reset()
        config_path = config_dir / f"task_config_{index}.json"
        config_path.write_text(json.dumps(task.config), encoding="utf-8")
        time.sleep(2)
        print(f"Task {index}: type {task.config['type']} -> {config_path}")
        sim.stopSimulation()
        time.sleep(2)


if __name__ == "__main__":
    main()
