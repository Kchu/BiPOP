#!/usr/bin/env python3
"""
Download BiPOP plans that plan_generation.py uploaded to a W&B run.

Lets plans be generated on one machine (--log_wandb) and executed with
plan_validation.py on the machine that runs CoppeliaSim:

    python scripts/wandb_plan_monitor.py \
        --run_url "https://wandb.ai/<entity>/BiPOP-StackBlock/runs/<run_id>" \
        --task_name StackBlock

Plans go to results/pddl_plans/<task>/ by default. A state file keeps track of
the downloaded plans, so the monitor can be restarted without downloading them again.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bipop.paths import RESULTS_DIR  # noqa: E402

try:
    from wandb import Api
except ImportError:
    sys.exit("wandb is required: pip install wandb")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> Dict[str, str]:
    """Load seen-file state: maps 'run_id/filename' -> updatedAt string."""
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_state(state_path: Path, state: Dict[str, str]) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# File matching
# ---------------------------------------------------------------------------

def _is_merged_plan(filename: str, task_name: str) -> bool:
    """Match only the final merged plan: {task_name}_pddl_plan_{index}.json.

    Excludes stage files like _pddl_plan_0_1.json / _pddl_plan_0_2.json.
    """
    return bool(re.fullmatch(
        rf"{re.escape(task_name)}_pddl_plan_(\d+)\.json",
        filename,
    ))


def _extract_index(filename: str, task_name: str) -> Optional[int]:
    m = re.fullmatch(rf"{re.escape(task_name)}_pddl_plan_(\d+)\.json", filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Core polling logic
# ---------------------------------------------------------------------------

def poll(
    api: Api,
    entity: str,
    project: str,
    task_name: str,
    output_dir: Path,
    state: Dict[str, str],
    run_id_filter: Optional[str],
) -> int:
    """Check W&B for new plan files. Returns the count of newly processed files."""
    processed = 0
    try:
        runs = api.runs(f"{entity}/{project}")
    except Exception as exc:
        print(f"[warn] Could not list runs: {exc}")
        return 0

    for run in runs:
        if run_id_filter and run.id != run_id_filter:
            continue
        cfg_task = run.config.get("task_name", "")
        if task_name and cfg_task and cfg_task != task_name:
            continue

        try:
            files = run.files()
        except Exception as exc:
            print(f"[warn] Could not list files for run {run.id}: {exc}")
            continue

        for wf in files:
            if not _is_merged_plan(wf.name, task_name):
                continue

            updated_at = str(getattr(wf, "updatedAt", ""))
            state_key = f"{run.id}/{wf.name}"
            if state.get(state_key) == updated_at:
                continue  # already processed this exact version

            index = _extract_index(wf.name, task_name)
            if index is None:
                continue

            try:
                wf.download(root=str(output_dir), replace=True)
            except Exception as exc:
                print(f"[error] Download failed for {wf.name} (run {run.id}): {exc}")
                continue

            local_path = output_dir / wf.name
            state[state_key] = updated_at
            processed += 1
            print(f"[new] run={run.id}  index={index}  file={wf.name}  -> {local_path}")

    return processed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_run_url(url: str) -> tuple[str, str, str]:
    """Parse entity, project, run_id from a W&B run URL.

    Accepts: https://wandb.ai/{entity}/{project}/runs/{run_id}[/...]
    """
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", url)
    if not m:
        sys.exit(
            f"Could not parse W&B run URL: {url!r}\n"
            "Expected format: https://wandb.ai/ENTITY/PROJECT/runs/RUN_ID"
        )
    return m.group(1), m.group(2), m.group(3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor a W&B run for new BiPOP plan files from plan_generation.py and download them."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--run_url",
        help="Full W&B run URL, e.g. https://wandb.ai/<entity>/BiPOP-ServeWater/runs/<run_id>",
    )
    src.add_argument(
        "--entity",
        help="W&B entity (username or team); also requires --project and --run_id",
    )
    parser.add_argument("--project", default="",
                        help="W&B project name (required when --entity is used)")
    parser.add_argument("--run_id", default="",
                        help="W&B run ID (required when --entity is used)")
    parser.add_argument("--task_name", required=True,
                        help="Task name used to filter plan file names, e.g. ServeWater")
    parser.add_argument("--output_dir", default=None,
                        help="Local directory for downloaded plan files (default: results/pddl_plans/<task>)")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Seconds between W&B polls (default: 60)")
    parser.add_argument("--state_file", default=None,
                        help="JSON file of the downloaded plans (default: results/wandb_monitor_state_<task>.json)")
    parser.add_argument("--once", action="store_true",
                        help="Poll once and exit instead of looping")
    args = parser.parse_args()

    if args.run_url:
        entity, project, run_id = parse_run_url(args.run_url)
    else:
        entity = args.entity
        project = args.project.strip()
        run_id = args.run_id.strip()
        if not project or not run_id:
            parser.error("--entity requires --project and --run_id")

    output_dir = Path(args.output_dir or RESULTS_DIR / "pddl_plans" / args.task_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file or RESULTS_DIR / f"wandb_monitor_state_{args.task_name}.json").resolve()

    api = Api()
    state = load_state(state_path)

    print(f"Monitoring  : {entity}/{project}  run={run_id}")
    print(f"Task filter : {args.task_name}")
    print(f"Poll interval: {args.poll_interval}s")
    print(f"Output dir  : {output_dir}")
    print(f"State file  : {state_path}")
    print()

    try:
        while True:
            n = poll(
                api=api,
                entity=entity,
                project=project,
                task_name=args.task_name,
                output_dir=output_dir,
                state=state,
                run_id_filter=run_id,
            )
            save_state(state_path, state)
            ts = time.strftime("%H:%M:%S")
            if n:
                print(f"[{ts}] {n} new plan(s) processed (total seen: {len(state)})")
            else:
                print(f"[{ts}] no new plans (total seen: {len(state)})")

            if args.once:
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        save_state(state_path, state)


if __name__ == "__main__":
    main()
