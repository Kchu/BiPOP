"""Repository paths and the locations of external tools.

Every output of the experiments is written below ``RESULTS_DIR``. External
binaries are configured through environment variables, see the README.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
TASK_CONFIGS_DIR = ROOT_DIR / "task_configs"
NICOL_DIR = ROOT_DIR / "NICOL"
EXECUTOR_PATH = NICOL_DIR / "nicol_executor.py"


def result_paths(results_dir: Path, task_name: str) -> Dict[str, Path]:
    """Output folders of one task below ``results_dir``.

    results/
      direct_plans/<task>/   LLM-agent plans
      pddl_plans/<task>/     BiPOP plans (+ intermediate/ planner artifacts)
      react_plans/<task>/    ReAct-agent plans and step logs
      logs/                  per-run CSV logs and execution logs
    """
    results_dir = Path(results_dir).resolve()
    return {
        "root": results_dir,
        "direct_plans": results_dir / "direct_plans" / task_name,
        "pddl_plans": results_dir / "pddl_plans" / task_name,
        "react_plans": results_dir / "react_plans" / task_name,
        "logs": results_dir / "logs",
    }


def tfd_dir() -> Path:
    """Directory of Temporal Fast Downward that holds translate/, preprocess/ and search/.

    Set with ``TFD_DIR`` (for example ``.../tfd/downward``).
    """
    value = os.environ.get("TFD_DIR")
    if not value:
        raise RuntimeError(
            "TFD_DIR is not set. Point it to the Temporal Fast Downward 'downward' "
            "directory that contains translate/, preprocess/ and search/."
        )
    return Path(value).expanduser().resolve()


def default_val_bin() -> Optional[str]:
    """Path of the VAL ``Validate`` binary: ``$VAL_BIN``, otherwise ``Validate`` on PATH."""
    return os.environ.get("VAL_BIN") or shutil.which("Validate")
