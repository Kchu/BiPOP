"""Temporal planning with Temporal Fast Downward (TFD) and plan validation with VAL.

A generated planning script is run to write the PDDL domain/problem, TFD is
started once per search preset in parallel, and every plan TFD finds is
checked with VAL. The VAL-valid plan with the smallest makespan is selected.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from ..paths import default_val_bin, tfd_dir
from ..plan_format import parse_makespan, parse_plan_length
from .code_blocks import domain_definition, problem_initialization
from .log import debug_print

# ---------------------------------------------------------------------------
# TFD search presets (search options joined with "+")
# ---------------------------------------------------------------------------

DEFAULT_PRESETS = {
    "first": "y+Y+a+e+r+G+m+C+1+v+t+10+T+60",
    "second": "y+Y+a+e+r+G+m+C+2+O+1+v+t+10+T+60",
    "third": "y+Y+a+e+r+G+m+C+2+E+2+v+t+10+T+60",
}

STACKBLOCK_STAGE_PRESETS = {
    "stage1": {
        "1": "y+Y+a+e+r+S+Q+p+G+m+C+1+M+v+t+5+T+5",
        "2": "y+Y+a+e+r+Q+p+G+m+O+1+M+v+t+5+T+5",
        "3": "y+Y+a+e+r+Q+p+G+c+C+2+M+v+t+5+T+5",
        "5": "y+Y+a+e+r+S+O+1+C+1+M+v+t+5+T+5",
        "6": "y+Y+a+e+r+S+C+1+v+t+5+T+5",
        "7": "y+Y+a+e+r+S+G+c+O+1+v+t+5+T+5",
        "8": "y+Y+a+e+r+S+C+2+v+t+5+T+5",
    },
    "stage2": {
        "1": "y+Y+a+e+r+Q+p+C+1+M+v+t+20+T+20",
        "2": "y+Y+a+e+r+Q+p+O+1+M+v+t+20+T+20",
        "3": "y+Y+a+e+r+Q+p+G+c+C+2+M+v+t+20+T+20",
        "4": "y+Y+a+e+r+S+C+2+O+1+M+v+t+20+T+20",
        "5": "y+Y+a+e+r+S+O+1+C+1+v+t+20+T+20",
        "6": "y+Y+a+e+r+S+C+1+v+t+20+T+20",
        "7": "y+Y+a+e+r+S+G+c+O+1+v+t+20+T+20",
        "8": "y+Y+a+e+r+S+C+2+v+t+20+T+20",
    },
    "single": {
        "3": "y+Y+a+e+r+Q+p+G+c+C+2+M+v+t+25+T+25",
        "6": "y+Y+a+e+r+S+C+1+v+t+25+T+25",
        "7": "y+Y+a+e+r+S+G+c+O+1+v+t+25+T+25",
        "8": "y+Y+a+e+r+S+C+2+v+t+25+T+25",
    },
}

PRESETS_BY_TASK = {
    "__default__": DEFAULT_PRESETS,
    "StackBlock-super": {
        "first": "y+Y+a+e+r+Q+p+G+m+C+1+v+M+v+t+10+T+15",
        "second": "y+Y+a+e+r+G+c+C+2+O+1+v+M+v+t+10+T+15",
        "third": "y+Y+a+e+r+G+m+C+2+E+2+v+M+v+t+10+T+15",
        "fourth": "y+Y+a+e+r+G+c+C+2+v+M+v+t+10+T+15",
        "fifth": "y+Y+a+e+r+S+C+1+O+1+v+t+10+T+15",
    },
    "ServeWater": {
        "first": "y+Y+a+e+r+Q+p+G+m+C+1+v+t+1+T+1",
        "second": "y+Y+a+e+r+S+C+2+O+1+v+t+1+T+1",
        "third": "y+Y+a+e+r+Q+p+G+c+C+2+E+2+v+t+1+T+1",
    },
    "ServeFruit": {
        "first": "y+Y+a+e+r+Q+p+G+m+C+1+v+t+1+T+1",
        "second": "y+Y+a+e+r+S+C+2+O+1+v+t+1+T+1",
        "third": "y+Y+a+e+r+Q+p+G+c+C+2+E+2+v+t+1+T+1",
    },
}


def get_presets_for_task(task_name: str, stage_label: str = "single") -> Dict[str, str]:
    """TFD presets of a task; StackBlock has separate presets per stage."""
    normalized = task_name.replace(" ", "")
    if normalized == "StackBlock":
        return STACKBLOCK_STAGE_PRESETS.get(stage_label, STACKBLOCK_STAGE_PRESETS["single"])
    return (
        PRESETS_BY_TASK.get(task_name)
        or PRESETS_BY_TASK.get(normalized)
        or PRESETS_BY_TASK["__default__"]
    )


# ---------------------------------------------------------------------------
# VAL
# ---------------------------------------------------------------------------

# VAL's default tolerance (0.01) is 10x larger than TFD's inter-action epsilon
# (0.001). VAL snaps timestamps onto a tolerance-sized grid, so two actions TFD
# scheduled one epsilon apart can collapse onto the same happening and become
# simultaneous, producing spurious mutex violations. A tolerance well below the
# epsilon keeps 0.001-separated actions distinct. Overridable via the environment.
VAL_TOLERANCE = os.environ.get("VAL_TOLERANCE", "0.0001")



def val_output_is_valid(output: str) -> bool:
    """VAL's verdict on a plan.

    VAL prints "Plan executed successfully - checking goal" *before* it tests the goal,
    so that line is no verdict: "Goal not satisfied" and "Plan invalid" may follow it.
    Only "Plan valid" (and no "Plan invalid" / "Goal not satisfied") means valid.
    """
    return "Plan valid" in output and "Plan invalid" not in output and "Goal not satisfied" not in output


VAL_VALUE_RE = re.compile(r"(?:Final\s+value|Value)\s*:\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
VAL_REASON_RE = re.compile(
    r"(plan valid|plan executed successfully|goal|final value|value\s*:|metric)",
    re.IGNORECASE,
)
VAL_FAILURE_RE = re.compile(
    (
        r"(object with unknown type|unknown type|unsatisfied precondition|"
        r"plan failed|failed to execute|failed because|invalid|violation|"
        r"failed|error|unsatisfied)"
    ),
    re.IGNORECASE,
)
VAL_INVALID_RE = re.compile(
    r"(invalid|violation|failed|error|unsatisfied|unknown type|type-checking)",
    re.IGNORECASE,
)
MAX_VAL_FAILURE_CANDIDATES_FOR_LLM = 4
MAX_VAL_EXCERPT_LINES_FOR_LLM = 20


def _extract_val_evidence(output: str, is_valid: bool) -> List[str]:
    """Lines of the VAL output that explain the verdict (failure lines first)."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return []

    if is_valid:
        evidences = [line for line in lines if VAL_REASON_RE.search(line)]
    else:
        evidences = [line for line in lines if VAL_FAILURE_RE.search(line)]
        if not evidences:
            evidences = [line for line in lines if VAL_INVALID_RE.search(line)]
    if not evidences:
        evidences = lines[-8:]
    return evidences[:12]


def _extract_val_origin_excerpt(output: str, is_valid: bool) -> List[str]:
    """Numbered excerpt (``L<n>: ...``) of the VAL output around the relevant lines."""
    raw_lines = output.splitlines()
    if not raw_lines:
        return []

    if is_valid:
        anchor_indices = [idx for idx, line in enumerate(raw_lines) if VAL_REASON_RE.search(line)]
    else:
        # Prefer substantive failure lines over routine "Type-checking ..." chatter.
        anchor_indices = [idx for idx, line in enumerate(raw_lines) if VAL_FAILURE_RE.search(line)]
        if not anchor_indices:
            anchor_indices = [idx for idx, line in enumerate(raw_lines) if VAL_INVALID_RE.search(line)]

    if not anchor_indices:
        start = max(0, len(raw_lines) - MAX_VAL_EXCERPT_LINES_FOR_LLM)
        return [f"L{idx + 1}: {raw_lines[idx]}" for idx in range(start, len(raw_lines))]

    selected: List[int] = []
    for idx in anchor_indices:
        selected.extend(range(max(0, idx - 1), min(len(raw_lines), idx + 4)))

    seen: set[int] = set()
    excerpt_indices: List[int] = []
    for idx in selected:
        if idx in seen:
            continue
        seen.add(idx)
        excerpt_indices.append(idx)
        if len(excerpt_indices) >= MAX_VAL_EXCERPT_LINES_FOR_LLM:
            break
    return [f"L{idx + 1}: {raw_lines[idx]}" for idx in excerpt_indices]


def run_val(
    domain: str | os.PathLike[str],
    problem: str | os.PathLike[str],
    plan_file: str | os.PathLike[str],
    val_bin: str | os.PathLike[str] | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate a temporal plan with VAL.

    Returns ``available`` (VAL could be run), ``is_valid``, ``value`` (plan
    metric), the raw ``output`` and the lines that explain the verdict.
    """
    effective_val_bin = str(val_bin or default_val_bin() or "Validate")
    command = [effective_val_bin, "-t", VAL_TOLERANCE]
    if verbose:
        command.append("-v")
    command += [str(domain), str(problem), str(plan_file)]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
        )
    except OSError as exc:  # missing or not executable
        message = f"VAL binary not found at: {effective_val_bin} ({type(exc).__name__})"
        return {
            "available": False,
            "is_valid": False,
            "value": None,
            "output": message,
            "evidence_lines": [message],
            "origin_excerpt_lines": [f"L1: {message}"],
        }

    output = result.stdout or ""
    is_valid = val_output_is_valid(output)
    matched = VAL_VALUE_RE.search(output)
    return {
        "available": True,
        "is_valid": is_valid,
        "value": float(matched.group(1)) if (matched and is_valid) else None,
        "output": output,
        "evidence_lines": _extract_val_evidence(output, is_valid),
        "origin_excerpt_lines": _extract_val_origin_excerpt(output, is_valid),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_optional_float(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def format_best_valid_preset(value: Any) -> str:
    if not isinstance(value, str):
        return "NA"
    value = value.strip()
    return value if value and value.lower() not in {"none", "null", "na", "n/a"} else "NA"


def plan_quality_key(
    makespan: float | None,
    plan_length: int | None,
    val_value: float | None = None,
) -> tuple[float, float, float, float]:
    """Sort key of a plan: has a makespan, then makespan, plan length and VAL value."""
    has_makespan = 0.0 if makespan is not None else 1.0
    ms_key = float("inf") if makespan is None else makespan
    len_key = float("inf") if plan_length is None else float(plan_length)
    val_key = float("inf") if val_value is None else val_value
    return (has_makespan, ms_key, len_key, val_key)


def _best_valid_preset_from_val(val_result: Dict[str, Any]) -> str | None:
    if not val_result.get("val_available", True) or val_result.get("best_plan_path") is None:
        return None
    best_preset = val_result.get("best_preset")
    return best_preset if isinstance(best_preset, str) and best_preset.strip() else None


def _format_invalid_val_candidates_for_llm(invalid_candidates: List[Dict[str, Any]]) -> List[str]:
    """VAL failure details of invalid plans, fed back to the LLM on a retry."""
    if not invalid_candidates:
        return []

    lines = [
        "VAL failure details for invalid planner candidates "
        f"(showing up to {MAX_VAL_FAILURE_CANDIDATES_FOR_LLM}):"
    ]
    for candidate in invalid_candidates[:MAX_VAL_FAILURE_CANDIDATES_FOR_LLM]:
        val_report = candidate.get("val_report") or {}
        lines.append(
            f"preset={candidate.get('preset', 'NA')} "
            f"plan={candidate.get('plan_path', 'NA')} "
            f"makespan={format_optional_float(candidate.get('makespan'))} "
            f"plan_length={candidate.get('plan_length', 'NA')}"
        )
        evidence_lines = val_report.get("evidence_lines") or []
        if evidence_lines:
            lines.append("  VAL error evidence:")
            lines.extend(f"    - {evidence}" for evidence in evidence_lines[:6])
        origin_excerpt_lines = val_report.get("origin_excerpt_lines") or []
        if origin_excerpt_lines:
            lines.append("  VAL original output excerpt:")
            lines.extend(f"    {line}" for line in origin_excerpt_lines[:MAX_VAL_EXCERPT_LINES_FOR_LLM])
    return lines


def _print_val_short_summary(stage_label: str, val_result: Dict[str, Any]) -> None:
    if not val_result.get("val_available", True):
        reason = val_result.get("val_log_lines") or ["VAL unavailable"]
        debug_print(f"[VAL] {stage_label}: unavailable | {reason[0]}", flush=True)
        return

    total = val_result.get("total_candidates", 0)
    valid = val_result.get("valid_candidates", 0)
    invalid = max(total - valid, 0)
    if val_result.get("best_plan_path") is None:
        debug_print(f"[VAL] {stage_label}: no valid plan | checked={total} | invalid={invalid}", flush=True)
        return

    value = val_result.get("best_val_value")
    value_text = f" | value={value}" if value is not None else ""
    debug_print(
        f"[VAL] {stage_label}: valid | checked={total} | valid={valid} | invalid={invalid} "
        f"| makespan={format_optional_float(val_result.get('best_makespan'))} "
        f"| length={val_result.get('best_plan_length')}{value_text}",
        flush=True,
    )


def _print_preset_val_summary(
    stage_label: str,
    preset_names: List[str] | None,
    candidates_by_preset: Dict[str, List[str]],
    preset_has_valid: Dict[str, bool],
    preset_best_valid_makespan: Dict[str, float | None],
) -> None:
    for preset_name in preset_names or sorted(candidates_by_preset):
        if preset_has_valid.get(preset_name):
            makespan_text = format_optional_float(preset_best_valid_makespan.get(preset_name))
            print(
                f"[VAL] stage={stage_label} preset={preset_name} status=valid "
                f"best_makespan={makespan_text}",
                flush=True,
            )
        else:
            print(f"[VAL] stage={stage_label} preset={preset_name} status=invalid", flush=True)


# ---------------------------------------------------------------------------
# TFD
# ---------------------------------------------------------------------------

def check_planner_setup(val_bin: str | os.PathLike[str] | None) -> None:
    """Fail before any LLM call if TFD is missing; warn if VAL is missing.

    Without VAL, plans are not validated and the shortest TFD plan is used as is.
    """
    downward_dir = tfd_dir()
    for relative in ("translate/translate.py", "preprocess/preprocess", "search/search"):
        if not (downward_dir / relative).is_file():
            raise RuntimeError(f"TFD_DIR={downward_dir} has no {relative}; build TFD and check TFD_DIR.")
    if not val_bin or not Path(val_bin).is_file():
        print(
            f"[WARNING] VAL binary not found ({val_bin or 'VAL_BIN not set, Validate not on PATH'}): "
            "plans will NOT be validated. Set VAL_BIN or pass --val_bin.",
            flush=True,
        )


def find_all_plan_files(base_name: str = "plan.txt") -> List[str]:
    """TFD plan files ``<base_name>``, ``<base_name>.1``, ``<base_name>.2``, ... in order."""
    candidates = [p for p in glob.glob(f"{base_name}*") if os.path.isfile(p)]

    def _suffix_index(path: str) -> int:
        if path == base_name:
            return 0
        prefix = f"{base_name}."
        if path.startswith(prefix) and path[len(prefix):].isdigit():
            return int(path[len(prefix):])
        return 10**9

    return sorted(candidates, key=lambda p: (_suffix_index(p), p))


def cleanup_files(pattern: str) -> None:
    for file_path in glob.glob(pattern):
        if os.path.isfile(file_path):
            os.remove(file_path)


def _run_tfd_preset_isolated(
    preset_name: str,
    config: str,
    domain: str,
    problem: str,
    plan_prefix: str,
    task_plan_dir: Path,
    index: int | str,
) -> str:
    """Run translate -> preprocess -> search for one preset in its own work directory."""
    downward_dir = tfd_dir()
    work_dir = task_plan_dir / f"{preset_name}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    for temp_name in ("all.groups", "output", "output.sas", "variables.groups"):
        temp_path = work_dir / temp_name
        if temp_path.exists():
            temp_path.unlink()

    report_path = task_plan_dir / f"{preset_name}_tfd_{index}.log"
    report_lines = [
        f"preset: {preset_name}",
        "mode: isolated-parallel",
        f"work_dir: {work_dir}",
        f"domain: {domain}",
        f"problem: {problem}",
        f"plan_prefix: {plan_prefix}",
        f"config: {config}",
    ]
    error_summaries: List[str] = []

    def run_step(name: str, command: List[str], input_file: str | None = None) -> int:
        report_lines.append(f"\n=== {name.upper()} COMMAND ===")
        report_lines.append(" ".join(command))
        stdin_handle = None
        try:
            if input_file is not None:
                stdin_handle = (work_dir / input_file).open("r", encoding="utf-8", errors="ignore")
            result = subprocess.run(
                command,
                cwd=work_dir,
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            summary = f"{name}: {type(exc).__name__}: {exc}"
            error_summaries.append(summary)
            report_lines.append(f"exception: {summary}")
            return 1
        finally:
            if stdin_handle is not None:
                stdin_handle.close()

        report_lines.append(f"returncode: {result.returncode}")
        report_lines.append(f"\n=== {name.upper()} STDOUT ===")
        report_lines.append(result.stdout or "")
        report_lines.append(f"\n=== {name.upper()} STDERR ===")
        report_lines.append(result.stderr or "")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            summary = detail[0] if detail else f"returncode={result.returncode}"
            error_summaries.append(f"{name}: {summary}")
        return result.returncode

    steps = [
        ("translate", [str(downward_dir / "translate" / "translate.py"), domain, problem], None),
        ("preprocess", [str(downward_dir / "preprocess" / "preprocess")], "output.sas"),
        ("search", [str(downward_dir / "search" / "search"), *config.split("+"), "p", plan_prefix], "output"),
    ]

    final_returncode = 0
    for step_name, command, input_file in steps:
        final_returncode = run_step(step_name, command, input_file)
        if final_returncode != 0:
            break

    discovered = find_all_plan_files(plan_prefix)
    report_lines.append("\n=== DISCOVERED PLAN FILES ===")
    report_lines.extend(discovered or ["<none>"])
    if not discovered and not error_summaries:
        error_summaries.append("search: no plan files discovered")
    report_lines.append(f"\nfinal_returncode: {final_returncode}")
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    summary_text = f" error={' | '.join(error_summaries)}" if error_summaries else ""
    return f"{preset_name}: returncode={final_returncode} plans={len(discovered)}{summary_text} log={report_path}"


def run_tfd_parallel(
    domain: str,
    problem: str,
    plan_dir: str,
    task_name: str,
    index: int | str,
    stage_label: str = "single",
) -> Dict[str, Any]:
    """Run all TFD presets of the task in parallel and collect every plan found.

    Plans go to ``<plan_dir>/task_<index>/<preset>_plan_<index>.txt.<n>``.
    """
    task_plan_dir = Path(plan_dir) / f"task_{index}"
    task_plan_dir.mkdir(parents=True, exist_ok=True)

    presets = get_presets_for_task(task_name, stage_label)
    for preset_name in presets:
        cleanup_files(str(task_plan_dir / f"{preset_name}_plan_{index}.txt*"))

    preset_reports: List[str] = []
    prefixes: Dict[str, str] = {}
    futures = {}
    debug_print(
        f"[TFD] task={index} stage={stage_label}: launching {len(presets)} isolated preset worker(s) "
        "| presets=" + ", ".join(f"{name}={config}" for name, config in presets.items()),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, len(presets))) as executor:
        for preset_name, config in presets.items():
            plan_prefix = str(task_plan_dir / f"{preset_name}_plan_{index}.txt")
            prefixes[preset_name] = plan_prefix
            future = executor.submit(
                _run_tfd_preset_isolated,
                preset_name, config, domain, problem, plan_prefix, task_plan_dir, index,
            )
            futures[future] = preset_name

        for future in as_completed(futures):
            preset_name = futures[future]
            try:
                report = future.result()
            except Exception as exc:
                report = f"{preset_name}: exception={type(exc).__name__}: {exc}"
            preset_reports.append(report)
            debug_print(f"[TFD] task={index}: {report}", flush=True)
    debug_print(f"[TFD] task={index}: presets complete", flush=True)

    best: Dict[str, Any] = {"preset": None, "makespan": None, "plan_length": None, "path": None, "text": None}
    all_plan_paths: Dict[str, List[str]] = {}
    for preset_name, plan_prefix in prefixes.items():
        plan_files = find_all_plan_files(plan_prefix)
        if not plan_files:
            continue
        all_plan_paths[preset_name] = plan_files
        for plan_path in plan_files:
            text = Path(plan_path).read_text(encoding="utf-8", errors="ignore")
            makespan = parse_makespan(text)
            plan_length = parse_plan_length(text)
            if best["path"] is None or plan_quality_key(makespan, plan_length) < plan_quality_key(
                best["makespan"], best["plan_length"]
            ):
                best = {
                    "preset": preset_name,
                    "makespan": makespan,
                    "plan_length": plan_length,
                    "path": plan_path,
                    "text": text,
                }

    return {
        "best_preset": best["preset"],
        "best_ms": best["makespan"],
        "best_plan_len": best["plan_length"],
        "best_plan_path": best["path"],
        "best_plan_text": best["text"],
        "all_plan_paths": all_plan_paths,
        "preset_reports": preset_reports,
        "preset_names": list(presets.keys()),
    }


def select_val_valid_plan(
    domain: str,
    problem: str,
    candidates_by_preset: Dict[str, List[str]],
    val_bin: str | os.PathLike[str] | None = None,
    stage_label: str = "single",
    preset_names: List[str] | None = None,
) -> Dict[str, Any]:
    """Validate every candidate plan with VAL and select the best valid one."""
    valid_candidates: List[Dict[str, Any]] = []
    invalid_candidates: List[Dict[str, Any]] = []
    total_candidates = 0
    preset_has_valid: Dict[str, bool] = {}
    preset_best_valid_makespan: Dict[str, float | None] = {}

    for preset_name, plan_paths in candidates_by_preset.items():
        for plan_path in plan_paths:
            total_candidates += 1
            plan_text = Path(plan_path).read_text(encoding="utf-8", errors="ignore")
            makespan = parse_makespan(plan_text)
            plan_length = parse_plan_length(plan_text)
            if val_bin is None:
                val_report = run_val(domain, problem, plan_path)
            else:
                val_report = run_val(domain, problem, plan_path, val_bin)

            if not val_report["available"]:
                print(f"[VAL] stage={stage_label} preset-summary skipped: VAL unavailable", flush=True)
                return {
                    "val_available": False,
                    "best_plan_path": None,
                    "best_preset": None,
                    "best_makespan": None,
                    "best_plan_length": None,
                    "best_val_value": None,
                    "total_candidates": total_candidates,
                    "valid_candidates": len(valid_candidates),
                    "val_log_lines": [f"VAL unavailable: {val_report['output']}"],
                }

            if val_report["is_valid"]:
                preset_has_valid[preset_name] = True
                if makespan is not None:
                    best_ms = preset_best_valid_makespan.get(preset_name)
                    if best_ms is None or makespan < best_ms:
                        preset_best_valid_makespan[preset_name] = makespan
                else:
                    preset_best_valid_makespan.setdefault(preset_name, None)
                valid_candidates.append(
                    {
                        "preset": preset_name,
                        "plan_path": plan_path,
                        "makespan": makespan,
                        "plan_length": plan_length,
                        "val_value": val_report["value"],
                    }
                )
            else:
                invalid_candidates.append(
                    {
                        "preset": preset_name,
                        "plan_path": plan_path,
                        "makespan": makespan,
                        "plan_length": plan_length,
                        "val_report": val_report,
                    }
                )

    _print_preset_val_summary(
        stage_label, preset_names, candidates_by_preset, preset_has_valid, preset_best_valid_makespan
    )

    if not valid_candidates:
        return {
            "val_available": True,
            "best_plan_path": None,
            "best_preset": None,
            "best_makespan": None,
            "best_plan_length": None,
            "best_val_value": None,
            "total_candidates": total_candidates,
            "valid_candidates": 0,
            "val_log_lines": _format_invalid_val_candidates_for_llm(invalid_candidates),
        }

    valid_candidates.sort(
        key=lambda item: plan_quality_key(item["makespan"], item["plan_length"], item["val_value"])
    )
    best = valid_candidates[0]
    return {
        "val_available": True,
        "best_plan_path": best["plan_path"],
        "best_preset": best["preset"],
        "best_makespan": best["makespan"],
        "best_plan_length": best["plan_length"],
        "best_val_value": best["val_value"],
        "total_candidates": total_candidates,
        "valid_candidates": len(valid_candidates),
        "val_log_lines": [],
    }


# ---------------------------------------------------------------------------
# From generated code to a VAL-valid plan
# ---------------------------------------------------------------------------

def write_plan_file(
    plan_file_path: str,
    domain_definition_block: str,
    code_string: str,
    action_definition: str,
    pddl_writer_definition: str,
) -> None:
    """Write the complete planning script: domain, problem init, LLM code, actions, PDDL writer."""
    plan_content = f"""
## System code string:
{domain_definition_block}
## Problem code string:
{problem_initialization}
## The following is generated by GPT.
{"#" * 60}
{code_string}
## Action code string:
{action_definition}
## write domain and problem to pddl files
{pddl_writer_definition}
"""
    Path(plan_file_path).write_text(plan_content, encoding="utf-8")


def emit_tfd_val_report(
    report_dir: str,
    index: int,
    stage_label: str,
    trial: int | None,
    report_text: str,
    cached: bool = False,
) -> None:
    """Write the planner/VAL report of one stage and trial to ``tfd_val_<index>_<stage>_trial_<n>.log``."""
    trial_text = "NA" if trial is None else str(trial)
    cache_text = " cached" if cached else ""
    header = f"=== TFD/VAL {stage_label} trial {trial_text}{cache_text} ==="
    footer = f"=== END TFD/VAL {stage_label} trial {trial_text}{cache_text} ==="

    name = f"tfd_val_{index}_{stage_label}_trial_{trial_text}{'_cached' if cached else ''}.log"
    report_path = Path(report_dir) / name
    report_path.write_text(f"{header}\n{report_text.rstrip()}\n{footer}\n", encoding="utf-8")
    first_line = report_text.strip().splitlines()[0] if report_text.strip() else "no report"
    debug_print(f"[TFD/VAL] {stage_label} trial {trial_text}{cache_text}: {first_line} | log={report_path}", flush=True)


def solve_code_string(
    code_string: str,
    plan_file_path: str,
    task_name: str,
    index: int,
    stage_label: str,
    action_definition: str,
    trial: int | None = None,
    val_bin: str | os.PathLike[str] | None = None,
    intermediate_dir: str | None = None,
) -> Dict[str, Any]:
    """Turn LLM-generated UP code into a VAL-valid temporal plan.

    Writes the planning script ``<plan_file_path>/<task>_llm_pddl_<index>[_<stage>].py``,
    runs it to produce the PDDL files in ``intermediate_dir``, plans with all
    TFD presets and selects the best VAL-valid plan. Raises
    ``subprocess.CalledProcessError`` when the generated script crashes.
    """
    suffix = "" if stage_label == "single" else f"_{stage_label}"
    idir = intermediate_dir if intermediate_dir is not None else plan_file_path
    py_path = Path(plan_file_path) / f"{task_name}_llm_pddl_{index}{suffix}.py"
    domain = f"{idir}/{task_name}_domain_{index}{suffix}.pddl"
    problem = f"{idir}/{task_name}_problem_{index}{suffix}.pddl"
    pddl_writer_definition = f"""
writer = PDDLWriter(problem)
writer.write_domain({domain!r})
writer.write_problem({problem!r})
"""
    trial_text = trial if trial is not None else "NA"
    debug_print(f"[PDDL] stage={stage_label} trial={trial_text} writing generator script: {py_path}", flush=True)
    write_plan_file(str(py_path), domain_definition, code_string, action_definition, pddl_writer_definition)
    # A crashing script must not leave the PDDL of an earlier trial behind.
    for pddl_path in (domain, problem):
        Path(pddl_path).unlink(missing_ok=True)

    debug_print(f"[PDDL] stage={stage_label} trial={trial_text} running generator script", flush=True)
    subprocess.run([sys.executable, str(py_path)], capture_output=True, text=True, check=True)
    debug_print(f"[PDDL] stage={stage_label} trial={trial_text} generated domain/problem: {domain} | {problem}", flush=True)

    tfd_index = index if stage_label == "single" else f"{index}_{stage_label}"
    planner_start = time.time()
    parallel_result = run_tfd_parallel(domain, problem, idir, task_name, tfd_index, stage_label)
    planner_time_cost = time.time() - planner_start

    if parallel_result["all_plan_paths"]:
        val_result = select_val_valid_plan(
            domain,
            problem,
            parallel_result["all_plan_paths"],
            val_bin=val_bin,
            stage_label=stage_label,
            preset_names=parallel_result.get("preset_names"),
        )
    else:
        val_result = {
            "val_available": True,
            "best_plan_path": None,
            "best_preset": None,
            "best_makespan": None,
            "best_plan_length": None,
            "best_val_value": None,
            "total_candidates": 0,
            "valid_candidates": 0,
            "val_log_lines": [],
        }
        _print_preset_val_summary(
            stage_label, parallel_result.get("preset_names"), parallel_result.get("all_plan_paths", {}), {}, {}
        )

    _print_val_short_summary(stage_label, val_result)
    best_valid_preset = _best_valid_preset_from_val(val_result)
    best_valid_preset_text = format_best_valid_preset(best_valid_preset)

    if val_result["val_available"]:
        selected_plan_path = val_result["best_plan_path"]
        selected_makespan = val_result["best_makespan"]
        selected_plan_length = val_result["best_plan_length"]
        selected_val_value = val_result["best_val_value"]
        failure_reason = "No VAL-valid plan was found.\n"
    else:
        selected_plan_path = parallel_result["best_plan_path"]
        selected_makespan = parallel_result["best_ms"]
        selected_plan_length = parallel_result["best_plan_len"]
        selected_val_value = None
        failure_reason = "No plan was found.\n"

    if selected_plan_path is None:
        val_log_lines = val_result.get("val_log_lines") or []
        val_log_text = "\n".join(val_log_lines)
        if val_log_text:
            val_log_text += "\n"
            print("[VAL] error details sent to LLM:", flush=True)
            for line in val_log_lines:
                print(line, flush=True)
        report = (
            failure_reason
            + f"Checked plans: {val_result.get('total_candidates', 0)}\n"
            + f"Valid plans: {val_result.get('valid_candidates', 0)}\n"
            + f"Best valid planner preset: {best_valid_preset_text}\n"
            + f"Planner time: {planner_time_cost:.6f}\n"
            + val_log_text
            + "\n".join(parallel_result.get("preset_reports", []))
            + "\n"
        )
        emit_tfd_val_report(idir, index, stage_label, trial, report)
        return {
            "success": False,
            "domain": domain,
            "problem": problem,
            "plan_path": None,
            "plan_text": None,
            "output": report,
            "val": val_result,
            "best_valid_preset": best_valid_preset,
            "planner_time_cost": planner_time_cost,
        }

    plan_text = Path(selected_plan_path).read_text(encoding="utf-8", errors="ignore")
    output = (
        "New solution has been found.\n"
        f"Checked plans: {val_result.get('total_candidates', 0)}\n"
        f"Valid plans: {val_result.get('valid_candidates', 0)}\n"
        f"Makespan: {format_optional_float(selected_makespan)}\n"
        f"Plan length: {selected_plan_length}\n"
        f"Best valid planner preset: {best_valid_preset_text}\n"
        f"Planner time: {planner_time_cost:.6f}\n"
        + (
            f"VAL valid: True\nVAL value: {selected_val_value}\n"
            if val_result["val_available"]
            else "VAL valid: skipped (VAL unavailable)\n"
        )
    )
    emit_tfd_val_report(idir, index, stage_label, trial, output)
    return {
        "success": True,
        "domain": domain,
        "problem": problem,
        "py_path": str(py_path),
        "plan_path": selected_plan_path,
        "plan_text": plan_text,
        "output": output,
        "val": val_result,
        "best_valid_preset": best_valid_preset,
        "makespan": selected_makespan,
        "plan_length": selected_plan_length,
        "planner_time_cost": planner_time_cost,
    }
