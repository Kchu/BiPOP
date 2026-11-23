"""LLM-agent baseline: the LLM writes the step-aligned bimanual plan directly as JSON."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .client import build_client, chat
from .log import debug_print
from .prompts import choose_skill_desc, direct_rule_message, direct_system_message


def _extract_json_payloads(raw_content: str) -> List[Any]:
    """All parseable ```json``` blocks, or the whole answer if it is JSON itself."""
    payloads: List[Any] = []
    for block in re.findall(r"```json\s*(.*?)\s*```", raw_content, re.DOTALL):
        try:
            payloads.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    if payloads:
        return payloads
    try:
        return [json.loads(raw_content)]
    except json.JSONDecodeError:
        return []


def _normalize_plan_payload(payloads: List[Any]) -> Dict[str, List[str]] | None:
    """Accept ``{"Left": [...], "Right": [...]}`` or a list of ``{"Left": .., "Right": ..}`` steps."""
    if not payloads:
        return None

    if len(payloads) == 1:
        item = payloads[0]
        if isinstance(item, dict):
            left, right = item.get("Left"), item.get("Right")
            if isinstance(left, list) and isinstance(right, list):
                return {"Left": [str(step) for step in left], "Right": [str(step) for step in right]}
            if "Left" in item or "Right" in item:
                steps = [item]
            else:
                return None
        elif isinstance(item, list):
            steps = item
        else:
            return None
    else:
        steps = payloads

    plan_data: Dict[str, List[str]] = {"Left": [], "Right": []}
    for step in steps:
        if not isinstance(step, dict) or ("Left" not in step and "Right" not in step):
            continue
        left = step.get("Left", "wait")
        right = step.get("Right", "wait")
        plan_data["Left"].append("wait" if left is None else str(left))
        plan_data["Right"].append("wait" if right is None else str(right))

    if not plan_data["Left"] and not plan_data["Right"]:
        return None
    return plan_data


def llm_direct(
    model_name: str, task_name: str, task_description: str, plan_path: str, temperature: float = 0.0
) -> Tuple[Dict[str, List[str]] | None, float]:
    """Ask the LLM for a direct plan and save it to ``plan_path``. Returns (plan or None, LLM time)."""
    messages = [
        {"role": "user", "content": direct_system_message},
        {"role": "user", "content": task_description + choose_skill_desc(task_name)},
        {"role": "user", "content": direct_rule_message},
    ]

    start_time = time.time()
    client = build_client(model_name)
    response_content = chat(client, model_name, messages, temperature=temperature)
    debug_print(response_content)
    time_cost = time.time() - start_time

    try:
        plan_data = _normalize_plan_payload(_extract_json_payloads(response_content))
        if plan_data is None:
            raise ValueError("No JSON plan payload detected.")
        Path(plan_path).write_text(json.dumps(plan_data, indent=4), encoding="utf-8")
    except Exception as exc:
        plan_data = None
        debug_print(f"No valid plan found! {exc}")
    return plan_data, time_cost
