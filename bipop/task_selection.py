"""Selection of task indices from command-line arguments."""

from __future__ import annotations

from typing import List


def build_task_indices(start_index: int, task_nums: int) -> List[int]:
    """Return ``task_nums`` consecutive indices starting at ``start_index``."""
    if start_index < 0:
        raise ValueError("--start_index must be greater than or equal to 0.")
    if task_nums < 1:
        raise ValueError("--task_nums must be greater than or equal to 1.")
    return list(range(start_index, start_index + task_nums))


def parse_task_indices(raw_indices: str) -> List[int]:
    """Parse explicit comma-separated indices, dropping duplicates but keeping the order."""
    if not raw_indices.strip():
        raise ValueError("--task_indices must not be empty when provided.")

    indices: List[int] = []
    seen: set[int] = set()
    for raw_part in raw_indices.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("--task_indices contains an empty index.")
        try:
            index = int(part)
        except ValueError as exc:
            raise ValueError(f"--task_indices contains a non-integer index: {part!r}.") from exc
        if index < 0:
            raise ValueError("--task_indices must contain only non-negative indices.")
        if index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return indices


def select_task_indices(start_index: int, task_nums: int, task_indices: str = "") -> List[int]:
    """Use the explicit indices when given, otherwise the ``start_index``/``task_nums`` range."""
    if task_indices.strip():
        return parse_task_indices(task_indices)
    return build_task_indices(start_index, task_nums)


def describe_task_selection(indices: List[int], explicit: bool) -> str:
    """One-line description of the selected indices for console output."""
    if explicit:
        return f"Task Indices: {', '.join(str(index) for index in indices)} (count={len(indices)})"
    return f"Task Range: {indices[0]} to {indices[-1]} (count={len(indices)})"
