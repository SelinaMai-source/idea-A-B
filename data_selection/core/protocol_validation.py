"""Refuse mixed observation spaces and inapplicable forgetting claims."""

from __future__ import annotations


class ProtocolError(ValueError):
    pass


def validate_comparison_rows(rows: list[dict]) -> None:
    if not rows:
        raise ProtocolError("comparison is empty")
    spaces = {row.get("observation_mode") for row in rows}
    controls = {row.get("action_convention") for row in rows}
    models = {row.get("policy_model") for row in rows}
    if len(spaces) != 1 or len(controls) != 1 or len(models) != 1:
        raise ProtocolError(
            "a comparison table mixes observation modes, action conventions, or policy models"
        )


def forgetting_cell(benchmark_name: str, value):
    from core.registry import discover_benchmarks

    matches = [item for item in discover_benchmarks() if item.name == benchmark_name]
    if not matches:
        raise ProtocolError(f"unknown benchmark {benchmark_name}")
    if not matches[0].forgetting_applicable:
        if value not in (None, "N/A"):
            raise ProtocolError("forgetting is N/A outside the task-incremental LIBERO protocol")
        return "N/A"
    return "NOT_RUN" if value is None else value
