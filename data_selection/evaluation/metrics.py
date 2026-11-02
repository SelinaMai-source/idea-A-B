"""Canonical metrics. Inapplicable cells stay N/A. Missing runs stay NOT_RUN."""

from __future__ import annotations

from core.status import NA, NOT_RUN

METRIC_FIELDS = [
    "final_success_rate",
    "success_auc",
    "selected_episodes",
    "selected_transitions",
    "retention_ratio",
    "environment_steps",
    "training_optimizer_steps",
    "selection_wallclock_seconds",
    "training_wallclock_seconds",
    "rollout_wallclock_seconds",
    "total_wallclock_seconds",
    "gpu_hours",
    "cpu_hours",
    "expert_query_count",
    "seed",
    "benchmark",
    "task",
    "policy_model",
    "method",
    "protocol",
    "adaptation_status",
    "forgetting",
    "backward_transfer",
    "average_success",
]


def final_success_rate(outcomes: list[bool]) -> float:
    if not outcomes:
        raise ValueError("success rate needs at least one evaluated episode")
    if any(not isinstance(item, bool) for item in outcomes):
        raise ValueError("outcomes must be booleans")
    return sum(1.0 for item in outcomes if item) / len(outcomes)


def success_auc(
    curve: list[tuple[int, float]],
    eval_steps: list[int],
    *,
    horizon: int | None = None,
) -> float:
    """Trapezoid over the predeclared environment-step grid only."""
    if list(eval_steps) != sorted(set(eval_steps)):
        raise ValueError("eval_steps must be strictly increasing")
    steps = [step for step, _rate in curve]
    rates = [rate for _step, rate in curve]
    if steps != list(eval_steps):
        raise ValueError("curve steps must equal the predeclared evaluation grid")
    if any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError("success rates must lie in [0, 1]")
    if len(steps) < 2:
        raise ValueError("AUC needs at least two evaluation points")
    area = 0.0
    for left, right, y0, y1 in zip(steps, steps[1:], rates, rates[1:]):
        area += (right - left) * (y0 + y1) / 2.0
    if horizon is None:
        return area
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return area / float(horizon)


def _square(matrix: list[list]) -> int:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("A[t, j] must be square")
    return len(matrix)


def _learned_tasks(matrix: list[list]) -> list[int]:
    size = _square(matrix)
    learned = []
    for task in range(size - 1):
        if matrix[task][task] is not None:
            learned.append(task)
    return learned


def average_success(matrix: list[list]) -> float:
    size = _square(matrix)
    finals = [matrix[size - 1][task] for task in range(size)]
    if any(value is None for value in finals):
        raise ValueError("final row is incomplete; refusing to treat missing tasks as zero")
    return sum(finals) / size


def forgetting(matrix: list[list]) -> float:
    """Mean of peak-after-learning minus final success, excluding the last task.

    Task j is learned only when A[j, j] is present. Tasks that were never learned
    are omitted. None cells are not zeros.
    """
    size = _square(matrix)
    learned = _learned_tasks(matrix)
    if not learned:
        raise ValueError("no previously learned task; forgetting is undefined")
    penalties = []
    for task in learned:
        observed = []
        for round_index in range(task, size - 1):
            value = matrix[round_index][task]
            if value is None:
                raise ValueError("missing intermediate success; refusing to impute zero")
            observed.append(value)
        final = matrix[size - 1][task]
        if final is None:
            raise ValueError("missing final success; refusing to impute zero")
        penalties.append(max(observed) - final)
    return sum(penalties) / len(penalties)


def backward_transfer(matrix: list[list]) -> float:
    """Mean of final success minus end-of-task success A[j, j] for earlier tasks."""
    size = _square(matrix)
    learned = _learned_tasks(matrix)
    if not learned:
        raise ValueError("no previously learned task; BWT is undefined")
    transfers = []
    for task in learned:
        end_of_task = matrix[task][task]
        final = matrix[size - 1][task]
        if end_of_task is None or final is None:
            raise ValueError("BWT requires end-of-task and final success")
        transfers.append(final - end_of_task)
    return sum(transfers) / len(transfers)


def blank_record(**known) -> dict:
    row = {key: NOT_RUN for key in METRIC_FIELDS}
    row["forgetting"] = NA
    row["backward_transfer"] = NA
    row["average_success"] = NA
    row.update(known)
    return row
