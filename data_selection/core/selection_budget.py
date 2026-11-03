"""Episode-complete budgets.

All Data is the whole eligible pool. Whole-demonstration methods do not cut an
episode. Sub-trajectory cuts are allowed only when the caller sets
``allow_partial`` (STRAP).
"""

from __future__ import annotations

from core.status import BLOCKED, OK
from core.trajectories import validate_pool
from core.types import SelectionBudget, SelectionResult, Trajectory


def guard_pool(trajectories: list[Trajectory]) -> None:
    validate_pool(trajectories)


def blocked_result(method: str, reason: str, extra: dict | None = None) -> SelectionResult:
    cost = {"heuristic_fallback": False}
    if extra:
        cost.update(extra)
    return SelectionResult(
        method=method,
        status=BLOCKED,
        reason=reason,
        selected_ids=[],
        selected_episodes=0,
        selected_transitions=0,
        truncated=False,
        information_used=[],
        extra_cost=cost,
    )


def take_prefix(
    ordered: list[Trajectory],
    budget: SelectionBudget,
    *,
    allow_partial: bool = False,
) -> tuple[list[Trajectory], bool, str]:
    """Keep a prefix of complete episodes.

    ``allow_partial`` may cut the last episode. Whole-demonstration methods
    leave that episode out instead of slicing it.
    """
    episode_cap = budget.episode_cap(len(ordered))
    transition_cap = budget.max_transitions
    if transition_cap is not None and transition_cap < 0:
        raise ValueError("max_transitions must be >= 0")
    kept: list[Trajectory] = []
    used = 0
    truncated = False
    stopped = ""
    for traj in ordered:
        if episode_cap is not None and len(kept) >= episode_cap:
            break
        width = traj.n_transitions
        if transition_cap is not None and used + width > transition_cap:
            room = transition_cap - used
            if allow_partial and 0 < room < width:
                kept.append(traj.truncated_copy(room))
                truncated = True
                stopped = "subtrajectory prefix cut to the transition budget"
                break
            stopped = (
                "next episode does not fit in the remaining transition budget; "
                "refusing to cut a whole demonstration"
            )
            break
        kept.append(traj)
        used += width
    return kept, truncated, stopped


def accounting(
    method: str,
    pool: list[Trajectory],
    kept: list[Trajectory],
    *,
    granularity: str,
    information_used: list[str],
    reason: str,
    status: str = OK,
    truncated: bool = False,
    extra: dict | None = None,
) -> SelectionResult:
    extra_cost = {
        "heuristic_fallback": False,
        "eligible_pool_episodes": len(pool),
        "eligible_pool_transitions": sum(item.n_transitions for item in pool),
        "granularity": granularity,
        "original_boundaries_preserved": not truncated,
    }
    if extra:
        extra_cost.update(extra)
    return SelectionResult(
        method=method,
        status=status,
        reason=reason,
        selected_ids=[item.trajectory_id for item in kept],
        selected_episodes=len(kept),
        selected_transitions=sum(item.n_transitions for item in kept),
        truncated=truncated,
        information_used=list(information_used),
        extra_cost=extra_cost,
    )


def result_from_ordered(
    method: str,
    ordered: list[Trajectory],
    budget: SelectionBudget,
    information_used: list[str],
    *,
    reason: str,
    extra_cost: dict | None = None,
    allow_partial: bool = False,
    pool: list[Trajectory] | None = None,
) -> SelectionResult:
    source = pool if pool is not None else ordered
    kept, truncated, stopped = take_prefix(ordered, budget, allow_partial=allow_partial)
    if stopped and not allow_partial and not kept:
        return blocked_result(method, stopped, extra_cost)
    detail = reason if not stopped else f"{reason} {stopped}"
    return accounting(
        method,
        source,
        kept,
        granularity="subtrajectory" if allow_partial else "episode",
        information_used=information_used,
        reason=detail,
        truncated=truncated,
        extra=extra_cost,
    )


def budget_is_explicit(budget: SelectionBudget) -> bool:
    return (
        budget.max_episodes is not None
        or budget.max_transitions is not None
        or budget.retention_ratio is not None
    )
