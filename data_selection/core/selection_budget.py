"""Apply an episode cap and a transition cap to an already ordered pool."""

from __future__ import annotations

from core.status import OK
from core.trajectories import validate_pool
from core.types import SelectionBudget, SelectionResult, Trajectory


def take_prefix(ordered: list[Trajectory], budget: SelectionBudget) -> tuple[list[Trajectory], bool]:
    episode_cap = budget.episode_cap(len(ordered))
    transition_cap = budget.max_transitions
    if transition_cap is not None and transition_cap < 0:
        raise ValueError("max_transitions must be >= 0")
    kept: list[Trajectory] = []
    used = 0
    truncated = False
    for traj in ordered:
        if episode_cap is not None and len(kept) >= episode_cap:
            break
        width = traj.n_transitions
        if transition_cap is not None and used + width > transition_cap:
            room = transition_cap - used
            if room <= 0:
                break
            kept.append(traj.truncated_copy(room))
            truncated = True
            break
        kept.append(traj)
        used += width
    return kept, truncated


def result_from_ordered(
    method: str,
    ordered: list[Trajectory],
    budget: SelectionBudget,
    information_used: list[str],
    *,
    reason: str,
    extra_cost: dict | None = None,
) -> SelectionResult:
    kept, truncated = take_prefix(ordered, budget)
    transitions = sum(item.n_transitions for item in kept)
    episodes = len(kept)
    return SelectionResult(
        method=method,
        status=OK,
        reason=reason,
        selected_ids=[item.trajectory_id for item in kept],
        selected_episodes=episodes,
        selected_transitions=transitions,
        truncated=truncated,
        information_used=list(information_used),
        extra_cost=extra_cost or {"heuristic_fallback": False},
    )


def guard_pool(trajectories: list[Trajectory]) -> None:
    validate_pool(trajectories)
