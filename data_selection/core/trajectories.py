"""Trajectory schema checks. One trajectory is one episode."""

from __future__ import annotations

from core.types import PROVENANCE, Trajectory


class TrajectorySchemaError(ValueError):
    pass


def validate_trajectory(traj: Trajectory) -> None:
    if not traj.trajectory_id or not traj.episode_id or not traj.task_id:
        raise TrajectorySchemaError("trajectory_id, episode_id, and task_id are required")
    if traj.dataset_id is None or traj.dataset_id == "":
        raise TrajectorySchemaError("dataset_id is required")
    if traj.label_provenance not in PROVENANCE or traj.data_origin not in PROVENANCE:
        raise TrajectorySchemaError(f"provenance must be one of {PROVENANCE}")
    if traj.n_transitions != len(traj.transitions):
        raise TrajectorySchemaError("n_transitions does not match the transition list")
    if not traj.transitions:
        raise TrajectorySchemaError("episode is empty")
    widths = set()
    for index, step in enumerate(traj.transitions):
        if step.t != index:
            raise TrajectorySchemaError("observation timestamps must be contiguous from 0")
        if step.terminal and index != len(traj.transitions) - 1:
            raise TrajectorySchemaError("terminal flag is set before the episode boundary")
        if not isinstance(step.action, tuple) or len(step.action) == 0:
            raise TrajectorySchemaError("action must be a non-empty tuple")
        if any(not isinstance(value, (int, float)) for value in step.action):
            raise TrajectorySchemaError("action entries must be numeric")
        widths.add(len(step.action))
        if "proprio" not in step.observation:
            raise TrajectorySchemaError("observation is missing proprio")
    if len(widths) != 1:
        raise TrajectorySchemaError("action dimension changes inside the episode")
    if not traj.transitions[-1].terminal:
        raise TrajectorySchemaError("episode must end with a terminal flag")


def validate_pool(trajectories: list[Trajectory]) -> None:
    if not trajectories:
        raise TrajectorySchemaError("trajectory pool is empty")
    seen = set()
    widths = set()
    modes = set()
    for traj in trajectories:
        validate_trajectory(traj)
        if traj.trajectory_id in seen:
            raise TrajectorySchemaError(f"duplicate trajectory_id {traj.trajectory_id}")
        seen.add(traj.trajectory_id)
        widths.add(len(traj.transitions[0].action))
        modes.add(traj.observation_mode)
    if len(widths) != 1:
        raise TrajectorySchemaError("pool mixes action dimensions")
    if len(modes) != 1:
        raise TrajectorySchemaError("pool mixes observation modes")


def make_synthetic_trajectory(
    trajectory_id: str,
    n_transitions: int,
    *,
    task_id: str = "synthetic_task",
    success: bool = False,
    score: float | None = None,
    origin: str = "external_expert",
    collected_step: int = 0,
    action_dim: int = 2,
    dataset_id: str = "synthetic",
) -> Trajectory:
    from core.types import Transition

    if n_transitions < 1:
        raise TrajectorySchemaError("n_transitions must be positive")
    steps = []
    for index in range(n_transitions):
        steps.append(
            Transition(
                t=index,
                observation={"proprio": (float(index),), "image": None},
                action=tuple(0.0 for _ in range(action_dim)),
                reward=1.0 if success and index == n_transitions - 1 else 0.0,
                terminal=index == n_transitions - 1,
                success=bool(success) if index == n_transitions - 1 else False,
            )
        )
    traj = Trajectory(
        trajectory_id=trajectory_id,
        task_id=task_id,
        episode_id=trajectory_id,
        transitions=steps,
        task_success=success,
        data_origin=origin,
        label_provenance=origin,
        policy_checkpoint_source=None,
        dataset_id=dataset_id,
        n_transitions=n_transitions,
        score=score,
        collected_step=collected_step,
        observation_mode="synthetic_vector",
    )
    validate_trajectory(traj)
    return traj
