import pytest

from core.trajectories import TrajectorySchemaError, make_synthetic_trajectory, validate_pool, validate_trajectory
from core.types import Transition


def test_episode_must_end_on_terminal_and_keep_action_shape():
    traj = make_synthetic_trajectory("ep0", 3, action_dim=4)
    validate_trajectory(traj)
    assert traj.transitions[-1].terminal
    assert all(len(step.action) == 4 for step in traj.transitions)


def test_terminal_before_boundary_is_rejected():
    traj = make_synthetic_trajectory("ep0", 3)
    traj.transitions[0] = Transition(
        t=0,
        observation=traj.transitions[0].observation,
        action=traj.transitions[0].action,
        reward=0.0,
        terminal=True,
        success=False,
    )
    with pytest.raises(TrajectorySchemaError, match="episode boundary"):
        validate_trajectory(traj)


def test_pool_rejects_mixed_action_dimensions():
    left = make_synthetic_trajectory("a", 2, action_dim=2)
    right = make_synthetic_trajectory("b", 2, action_dim=7)
    with pytest.raises(TrajectorySchemaError, match="action dimensions"):
        validate_pool([left, right])
