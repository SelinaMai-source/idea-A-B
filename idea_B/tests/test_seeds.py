"""Seed determinism for the closed-loop runner."""

import numpy as np

from idea_B.benchmarks.Synthetic_Closed_Loop import make_environment
from idea_B.methods import run_closed_loop


def _run(seed: int) -> dict:
    env = make_environment("B2", population_per_round=16)
    return run_closed_loop(
        env,
        "ObservabilityController",
        seed=seed,
        budget=0.10,
        horizon=6,
    )


def test_same_seed_reproduces_trajectory():
    first = _run(0)
    second = _run(0)
    np.testing.assert_allclose(
        first["trajectory"]["true_performance"],
        second["trajectory"]["true_performance"],
    )
    np.testing.assert_allclose(first["W_T"], second["W_T"])
    np.testing.assert_allclose(first["theta_final"], second["theta_final"])


def test_different_seeds_change_trajectory():
    first = _run(0)
    other = _run(1)
    assert not np.allclose(first["theta_final"], other["theta_final"])
