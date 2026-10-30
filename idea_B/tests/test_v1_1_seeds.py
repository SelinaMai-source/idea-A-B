"""Seed determinism for the v1.1 closed-loop runner and Env3 change point."""

import numpy as np

from idea_B.benchmarks.B_Env1_Endogenous_Selection import make_environment as make_env1
from idea_B.benchmarks.B_Env3_Recovery_After_Latent_Change import (
    make_environment as make_env3,
)
from idea_B.benchmarks.B_Env3_Recovery_After_Latent_Change import predraw_theta_after
from idea_B.methods.v1_1 import run_closed_loop


def _run(seed: int, method: str = "B-v1.1c") -> dict:
    env = make_env1(population_per_round=16)
    return run_closed_loop(env, method, seed=seed, budget=0.10, horizon=6)


def test_same_seed_reproduces_trajectory():
    first = _run(0)
    second = _run(0)
    np.testing.assert_allclose(
        first["trajectory"]["true_mse"],
        second["trajectory"]["true_mse"],
    )
    np.testing.assert_allclose(first["W_T"], second["W_T"])
    np.testing.assert_allclose(first["theta_final"], second["theta_final"])
    np.testing.assert_allclose(first["W_T_candidates"], second["W_T_candidates"])


def test_different_seeds_change_trajectory():
    first = _run(0)
    other = _run(1)
    assert not np.allclose(first["theta_final"], other["theta_final"])


def test_env3_change_is_a_function_of_seed_not_of_the_method():
    after_a = predraw_theta_after(4, (1.2, 1.05, 1.0), 2)
    after_b = predraw_theta_after(4, (1.2, 1.05, 1.0), 2)
    np.testing.assert_allclose(after_a, after_b)
    env_u = make_env3(unrecoverable=False, seed=4, horizon=32, population_per_round=20)
    env_e = make_env3(unrecoverable=False, seed=4, horizon=32, population_per_round=20)
    np.testing.assert_allclose(env_u.theta_star_after, env_e.theta_star_after)
    assert env_u.change_round == 16
    uniform = run_closed_loop(env_u, "UniformExploration", seed=4, budget=0.05, horizon=10)
    eopt = run_closed_loop(env_e, "B-v1.1b", seed=4, budget=0.05, horizon=10)
    assert uniform["recovery"]["recovery_time"] is not None
    assert eopt["config"]["seed"] == 4
    assert env_u.change_round == env_e.change_round
