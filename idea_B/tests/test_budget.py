"""Audit-budget accounting for generic controls and the controller."""

import numpy as np

from idea_B.benchmarks.Synthetic_Closed_Loop import make_environment
from idea_B.methods import run_closed_loop
from shared.evaluation.budget import assert_budget


def _audit_series(result: dict) -> np.ndarray:
    return np.asarray(result["trajectory"]["realized_audit_fraction"], dtype=float)


def test_uniform_realized_audit_matches_budget():
    env = make_environment("B1", population_per_round=40)
    result = run_closed_loop(
        env,
        "UniformExploration",
        seed=0,
        budget=0.20,
        horizon=12,
    )
    realized = float(result["summary"]["realized_audit_fraction"])
    assert_budget(realized, 0.20, 0.08)
    per_round = _audit_series(result)
    np.testing.assert_allclose(np.mean(per_round), realized)


def test_controller_realized_audit_matches_budget():
    env = make_environment("B2", population_per_round=40)
    result = run_closed_loop(
        env,
        "ObservabilityController",
        seed=1,
        budget=0.10,
        horizon=12,
    )
    assert_budget(float(result["summary"]["realized_audit_fraction"]), 0.10, 0.08)


def test_no_controller_spends_no_audit_budget():
    env = make_environment("B2", population_per_round=24)
    result = run_closed_loop(
        env,
        "NoFeedbackController",
        seed=0,
        budget=0.20,
        horizon=6,
    )
    assert_budget(float(result["summary"]["realized_audit_fraction"]), 0.0, 1e-12)


def test_oracle_observes_everything():
    env = make_environment("B1", population_per_round=16)
    result = run_closed_loop(
        env,
        "FullFeedbackOracle",
        seed=0,
        budget=0.05,
        horizon=5,
    )
    assert_budget(float(result["summary"]["realized_audit_fraction"]), 1.0, 1e-12)
    n_obs = np.asarray(result["trajectory"]["n_observed"], dtype=float)
    assert np.all(n_obs == 16.0)
