"""Audit-budget accounting for v1.1 controllers and generic controls."""

import numpy as np

from idea_B.benchmarks.B_Env1_Endogenous_Selection import make_environment as make_env1
from idea_B.methods.v1_1 import run_closed_loop
from shared.evaluation.budget import assert_budget


def _run(method: str, budget: float, seed: int = 0, horizon: int = 8) -> dict:
    env = make_env1(population_per_round=40)
    return run_closed_loop(env, method, seed=seed, budget=budget, horizon=horizon)


def test_uniform_expected_and_realized_match_budget():
    result = _run("UniformExploration", 0.20)
    assert_budget(float(result["summary"]["expected_audit_fraction"]), 0.20, 1e-12)
    assert_budget(float(result["summary"]["realized_audit_fraction"]), 0.20, 0.08)


def test_equal_skill_expected_matches_budget():
    result = _run("B-v1.1a", 0.10)
    assert_budget(float(result["summary"]["expected_audit_fraction"]), 0.10, 1e-8)
    assert_budget(float(result["summary"]["realized_audit_fraction"]), 0.10, 0.08)


def test_e_optimal_and_d_optimal_and_v11c_match_budget():
    for method in ("B-v1.1b", "D-optimal", "B-v1.1c"):
        result = _run(method, 0.10, seed=1, horizon=6)
        assert_budget(float(result["summary"]["expected_audit_fraction"]), 0.10, 1e-6)
        assert_budget(float(result["summary"]["realized_audit_fraction"]), 0.10, 0.10)


def test_no_controller_spends_no_audit_budget():
    result = _run("NoFeedbackController", 0.20, horizon=5)
    assert_budget(float(result["summary"]["realized_audit_fraction"]), 0.0, 1e-12)


def test_oracle_observes_everything():
    env = make_env1(population_per_round=16)
    result = run_closed_loop(env, "FullFeedbackOracle", seed=0, budget=0.05, horizon=4)
    assert_budget(float(result["summary"]["realized_audit_fraction"]), 1.0, 1e-12)
    n_obs = np.asarray(result["trajectory"]["n_observed"], dtype=float)
    assert np.all(n_obs == 16.0)


def test_v10_wrapper_matches_budget():
    result = _run("B-v1.0", 0.10, seed=2, horizon=6)
    assert_budget(float(result["summary"]["expected_audit_fraction"]), 0.10, 0.02)
    assert_budget(float(result["summary"]["realized_audit_fraction"]), 0.10, 0.10)
    assert result["method_meta"]["v1_0_controller_wrapped"] is True
