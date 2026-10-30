"""Audit inclusion probabilities and E[q]=b for v1.1 configs."""

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.methods.v1_load import load_v1_module, load_v1_1_module
from shared.evaluation.budget import assert_budget, expected_audit_fraction

_alg = load_v1_1_module("algorithm")
_alloc = load_v1_1_module("adaptive_allocation")
_v1_alg = load_v1_module("algorithm")
_v1_est = load_v1_module("estimators")
_v1_int = load_v1_module("intervention")


@pytest.mark.parametrize("budget", [0.05, 0.10, 0.20])
def test_uniform_and_aipw_variance_match_budget(budget):
    n = 400
    q_u = _alloc.allocate_uniform(n, budget)
    assert expected_audit_fraction(q_u) == pytest.approx(budget)
    assert_budget(expected_audit_fraction(q_u), budget, 1e-12)
    stratum = np.repeat(np.arange(8), n // 8)
    sigma = np.linspace(0.2, 2.0, 8)
    q_b = _alloc.allocate_aipw_variance(stratum, 8, budget, sigma)
    assert_budget(expected_audit_fraction(q_b), budget, 1e-9)
    q_c = _alloc.allocate_aipw_variance_remaining_drift(
        stratum, 8, budget * n, n, sigma, None
    )
    assert_budget(expected_audit_fraction(q_c), budget, 1e-9)


def test_aipw_variance_puts_more_q_on_high_residual_sigma():
    stratum = np.array([0] * 80 + [1] * 80)
    sigma = np.array([0.1, 2.0])
    q = _alloc.allocate_aipw_variance(stratum, 2, 0.20, sigma)
    assert float(np.mean(q[stratum == 1])) > float(np.mean(q[stratum == 0]))
    assert_budget(expected_audit_fraction(q), 0.20, 1e-9)


def test_residual_sigma_ignores_non_audit_entries():
    stratum = np.array([0, 0, 0, 1, 1, 1])
    resid = np.array([1.0, 999.0, 1.0, 4.0, 888.0, 4.0])
    audit = np.array([1, 0, 1, 1, 0, 1])
    sigma = _alloc.residual_sigma_by_stratum(stratum, resid, audit, 2)
    assert sigma[0] == pytest.approx(1.0)
    assert sigma[1] == pytest.approx(4.0)


def test_a_v10_matches_v1_neyman_ipw():
    world = simulate("mnar", 0, n=400)
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    row = _alg.run_static_config("A-v1.0", world, 0.10, rng_a)
    y_nat = world.observe(np.zeros(world.n, dtype=int))
    q = _v1_alg.allocate_neyman(world.stratum, y_nat, world.s, 0.10, world.n_strata, exploration_mix=0.15)
    a = _v1_int.draw_audit(q, rng_b)
    est = _v1_est.estimate_population_mean(world.observe(a), a, q, "ipw")
    np.testing.assert_allclose(row["q"], q)
    np.testing.assert_array_equal(row["a"], a)
    assert row["population_mean_reward"]["estimate"] == pytest.approx(est)
    assert row["estimator"] == "ipw"


@pytest.mark.parametrize("config_id", ["A-v1.1a", "A-v1.1b", "A-v1.1c"])
def test_static_configs_respect_budget(config_id):
    world = simulate("mnar", 0, n=400)
    row = _alg.run_static_config(config_id, world, 0.10, np.random.default_rng(0))
    assert_budget(row["expected_audit_fraction"], 0.10, 1e-6)


def test_audit_q_is_inclusion_probability_and_draw_does_not_read_y():
    world = simulate("mnar", 0, n=400)
    q = np.full(world.n, 0.2)
    hidden = world.s == 0
    y_orig = world.y.copy()
    rng = np.random.default_rng(3)
    a1 = _v1_int.draw_audit(q, rng)
    world.y[hidden] = 1.0e9
    a2 = _v1_int.draw_audit(q, np.random.default_rng(3))
    np.testing.assert_array_equal(a1, a2)
    world.y[:] = y_orig
    assert 0.12 < float(np.mean(a1)) < 0.28
