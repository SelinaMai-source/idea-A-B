"""v1.1 estimand-specific nuisances: mu_Y, mu_P, mu_V = w mu_Y."""

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.methods.v1_load import load_v1_1_module
from shared.metrics.estimators import dr_mean

_outcome = load_v1_1_module("outcome_model")
_ident = load_v1_1_module("propensity")


def test_estimand_definitions_match_v1_targets():
    world = simulate("mcar", 0, n=200)
    assert world.truths["population_mean_reward"] == pytest.approx(float(np.mean(world.y)))
    assert world.truths["preference_rate"] == pytest.approx(float(np.mean(world.y > 0.0)))
    assert world.truths["policy_value_target_distribution"] == pytest.approx(
        float(np.mean(world.target_weights * world.y))
    )


def test_mu_p_is_a_probability_and_is_not_mu_y():
    world = simulate("mcar", 0, n=400)
    rng = np.random.default_rng(0)
    q = np.full(world.n, 0.3)
    a = (rng.random(world.n) < q).astype(int)
    y_vis = world.observe(a)
    mu_y = _outcome.fit_mu_y(y_vis, a, world.x, world.pi, world.stratum, world.n_strata).predict(
        world.x, world.pi, world.stratum
    )
    mu_p = _outcome.fit_mu_p(y_vis, a, world.x, world.pi, world.stratum, world.n_strata).predict(
        world.x, world.pi, world.stratum
    )
    assert np.all((mu_p > 0.0) & (mu_p < 1.0))
    assert np.any(mu_y < 0.0) or np.any(mu_y > 1.0)
    assert not np.allclose(mu_p, mu_y)
    assert not np.allclose(mu_p, 1.0 / (1.0 + np.exp(-mu_y)))


def test_policy_value_uses_weighted_mu_y_not_mu_y_in_weighted_residual():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    a = np.array([1, 0, 1, 0])
    q = np.array([0.5, 0.5, 0.25, 0.25])
    mu_y = np.array([0.0, 10.0, 0.0, 10.0])
    w = np.array([2.0, 2.0, 0.5, 0.5])
    right = _outcome.aipw_policy_value(y, a, q, mu_y, w)
    wrong = dr_mean(w * y, a, q, mu_y)
    assert right == pytest.approx(dr_mean(w * y, a, q, w * mu_y))
    assert right == pytest.approx(8.75)
    assert wrong == pytest.approx(7.5)
    assert right != pytest.approx(wrong)


def test_mu_v_is_w_times_mu_y():
    mu_y = np.array([1.5, -0.5, 2.0])
    w = np.array([0.5, 2.0, 1.0])
    np.testing.assert_allclose(_outcome.mu_v_from_mu_y(mu_y, w), w * mu_y)


def test_combined_o_formula_rejected_when_p_s_depends_on_y():
    p_s = np.full(4, 0.3)
    q = np.full(4, 0.2)
    with pytest.raises(ValueError, match="does not convert MNAR"):
        _ident.combined_observation_probability(p_s, q, p_s_depends_on_unobserved_y=True)
    combined = _ident.combined_observation_probability(p_s, q, p_s_depends_on_unobserved_y=False)
    np.testing.assert_allclose(combined, 1.0 - (1.0 - p_s) * (1.0 - q))


def test_ordinary_dr_scope_is_audit_missingness_only():
    assert _ident.dr_double_robustness_applies(
        missingness="audit", q_known=True, audit_independent_of_y_given_history=True
    )
    assert not _ident.dr_double_robustness_applies(
        missingness="natural_mnar", q_known=True, audit_independent_of_y_given_history=True
    )
    assert not _ident.dr_double_robustness_applies(
        missingness="audit", q_known=False, audit_independent_of_y_given_history=True
    )
    notes = _ident.identification_notes()
    assert "does not make the natural channel MAR" in notes["mnar_not_mar"]
