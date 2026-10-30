"""AIPW scores match the shared dr_mean and do not read A=0 outcomes."""

import numpy as np
import pytest

from idea_A.methods.v1_load import load_v1_1_module
from shared.metrics.estimators import dr_mean

_seq = load_v1_1_module("sequential_estimator")
_outcome = load_v1_1_module("outcome_model")


Y = np.array([1.0, 2.0, 3.0, 4.0])
A = np.array([1, 0, 1, 0])
Q = np.array([0.5, 0.5, 0.25, 0.25])
MU = np.array([0.0, 10.0, 0.0, 10.0])


def test_aipw_influence_matches_hand_computation_and_dr_mean():
    psi = _seq.aipw_influence(Y, A, Q, MU)
    np.testing.assert_allclose(psi, np.array([2.0, 10.0, 12.0, 10.0]))
    assert float(np.mean(psi)) == pytest.approx(dr_mean(Y, A, Q, MU))
    assert _outcome.aipw_population_mean(Y, A, Q, MU) == pytest.approx(8.5)


def test_aipw_does_not_read_unaudited_y():
    y_nan = np.array([1.0, np.nan, 3.0, np.nan])
    y_poison = np.array([1.0, 1.0e9, 3.0, 1.0e9])
    assert dr_mean(y_nan, A, Q, MU) == pytest.approx(dr_mean(y_poison, A, Q, MU))
    np.testing.assert_allclose(
        _seq.aipw_influence(y_nan, A, Q, MU),
        _seq.aipw_influence(y_poison, A, Q, MU),
    )


def test_preference_aipw_uses_mu_p_not_mu_y():
    pref = np.array([0.0, 1.0, 1.0, 0.0])
    mu_p = np.array([0.2, 0.8, 0.7, 0.1])
    mu_y = np.array([5.0, -3.0, 8.0, 0.0])
    with_p = _outcome.aipw_preference_rate(pref, A, Q, mu_p)
    with_y = dr_mean(pref, A, Q, mu_y)
    assert with_p == pytest.approx(dr_mean(pref, A, Q, mu_p))
    assert with_p != pytest.approx(with_y)


def test_wald_interval_formula_is_frozen():
    psi = np.array([1.0, 2.0, 3.0, 4.0])
    ci = _seq.wald_interval(psi)
    se = float(np.std(psi, ddof=1) / np.sqrt(4))
    z = 1.959963984540054
    assert ci["formula"] == "influence_wald_95"
    assert ci["estimate"] == pytest.approx(2.5)
    assert ci["se"] == pytest.approx(se)
    assert ci["ci_low"] == pytest.approx(2.5 - z * se)
    assert ci["ci_high"] == pytest.approx(2.5 + z * se)
    assert _seq.interval_covers(ci, 2.5)
    assert not _seq.interval_covers(ci, 10.0)
