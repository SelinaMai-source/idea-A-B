"""Algebraic checks for the shared mean estimators."""

import math

import numpy as np
import pytest

from shared.metrics.estimators import (
    dr_mean,
    ipw_mean,
    naive_ips_mean,
    observed_only_mean,
    snips_mean,
)


Y = np.array([1.0, 2.0, 3.0, 4.0])
A = np.array([1, 0, 1, 0])
Q = np.array([0.5, 0.5, 0.25, 0.25])
MU = np.array([0.0, 10.0, 0.0, 10.0])


def test_hand_computed_ipw_snips_and_naive_differ():
    # IPW terms on audited units: 1/0.5 and 3/0.25. Divisor is n=4.
    assert ipw_mean(Y, A, Q) == pytest.approx(3.5)
    # Weights 1/q are 2 and 4. Weighted outcomes are 2 and 12. SNIPS = 14/6.
    assert snips_mean(Y, A, Q) == pytest.approx(14.0 / 6.0)
    assert observed_only_mean(Y, A) == pytest.approx(2.0)
    # 0 + (1-0)/0.5, mu=10, 0+(3-0)/0.25, mu=10 -> mean 8.5
    assert dr_mean(Y, A, Q, MU) == pytest.approx(8.5)

    p_hat = np.full(4, 0.9)
    naive = naive_ips_mean(Y, A, p_hat)
    assert naive == pytest.approx((1.0 / 0.9 + 3.0 / 0.9) / 4.0)
    assert naive != pytest.approx(ipw_mean(Y, A, Q))
    assert naive_ips_mean is not ipw_mean


def test_no_oracle_leakage_when_unaudited_outcomes_are_nan():
    y = np.array([1.0, np.nan, 3.0, np.nan])
    assert ipw_mean(y, A, Q) == pytest.approx(3.5)
    assert snips_mean(y, A, Q) == pytest.approx(14.0 / 6.0)
    assert dr_mean(y, A, Q, MU) == pytest.approx(8.5)
    assert observed_only_mean(y, A) == pytest.approx(2.0)
    assert math.isnan(observed_only_mean(y, np.zeros(4, dtype=int)))


def test_estimators_do_not_mutate_inputs_or_clip_silently():
    y = np.array([1.0, np.nan, 3.0])
    a = np.array([1, 0, 1])
    q = np.array([1e-6, 0.2, 1e-6])
    mu = np.array([0.2, 9.0, 0.4])
    y0, a0, q0, mu0 = y.copy(), a.copy(), q.copy(), mu.copy()
    raw = ipw_mean(y, a, q)
    ipw_mean(y, a, q, clip=1e-3)
    snips_mean(y, a, q)
    dr_mean(y, a, q, mu)
    assert np.array_equal(y, y0, equal_nan=True)
    np.testing.assert_array_equal(a, a0)
    np.testing.assert_array_equal(q, q0)
    np.testing.assert_array_equal(mu, mu0)
    assert raw == pytest.approx(np.mean([1.0 / 1e-6, 0.0, 3.0 / 1e-6]))
    clipped = ipw_mean(y, a, q, clip=1e-3)
    assert clipped == pytest.approx(np.mean([1.0 / 1e-3, 0.0, 3.0 / 1e-3]))
    assert clipped != pytest.approx(raw)


def test_zero_propensity_on_an_audited_unit_raises_without_clip():
    y = np.array([1.0, 2.0])
    a = np.array([1, 0])
    q = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        ipw_mean(y, a, q)
    assert ipw_mean(y, a, q, clip=0.5) == pytest.approx(np.mean([1.0 / 0.5, 0.0]))
    with pytest.raises(TypeError):
        ipw_mean(y, a, np.array([0.5, 0.5]), clip=True)


def test_scalar_propensity_broadcast_and_shape_checks():
    y = np.array([2.0, 4.0])
    assert ipw_mean(y, np.array([1, 1]), 0.5) == pytest.approx(np.mean([2.0 / 0.5, 4.0 / 0.5]))
    with pytest.raises(ValueError):
        ipw_mean(y, np.array([1, 0, 1]), np.array([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError):
        ipw_mean(y, np.array([1, 2]), np.array([0.5, 0.5]))


def test_snips_is_nan_when_nothing_is_audited():
    y = np.array([1.0, 2.0, 3.0])
    assert math.isnan(snips_mean(y, np.zeros(3), np.full(3, 0.2)))
