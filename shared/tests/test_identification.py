"""Identification checks that use only the shared estimators.

Full feedback (every unit audited at q=1) recovers the population mean.
Zero feedback does not: IPW and DR are invariant to the unseen outcomes.
A separate Monte Carlo checks that independent audits with a known q are
approximately unbiased for a fixed finite population. That Monte Carlo is an
estimator sanity check, not an Idea A simulator.
"""

import math

import numpy as np
import pytest

from shared.metrics.errors import bias
from shared.metrics.estimators import dr_mean, ipw_mean, observed_only_mean, snips_mean


def test_full_feedback_recovers_population_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    truth = float(np.mean(y))
    audited = np.ones(4)
    propensity = np.ones(4)
    wrong_prediction = np.array([9.0, -3.0, 0.0, 4.0])
    assert observed_only_mean(y, audited) == pytest.approx(truth)
    assert ipw_mean(y, audited, propensity) == pytest.approx(truth)
    assert snips_mean(y, audited, propensity) == pytest.approx(truth)
    assert dr_mean(y, audited, propensity, wrong_prediction) == pytest.approx(truth)
    assert dr_mean(y, audited, 1.0, 99.0) == pytest.approx(truth)


def test_zero_feedback_does_not_use_the_outcome():
    y_left = np.array([1.0, 2.0, 3.0])
    y_right = np.array([8.0, -1.0, 40.0])
    unaudited = np.zeros(3, dtype=int)
    known_q = np.full(3, 0.4)
    no_overlap = np.zeros(3)
    mu = np.array([5.0, 1.0, 0.0])
    mu_mean = float(np.mean(mu))

    assert ipw_mean(y_left, unaudited, known_q) == 0.0
    assert ipw_mean(y_right, unaudited, known_q) == 0.0
    assert ipw_mean(y_left, unaudited, known_q) != pytest.approx(float(np.mean(y_left)))
    assert dr_mean(y_left, unaudited, known_q, mu) == pytest.approx(mu_mean)
    assert dr_mean(y_right, unaudited, known_q, mu) == pytest.approx(mu_mean)

    assert ipw_mean(y_left, unaudited, no_overlap) == ipw_mean(y_right, unaudited, no_overlap)
    assert dr_mean(np.array([np.nan, np.nan, np.nan]), unaudited, no_overlap, mu) == pytest.approx(
        mu_mean
    )
    assert math.isnan(snips_mean(y_left, unaudited, no_overlap))


def test_independent_audit_ipw_is_approximately_unbiased():
    y = np.random.default_rng(0).normal(loc=1.7, size=2000)
    truth = float(np.mean(y))
    q = np.full(y.shape, 0.2)
    estimates = []
    for replicate in range(400):
        audited = np.random.default_rng(10_000 + replicate).random(y.shape[0]) < q
        estimates.append(ipw_mean(y, audited, q))
    assert bias(estimates, truth) == pytest.approx(0.0, abs=0.03)
