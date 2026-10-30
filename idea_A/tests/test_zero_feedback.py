"""Zero feedback does not identify the finite-population mean."""

import math

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.methods.v1_load import load_v1_module
from shared.metrics.estimators import dr_mean, ipw_mean, observed_only_mean, snips_mean

_intervention = load_v1_module("intervention")


def test_zero_audit_and_zero_natural_do_not_identify():
    world = simulate("mcar", 0, n=300)
    q, a = _intervention.no_audit(world.n)
    # Force the natural channel off as well.
    s = np.zeros(world.n, dtype=int)
    y_left = _intervention.reveal_labels(world.y, s, a)
    y_right = _intervention.reveal_labels(world.y + 10.0, s, a)
    assert np.all(np.isnan(y_left))
    assert ipw_mean(y_left, a, np.full(world.n, 0.2)) == 0.0
    assert ipw_mean(y_right, a, np.full(world.n, 0.2)) == 0.0
    assert ipw_mean(y_left, a, np.full(world.n, 0.2)) != pytest.approx(float(np.mean(world.y)))
    assert math.isnan(observed_only_mean(y_left, s))
    assert math.isnan(snips_mean(y_left, a, q))
    mu = np.linspace(-1.0, 1.0, world.n)
    assert dr_mean(y_left, a, np.full(world.n, 0.2), mu) == pytest.approx(float(np.mean(mu)))
    assert dr_mean(y_right, a, np.full(world.n, 0.2), mu) == pytest.approx(float(np.mean(mu)))


def test_zero_overlap_natural_channel_is_identification_failure():
    world = simulate("zero_overlap", 0, n=2000)
    assert world.identification_natural["status"] == "identification_failure"
    empty = world.identification_natural["zero_overlap_strata"]
    assert empty
    for h in empty:
        assert int(np.sum(world.s[world.stratum == h])) == 0
