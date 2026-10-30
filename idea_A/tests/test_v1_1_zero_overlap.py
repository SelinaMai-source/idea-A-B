"""Zero-overlap natural channel is identification failure; known q still identifies."""

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.methods.v1_load import load_v1_1_module
from shared.metrics.estimators import observed_only_mean

_alg = load_v1_1_module("algorithm")
_ident = load_v1_1_module("propensity")


def test_zero_overlap_natural_channel_fails_audit_channel_identifies():
    world = simulate("zero_overlap", 0, n=400)
    assert world.identification_natural["status"] == "identification_failure"
    empty = world.identification_natural["zero_overlap_strata"]
    assert empty
    for h in empty:
        assert int(np.sum(world.s[world.stratum == h])) == 0
        y_nat = world.observe(np.zeros(world.n, dtype=int))
        mask = world.stratum == h
        assert np.all(np.isnan(y_nat[mask]))

    row = _alg.run_static_config("A-v1.1a", world, 0.20, np.random.default_rng(0))
    audit = _ident.audit_channel_status(row["q"])
    assert audit["status"] == "identified"
    assert np.isfinite(row["population_mean_reward"]["estimate"])
    y_nat = world.observe(np.zeros(world.n, dtype=int))
    natural_mean = observed_only_mean(y_nat, world.s)
    # Natural mean drops the zero-overlap strata; it is not the finite-pop mean.
    assert natural_mean != pytest.approx(world.truths["population_mean_reward"], abs=1e-6)
