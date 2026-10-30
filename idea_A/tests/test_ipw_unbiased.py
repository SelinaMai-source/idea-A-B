"""Independent-audit IPW is approximately unbiased under MCAR and MNAR."""

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from shared.metrics.errors import bias
from shared.metrics.estimators import ipw_mean, observed_only_mean


def _audit_ipw_replicates(mechanism: str, n_rep: int = 60) -> tuple[float, list[float], float]:
    world = simulate(mechanism, 0, n=2000)
    truth = world.truths["population_mean_reward"]
    q = np.full(world.n, 0.20)
    estimates = []
    for replicate in range(n_rep):
        a = (np.random.default_rng(50_000 + replicate).random(world.n) < q).astype(int)
        estimates.append(ipw_mean(world.observe(a), a, q))
    return truth, estimates, bias(estimates, truth)


def test_independent_audit_ipw_approximately_unbiased_under_mcar():
    _truth, _estimates, err = _audit_ipw_replicates("mcar")
    assert err == pytest.approx(0.0, abs=0.04)


def test_independent_audit_ipw_approximately_unbiased_under_mnar():
    truth, _estimates, err = _audit_ipw_replicates("mnar")
    world = simulate("mnar", 0, n=2000)
    observed = observed_only_mean(world.observe(np.zeros(world.n, dtype=int)), world.s)
    assert err == pytest.approx(0.0, abs=0.04)
    # Natural channel is MNAR: observed-only is identification failure, not a bug.
    assert abs(observed - truth) > 0.05
    assert world.identification_natural["status"] == "identification_failure"
