"""The allocator and estimators may read Y only where O = S OR A."""

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.methods.v1_load import load_v1_module
from shared.metrics.estimators import dr_mean, ipw_mean, snips_mean

_algorithm = load_v1_module("algorithm")
_estimators = load_v1_module("estimators")


def test_observe_hides_labels_unless_s_or_a():
    world = simulate("mnar", 0, n=400)
    a = np.zeros(world.n, dtype=int)
    a[:25] = 1
    visible = world.observe(a)
    revealed = world.s.astype(bool) | a.astype(bool)
    assert np.all(np.isnan(visible[~revealed]))
    np.testing.assert_allclose(visible[revealed], world.y[revealed])
    assert not np.any(np.isnan(visible[revealed]))


def test_neyman_allocation_ignores_hidden_labels():
    world = simulate("mnar", 1, n=400)
    y_vis = world.observe(np.zeros(world.n, dtype=int))
    q1 = _algorithm.allocate_neyman(
        world.stratum, y_vis, world.s, 0.10, world.n_strata, exploration_mix=0.15
    )
    hidden = world.s == 0
    world.y[hidden] = 1.0e6
    q2 = _algorithm.allocate_neyman(
        world.stratum, y_vis, world.s, 0.10, world.n_strata, exploration_mix=0.15
    )
    np.testing.assert_allclose(q1, q2)
    assert np.all(np.isfinite(q1))
    assert np.all(q1 > 0.0)


def test_ipw_snips_dr_do_not_read_unaudited_entries():
    world = simulate("mcar", 2, n=200)
    rng = np.random.default_rng(7)
    q = np.full(world.n, 0.2)
    a = (rng.random(world.n) < q).astype(int)
    y_vis = world.observe(a)
    y_poison = y_vis.copy()
    y_poison[a == 0] = 1.0e9
    mu = _estimators.fit_mu_hat(y_vis, a, world.x, world.pi, world.stratum, world.n_strata)
    mu_poison_fit = _estimators.fit_mu_hat(
        y_poison, a, world.x, world.pi, world.stratum, world.n_strata
    )
    np.testing.assert_allclose(mu, mu_poison_fit)
    assert ipw_mean(y_vis, a, q) == pytest.approx(ipw_mean(y_poison, a, q))
    assert snips_mean(y_vis, a, q) == pytest.approx(snips_mean(y_poison, a, q))
    assert dr_mean(y_vis, a, q, mu) == pytest.approx(dr_mean(y_poison, a, q, mu))
