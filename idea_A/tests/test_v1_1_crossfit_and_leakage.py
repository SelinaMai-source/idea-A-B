"""Cross-fit is out-of-fold; hidden labels never enter v1.1 nuisances or DR."""

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.methods.v1_load import load_v1_1_module
from shared.metrics.estimators import dr_mean

_cf = load_v1_1_module("cross_fitting")
_outcome = load_v1_1_module("outcome_model")
_seq = load_v1_1_module("sequential_estimator")


def test_shuffled_cv_leakage_is_documented():
    assert "leaks under adaptive sampling" in _cf.SHUFFLED_CV_LEAKAGE_NOTE
    assert "past rounds" in _seq.sequential_leakage_note()


def test_cross_fit_prediction_does_not_use_own_label():
    world = simulate("mcar", 0, n=240)
    rng = np.random.default_rng(0)
    q = np.full(world.n, 0.4)
    a = (rng.random(world.n) < q).astype(int)
    y_vis = world.observe(a)
    fold_ids = _cf.assign_folds(world.n, n_folds=2, rng=np.random.default_rng(1))
    mu1 = _cf.cross_fit_mu_y(y_vis, a, world.x, world.pi, world.stratum, world.n_strata, fold_ids)
    i = int(np.flatnonzero(a.astype(bool))[0])
    y_tweak = y_vis.copy()
    y_tweak[i] = y_vis[i] + 50.0
    mu2 = _cf.cross_fit_mu_y(y_tweak, a, world.x, world.pi, world.stratum, world.n_strata, fold_ids)
    assert mu1[i] == pytest.approx(mu2[i])
    other = fold_ids == fold_ids[i]
    # Units in the same fold as i are also predicted from the other fold, so
    # changing Y_i must not change those predictions either.
    np.testing.assert_allclose(mu1[other], mu2[other])


def test_cross_fit_and_ridge_ignore_hidden_and_unaudited_labels():
    world = simulate("mnar", 1, n=300)
    rng = np.random.default_rng(4)
    q = np.full(world.n, 0.25)
    a = (rng.random(world.n) < q).astype(int)
    y_vis = world.observe(a)
    y_poison = y_vis.copy()
    y_poison[a == 0] = 1.0e9
    folds = _cf.assign_folds(world.n, rng=np.random.default_rng(2))
    mu_a = _cf.cross_fit_mu_y(y_vis, a, world.x, world.pi, world.stratum, world.n_strata, folds)
    mu_b = _cf.cross_fit_mu_y(y_poison, a, world.x, world.pi, world.stratum, world.n_strata, folds)
    np.testing.assert_allclose(mu_a, mu_b)
    ridge_a = _outcome.fit_mu_y(y_vis, a, world.x, world.pi, world.stratum, world.n_strata).predict(
        world.x, world.pi, world.stratum
    )
    ridge_b = _outcome.fit_mu_y(y_poison, a, world.x, world.pi, world.stratum, world.n_strata).predict(
        world.x, world.pi, world.stratum
    )
    np.testing.assert_allclose(ridge_a, ridge_b)
    assert dr_mean(y_vis, a, q, mu_a) == pytest.approx(dr_mean(y_poison, a, q, mu_a))
    np.testing.assert_allclose(
        _seq.aipw_influence(y_vis, a, q, mu_a),
        _seq.aipw_influence(y_poison, a, q, mu_a),
    )


def test_assign_folds_does_not_read_outcomes():
    folds = _cf.assign_folds(20, rng=np.random.default_rng(0))
    assert set(folds.tolist()) == {0, 1}
    assert folds.shape == (20,)


def test_mu_y_fit_ignores_natural_only_labels():
    world = simulate("mnar", 0, n=300)
    rng = np.random.default_rng(5)
    a = (rng.random(world.n) < 0.15).astype(int)
    y_vis = world.observe(a)
    y_poison = y_vis.copy()
    natural_only = (world.s == 1) & (a == 0)
    assert np.any(natural_only)
    y_poison[natural_only] = 1.0e9
    mu1 = _outcome.fit_mu_y(y_vis, a, world.x, world.pi, world.stratum, world.n_strata).predict(
        world.x, world.pi, world.stratum
    )
    mu2 = _outcome.fit_mu_y(y_poison, a, world.x, world.pi, world.stratum, world.n_strata).predict(
        world.x, world.pi, world.stratum
    )
    np.testing.assert_allclose(mu1, mu2)
