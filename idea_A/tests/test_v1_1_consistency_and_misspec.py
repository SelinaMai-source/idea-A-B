"""Consistency under suitable conditions, and DR misspecification cases."""

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.methods.v1_load import load_v1_1_module
from shared.metrics.estimators import dr_mean, ipw_mean

_alg = load_v1_1_module("algorithm")
_seq = load_v1_1_module("sequential_estimator")
_cf = load_v1_1_module("cross_fitting")


def _linear_dgp(n: int = 400, seed: int = 0, q_true: float = 0.40):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    stratum = rng.integers(0, 8, size=n)
    pi = rng.integers(0, 2, size=n)
    y = 1.0 + x[:, 0] + 0.5 * x[:, 1] + 0.15 * rng.normal(size=n)
    q = np.full(n, q_true)
    a = (rng.random(n) < q).astype(int)
    y_vis = np.full(n, np.nan, dtype=float)
    y_vis[a == 1] = y[a == 1]
    return {
        "x": x,
        "stratum": stratum,
        "pi": pi,
        "y": y,
        "q": q,
        "a": a,
        "y_vis": y_vis,
        "truth": float(np.mean(y)),
        "oracle_mu": 1.0 + x[:, 0] + 0.5 * x[:, 1],
        "n_strata": 8,
    }


def test_cross_fit_dr_consistent_under_mcar_known_q():
    world = simulate("mcar", 0, n=400)
    # Audit RNG must not reuse the world Generator seed; the protocol salts it.
    row = _alg.run_static_config("A-v1.1a", world, 0.35, np.random.default_rng(100_003))
    err = abs(row["population_mean_reward"]["estimate"] - world.truths["population_mean_reward"])
    assert err < 0.25


def test_linear_dgp_cross_fit_dr_near_truth():
    d = _linear_dgp()
    folds = _cf.assign_folds(d["y"].shape[0], rng=np.random.default_rng(1))
    mu = _cf.cross_fit_mu_y(
        d["y_vis"], d["a"], d["x"], d["pi"], d["stratum"], d["n_strata"], folds
    )
    est = dr_mean(d["y_vis"], d["a"], d["q"], mu)
    assert abs(est - d["truth"]) < 0.12


def test_correct_propensity_misspecified_outcome_still_close():
    d = _linear_dgp()
    mu_wrong = np.zeros(d["y"].shape[0])
    est = dr_mean(d["y_vis"], d["a"], d["q"], mu_wrong)
    ipw = ipw_mean(d["y_vis"], d["a"], d["q"])
    assert est == pytest.approx(ipw)
    assert abs(est - d["truth"]) < 0.20


def test_misspecified_propensity_correct_outcome_still_close():
    d = _linear_dgp()
    q_wrong = np.clip(0.15 + 0.70 * (d["stratum"] >= 4).astype(float), 1e-3, 1.0)
    est = dr_mean(d["y_vis"], d["a"], q_wrong, d["oracle_mu"])
    assert abs(est - d["truth"]) < 0.08


def test_both_nuisances_misspecified_is_biased():
    d = _linear_dgp()
    q_wrong = np.full(d["y"].shape[0], 0.90)
    mu_wrong = np.full(d["y"].shape[0], 10.0)
    est = dr_mean(d["y_vis"], d["a"], q_wrong, mu_wrong)
    assert abs(est - d["truth"]) > 1.0


def test_history_respecting_fit_ignores_current_round_labels():
    d0 = _linear_dgp(n=200, seed=0)
    d1 = _linear_dgp(n=200, seed=1)
    state = _seq.SequentialNuisanceState(n_strata=8)
    snap0 = _seq.RoundSnapshot(
        y_visible=d0["y_vis"],
        pref_visible=(d0["y_vis"] > 0).astype(float),
        a=d0["a"],
        q=d0["q"],
        x=d0["x"],
        pi=d0["pi"],
        stratum=d0["stratum"],
        target_weights=np.ones(200),
        n_strata=8,
    )
    state.append(snap0)
    snap1 = _seq.RoundSnapshot(
        y_visible=d1["y_vis"],
        pref_visible=(d1["y_vis"] > 0).astype(float),
        a=d1["a"],
        q=d1["q"],
        x=d1["x"],
        pi=d1["pi"],
        stratum=d1["stratum"],
        target_weights=np.ones(200),
        n_strata=8,
    )
    est_a = _seq.estimate_current_round(snap1, state)
    y_poison = d1["y_vis"].copy()
    y_poison[d1["a"] == 1] = 1.0e6
    snap1_bad = _seq.RoundSnapshot(
        y_visible=y_poison,
        pref_visible=snap1.pref_visible,
        a=d1["a"],
        q=d1["q"],
        x=d1["x"],
        pi=d1["pi"],
        stratum=d1["stratum"],
        target_weights=np.ones(200),
        n_strata=8,
    )
    est_b = _seq.estimate_current_round(snap1_bad, state)
    np.testing.assert_allclose(est_a["mu_y"], est_b["mu_y"])
    # Current-round outcomes still enter the residual where A=1, so the
    # point estimate may change; the nuisance must not.
    assert not np.allclose(est_a["influence"], est_b["influence"])
