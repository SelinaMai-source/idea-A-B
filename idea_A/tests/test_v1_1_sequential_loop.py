"""Closed loop: Continual Adaptation changes π, then S. Tracks stay separate."""

import numpy as np
import pytest

from idea_A.benchmarks.Sequential_Closed_Loop.continual_adaptation.track import (
    TRACK_METRICS as CONTINUAL_METRICS,
)
from idea_A.benchmarks.Sequential_Closed_Loop.continual_adaptation.track import (
    TRACK_NAME as CONTINUAL_NAME,
)
from idea_A.benchmarks.Sequential_Closed_Loop.continual_adaptation.track import run_continual_adaptation
from idea_A.benchmarks.Sequential_Closed_Loop.environment import natural_propensity
from idea_A.benchmarks.Sequential_Closed_Loop.static_estimation.track import (
    TRACK_METRICS as STATIC_METRICS,
)
from idea_A.benchmarks.Sequential_Closed_Loop.static_estimation.track import TRACK_NAME as STATIC_NAME
from idea_A.benchmarks.Sequential_Closed_Loop.static_estimation.track import run_static_estimation


def test_tracks_are_separate_experiments():
    assert STATIC_NAME != CONTINUAL_NAME
    assert STATIC_NAME == "static_estimation"
    assert CONTINUAL_NAME == "continual_adaptation"
    assert "mean_pi_by_round" not in STATIC_METRICS
    assert "mean_pi_by_round" in CONTINUAL_METRICS
    assert "pi_changed" in CONTINUAL_METRICS


def test_policy_drift_natural_observation_depends_on_pi_not_hidden_y():
    pi0 = np.zeros(80, dtype=int)
    pi1 = np.ones(80, dtype=int)
    y = np.linspace(-2.0, 2.0, 80)
    p0 = natural_propensity("policy_drift", pi0, y)
    p1 = natural_propensity("policy_drift", pi1, y)
    assert float(np.mean(p1)) > float(np.mean(p0)) + 0.5
    np.testing.assert_allclose(p0, 0.08)
    np.testing.assert_allclose(p1, 0.72)
    p_mnar_a = natural_propensity("mnar_audit", pi0, y)
    p_mnar_b = natural_propensity("mnar_audit", pi1, y)
    np.testing.assert_allclose(p_mnar_a, p_mnar_b)


def test_static_track_does_not_adapt_policy():
    row = run_static_estimation(
        "policy_drift",
        0,
        0.20,
        "A-v1.1a",
        n_per_round=40,
        n_rounds=4,
    )
    assert row["track"] == "static_estimation"
    assert row["policy_adapted"] is False
    assert row["n"] == 160
    assert np.isfinite(row["population_mean_rmse"])


def test_continual_loop_changes_pi_then_natural_observation():
    row = run_continual_adaptation(
        "policy_drift",
        0,
        0.35,
        "A-v1.1c",
        n_per_round=40,
        n_rounds=4,
        extra_tau=4.0,
        noise_scale=0.25,
        policy_kind="linear",
        config={"policy_explore_mix": 0.05},
    )
    assert row["track"] == "continual_adaptation"
    assert row["policy_adapted"] is True
    pi_path = np.asarray(row["mean_pi_by_round"], dtype=float)
    p_path = np.asarray(row["mean_p_natural_by_round"], dtype=float)
    assert pi_path.shape == (4,)
    assert float(pi_path[-1]) > float(pi_path[0]) + 0.08
    assert float(p_path[-1]) > float(p_path[0]) + 0.04
    assert row["pi_changed"] is True
    assert row["p_natural_changed"] is True
    # S is a noisy draw from p_natural; the identification channel is q.
    assert np.isfinite(row["estimands"]["population_mean_reward"]["estimate"])


def test_static_and_continual_are_not_the_same_result_object():
    static = run_static_estimation("mnar_audit", 0, 0.20, "A-v1.1a", n_per_round=40, n_rounds=3)
    continual = run_continual_adaptation(
        "mnar_audit", 0, 0.20, "A-v1.1a", n_per_round=40, n_rounds=3, extra_tau=3.0, noise_scale=0.3
    )
    assert static["track"] != continual["track"]
    assert static["policy_adapted"] is False
    assert continual["policy_adapted"] is True
    assert "pi_changed" in continual
    assert "pi_changed" not in static
