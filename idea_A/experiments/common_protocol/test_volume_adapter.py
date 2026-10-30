"""Tests for the adapted volume allocation (not AMPLe official)."""

from __future__ import annotations

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.experiments.common_protocol.volume_adapter import allocate_volume
from idea_A.experiments.protocol import allocate_q
from idea_A.methods.v1_load import load_v1_module
from shared.evaluation.budget import expected_audit_fraction

_algorithm = load_v1_module("algorithm")


def test_volume_matches_budget_and_stays_in_unit_interval():
    world = simulate("mnar", 0, n=400)
    y_vis = world.observe(np.zeros(world.n, dtype=int))
    q = allocate_volume(world.stratum, y_vis, world.s, 0.05, world.n_strata)
    assert np.all(np.isfinite(q))
    assert np.all(q > 0.0)
    assert np.all(q <= 1.0)
    assert expected_audit_fraction(q) == pytest.approx(0.05, abs=1e-8)


def test_volume_ignores_hidden_labels():
    world = simulate("mnar", 1, n=400)
    y_vis = world.observe(np.zeros(world.n, dtype=int))
    q1 = allocate_volume(world.stratum, y_vis, world.s, 0.05, world.n_strata)
    hidden = world.s == 0
    world.y[hidden] = 1.0e6
    q2 = allocate_volume(world.stratum, y_vis, world.s, 0.05, world.n_strata)
    np.testing.assert_allclose(q1, q2)


def test_volume_differs_from_uniform_and_neyman_on_mnar():
    world = simulate("mnar", 2, n=800)
    config = {"exploration_mix": 0.15, "q_floor": 1e-6, "ridge_lambda": 1e-3}
    y_vis = world.observe(np.zeros(world.n, dtype=int))
    q_vol = allocate_volume(world.stratum, y_vis, world.s, 0.05, world.n_strata)
    q_uni = allocate_q("uniform", world, 0.05, config)
    q_ney = allocate_q("neyman", world, 0.05, config)
    assert not np.allclose(q_vol, q_uni)
    assert not np.allclose(q_vol, q_ney)


def test_zero_overlap_empty_strata_get_positive_q():
    world = simulate("zero_overlap", 0, n=800)
    y_vis = world.observe(np.zeros(world.n, dtype=int))
    q = allocate_volume(world.stratum, y_vis, world.s, 0.05, world.n_strata)
    for h in (6, 7):
        mask = world.stratum == h
        if np.any(mask):
            assert float(np.min(q[mask])) > 0.0
