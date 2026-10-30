"""Expected audit fraction tracks the requested budget."""

import numpy as np
import pytest

from idea_A.baselines.Uniform_Audit.adapter import allocate as allocate_uniform
from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.methods.v1_load import load_v1_module
from shared.evaluation.budget import (
    assert_budget,
    expected_audit_fraction,
    realized_audit_fraction,
)

_algorithm = load_v1_module("algorithm")
_intervention = load_v1_module("intervention")


@pytest.mark.parametrize("budget", [0.01, 0.05, 0.10, 0.20])
def test_uniform_expected_fraction_matches_budget(budget):
    q = allocate_uniform(2000, budget)
    expected = expected_audit_fraction(q)
    assert expected == pytest.approx(budget)
    assert_budget(expected, budget, 1e-12)


@pytest.mark.parametrize("budget", [0.01, 0.05, 0.10, 0.20])
def test_neyman_expected_fraction_matches_budget(budget):
    world = simulate("mnar", 0, n=2000)
    y_vis = world.observe(np.zeros(world.n, dtype=int))
    q = _algorithm.allocate_neyman(
        world.stratum, y_vis, world.s, budget, world.n_strata, exploration_mix=0.15
    )
    expected = expected_audit_fraction(q)
    assert_budget(expected, budget, 1e-6)
    rng = np.random.default_rng(0)
    a = _intervention.draw_audit(q, rng)
    realized = realized_audit_fraction(a)
    # Binomial noise around mean(q); n=2000, b=0.01 still has SD ~0.002.
    assert abs(realized - expected) < 0.03
