"""Wealth / overshoot refund identities on tiny sequences."""

from __future__ import annotations

import numpy as np

from idea_C.baselines.score.wealth import (
    lond_refund,
    overshoot,
    saffron_score_cost,
    saffron_tight_cost,
    score_cost,
)
from idea_C.baselines.score import score_lond, e_lond, score_lord, e_lord


def test_overshoot_zero_below_threshold() -> None:
    assert overshoot(0.1, 5.0) == 0.0  # αe = 0.5 < 1
    assert overshoot(0.1, 10.0) == 0.0  # exactly on the boundary


def test_overshoot_positive_when_exceeding() -> None:
    # α=0.1, e=20 → αe-1 = 1
    assert overshoot(0.1, 20.0) == 1.0
    # α=0.1, e=100 → 9
    assert overshoot(0.1, 100.0) == 9.0


def test_score_cost_is_positive_part() -> None:
    assert score_cost(0.1, 9.0) == 0.0
    assert score_cost(0.1, 0.05) == 0.05
    assert score_cost(0.1, 0.0) == 0.1


def test_lond_refund_identity() -> None:
    for alpha in (0.01, 0.05, 0.2):
        for o in (0.0, 0.5 * alpha, alpha, 2.0 * alpha, 10.0):
            left = lond_refund(alpha, o)
            right = alpha - score_cost(alpha, o)
            assert abs(left - right) < 1e-12
            assert abs(left - min(max(o, 0.0), alpha)) < 1e-12


def test_saffron_tight_cost_zero_for_candidates() -> None:
    # λ=0.5, e=3 > 1/λ=2 → candidate, C'=0
    assert saffron_tight_cost(0.05, 3.0, 0.5) == 0.0
    # e=1 < 2 → not candidate, C' = 0.05 * (1 - 0.5*1) / 0.5 = 0.05
    assert abs(saffron_tight_cost(0.05, 1.0, 0.5) - 0.05) < 1e-12


def test_saffron_score_cost_subtracts_overshoot() -> None:
    alpha, e, lam = 0.2, 10.0, 0.5
    o = overshoot(alpha, e)
    expected = max(alpha * (1.0 - lam * e) / (1.0 - lam) - o, 0.0)
    assert abs(saffron_score_cost(alpha, e, lam) - expected) < 1e-12


def test_score_lond_refunds_raise_next_threshold() -> None:
    alpha = 0.2

    def gamma(t: int) -> float:
        return {1: 0.5, 2: 0.25, 3: 0.25}[t]

    e = [100.0, 1.0]
    base = e_lond(e, alpha=alpha, gamma=gamma)
    sc = score_lond(e, alpha=alpha, gamma=gamma)
    assert sc.overshoots[0] > 0.0
    assert sc.costs[0] == score_cost(sc.thresholds[0], sc.overshoots[0])
    assert sc.thresholds[1] > base.thresholds[1] - 1e-12
    # refund identity on step 1
    o0 = sc.overshoots[0]
    a0 = sc.thresholds[0]
    assert abs(lond_refund(a0, o0) + sc.costs[0] - a0) < 1e-12


def test_score_lord_spent_uses_refunded_cost() -> None:
    e = [1e6, 1.0, 1.0]
    base = e_lord(e, alpha=0.05, omega=0.05)
    sc = score_lord(e, alpha=0.05, omega=0.05)
    assert sc.overshoots[0] > 0.0
    assert sc.costs[0] < base.costs[0] + 1e-12
    assert sc.thresholds[1] >= base.thresholds[1] - 1e-12
