"""Wealth / overshoot refund identities on tiny sequences."""

from __future__ import annotations

import numpy as np
import pytest

from idea_C.baselines.score import e_lond, e_lord, score_lond, score_lord, score_plus_lord, score_plus_saffron
from idea_C.baselines.score.metrics import fdr_terminal
from idea_C.baselines.score.procedures import ScorePlusLORD
from idea_C.baselines.score.wealth import (
    lond_refund,
    overshoot,
    rejects,
    saffron_score_cost,
    saffron_tight_cost,
    score_cost,
    score_plus_lord_alpha,
    score_plus_saffron_alpha,
    score_plus_wealth,
)


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


def test_score_plus_handmade_stepwise_wealth_matches_prop5() -> None:
    """Hand-computed Prop. 5.3 path. Formula is the paper update, not a shortcut."""
    alpha, omega = 0.2, 0.5
    evalues = [2.0, 80.0, 1.0, 50.0]
    result = score_plus_lord(evalues, alpha=alpha, omega=omega, assume_cpqd=True)
    assert result.status == "ok"
    assert result.protocol_status == "ok"

    # t=1: R=0, W=0.2, α=0.5*1*0.2=0.1; e=2 → no reject, O=0, C=0.1
    assert result.thresholds[0] == pytest.approx(0.1)
    assert not bool(result.rejected[0])
    assert result.overshoots[0] == pytest.approx(0.0)
    assert result.costs[0] == pytest.approx(0.1)
    assert result.extra["wealth"][0] == pytest.approx(0.2)

    # t=2: R=0, W=0.2-0.1=0.1, α=0.05; e=80 → reject, O=3, C=0
    assert result.thresholds[1] == pytest.approx(0.05)
    assert bool(result.rejected[1])
    assert result.overshoots[1] == pytest.approx(3.0)
    assert result.costs[1] == pytest.approx(0.0)
    assert result.extra["wealth"][1] == pytest.approx(0.1)

    # t=3: R=1, denom=1, spent=0.1, W=0.1, α=0.05; e=1 → no reject, C=0.05
    assert result.thresholds[2] == pytest.approx(0.05)
    assert not bool(result.rejected[2])
    assert result.overshoots[2] == pytest.approx(0.0)
    assert result.costs[2] == pytest.approx(0.05)
    assert result.extra["wealth"][2] == pytest.approx(0.1)

    # t=4: R=1, spent=0.15, W=0.05, α=0.025; e=50 → reject, O=0.25, C=0
    assert result.thresholds[3] == pytest.approx(0.025)
    assert bool(result.rejected[3])
    assert result.overshoots[3] == pytest.approx(0.25)
    assert result.costs[3] == pytest.approx(0.0)
    assert result.extra["wealth"][3] == pytest.approx(0.05)

    # Reconstruct each step from the public formula helpers.
    costs: list[float] = []
    r_prev = 0
    for t, e in enumerate(evalues, start=1):
        wealth = score_plus_wealth(alpha, costs, r_prev)
        thresh = score_plus_lord_alpha(omega, r_prev, wealth)
        o = overshoot(thresh, e)
        c = score_cost(thresh, o)
        assert thresh == pytest.approx(result.thresholds[t - 1])
        assert o == pytest.approx(result.overshoots[t - 1])
        assert c == pytest.approx(result.costs[t - 1])
        costs.append(c)
        r_prev += int(rejects(thresh, e))


def test_score_plus_retroactive_denominator_after_two_rejections() -> None:
    alpha, omega = 0.2, 0.5
    result = score_plus_lord([1000.0, 1000.0, 1.0], alpha=alpha, omega=omega, assume_cpqd=True)
    # After two overshooting rejections, W stays α and α_3 = ω * 2 * α = 0.2
    assert result.costs[0] == pytest.approx(0.0)
    assert result.costs[1] == pytest.approx(0.0)
    assert result.extra["wealth"][2] == pytest.approx(0.2)
    assert result.thresholds[2] == pytest.approx(0.2)
    base = e_lord([1000.0, 1000.0, 1.0], alpha=alpha, omega=omega)
    assert result.thresholds[2] > base.thresholds[2] + 1e-12


def test_score_plus_saffron_handmade_alpha1_and_cost() -> None:
    alpha, omega, lam = 0.2, 0.5, 0.5
    evalues = [2.0, 20.0]
    result = score_plus_saffron(
        evalues, alpha=alpha, omega=omega, lam=lam, assume_cpqd=True
    )
    # α_1 = ω(1-λ)α = 0.05
    assert result.thresholds[0] == pytest.approx(score_plus_saffron_alpha(omega, lam, 0, alpha))
    assert result.thresholds[0] == pytest.approx(0.05)
    o0 = overshoot(result.thresholds[0], evalues[0])
    assert result.overshoots[0] == pytest.approx(o0)
    assert result.costs[0] == pytest.approx(saffron_score_cost(result.thresholds[0], evalues[0], lam))
    wealth1 = score_plus_wealth(alpha, [result.costs[0]], int(result.rejected[0]))
    assert result.thresholds[1] == pytest.approx(
        score_plus_saffron_alpha(omega, lam, int(result.rejected[0]), wealth1)
    )


def test_score_plus_per_step_cost_overshoot_identity() -> None:
    proc = ScorePlusLORD(alpha=0.05, omega=0.05)
    for e in (0.4, 80.0, 1.2, 30.0, 0.2):
        proc.process_next(e)
    for alpha, evalue, o, cost in zip(proc.alphas, proc.evalues, proc.overshoots, proc.costs):
        assert o == pytest.approx(overshoot(alpha, evalue))
        assert cost == pytest.approx(score_cost(alpha, o))
        assert cost + min(o, alpha) == pytest.approx(alpha)


def test_score_plus_null_only_fdr_diagnostic() -> None:
    rng = np.random.default_rng(1)
    fdps: list[float] = []
    for seed in range(20):
        del seed
        x = rng.normal(0.0, 1.0, size=40)
        e = np.exp(3.0 * x - 4.5)  # LR of N(3,1) vs N(0,1) under the null
        is_null = np.ones(40, dtype=bool)
        result = score_plus_lord(e, alpha=0.05, omega=0.05, assume_cpqd=True)
        fdps.append(fdr_terminal(result.rejected, is_null))
        assert result.n_rejections < 40
    assert float(np.mean(fdps)) <= 0.15
    blocked = score_plus_lord(e, alpha=0.05, assume_cpqd=False)
    assert blocked.status == "N/A"
    assert blocked.protocol_status == "INVALID_PROTOCOL"
    assert blocked.extra.get("fdr_guarantee") is False
