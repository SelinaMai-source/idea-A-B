"""SCORE procedures: CPQD gate, no future peeking, reproducibility, null FDR."""

from __future__ import annotations

import numpy as np

from idea_C.baselines.score import (
    CODE_STATUS,
    average_power_path,
    e_lond,
    e_lord,
    e_saffron,
    fdr_terminal,
    independent_gaussian_mixture,
    score_lond,
    score_lord,
    score_plus_lord,
    score_plus_saffron,
    score_saffron,
)
from idea_C.baselines.score.procedures import ELORD, ScoreLORD


def test_code_status_is_independent() -> None:
    assert CODE_STATUS == "independent_reimplementation"


def test_score_plus_na_without_cpqd() -> None:
    e = [2.0, 3.0, 4.0]
    r1 = score_plus_lord(e, alpha=0.05, assume_cpqd=False)
    r2 = score_plus_saffron(e, alpha=0.05, assume_cpqd=False)
    assert r1.status == "N/A"
    assert r2.status == "N/A"
    assert r1.rejected.size == 0
    assert "CPQD" in (r1.na_reason or "")
    assert r1.assume_cpqd is False


def test_score_plus_runs_when_cpqd_assumed() -> None:
    e = [2.0, 50.0, 4.0, 80.0]
    r = score_plus_lord(e, alpha=0.05, omega=0.05, assume_cpqd=True)
    assert r.status == "ok"
    assert r.assume_cpqd is True
    assert r.thresholds.shape == (4,)
    assert r.rejected.shape == (4,)


def test_same_seed_reproducible_dgp_and_decisions() -> None:
    a = independent_gaussian_mixture(T=40, pi1=0.3, seed=7)
    b = independent_gaussian_mixture(T=40, pi1=0.3, seed=7)
    np.testing.assert_allclose(a.evalues, b.evalues)
    np.testing.assert_array_equal(a.is_null, b.is_null)
    r1 = score_lord(a.evalues, alpha=0.05, omega=0.05)
    r2 = score_lord(b.evalues, alpha=0.05, omega=0.05)
    np.testing.assert_array_equal(r1.rejected, r2.rejected)
    np.testing.assert_allclose(r1.thresholds, r2.thresholds)


def test_no_future_information_in_thresholds() -> None:
    prefix = [1.5, 8.0, 0.4]
    e1 = prefix + [100.0, 0.2]
    e2 = prefix + [0.01, 50.0]
    for fn in (e_lond, score_lond, e_lord, score_lord, e_saffron, score_saffron):
        r1 = fn(e1, alpha=0.05)
        r2 = fn(e2, alpha=0.05)
        np.testing.assert_allclose(r1.thresholds[:3], r2.thresholds[:3])
        np.testing.assert_array_equal(r1.rejected[:3], r2.rejected[:3])


def test_next_alpha_does_not_see_current_e() -> None:
    proc = ScoreLORD(alpha=0.05, omega=0.05)
    a_before = proc.next_alpha()
    # calling next_alpha twice without process_next is stable
    assert proc.next_alpha() == a_before
    proc.process_next(1e9)
    a_after = proc.next_alpha()
    # after a rejection+overshoot, wealth should not decrease vs baseline ELORD
    base = ELORD(alpha=0.05, omega=0.05)
    base.process_next(1e9)
    assert a_after >= base.next_alpha() - 1e-12


def test_null_only_fdr_sanity() -> None:
    rng = np.random.default_rng(0)
    # valid null e-values: LR of N(3,1) vs N(0,1) with X~N(0,1)
    x = rng.normal(0.0, 1.0, size=30)
    e = np.exp(3.0 * x - 4.5)
    is_null = np.ones(30, dtype=bool)
    for fn in (e_lond, score_lond, e_lord, score_lord):
        r = fn(e, alpha=0.05)
        fdp = fdr_terminal(r.rejected, is_null)
        assert 0.0 <= fdp <= 1.0
        # with T=30 and α=0.05, should not reject the entire stream
        assert r.n_rejections < 30


def test_never_reject_tiny_evalues() -> None:
    e = np.full(20, 0.1)
    r = score_lord(e, alpha=0.05, omega=0.05)
    assert r.n_rejections == 0
    r2 = e_lond(e, alpha=0.05)
    assert r2.n_rejections == 0


def test_average_power_matches_paper_definition() -> None:
    rejected = np.array([True, False, True, False])
    is_null = np.array([True, True, False, False])
    ap = average_power_path(rejected, is_null)
    # t=1,2: no non-nulls yet → NaN
    assert np.isnan(ap[0]) and np.isnan(ap[1])
    # t=3: one non-null arrived, rejected → 1
    assert ap[2] == 1.0
    # t=4: two non-nulls, one rejected → 0.5
    assert ap[3] == 0.5


def test_ground_truth_not_required_for_run() -> None:
    # algorithms take only e-values
    e = independent_gaussian_mixture(T=15, pi1=0.8, seed=1).evalues
    r = score_saffron(e, alpha=0.05, omega=0.05, lam=0.5)
    assert r.status == "ok"
    assert r.thresholds.shape == (15,)
