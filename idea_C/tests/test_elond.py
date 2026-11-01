"""e-LOND / donation / closed e-LOND invariants (UAI 2026, NOT CCF-A Main)."""

from __future__ import annotations

import numpy as np

from idea_C.baselines.eclosure import (
    ONLINE_CODE_STATUS,
    VENUE,
    closed_e_lond,
    donation_e_lond,
    e_lond,
)
from idea_C.baselines.eclosure.online_elond import ClosedELOND, DonationELOND, ELOND


def test_venue_is_not_ccf_a() -> None:
    assert "NOT CCF-A" in VENUE
    assert ONLINE_CODE_STATUS == "independent_reimplementation"


def test_elond_first_threshold() -> None:
    proc = ELOND(delta=0.1)
    # γ_1 = 1/2, α_1 = 0.1 * 0.5 * 1 = 0.05
    assert abs(proc.next_alpha() - 0.05) < 1e-12


def test_closed_elond_t2_matches_paper_example() -> None:
    # Paper Theorem 2 sketch: E1 = 1/(δ γ_1) = 20, both reject H1,
    # α_closed_2 = 2 δ γ_1 = 0.1, α_elond_2 = 2 δ γ_2 = 0.1/3.
    delta = 0.1
    e1 = 1.0 / (delta * 0.5)  # 20
    e = [e1, 1.0]
    base = e_lond(e, delta=delta)
    closed = closed_e_lond(e, delta=delta)
    assert base.rejected[0] and closed.rejected[0]
    assert abs(base.thresholds[1] - 2 * delta / 6.0) < 1e-12
    assert abs(closed.thresholds[1] - 2 * delta * 0.5) < 1e-9
    assert closed.thresholds[1] > base.thresholds[1]


def test_donation_elond_dominates_elond_thresholds() -> None:
    rng = np.random.default_rng(1)
    e = np.exp(rng.normal(0.0, 1.5, size=25))
    base = e_lond(e, delta=0.1)
    don = donation_e_lond(e, delta=0.1)
    # donation α_t ≥ e-LOND α_t, so the rejection set contains e-LOND's
    assert np.all(don.rejected | ~base.rejected)
    # first step W̄=0 so α matches
    assert abs(don.thresholds[0] - base.thresholds[0]) < 1e-12


def test_closed_elond_rejection_superset_of_elond() -> None:
    rng = np.random.default_rng(2)
    e = np.exp(rng.normal(0.5, 1.0, size=20))
    base = e_lond(e, delta=0.1)
    closed = closed_e_lond(e, delta=0.1)
    # γ decreasing ⇒ closed dominates e-LOND (Theorem 2)
    assert np.all(closed.rejected | ~base.rejected)


def test_donation_wealth_nonnegative() -> None:
    e = [20.0, 5.0, 0.2, 30.0, 1.0]
    proc = DonationELOND(delta=0.1)
    for x in e:
        proc.process_next(x)
    assert all(w >= -1e-12 for w in proc.wealth_trace)


def test_no_future_peeking_elond_family() -> None:
    prefix = [12.0, 0.5, 8.0]
    e1 = prefix + [100.0]
    e2 = prefix + [0.01]
    for fn in (e_lond, closed_e_lond, donation_e_lond):
        r1 = fn(e1, delta=0.1)
        r2 = fn(e2, delta=0.1)
        np.testing.assert_allclose(r1.thresholds[:3], r2.thresholds[:3])
        np.testing.assert_array_equal(r1.rejected[:3], r2.rejected[:3])


def test_next_alpha_ignores_current_e() -> None:
    for cls in (ELOND, ClosedELOND, DonationELOND):
        proc = cls(delta=0.1)
        a = proc.next_alpha()
        assert proc.next_alpha() == a
        proc.process_next(0.01)
        _ = proc.next_alpha()
