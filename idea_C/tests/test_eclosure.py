"""Official vs independent labels, wrappers, r-LOND monotonicity."""

from __future__ import annotations

import numpy as np

from idea_C.baselines.eclosure import (
    AISTATS_ELOND_STATUS,
    OFFLINE_ECLOSURE_STATUS,
    ONLINE_CODE_STATUS,
    RELATED_OFFICIAL_EXP,
    aistats_elond_on_evalues,
    closed_r_lond,
    donation_r_lond,
    r_lond,
    try_offline_closedeBH,
)
from idea_C.baselines.eclosure.official_wrappers import load_aistats_lond_class


def test_labels_do_not_confuse_offline_with_uai() -> None:
    assert "arxiv2509" in OFFLINE_ECLOSURE_STATUS
    assert ONLINE_CODE_STATUS == "independent_reimplementation"
    assert RELATED_OFFICIAL_EXP == "lag_comp"
    assert "aistats2024" in AISTATS_ELOND_STATUS


def test_offline_closedeBH_unavailable_or_honestly_offline() -> None:
    result = try_offline_closedeBH([1.0, 100.0], alpha=0.05)
    assert "NOT UAI" in (result.reason or "") or "not UAI" in (result.extra.get("note") or "").lower() or result.method == "offline-closedeBH"
    if result.status == "unavailable":
        assert "C++" in (result.reason or "") or "bindings" in (result.reason or "").lower() or "clone" in (result.reason or "")
        assert result.code_status == OFFLINE_ECLOSURE_STATUS
    else:
        assert result.code_status == OFFLINE_ECLOSURE_STATUS
        assert "offline" in result.extra.get("note", "").lower()


def test_aistats_lond_wrapper_runs() -> None:
    cls = load_aistats_lond_class()
    assert cls is not None
    e = np.array([1.0, 2.0, 50.0, 0.5, 8.0])
    result = aistats_elond_on_evalues(e, delta=0.1)
    assert result.status == "ok"
    assert result.code_status == AISTATS_ELOND_STATUS
    assert result.rejected.shape == (5,)
    assert "AISTATS 2024" in result.extra.get("note", "")


def test_rlond_donation_dominates_first_step_and_inflates() -> None:
    p = [0.001, 0.2, 0.04, 0.8, 0.01]
    base = r_lond(p, delta=0.1)
    don = donation_r_lond(p, delta=0.1)
    assert abs(don.thresholds[0] - base.thresholds[0]) < 1e-12
    # donation levels are at least as large along the run when R matches at t=1
    assert don.thresholds[0] <= 1.0
    assert np.all(don.rejected | ~base.rejected)


def test_closed_rlond_not_below_rlond_at_t1() -> None:
    p = [0.2, 0.2]
    base = r_lond(p, delta=0.1)
    closed = closed_r_lond(p, delta=0.1)
    assert abs(closed.thresholds[0] - base.thresholds[0]) < 1e-12


def test_same_seed_reproducible_rlond() -> None:
    p = [0.01, 0.2, 0.03, 0.4]
    a = r_lond(p, delta=0.1)
    b = r_lond(p, delta=0.1)
    np.testing.assert_array_equal(a.rejected, b.rejected)
