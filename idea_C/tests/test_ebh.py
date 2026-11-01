"""e-BH correctness: threshold, nested levels, no label use."""

from __future__ import annotations

import numpy as np

from idea_C.shared.ebh import clip_evalues, ebh_rejection_set, ebh_rejections, ebh_threshold_index
from idea_C.shared.metrics import fdr


def test_empty_when_all_evalues_one() -> None:
    e = np.ones(10)
    assert ebh_threshold_index(e, 0.05) == 0
    assert ebh_rejection_set(e, 0.05) == set()


def test_rejects_large_evalue() -> None:
    e = np.array([1.0, 1.0, 1000.0, 1.0])
    mask = ebh_rejections(e, 0.05)
    assert mask[2]
    assert int(mask.sum()) >= 1


def test_k_star_matches_wang_ramdas() -> None:
    e = np.array([40.0, 20.0, 5.0, 1.0])
    alpha = 0.2
    k = e.size
    order = np.sort(e)[::-1]
    expected = 0
    for i in range(1, k + 1):
        if order[i - 1] >= k / (alpha * i):
            expected = i
    assert ebh_threshold_index(e, alpha) == expected


def test_nested_rejections_larger_alpha_is_superset() -> None:
    rng = np.random.default_rng(7)
    e = np.exp(rng.normal(0.0, 1.0, size=20))
    small = ebh_rejections(e, 0.05)
    mid = ebh_rejections(e, 0.2)
    large = ebh_rejections(e, 0.5)
    assert np.all(small <= mid)
    assert np.all(mid <= large)


def test_clip_nonfinite_does_not_auto_reject() -> None:
    e = np.array([np.inf, np.nan, -3.0, 1.0])
    clipped = clip_evalues(e)
    assert np.all(np.isfinite(clipped))
    assert np.all(clipped > 0.0)
    assert not np.any(ebh_rejections(np.array([np.nan, np.inf, -1.0]), 0.05))


def test_null_only_synthetic_evalues_fdr_control() -> None:
    """Valid null e-values: e = exp(Z − 1/2), Z~N(0,1), E[e]=1."""
    rng = np.random.default_rng(11)
    alpha = 0.2
    k = 8
    fdps = []
    is_null = np.ones(k, dtype=bool)
    for _ in range(80):
        z = rng.normal(size=k)
        e = np.exp(z - 0.5)
        fdps.append(fdr(ebh_rejections(e, alpha), is_null))
    assert float(np.mean(fdps)) <= alpha + 0.12
