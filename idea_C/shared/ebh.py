"""e-Benjamini-Hochberg (Wang and Ramdas, 2022).

Given nonnegative e-values ``e_1, …, e_K`` and level ``alpha`` in ``(0, 1]``,
sort them decreasingly as ``e_(1) ≥ ⋯ ≥ e_(K)`` and set

    k* = max { k ∈ [K] : e_(k) ≥ K / (alpha * k) },

with the convention ``k* = 0`` (empty rejection set) if the set is empty.
Reject every hypothesis ``i`` with ``e_i ≥ K / (alpha * k*)``.

This is identical to AAAI 2026 Eq. (3)–(4) with ``δ = alpha``, and to
arXiv:2609.26651 Eq. (2). The procedure does not use ground-truth labels
and does not look at future observations.

Nestedness in the level: if ``0 < alpha_small ≤ alpha_large ≤ 1``, then the
rejection set at ``alpha_small`` is a subset of the rejection set at
``alpha_large``.
"""

from __future__ import annotations

import numpy as np

_E_MIN = 1e-300
_E_MAX = 1e300


def clip_evalues(evalues: np.ndarray) -> np.ndarray:
    """Map e-values to a finite positive range; non-finite entries become ``_E_MIN``."""
    arr = np.asarray(evalues, dtype=float).reshape(-1)
    out = np.array(arr, copy=True)
    bad = ~np.isfinite(out) | (out <= 0.0)
    out[bad] = _E_MIN
    return np.clip(out, _E_MIN, _E_MAX)


def ebh_threshold_index(evalues: np.ndarray, alpha: float) -> int:
    """Return ``k*`` (number of largest e-values rejected), or 0 if none."""
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    e = clip_evalues(evalues)
    k = int(e.size)
    if k == 0:
        return 0
    order = np.sort(e)[::-1]
    cutoff = (k / float(alpha)) / np.arange(1, k + 1)
    hits = np.nonzero(order >= cutoff)[0]
    if hits.size == 0:
        return 0
    return int(hits[-1] + 1)


def ebh_rejections(evalues: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask of e-BH rejections at level ``alpha``.

    Parameters
    ----------
    evalues:
        Length-``K`` nonnegative e-values. Non-finite / non-positive entries
        are clipped; they are never treated as automatic discoveries.
    alpha:
        Target FDR level in ``(0, 1]``.
    """
    e = clip_evalues(evalues)
    k_star = ebh_threshold_index(e, alpha)
    if k_star == 0:
        return np.zeros(e.size, dtype=bool)
    thresh = e.size / (float(alpha) * k_star)
    return e >= thresh


def ebh_rejection_set(evalues: np.ndarray, alpha: float) -> set[int]:
    """Index set of rejected hypotheses (0-based)."""
    mask = ebh_rejections(evalues, alpha)
    return set(int(i) for i in np.flatnonzero(mask))
