"""Audit intervention: draw ``A`` from a known propensity and reveal labels.

The independent audit is the identification anchor. It does not read hidden
``Y`` when forming ``A``; ``A_t ~ Bernoulli(q_t)`` given the logged history.
"""

from __future__ import annotations

import numpy as np


def draw_audit(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent Bernoulli audits with known per-unit propensity ``q``."""
    propensity = np.asarray(q, dtype=float)
    if propensity.ndim != 1 or propensity.size == 0:
        raise ValueError("q must be a non-empty 1-D vector")
    if np.any(~np.isfinite(propensity)):
        raise ValueError("q must be finite")
    if np.any(propensity < 0.0) or np.any(propensity > 1.0):
        raise ValueError("q must lie in [0, 1]")
    u = rng.random(propensity.shape[0])
    return (u < propensity).astype(int)


def observed_mask(s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``O = S ∨ A``. Integer 0/1."""
    return (np.asarray(s).astype(bool) | np.asarray(a).astype(bool)).astype(int)


def reveal_labels(y: np.ndarray, s: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Copy ``Y`` onto units with ``O=1``; write NaN elsewhere.

    Callers that already hold a ``World`` should prefer ``World.observe``.
    This helper exists so tests can pass an explicit ``y`` array without
    exposing unaudited / unobserved entries.
    """
    y_arr = np.asarray(y, dtype=float)
    mask = observed_mask(s, a).astype(bool)
    out = np.full(y_arr.shape[0], np.nan, dtype=float)
    out[mask] = y_arr[mask]
    return out


def no_audit(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero intervention: ``q=0``, ``A=0``."""
    zeros = np.zeros(int(n), dtype=float)
    return zeros, zeros.astype(int)


def full_feedback(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Oracle intervention that does not respect a budget: ``q=1``, ``A=1``."""
    ones = np.ones(int(n), dtype=float)
    return ones, ones.astype(int)
