"""Predictable spending sequences. No horizon renormalization (that would peek T)."""

from __future__ import annotations

import numpy as np


def gamma_inv_t_tplus1(t: int) -> float:
    """γ_t = 1/(t(t+1)), sums to 1 over t=1…∞. Default for SCORE-LOND / UAI e-LOND."""
    if t < 1:
        raise ValueError("t is 1-based")
    return 1.0 / (t * (t + 1.0))


def gamma_sequence(n: int, kind: str = "inv_t_tplus1") -> np.ndarray:
    t = np.arange(1, n + 1, dtype=float)
    if kind == "inv_t_tplus1":
        return 1.0 / (t * (t + 1.0))
    if kind == "inv_t2":
        raw = 1.0 / (t * t)
        # infinite-horizon normalizer  π²/6, not finite-T renormalization
        return raw * (6.0 / (np.pi ** 2))
    raise ValueError(f"unknown gamma kind {kind}")
