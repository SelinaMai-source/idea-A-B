"""SCORE metrics.

Power matches ICML 2026 SCORE §2 and §6:

    AP(t) = E[  (sum_{j in H_1(t)} δ_j) / |H_1(t)|  ]

i.e. the expected fraction of *arrived* non-nulls that have been rejected
by time t (time-point average power). Empirical AP is the Monte Carlo mean
of that ratio at each t. Power ratio is pointwise AP_method(t) / AP_baseline(t)
within a family (e-LORD for LORD-type, e-SAFFRON for SAFFRON-type).

FDR(t) = E[ V(t) / (R(t) ∨ 1) ] with V the false rejections among 1..t.

These helpers never leak into wealth updates; they are evaluation-only.
Temporary local copy: idea_C.shared is owned by another agent.
"""

from __future__ import annotations

import numpy as np

try:  # optional shared metrics if the other agent lands a stable API
    from idea_C.shared.metrics import fdp as _shared_fdp  # type: ignore
except Exception:  # pragma: no cover - shared is empty at this commit
    _shared_fdp = None


def fdp_path(rejected: np.ndarray, is_null: np.ndarray) -> np.ndarray:
    rejected = np.asarray(rejected, dtype=bool)
    is_null = np.asarray(is_null, dtype=bool)
    if rejected.shape != is_null.shape:
        raise ValueError("rejected and is_null shapes differ")
    false_disc = np.cumsum(rejected & is_null, dtype=float)
    n_rej = np.cumsum(rejected, dtype=float)
    return false_disc / np.maximum(n_rej, 1.0)


def fdr_terminal(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """Single-stream terminal FDP (unbiased for FDR after averaging trials)."""
    path = fdp_path(rejected, is_null)
    if path.size == 0:
        return 0.0
    return float(path[-1])


def average_power_path(rejected: np.ndarray, is_null: np.ndarray) -> np.ndarray:
    """TPR(t) = true discoveries up to t / non-nulls arrived by t.

    If |H_1(t)|=0 the ratio is undefined and we store NaN (not 0 or 1).
    """
    rejected = np.asarray(rejected, dtype=bool)
    is_null = np.asarray(is_null, dtype=bool)
    tp = np.cumsum(rejected & ~is_null, dtype=float)
    n_alt = np.cumsum(~is_null, dtype=float)
    out = np.full(rejected.shape, np.nan, dtype=float)
    np.divide(tp, n_alt, out=out, where=n_alt > 0)
    return out


def average_power_terminal(rejected: np.ndarray, is_null: np.ndarray) -> float:
    path = average_power_path(rejected, is_null)
    finite = path[np.isfinite(path)]
    if finite.size == 0:
        return float("nan")
    return float(path[-1]) if np.isfinite(path[-1]) else float("nan")


def power_ratio(power_method: np.ndarray, power_baseline: np.ndarray) -> np.ndarray:
    method = np.asarray(power_method, dtype=float)
    baseline = np.asarray(power_baseline, dtype=float)
    out = np.full(method.shape, np.nan, dtype=float)
    np.divide(method, baseline, out=out, where=np.isfinite(baseline) & (baseline > 0))
    return out
