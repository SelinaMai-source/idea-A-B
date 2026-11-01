"""FDR vs SupFDR (UAI 2026 Xu, Fischer, Ramdas). They are distinct.

    FDP_t = V_t / (R_t ∨ 1)
    FDR   = E[FDP_T]                 (terminal, or E[FDP_t] at a fixed t)
    SupFDR = E[ sup_{t} FDP_t ]      (Fischer / Xu online criterion)

Never mix the two. Temporary local copy: idea_C.shared is owned elsewhere.
"""

from __future__ import annotations

import numpy as np

try:
    from idea_C.shared.metrics import fdp as _shared_fdp  # type: ignore
except Exception:  # pragma: no cover
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
    """Single-stream terminal FDP; average over trials to estimate FDR."""
    path = fdp_path(rejected, is_null)
    return float(path[-1]) if path.size else 0.0


def sup_fdp(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """Single-stream sup_t FDP_t; average over trials to estimate SupFDR."""
    path = fdp_path(rejected, is_null)
    return float(np.max(path)) if path.size else 0.0


def empirical_fdr(rejected_mat: np.ndarray, is_null_mat: np.ndarray) -> float:
    rejected_mat = np.asarray(rejected_mat, dtype=bool)
    is_null_mat = np.asarray(is_null_mat, dtype=bool)
    terminals = [
        fdr_terminal(rejected_mat[i], is_null_mat[i])
        for i in range(rejected_mat.shape[0])
    ]
    return float(np.mean(terminals)) if terminals else 0.0


def empirical_supfdr(rejected_mat: np.ndarray, is_null_mat: np.ndarray) -> float:
    rejected_mat = np.asarray(rejected_mat, dtype=bool)
    is_null_mat = np.asarray(is_null_mat, dtype=bool)
    values = [
        sup_fdp(rejected_mat[i], is_null_mat[i])
        for i in range(rejected_mat.shape[0])
    ]
    return float(np.mean(values)) if values else 0.0


def average_power_terminal(rejected: np.ndarray, is_null: np.ndarray) -> float:
    rejected = np.asarray(rejected, dtype=bool)
    is_null = np.asarray(is_null, dtype=bool)
    n_alt = int(np.sum(~is_null))
    if n_alt == 0:
        return float("nan")
    return float(np.sum(rejected & ~is_null) / n_alt)
