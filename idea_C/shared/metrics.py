"""FDR / TPR / TPP / Power / SupFDR for Idea C v0.

These are *evaluation* functionals. They may look at ground-truth null /
non-null labels. Sampling rules and e-processes must never import this
module for decisions.

Definitions (single rejection set ``S``, truth partition ``H0``, ``H1``):

- **FDP** = ``|S ∩ H0| / max(|S|, 1)``
- **FDR** = ``E[FDP]``. In code, ``fdr`` on one set is the realized FDP;
  Monte Carlo FDR is the mean of per-trial FDP.
- **TPP** = ``|S ∩ H1| / max(|H1|, 1)`` (true positive *proportion*,
  AAAI Def. 2 without the outer expectation; e-PS §6 reports averaged TPP).
- **TPR** = ``E[TPP]``. On one set, ``tpr`` equals TPP. When ``|H1| = 0``,
  both are 0.0 (no true positives exist).
- **Power** = same as TPR / TPP on a fixed set: the fraction of non-nulls
  that are rejected. Streaming papers (SCORE / UAI) average this over time;
  we expose ``mean_power_path`` for that.
- **SupFDR** = ``E[ sup_t FDP(S_t) ]``. Requires a *path* of rejection sets.
  Do not mix SupFDR with FDR in tables.

``samples_at_target_tpr`` is the first time index ``t`` (1-based sample
count) at which TPP ≥ ``1 - delta``. Returns NaN if the target is never
met — callers must serialize that as missing / ``NOT_RUN``, never as 0.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_bool_mask(values: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == bool and arr.shape == (n,):
        return arr
    if arr.shape == (n,):
        return arr.astype(bool)
    raise ValueError(f"{name} must have shape ({n},)")


def fdp(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """Realized false discovery proportion of one rejection set."""
    rej = np.asarray(rejected, dtype=bool).reshape(-1)
    null = _as_bool_mask(np.asarray(is_null), rej.size, "is_null")
    n_rej = int(rej.sum())
    if n_rej == 0:
        return 0.0
    return float(np.count_nonzero(rej & null) / n_rej)


def fdr(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """FDR on a single set equals FDP; average across trials separately."""
    return fdp(rejected, is_null)


def tpp(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """True positive proportion ``|S ∩ H1| / max(|H1|, 1)``."""
    rej = np.asarray(rejected, dtype=bool).reshape(-1)
    null = _as_bool_mask(np.asarray(is_null), rej.size, "is_null")
    n_alt = int((~null).sum())
    if n_alt == 0:
        return 0.0
    return float(np.count_nonzero(rej & ~null) / n_alt)


def tpr(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """Single-set TPR (equals TPP). Monte Carlo TPR is the mean of TPP."""
    return tpp(rejected, is_null)


def power(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """Fraction of non-nulls rejected; identical to TPP on one set."""
    return tpp(rejected, is_null)


def mean_power_path(rejected_path: np.ndarray, is_null: np.ndarray) -> float:
    """Time-average of Power along a ``(T, K)`` rejection path."""
    path = np.asarray(rejected_path, dtype=bool)
    if path.ndim != 2:
        raise ValueError("rejected_path must be (T, K)")
    if path.size == 0:
        return 0.0
    return float(np.mean([power(row, is_null) for row in path]))


def sup_fdp(rejected_path: np.ndarray, is_null: np.ndarray) -> float:
    """``sup_t FDP(S_t)`` along a ``(T, K)`` path (one trial)."""
    path = np.asarray(rejected_path, dtype=bool)
    if path.ndim != 2:
        raise ValueError("rejected_path must be (T, K)")
    if path.size == 0:
        return 0.0
    return float(max(fdp(row, is_null) for row in path))


def sup_fdr(rejected_paths: Iterable[np.ndarray], is_null: np.ndarray) -> float:
    """Monte Carlo SupFDR: mean of per-trial ``sup_t FDP``."""
    values = [sup_fdp(path, is_null) for path in rejected_paths]
    if not values:
        return float("nan")
    return float(np.mean(values))


def samples_at_target_tpr(
    tpp_path: np.ndarray,
    delta: float = 0.05,
    sample_index: np.ndarray | None = None,
) -> float:
    """First sample count at which TPP ≥ ``1 - delta``.

    Returns ``nan`` if the target is never reached so serializers can leave
    the cell as ``NOT_RUN`` rather than writing a fake 0.
    """
    path = np.asarray(tpp_path, dtype=float).reshape(-1)
    target = 1.0 - float(delta)
    hits = np.flatnonzero(path >= target)
    if hits.size == 0:
        return float("nan")
    idx = int(hits[0])
    if sample_index is None:
        return float(idx + 1)
    index = np.asarray(sample_index, dtype=float).reshape(-1)
    return float(index[idx])
