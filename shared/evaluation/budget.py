"""Audit-budget accounting.

``expected_audit_fraction`` is the mean of the known propensities ``q``.
``realized_audit_fraction`` is the mean of the realized audit indicators.
``assert_budget`` checks a scalar realized fraction against a scalar target.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _indicator_values(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.dtype == bool:
        return np.array(np.reshape(arr, -1), dtype=float, copy=True)
    numeric = np.asarray(arr, dtype=float).reshape(-1)
    valid = np.isfinite(numeric) & ((numeric == 0.0) | (numeric == 1.0))
    if not np.all(valid):
        raise ValueError(f"{name} must contain only 0/1 values")
    return numeric


def _scalar(name: str, value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.shape != ():
        raise ValueError(f"{name} must be a scalar")
    out = float(arr)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def expected_audit_fraction(q: Any) -> float:
    """Expected audit fraction ``mean(q)`` for known per-unit propensities.

    A scalar ``q`` is the expected fraction of a constant-propensity audit.
    """
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("q must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("q must be finite")
    return float(np.mean(arr))


def realized_audit_fraction(a: Any) -> float:
    """Realized audit fraction ``mean(a)`` for 0/1 audit indicators."""
    return float(np.mean(_indicator_values("a", a)))


def assert_budget(realized: Any, target: Any, tol: Any) -> None:
    """Raise ``AssertionError`` when ``abs(realized - target) > tol``.

    The comparison is inclusive: a gap equal to ``tol`` passes. ``realized``,
    ``target``, and ``tol`` are scalars. ``tol`` must be finite and ``>= 0``.
    """
    realized_f = _scalar("realized", realized)
    target_f = _scalar("target", target)
    tol_f = _scalar("tol", tol)
    if tol_f < 0.0:
        raise ValueError("tol must be >= 0")
    gap = abs(realized_f - target_f)
    if gap > tol_f:
        raise AssertionError(
            "audit budget mismatch: "
            f"|realized - target|={gap} > tol={tol_f} "
            f"(realized={realized_f}, target={target_f})"
        )
