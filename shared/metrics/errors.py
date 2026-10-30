"""Error of a scalar estimand against one or more estimates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_estimates(estimates: Any) -> np.ndarray:
    arr = np.asarray(estimates, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2 and 1 in arr.shape:
        arr = np.reshape(arr, -1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("estimates must be a non-empty scalar or 1-D vector")
    return arr


def _as_truth(truth: Any) -> float:
    arr = np.asarray(truth, dtype=float)
    if arr.shape != ():
        raise ValueError("truth must be a scalar")
    return float(arr)


def bias(estimates: Any, truth: Any) -> float:
    """``mean(estimates) - truth`` for a scalar truth."""
    return float(np.mean(_as_estimates(estimates)) - _as_truth(truth))


def mae(estimates: Any, truth: Any) -> float:
    """Mean absolute error versus a scalar truth."""
    return float(np.mean(np.abs(_as_estimates(estimates) - _as_truth(truth))))


def rmse(estimates: Any, truth: Any) -> float:
    """Root mean squared error versus a scalar truth."""
    errors = _as_estimates(estimates) - _as_truth(truth)
    return float(np.sqrt(np.mean(np.square(errors))))
