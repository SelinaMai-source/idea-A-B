"""Parameter updates from observed linear-model data (Idea B v1.1).

Ridge uses the cumulative observed design Gramian ``W`` and the cumulative
vector ``b = sum S y phi``. SGD uses only the current observed batch.
Discounting of stale information is applied to ``W`` by the runner before
the ridge solve when a controller requests it; this module does not change
the v1.0 update algebra.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_theta(theta: Any, dim: int) -> np.ndarray:
    array = np.asarray(theta, dtype=float).reshape(-1)
    if array.shape[0] != dim:
        raise ValueError(f"theta length {array.shape[0]} != d={dim}")
    return array


def ridge_update(gramian: Any, xy_vector: Any, ridge_lambda: float) -> np.ndarray:
    """Return ``(W + ridge * I)^{-1} b`` for the linear model."""
    matrix = np.asarray(gramian, dtype=float)
    vector = np.asarray(xy_vector, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gramian must be square")
    if vector.shape[0] != matrix.shape[0]:
        raise ValueError("xy_vector length must match gramian dimension")
    ridge = float(ridge_lambda)
    if ridge < 0.0:
        raise ValueError("ridge_lambda must be >= 0")
    regularized = matrix + ridge * np.eye(matrix.shape[0])
    try:
        return np.linalg.solve(regularized, vector)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized) @ vector


def sgd_update(
    theta: Any,
    phi: Any,
    y: Any,
    observed_mask: Any,
    learning_rate: float,
) -> np.ndarray:
    """One pass of squared-loss SGD on currently observed rows."""
    features = np.asarray(phi, dtype=float)
    if features.ndim != 2:
        raise ValueError("phi must be 2-D")
    outcomes = np.asarray(y, dtype=float).reshape(-1)
    mask = np.asarray(observed_mask, dtype=float).reshape(-1)
    if outcomes.shape[0] != features.shape[0] or mask.shape[0] != features.shape[0]:
        raise ValueError("phi, y, and observed_mask must share the batch length")
    updated = _as_theta(theta, features.shape[1]).copy()
    rate = float(learning_rate)
    if rate < 0.0:
        raise ValueError("learning_rate must be >= 0")
    observed_index = np.flatnonzero(mask > 0.5)
    for index in observed_index:
        residual = float(features[index] @ updated - outcomes[index])
        updated -= rate * residual * features[index]
    return updated


def accumulate_xy(xy_vector: Any, phi: Any, y: Any, observed_mask: Any) -> np.ndarray:
    """Add ``sum_i S_i y_i phi_i`` to the running right-hand side."""
    features = np.asarray(phi, dtype=float)
    outcomes = np.asarray(y, dtype=float).reshape(-1)
    mask = np.asarray(observed_mask, dtype=float).reshape(-1)
    current = np.asarray(xy_vector, dtype=float).reshape(-1)
    if features.ndim != 2:
        raise ValueError("phi must be 2-D")
    if current.shape[0] != features.shape[1]:
        raise ValueError("xy_vector length must match feature dimension")
    if outcomes.shape[0] != features.shape[0] or mask.shape[0] != features.shape[0]:
        raise ValueError("phi, y, and observed_mask must share the batch length")
    return current + features.T @ (mask * outcomes)


def discount_xy(xy_vector: Any, rho: float) -> np.ndarray:
    factor = float(rho)
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("information discount rho must be finite and >= 0")
    return factor * np.asarray(xy_vector, dtype=float)
