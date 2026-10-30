"""Linear-model observed design Gramian and related information summaries.

The information matrix used throughout Idea B is the observed design Gramian
of a linear model

    W_T = sum_t S_t phi_t phi_t^T

and the conditional-expectation version

    W_T^{cond} = sum_t E[S_t | phi_t, policy_t] phi_t phi_t^T.

These are not an abstract "observability Gramian" of a dynamical system.
They measure how much of the linear feature space has actually been seen.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-12


def _as_2d_features(phi: Any) -> np.ndarray:
    array = np.asarray(phi, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("phi must be a 1-D or 2-D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("phi must be non-empty")
    return array


def _as_mask(mask: Any, n: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.size == 0:
        raise ValueError("observation mask must be non-empty")
    if array.dtype == bool:
        values = np.array(np.reshape(array, -1), dtype=float, copy=True)
    else:
        values = np.asarray(array, dtype=float).reshape(-1)
        valid = np.isfinite(values) & ((values == 0.0) | (values == 1.0))
        if not np.all(valid):
            raise ValueError("observation mask must contain only 0/1 values")
    if values.shape[0] != n:
        raise ValueError(f"observation mask length {values.shape[0]} != n={n}")
    return values


def _as_weights(weights: Any, n: int, name: str) -> np.ndarray:
    array = np.asarray(weights, dtype=float).reshape(-1)
    if array.shape[0] != n:
        raise ValueError(f"{name} length {array.shape[0]} != n={n}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def observed_design_gramian(phi: Any, observed_mask: Any) -> np.ndarray:
    """Return ``sum_i S_i phi_i phi_i^T`` for a batch.

    ``observed_mask`` is the realized observation indicator ``S``.
    """
    features = _as_2d_features(phi)
    mask = _as_mask(observed_mask, features.shape[0])
    weighted = features * mask[:, None]
    return weighted.T @ features


def accumulate_observed_gramian(
    gramian: Any,
    phi: Any,
    observed_mask: Any,
) -> np.ndarray:
    """Add a batch contribution to a running observed design Gramian ``W``."""
    current = np.asarray(gramian, dtype=float)
    increment = observed_design_gramian(phi, observed_mask)
    if current.shape != increment.shape:
        raise ValueError(
            f"gramian shape {current.shape} does not match increment {increment.shape}"
        )
    return current + increment


def accumulate_conditional_gramian(
    gramian: Any,
    phi: Any,
    observation_probability: Any,
) -> np.ndarray:
    """Add ``sum_i q_i phi_i phi_i^T`` to the conditional design Gramian.

    ``observation_probability`` is ``q_i = E[S_i | phi_i, policy]`` for the
    current batch, not an arbitrary matrix of scores.
    """
    features = _as_2d_features(phi)
    weights = _as_weights(observation_probability, features.shape[0], "observation_probability")
    current = np.asarray(gramian, dtype=float)
    weighted = features * weights[:, None]
    increment = weighted.T @ features
    if current.shape != increment.shape:
        raise ValueError(
            f"gramian shape {current.shape} does not match increment {increment.shape}"
        )
    return current + increment


def eigenspectrum(gramian: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending eigenvalues and corresponding eigenvectors of a Gramian."""
    matrix = np.asarray(gramian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gramian must be a square matrix")
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return values, vectors


def lambda_min(gramian: Any) -> float:
    """Smallest eigenvalue of a design Gramian."""
    values, _vectors = eigenspectrum(gramian)
    return float(values[0])


def condition_number(gramian: Any, ridge: float = 0.0) -> float:
    """``lambda_max / lambda_min`` of ``gramian + ridge * I``.

    Returns ``+inf`` when the smallest eigenvalue is non-positive after ridge.
    """
    matrix = np.asarray(gramian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gramian must be a square matrix")
    ridge_f = float(ridge)
    if ridge_f < 0.0:
        raise ValueError("ridge must be >= 0")
    values, _vectors = eigenspectrum(matrix + ridge_f * np.eye(matrix.shape[0]))
    smallest = float(values[0])
    largest = float(values[-1])
    if smallest <= EPS:
        return float("inf")
    return float(largest / smallest)


def skill_information(
    gramian: Any,
    skill_basis: Any | None = None,
) -> np.ndarray:
    """Per-direction information ``e_k^T W e_k``.

    Default basis is the coordinate axes, which match the synthetic skill
    one-hot features. This is a diagonal readout of the observed design
    Gramian, not a different information concept.
    """
    matrix = np.asarray(gramian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gramian must be a square matrix")
    dim = matrix.shape[0]
    if skill_basis is None:
        basis = np.eye(dim)
    else:
        basis = np.asarray(skill_basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != dim:
            raise ValueError("skill_basis must have shape (d, n_skills)")
    return np.einsum("ij,jk,ik->k", basis, matrix, basis)
