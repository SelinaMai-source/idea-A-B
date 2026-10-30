"""Linear-model observed design Gramian and information summaries for Idea B v1.1.

The information matrix remains the observed design Gramian of a linear model

    W_T = sum_t S_t phi_t phi_t^T

v1.1 also reports three distinct quantities that must not be conflated:

(a) realized observed information: W from the realized mask S
(b) information from all candidates: W from every row in the batch (S = 1)
(c) the conditional-expectation version: W from q = E[S | phi, policy, audit]

A sliding window of recent increments supplies lambda_min, condition number,
numerical rank, and skill-level information. A larger eigenvalue is a
descriptive statistic, not a success criterion.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

EPS = 1e-12
RANK_TOL = 1e-8


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


def _symmetrize(gramian: Any) -> np.ndarray:
    matrix = np.asarray(gramian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gramian must be a square matrix")
    return 0.5 * (matrix + matrix.T)


def observed_design_gramian(phi: Any, observed_mask: Any) -> np.ndarray:
    """Return ``sum_i S_i phi_i phi_i^T`` for a batch (realized information)."""
    features = _as_2d_features(phi)
    mask = _as_mask(observed_mask, features.shape[0])
    weighted = features * mask[:, None]
    return weighted.T @ features


def all_candidates_gramian(phi: Any) -> np.ndarray:
    """Return ``sum_i phi_i phi_i^T`` over every candidate, ignoring missingness."""
    features = _as_2d_features(phi)
    return features.T @ features


def conditional_design_gramian(phi: Any, observation_probability: Any) -> np.ndarray:
    """Return ``sum_i q_i phi_i phi_i^T`` for known observation probabilities."""
    features = _as_2d_features(phi)
    weights = _as_weights(observation_probability, features.shape[0], "observation_probability")
    weighted = features * weights[:, None]
    return weighted.T @ features


def accumulate_observed_gramian(
    gramian: Any,
    phi: Any,
    observed_mask: Any,
) -> np.ndarray:
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
    current = np.asarray(gramian, dtype=float)
    increment = conditional_design_gramian(phi, observation_probability)
    if current.shape != increment.shape:
        raise ValueError(
            f"gramian shape {current.shape} does not match increment {increment.shape}"
        )
    return current + increment


def accumulate_candidates_gramian(gramian: Any, phi: Any) -> np.ndarray:
    current = np.asarray(gramian, dtype=float)
    increment = all_candidates_gramian(phi)
    if current.shape != increment.shape:
        raise ValueError(
            f"gramian shape {current.shape} does not match increment {increment.shape}"
        )
    return current + increment


def discount_gramian(gramian: Any, rho: float) -> np.ndarray:
    """Return ``rho * W``. ``rho=1`` keeps the cumulative Gramian."""
    factor = float(rho)
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("information discount rho must be finite and >= 0")
    return factor * np.asarray(gramian, dtype=float)


def eigenspectrum(gramian: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending eigenvalues and corresponding eigenvectors of a Gramian."""
    symmetric = _symmetrize(gramian)
    values, vectors = np.linalg.eigh(symmetric)
    return values, vectors


def lambda_min(gramian: Any) -> float:
    values, _vectors = eigenspectrum(gramian)
    return float(values[0])


def lambda_max(gramian: Any) -> float:
    values, _vectors = eigenspectrum(gramian)
    return float(values[-1])


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


def numerical_rank(gramian: Any, tol: float = RANK_TOL) -> int:
    """Count eigenvalues strictly larger than ``tol`` after symmetrization."""
    values, _vectors = eigenspectrum(gramian)
    return int(np.sum(values > float(tol)))


def is_psd(gramian: Any, atol: float = 1e-10) -> bool:
    return lambda_min(gramian) >= -float(atol)


def skill_information(
    gramian: Any,
    skill_basis: Any | None = None,
) -> np.ndarray:
    """Per-direction information ``e_k^T W e_k``.

    Default basis is the coordinate axes, matching one-hot synthetic skills.
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


def information_bundle(
    *,
    realized: Any,
    conditional: Any,
    candidates: Any,
    ridge: float = 0.0,
) -> dict[str, Any]:
    """Summaries for the three information matrices. Eigenvalues are not success."""
    realized_m = np.asarray(realized, dtype=float)
    conditional_m = np.asarray(conditional, dtype=float)
    candidates_m = np.asarray(candidates, dtype=float)
    return {
        "lambda_min_realized": lambda_min(realized_m),
        "lambda_min_conditional": lambda_min(conditional_m),
        "lambda_min_candidates": lambda_min(candidates_m),
        "condition_number_realized": condition_number(realized_m, ridge=ridge),
        "condition_number_conditional": condition_number(conditional_m, ridge=ridge),
        "condition_number_candidates": condition_number(candidates_m, ridge=ridge),
        "rank_realized": numerical_rank(realized_m),
        "rank_conditional": numerical_rank(conditional_m),
        "rank_candidates": numerical_rank(candidates_m),
        "skill_information_realized": skill_information(realized_m).tolist(),
        "skill_information_conditional": skill_information(conditional_m).tolist(),
        "skill_information_candidates": skill_information(candidates_m).tolist(),
        "higher_eigenvalue_is_not_success": True,
    }


class SlidingWindowGramian:
    """Running sum of the last ``window`` Gramian increments."""

    def __init__(self, dim: int, window: int) -> None:
        if int(dim) < 1:
            raise ValueError("dim must be >= 1")
        if int(window) < 1:
            raise ValueError("window must be >= 1")
        self.dim = int(dim)
        self.window = int(window)
        self._increments: deque[np.ndarray] = deque()
        self._sum = np.zeros((self.dim, self.dim), dtype=float)

    def push(self, increment: Any) -> np.ndarray:
        matrix = np.asarray(increment, dtype=float)
        if matrix.shape != (self.dim, self.dim):
            raise ValueError(
                f"increment shape {matrix.shape} does not match ({self.dim}, {self.dim})"
            )
        self._increments.append(matrix)
        self._sum = self._sum + matrix
        if len(self._increments) > self.window:
            oldest = self._increments.popleft()
            self._sum = self._sum - oldest
        return self.gramian()

    def gramian(self) -> np.ndarray:
        return np.array(self._sum, dtype=float, copy=True)

    def summaries(self, ridge: float = 0.0) -> dict[str, Any]:
        gramian = self.gramian()
        return {
            "lambda_min": lambda_min(gramian),
            "condition_number": condition_number(gramian, ridge=ridge),
            "rank": numerical_rank(gramian),
            "skill_information": skill_information(gramian).tolist(),
            "n_increments": len(self._increments),
        }
