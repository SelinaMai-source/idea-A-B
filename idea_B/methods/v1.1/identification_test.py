"""Explicit likelihood and moment tests for observational equivalence (B-Env2).

Two latent states are observationally equivalent when they induce the same
Gaussian likelihood on the currently observed support and differ on a
coordinate that the support does not identify. Plots are not the proof:
the equal-likelihood identity and a minimax parameter-error lower bound
are computed from the observed-data log-likelihood and Fisher information.

A randomized audit restores identifiability when it puts positive Fisher
information on the missing coordinate. A structural unrecoverable case is
the same coordinate remaining in the kernel of every observed feature,
including audited rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-12

DEFAULT_THETA_A = np.array([1.2, 1.05, 1.0], dtype=float)
DEFAULT_THETA_B = np.array([1.2, 1.05, -0.5], dtype=float)
DEFAULT_WEAK = 2
DEFAULT_NOISE_STD = 0.15


def two_latent_states(
    d: int = 3,
    weak_skill: int = DEFAULT_WEAK,
    *,
    theta_a: Any | None = None,
    theta_b: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two parameter vectors that agree off ``weak_skill`` by construction."""
    if theta_a is None:
        a = np.zeros(int(d), dtype=float)
        b = np.zeros(int(d), dtype=float)
        shared = min(int(d), DEFAULT_THETA_A.shape[0])
        a[:shared] = DEFAULT_THETA_A[:shared]
        b[:shared] = DEFAULT_THETA_A[:shared]
        b[int(weak_skill)] = float(DEFAULT_THETA_B[min(int(weak_skill), DEFAULT_THETA_B.shape[0] - 1)])
        if int(d) != DEFAULT_THETA_A.shape[0]:
            b[int(weak_skill)] = -0.5
        return a, b
    a = np.asarray(theta_a, dtype=float).reshape(-1).copy()
    if theta_b is None:
        b = a.copy()
        b[int(weak_skill)] = -0.5 if a[int(weak_skill)] >= 0.0 else 0.5
        return a, b
    b = np.asarray(theta_b, dtype=float).reshape(-1).copy()
    if a.shape[0] != b.shape[0]:
        raise ValueError("theta_a and theta_b must have the same length")
    return a, b


def _observed_index(mask: Any, n: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype == bool:
        values = np.array(np.reshape(array, -1), dtype=bool, copy=True)
    else:
        values = np.asarray(array, dtype=float).reshape(-1) > 0.5
    if values.shape[0] != n:
        raise ValueError("mask length must match batch")
    return np.flatnonzero(values)


def observed_log_likelihood(
    theta: Any,
    phi: Any,
    y: Any,
    observed_mask: Any,
    noise_std: float = DEFAULT_NOISE_STD,
) -> float:
    """Gaussian log-likelihood of the observed rows under ``y | phi ~ N(phi θ, σ²)``."""
    features = np.asarray(phi, dtype=float)
    outcomes = np.asarray(y, dtype=float).reshape(-1)
    param = np.asarray(theta, dtype=float).reshape(-1)
    if features.ndim != 2:
        raise ValueError("phi must be 2-D")
    if outcomes.shape[0] != features.shape[0]:
        raise ValueError("y length must match phi")
    if param.shape[0] != features.shape[1]:
        raise ValueError("theta length must match feature dimension")
    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be > 0")
    index = _observed_index(observed_mask, features.shape[0])
    n_obs = int(index.shape[0])
    if n_obs == 0:
        return 0.0
    residual = features[index] @ param - outcomes[index]
    return float(
        -0.5 * n_obs * np.log(2.0 * np.pi * sigma**2)
        - 0.5 * np.sum(residual**2) / (sigma**2)
    )


def observed_score(
    theta: Any,
    phi: Any,
    y: Any,
    observed_mask: Any,
    noise_std: float = DEFAULT_NOISE_STD,
) -> np.ndarray:
    """Score ``(1/σ²) sum_i S_i phi_i (y_i - phi_i^T θ)``."""
    features = np.asarray(phi, dtype=float)
    outcomes = np.asarray(y, dtype=float).reshape(-1)
    param = np.asarray(theta, dtype=float).reshape(-1)
    sigma = float(noise_std)
    index = _observed_index(observed_mask, features.shape[0])
    if index.shape[0] == 0:
        return np.zeros(features.shape[1], dtype=float)
    residual = outcomes[index] - features[index] @ param
    return (features[index].T @ residual) / (sigma**2)


def fisher_information(
    phi: Any,
    observed_mask: Any,
    noise_std: float = DEFAULT_NOISE_STD,
) -> np.ndarray:
    """Observed-data Fisher information ``(1/σ²) sum_i S_i phi_i phi_i^T``."""
    features = np.asarray(phi, dtype=float)
    sigma = float(noise_std)
    if sigma <= 0.0:
        raise ValueError("noise_std must be > 0")
    index = _observed_index(observed_mask, features.shape[0])
    if index.shape[0] == 0:
        return np.zeros((features.shape[1], features.shape[1]), dtype=float)
    observed = features[index]
    return (observed.T @ observed) / (sigma**2)


def likelihoods_equal(
    theta_a: Any,
    theta_b: Any,
    phi: Any,
    y: Any,
    observed_mask: Any,
    noise_std: float = DEFAULT_NOISE_STD,
    *,
    atol: float = 1e-10,
) -> bool:
    ll_a = observed_log_likelihood(theta_a, phi, y, observed_mask, noise_std)
    ll_b = observed_log_likelihood(theta_b, phi, y, observed_mask, noise_std)
    return bool(np.isfinite(ll_a) and np.isfinite(ll_b) and abs(ll_a - ll_b) <= float(atol))


def coordinate_in_observed_kernel(
    fisher: Any,
    coordinate: int,
    *,
    tol: float = 1e-10,
) -> bool:
    """True when ``e_k`` is (numerically) in the kernel of the Fisher matrix."""
    matrix = np.asarray(fisher, dtype=float)
    k = int(coordinate)
    if k < 0 or k >= matrix.shape[0]:
        raise ValueError("coordinate out of range")
    direction = np.zeros(matrix.shape[0], dtype=float)
    direction[k] = 1.0
    quadratic = float(direction @ matrix @ direction)
    return bool(quadratic <= float(tol))


def minimax_identification_lower_bound(theta_a: Any, theta_b: Any) -> float:
    """``||θ_A - θ_B||_2 / 2``.

    Any estimator that cannot distinguish two observationally equivalent
    states has max parameter error at least this large on the pair.
    """
    a = np.asarray(theta_a, dtype=float).reshape(-1)
    b = np.asarray(theta_b, dtype=float).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("theta_a and theta_b must have the same length")
    return float(0.5 * np.linalg.norm(a - b))


def structural_unrecoverable(phi: Any, weak_skill: int, *, tol: float = 1e-12) -> bool:
    """True when every row has a vanishing component on ``weak_skill``."""
    features = np.asarray(phi, dtype=float)
    if features.ndim != 2:
        raise ValueError("phi must be 2-D")
    k = int(weak_skill)
    if k < 0 or k >= features.shape[1]:
        raise ValueError("weak_skill out of range")
    return bool(np.max(np.abs(features[:, k])) <= float(tol))


def audit_restores_identifiability(
    phi: Any,
    natural_mask: Any,
    audit_mask: Any,
    weak_skill: int,
    noise_std: float = DEFAULT_NOISE_STD,
    *,
    fisher_tol: float = 1e-10,
) -> dict[str, Any]:
    """Compare Fisher information on ``weak_skill`` with and without the audit.

    Restoration requires that the natural-only Fisher has the weak coordinate
    in its kernel, and that the audited Fisher does not. If the coordinate is
    absent from every feature row, the case is structural unrecoverable.
    """
    features = np.asarray(phi, dtype=float)
    natural = np.asarray(natural_mask, dtype=bool).reshape(-1)
    audit = np.asarray(audit_mask, dtype=bool).reshape(-1)
    observed = natural | audit
    fisher_nat = fisher_information(features, natural, noise_std)
    fisher_obs = fisher_information(features, observed, noise_std)
    k = int(weak_skill)
    unrecoverable = structural_unrecoverable(features, k)
    nat_kernel = coordinate_in_observed_kernel(fisher_nat, k, tol=fisher_tol)
    obs_kernel = coordinate_in_observed_kernel(fisher_obs, k, tol=fisher_tol)
    restored = bool(nat_kernel and (not obs_kernel) and (not unrecoverable))
    return {
        "structural_unrecoverable": unrecoverable,
        "weak_in_natural_kernel": nat_kernel,
        "weak_in_observed_kernel": obs_kernel,
        "restored_by_audit": restored,
        "fisher_weak_natural": float(fisher_nat[k, k]),
        "fisher_weak_observed": float(fisher_obs[k, k]),
        "n_natural": int(np.sum(natural)),
        "n_audited": int(np.sum(audit)),
        "n_observed": int(np.sum(observed)),
    }


def observational_equivalence_report(
    theta_a: Any,
    theta_b: Any,
    phi: Any,
    y: Any,
    observed_mask: Any,
    weak_skill: int,
    noise_std: float = DEFAULT_NOISE_STD,
    *,
    ll_atol: float = 1e-10,
    fisher_tol: float = 1e-10,
) -> dict[str, Any]:
    """Package the explicit likelihood/moment evidence for one batch."""
    a = np.asarray(theta_a, dtype=float).reshape(-1)
    b = np.asarray(theta_b, dtype=float).reshape(-1)
    fisher = fisher_information(phi, observed_mask, noise_std)
    k = int(weak_skill)
    equal = likelihoods_equal(a, b, phi, y, observed_mask, noise_std, atol=ll_atol)
    kernel = coordinate_in_observed_kernel(fisher, k, tol=fisher_tol)
    bound = minimax_identification_lower_bound(a, b)
    score_a = observed_score(a, phi, y, observed_mask, noise_std)
    score_b = observed_score(b, phi, y, observed_mask, noise_std)
    return {
        "likelihoods_equal": equal,
        "loglik_a": observed_log_likelihood(a, phi, y, observed_mask, noise_std),
        "loglik_b": observed_log_likelihood(b, phi, y, observed_mask, noise_std),
        "weak_coordinate_unidentified": kernel,
        "fisher_weak": float(fisher[k, k]),
        "identification_error_lower_bound": bound,
        "score_gap_l2": float(np.linalg.norm(score_a - score_b)),
        "positive_lower_bound": bool(bound > 0.0),
        "proof": "gaussian_observed_likelihood_and_fisher",
    }
