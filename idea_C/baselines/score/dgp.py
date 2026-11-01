"""Independent Gaussian mixture DGP from SCORE §6.1.

    θ_t iid ~ Ber(π1)
    μ_t | θ_t ~ (1-θ_t) δ_0 + θ_t N(3, 5)     # N(mean, variance)
    X_t | μ_t ~ N(μ_t, 1)
    e_t = φ(X_t; 3, 6) / φ(X_t; 0, 1)

Paper-scale: T=1000, 500 repetitions, π1 ∈ {0.3, 0.8}, α=0.05.
This module only *generates* one stream. Runners must refuse 500 reps
unless an explicit paper-scale flag is set.

Ground-truth (θ, μ) is returned for evaluation only and must not be fed
into wealth updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAPER_T = 1000
PAPER_REPS = 500
PAPER_PI1 = (0.3, 0.8)
PAPER_ALPHA = 0.05
PAPER_OMEGA = 0.05
PAPER_LAMBDA = 0.5


@dataclass
class GaussianMixtureStream:
    x: np.ndarray
    evalues: np.ndarray
    is_null: np.ndarray
    mu: np.ndarray
    theta: np.ndarray
    pi1: float
    seed: int
    T: int


def likelihood_ratio_evalue(x: np.ndarray) -> np.ndarray:
    """e = φ(x; mean=3, var=6) / φ(x; mean=0, var=1)."""
    x = np.asarray(x, dtype=float)
    log_e = (
        -0.5 * np.log(6.0)
        - (x - 3.0) ** 2 / 12.0
        + (x ** 2) / 2.0
    )
    return np.exp(log_e)


def independent_gaussian_mixture(
    T: int,
    pi1: float,
    seed: int,
) -> GaussianMixtureStream:
    if T < 1:
        raise ValueError("T must be positive")
    if not 0.0 <= pi1 <= 1.0:
        raise ValueError("pi1 must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    theta = rng.binomial(1, pi1, size=T).astype(int)
    mu = np.zeros(T, dtype=float)
    n_alt = int(np.sum(theta))
    if n_alt:
        mu[theta == 1] = rng.normal(loc=3.0, scale=np.sqrt(5.0), size=n_alt)
    x = rng.normal(loc=mu, scale=1.0)
    evalues = likelihood_ratio_evalue(x)
    return GaussianMixtureStream(
        x=x,
        evalues=evalues,
        is_null=(theta == 0),
        mu=mu,
        theta=theta,
        pi1=float(pi1),
        seed=int(seed),
        T=int(T),
    )
