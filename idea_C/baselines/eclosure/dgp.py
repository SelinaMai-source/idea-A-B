"""UAI 2026 Section 4 local-dependence Gaussian protocol.

Venue: UAI 2026 (**NOT CCF-A Main**).

Paper DGP (source of truth for P0):
    m=200, μ0=0, μ1=3, lag L=100,
    Σ_{ij} = 0.5^{|i-j|} if |i-j|≤L else 0 (PSD checked numerically),
    e-value = Gaussian likelihood ratio, p-value = one-sided Gaussian CDF,
    γ_t = 1/(t(t+1)), δ=0.1, 200 trials.

Mapping to evalue-omt:
    Official experiment name ``lag_comp`` lives in AISTATS 2024
    *Online multiple testing with e-values* (neilzxu/evalue-omt).
    That experiment uses a *different* DGP (bounded sequential samples per
    hypothesis, δ=0.3, hypotheses=1000, max_seq_len=200, lags on a grid).
    P0 must use the UAI Section 4 DGP above. ``lag_comp`` is recorded only
    as the related official experiment name, not as a drop-in simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

UAI_VENUE_NOTE = "UAI 2026 (NOT CCF-A Main)"
RELATED_OFFICIAL_EXP = "lag_comp"  # AISTATS 2024 evalue-omt, not this DGP
PAPER_M = 200
PAPER_TRIALS = 200
PAPER_LAG = 100
PAPER_RHO = 0.5
PAPER_DELTA = 0.1
PAPER_MU0 = 0.0
PAPER_MU1 = 3.0


@dataclass
class LocalDependenceStream:
    x: np.ndarray
    evalues: np.ndarray
    pvalues: np.ndarray
    is_null: np.ndarray
    cov: np.ndarray
    psd_adjusted: bool
    min_eigenvalue: float
    pi1: float
    seed: int
    lag: int
    rho: float
    source: str
    related_official_exp: str = RELATED_OFFICIAL_EXP


def banded_ar1_cov(m: int, lag: int, rho: float) -> np.ndarray:
    idx = np.arange(m)
    dist = np.abs(idx[:, None] - idx[None, :])
    cov = np.where(dist <= lag, rho ** dist.astype(float), 0.0)
    return cov


def _nearest_psd(cov: np.ndarray) -> tuple[np.ndarray, float, bool]:
    eigs = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigs))
    adjusted = False
    out = cov
    if min_eig < 0.0:
        out = cov + (-min_eig + 1e-8) * np.eye(cov.shape[0])
        adjusted = True
    else:
        out = cov + 1e-12 * np.eye(cov.shape[0])
    return out, min_eig, adjusted


def gaussian_lr_evalue(x: np.ndarray, mu0: float, mu1: float) -> np.ndarray:
    """Marginal N(μ,1) likelihood ratio. Valid e-value if X_i is marginally N(μ0,1)."""
    x = np.asarray(x, dtype=float)
    delta = mu1 - mu0
    return np.exp(delta * (x - mu0) - 0.5 * delta * delta)


def onesided_gaussian_pvalue(x: np.ndarray, mu0: float) -> np.ndarray:
    return 1.0 - norm.cdf(np.asarray(x, dtype=float), loc=mu0, scale=1.0)


def local_dependent_gaussian(
    m: int,
    pi1: float,
    seed: int,
    *,
    mu0: float = PAPER_MU0,
    mu1: float = PAPER_MU1,
    lag: int = PAPER_LAG,
    rho: float = PAPER_RHO,
) -> LocalDependenceStream:
    """UAI 2026 Section 4 local-dependence Gaussian (not evalue-omt lag_comp)."""
    if m < 1:
        raise ValueError("m must be positive")
    rng = np.random.default_rng(seed)
    cov0, min_eig, adjusted = _nearest_psd(banded_ar1_cov(m, lag, rho))
    is_alt = rng.random(m) < pi1
    mean = np.where(is_alt, mu1, mu0)
    x = rng.multivariate_normal(mean=mean, cov=cov0)
    return LocalDependenceStream(
        x=x,
        evalues=gaussian_lr_evalue(x, mu0, mu1),
        pvalues=onesided_gaussian_pvalue(x, mu0),
        is_null=~is_alt,
        cov=cov0,
        psd_adjusted=adjusted,
        min_eigenvalue=min_eig,
        pi1=float(pi1),
        seed=int(seed),
        lag=int(lag),
        rho=float(rho),
        source="UAI 2026 Section 4 / Appendix C.3 (NOT CCF-A Main)",
    )
