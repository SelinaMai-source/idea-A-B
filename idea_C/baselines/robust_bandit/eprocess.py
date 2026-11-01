"""Robust e-processes from AAAI 2026 Zhou & Liu.

Gaussian / sub-Gaussian (Eq. 5–6)
    ``E_{i,t} = ∏_s exp(φ(λ (R_{i,s} − μ0))) / (1 + λ² σ² / 2 + 1.5 ε)``
    Catoni influence ``φ`` as in Eq. (6). Recommended
    ``λ = √ε / (2σ)`` (simulations; Theorem 8 uses ``λ² = ε / (4σ²)``).

Heavy tail, ``1 < p < 2`` (Eq. 12–13)
    ``E^{(p)}_{i,t} = ∏_s exp(φ_p(λ (R_{i,s} − μ0))) / (1 + λ^p κ / p + (p − 1/p) ε)``
    with ``λ = (ε / κ)^{1/p}``. Figure 3: ``p = 1.85``, ``κ = 50``.
    Paper typesets ``E^{(p)}_{i,0} = 0`` then says ``E_{i,0} = 1``; we
    initialize at 1 (otherwise the process is identically zero).

Non-robust e-process (INVALIDITY STRESS TEST ONLY)
    ``e = exp(λ (R − μ0) − λ² σ² / 2)`` with no Catoni clip and no
    contamination denominator. This increment is **not** valid under
    Huber contamination. It must never be cited as a counterexample to
    the paper's theorems. Label: ``NOT_COMPARABLE``.
"""

from __future__ import annotations

import math

import numpy as np

LOG_E_MIN = -700.0
LOG_E_MAX = 700.0
E_MIN = 1e-300
E_MAX = 1e300


def catoni_phi(x: float) -> float:
    """Catoni influence (Eq. 6). Uses ``log1p`` on the quadratic pieces."""
    if x < -1.0:
        return -math.log(2.0)
    if x < 0.0:
        return math.log1p(x + 0.5 * x * x)
    if x < 1.0:
        return -math.log1p(-x + 0.5 * x * x)
    return math.log(2.0)


def catoni_phi_p(x: float, p: float) -> float:
    """Heavy-tail influence ``φ_p`` (Eq. 12)."""
    if x < -1.0:
        return -math.log(p)
    if x < 0.0:
        return math.log1p(x + (abs(x) ** p) / p)
    if x < 1.0:
        return -math.log1p(-x + (x**p) / p)
    return math.log(p)


def recommended_lambda(eps: float, sigma: float = 1.0) -> float:
    """``λ = √ε / (2σ)``. If ``ε = 0``, fall back to a small positive λ."""
    eps = float(eps)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if eps <= 0.0:
        return 0.1 / sigma
    return math.sqrt(eps) / (2.0 * sigma)


def heavytail_lambda(eps: float, kappa: float, p: float) -> float:
    """``λ = (ε / κ)^{1/p}`` as in Figure 3."""
    if kappa <= 0.0 or p <= 1.0:
        raise ValueError("need kappa > 0 and p > 1")
    eps = max(float(eps), 0.0)
    if eps == 0.0:
        return (1e-6 / kappa) ** (1.0 / p)
    return (eps / kappa) ** (1.0 / p)


def _clip_log_e(log_e: float) -> float:
    if not math.isfinite(log_e):
        return 0.0
    return min(max(log_e, LOG_E_MIN), LOG_E_MAX)


def exp_clip(log_e: float) -> float:
    return float(math.exp(_clip_log_e(log_e)))


class RobustEProcess:
    """Product e-process (Eq. 5) accumulated in log space."""

    def __init__(
        self,
        k: int,
        *,
        mu0: float = 0.0,
        eps: float = 0.1,
        sigma: float = 1.0,
        lam: float | None = None,
    ) -> None:
        self.k = int(k)
        self.mu0 = float(mu0)
        self.eps = float(eps)
        self.sigma = float(sigma)
        self.lam = float(lam) if lam is not None else recommended_lambda(self.eps, self.sigma)
        self.denom = 1.0 + 0.5 * (self.lam**2) * (self.sigma**2) + 1.5 * self.eps
        self.log_e = np.zeros(self.k, dtype=float)

    def increment(self, reward: float) -> float:
        phi = catoni_phi(self.lam * (float(reward) - self.mu0))
        # log increment = φ − log(denom); denom > 1 so this is finite.
        return phi - math.log(self.denom)

    def update(self, arm: int, reward: float) -> float:
        self.log_e[int(arm)] += self.increment(reward)
        self.log_e[int(arm)] = _clip_log_e(float(self.log_e[int(arm)]))
        return self.values[int(arm)]

    @property
    def values(self) -> np.ndarray:
        return np.exp(np.clip(self.log_e, LOG_E_MIN, LOG_E_MAX))


class HeavyTailEProcess:
    """Infinite-variance e-process (Eq. 13). ``E_0 = 1`` (paper typo says 0)."""

    def __init__(
        self,
        k: int,
        *,
        mu0: float = 0.0,
        eps: float = 0.1,
        p: float = 1.85,
        kappa: float = 50.0,
        lam: float | None = None,
    ) -> None:
        self.k = int(k)
        self.mu0 = float(mu0)
        self.eps = float(eps)
        self.p = float(p)
        self.kappa = float(kappa)
        self.lam = float(lam) if lam is not None else heavytail_lambda(self.eps, self.kappa, self.p)
        self.denom = 1.0 + (self.lam**self.p) * self.kappa / self.p + (self.p - 1.0 / self.p) * self.eps
        self.log_e = np.zeros(self.k, dtype=float)

    def increment(self, reward: float) -> float:
        phi = catoni_phi_p(self.lam * (float(reward) - self.mu0), self.p)
        return phi - math.log(self.denom)

    def update(self, arm: int, reward: float) -> float:
        self.log_e[int(arm)] += self.increment(reward)
        self.log_e[int(arm)] = _clip_log_e(float(self.log_e[int(arm)]))
        return self.values[int(arm)]

    @property
    def values(self) -> np.ndarray:
        return np.exp(np.clip(self.log_e, LOG_E_MIN, LOG_E_MAX))


class NonRobustEProcess:
    """Sub-Gaussian likelihood e-process **without** robustness.

    INVALIDITY STRESS TEST ONLY. Do not claim this refutes any theorem.
    Under Lévy contamination ``exp(λ R)`` overflows; we clip in log space
    so the procedure stays numerically defined, not theoretically valid.
    """

    STRESS_TEST_ONLY = True
    LABEL = "NOT_COMPARABLE"

    def __init__(
        self,
        k: int,
        *,
        mu0: float = 0.0,
        sigma: float = 1.0,
        lam: float | None = None,
    ) -> None:
        self.k = int(k)
        self.mu0 = float(mu0)
        self.sigma = float(sigma)
        self.lam = 0.5 / float(sigma) if lam is None else float(lam)
        self.log_e = np.zeros(self.k, dtype=float)

    def increment(self, reward: float) -> float:
        centered = float(reward) - self.mu0
        # log increment = λ(X−μ0) − λ²σ²/2  (standard sub-Gaussian supermartingale)
        return self.lam * centered - 0.5 * (self.lam**2) * (self.sigma**2)

    def update(self, arm: int, reward: float) -> float:
        self.log_e[int(arm)] += self.increment(reward)
        self.log_e[int(arm)] = _clip_log_e(float(self.log_e[int(arm)]))
        return self.values[int(arm)]

    @property
    def values(self) -> np.ndarray:
        return np.exp(np.clip(self.log_e, LOG_E_MIN, LOG_E_MAX))
