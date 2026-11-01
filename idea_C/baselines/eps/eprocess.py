"""e-PS increment e-values (arXiv:2609.26651).

Oracle LR (simple-vs-simple, Eq. 12)
    ``e_{k,t} = (dQ°_k / dP°_k)(Y_t)`` if ``A_t = k``, else 1.
    Gaussian: ``log LR = θ_alt Y − θ_alt² / 2`` with ``θ_alt = 0.02 k``
    (the *hypothesized* alternative, never the realized ``θ_k``).
    Student-t(5): location-shift density ratio
    ``f_5(Y − θ_alt) / f_5(Y)``.

Adaptive plug-in (simple-vs-composite, Eq. 17 / Corollary 5.2)
    ``e_{k,t} = (d Q̂_{k, n−1} / d P°_k)(Y_t)`` with ``Q̂_{k,0} = P°_k``
    so the first increment is 1. Gaussian plug-in uses the sample mean
    of *past* observations only:
    ``log e = θ̂ Y − θ̂² / 2``. Student-t plug-in uses the same ``θ̂``
    as a location. Oracle and Adaptive are separate classes.

Unsampled arms contribute increment 1 (Eq. 3). Ground-truth labels are
never used.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import t as student_t

LOG_E_MIN = -700.0
LOG_E_MAX = 700.0


def _clip_log(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return min(max(x, LOG_E_MIN), LOG_E_MAX)


def gaussian_log_lr(y: float, theta: float, var: float = 1.0) -> float:
    """``log φ(y; θ, var) − log φ(y; 0, var)``."""
    return (float(theta) * float(y) - 0.5 * float(theta) ** 2) / float(var)


def student_t_log_lr(y: float, theta: float, df: float = 5.0) -> float:
    """Location-shift Student-t log density ratio."""
    log_alt = float(student_t.logpdf(y, df, loc=float(theta), scale=1.0))
    log_null = float(student_t.logpdf(y, df, loc=0.0, scale=1.0))
    return log_alt - log_null


class ProductEProcess:
    """``E_{k,t} = ∏_s e_{k,s}`` in log space."""

    def __init__(self, k: int) -> None:
        self.k = int(k)
        self.log_e = np.zeros(self.k, dtype=float)

    def multiply(self, arm: int, log_increment: float) -> None:
        self.log_e[int(arm)] = _clip_log(float(self.log_e[int(arm)]) + float(log_increment))

    @property
    def values(self) -> np.ndarray:
        return np.exp(np.clip(self.log_e, LOG_E_MIN, LOG_E_MAX))


class OracleLR:
    """Simple-vs-simple likelihood-ratio increments. Separate from Adaptive."""

    name = "oracle"

    def __init__(
        self,
        alternatives: np.ndarray,
        *,
        family: str = "gaussian",
        student_df: float = 5.0,
        var: float = 1.0,
    ) -> None:
        self.alternatives = np.asarray(alternatives, dtype=float).reshape(-1)
        self.family = family
        self.student_df = float(student_df)
        self.var = float(var)
        self.eproc = ProductEProcess(self.alternatives.size)

    def log_increment(self, arm: int, y: float) -> float:
        theta = float(self.alternatives[int(arm)])
        if self.family == "gaussian":
            return gaussian_log_lr(y, theta, self.var)
        if self.family == "student_t":
            return student_t_log_lr(y, theta, self.student_df)
        raise ValueError(self.family)

    def update(self, arm: int, y: float) -> np.ndarray:
        self.eproc.multiply(arm, self.log_increment(arm, y))
        return self.eproc.values

    @property
    def values(self) -> np.ndarray:
        return self.eproc.values


class AdaptivePluginLR:
    """Simple-vs-composite plug-in LR. Past-only ``θ̂``; first pull increment 1."""

    name = "adaptive"

    def __init__(
        self,
        k: int,
        *,
        family: str = "gaussian",
        student_df: float = 5.0,
        var: float = 1.0,
    ) -> None:
        self.k = int(k)
        self.family = family
        self.student_df = float(student_df)
        self.var = float(var)
        self.eproc = ProductEProcess(self.k)
        self.n = np.zeros(self.k, dtype=int)
        self.sum_y = np.zeros(self.k, dtype=float)

    def theta_hat(self, arm: int) -> float:
        n = int(self.n[arm])
        if n <= 0:
            return 0.0
        return float(self.sum_y[arm] / n)

    def log_increment(self, arm: int, y: float) -> float:
        theta = self.theta_hat(arm)
        if theta == 0.0:
            return 0.0
        if self.family == "gaussian":
            return gaussian_log_lr(y, theta, self.var)
        if self.family == "student_t":
            return student_t_log_lr(y, theta, self.student_df)
        raise ValueError(self.family)

    def update(self, arm: int, y: float) -> np.ndarray:
        # θ̂ uses observations *before* Y_t (Eq. 17: Q̂_{k, n_{k,t-1}}).
        self.eproc.multiply(arm, self.log_increment(arm, y))
        self.sum_y[int(arm)] += float(y)
        self.n[int(arm)] += 1
        return self.eproc.values

    @property
    def values(self) -> np.ndarray:
        return self.eproc.values
