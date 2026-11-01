"""e-PS (arXiv:2609.26651) Section 6 1-D data-generating processes.

``K = 50``. Draw ``B_k ~ Bern(0.1)`` once and set
``θ_k = 0.02 · k · B_k`` (1-based ``k``). The paper holds ``{θ_k}``
fixed across the 500 Monte Carlo trials.

Families implemented here (v0):
    * univariate Gaussian ``N(θ_k, 1)``
    * univariate Student-t(5) as a **location shift** ``t_5 + θ_k``

Deviation: the paper calls ``θ_k`` a “noncentrality parameter” of
``t_{θ_k, 5}``. We implement location-shifted Student-t, which matches
the mean-testing story and the plug-in location estimator. A genuine
noncentral-t law is a different distribution.

5-D, Jester, and LLM watermark are intentionally not implemented.

Rewards are generated only on pull. Algorithms never see ``theta`` or
``is_null`` except through the observations they request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def alternative_mean(k_one_based: int) -> float:
    """Simple-vs-simple hypothesized alternative ``0.02 * k`` (1-based)."""
    return 0.02 * float(k_one_based)


def draw_theta(k: int, rng: np.random.Generator, pi1: float = 0.1) -> np.ndarray:
    """``θ_k = 0.02 k B_k``, ``B_k ~ Bern(π1)``, 1-based ``k``."""
    b = rng.random(k) < float(pi1)
    idx = np.arange(1, k + 1, dtype=float)
    return idx * 0.02 * b.astype(float)


@dataclass
class EPSBandit:
    theta: np.ndarray
    family: str = "gaussian"
    student_df: float = 5.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    n_generated: np.ndarray = field(init=False)
    history: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=float).reshape(-1)
        k = int(self.theta.size)
        self.n_generated = np.zeros(k, dtype=int)
        self.history = [[] for _ in range(k)]

    @property
    def k(self) -> int:
        return int(self.theta.size)

    @property
    def is_nonnull(self) -> np.ndarray:
        return self.theta != 0.0

    @property
    def is_null(self) -> np.ndarray:
        return ~self.is_nonnull

    @property
    def oracle_alternatives(self) -> np.ndarray:
        """Hypothesized simple alternatives ``0.02 k``, not the realized θ."""
        return 0.02 * np.arange(1, self.k + 1, dtype=float)

    def observe(self, arm: int) -> float:
        idx = int(arm)
        if idx < 0 or idx >= self.k:
            raise IndexError(arm)
        loc = float(self.theta[idx])
        if self.family == "gaussian":
            y = float(self.rng.normal(loc=loc, scale=1.0))
        elif self.family == "student_t":
            y = float(self.rng.standard_t(self.student_df) + loc)
        else:
            raise ValueError(f"unknown family {self.family}")
        if not np.isfinite(y):
            y = loc
        self.n_generated[idx] += 1
        self.history[idx].append(y)
        return y

    def assert_no_extra_generation(self, pulls: np.ndarray) -> None:
        pulls_arr = np.asarray(pulls, dtype=int).reshape(-1)
        if pulls_arr.shape != self.n_generated.shape or np.any(self.n_generated != pulls_arr):
            raise AssertionError(
                f"peeking or dropped pulls: generated={self.n_generated} pulls={pulls_arr}"
            )


def make_eps_instance(
    k: int,
    rng: np.random.Generator,
    *,
    family: str = "gaussian",
    student_df: float = 5.0,
    theta: np.ndarray | None = None,
    pi1: float = 0.1,
) -> EPSBandit:
    if theta is None:
        theta = draw_theta(k, rng, pi1=pi1)
    return EPSBandit(theta=np.asarray(theta, dtype=float), family=family, student_df=student_df, rng=rng)
