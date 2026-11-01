"""AAAI 2026 Zhou & Liu data-generating processes.

Gaussian Figure 1/2
    ``P_i = N(μ_i, 1)``, null ``μ = 0``, non-null ``μ = 1``.
    Huber contamination with probability ``ε = 1/10`` from a Lévy-stable
    law with location 1000 and skewness 0.5.

Heavy-tail Figure 3
    Location-shifted Student-t(2) (infinite variance), null mean 0,
    non-null mean 5, same Lévy contamination, ``K = 50``, ``|H1| = 2``.

Deviation (paper-omitted Lévy parameters)
    The main text specifies only location and skewness. Stability index
    and scale are unspecified. We use ``alpha=1.5``, ``scale=1.0``
    (``scipy.stats.levy_stable`` S0 parameterization) and mark every
    call site. Do not treat this as an official default of the paper.

Rewards are generated only when an arm is pulled. The environment never
returns future observations or ground-truth labels to the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import levy_stable

# Paper-omitted Lévy-stable parameters. See module docstring.
LEVY_ALPHA = 1.5
LEVY_SCALE = 1.0
LEVY_LOC = 1000.0
LEVY_BETA = 0.5
DEFAULT_EPS = 0.1


def n_nonnull_from_spec(spec: str | int, k: int) -> int:
    """Map ``{2, floor(log K), floor(sqrt(K))}`` onto an integer in ``[0, K]``."""
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        n = int(spec)
    else:
        text = str(spec).strip().lower().replace(" ", "")
        if text in {"2", "fixed2"}:
            n = 2
        elif text in {"floor(logk)", "floor(log k)", "logk", "log"}:
            n = int(np.floor(np.log(k)))
        elif text in {"floor(sqrtk)", "floor(sqrt(k))", "sqrtk", "sqrt"}:
            n = int(np.floor(np.sqrt(k)))
        else:
            n = int(spec)
    return int(np.clip(n, 0, k))


def assign_nonnulls(k: int, n_nonnull: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of non-null arms. First ``n_nonnull`` if rng is unused? No — random subset."""
    n = int(np.clip(n_nonnull, 0, k))
    mask = np.zeros(k, dtype=bool)
    if n:
        idx = rng.choice(k, size=n, replace=False)
        mask[idx] = True
    return mask


@dataclass
class ContaminatedBandit:
    """On-demand contaminated bandit. Pull ``observe(i)`` only.

    Attributes
    ----------
    is_nonnull:
        Ground-truth alternative mask. Algorithms must not read this.
    n_generated:
        Per-arm count of realized rewards (equals pulls if nobody peeks).
    """

    means: np.ndarray
    eps: float = DEFAULT_EPS
    family: str = "gaussian"
    student_df: float = 2.0
    levy_alpha: float = LEVY_ALPHA
    levy_beta: float = LEVY_BETA
    levy_loc: float = LEVY_LOC
    levy_scale: float = LEVY_SCALE
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    n_generated: np.ndarray = field(init=False)
    history: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float).reshape(-1)
        k = int(self.means.size)
        self.n_generated = np.zeros(k, dtype=int)
        self.history = [[] for _ in range(k)]

    @property
    def k(self) -> int:
        return int(self.means.size)

    @property
    def is_nonnull(self) -> np.ndarray:
        return self.means > 0.0

    @property
    def is_null(self) -> np.ndarray:
        return ~self.is_nonnull

    def _clean(self, arm: int) -> float:
        mu = float(self.means[arm])
        if self.family == "gaussian":
            return float(self.rng.normal(loc=mu, scale=1.0))
        if self.family == "student_t":
            return float(self.rng.standard_t(self.student_df) + mu)
        raise ValueError(f"unknown family {self.family}")

    def _levy(self) -> float:
        # Deviation: alpha/scale not in the paper; see module docstring.
        draw = levy_stable.rvs(
            self.levy_alpha,
            self.levy_beta,
            loc=self.levy_loc,
            scale=self.levy_scale,
            random_state=self.rng,
        )
        value = float(np.asarray(draw, dtype=float))
        if not np.isfinite(value):
            value = float(self.levy_loc)
        return value

    def observe(self, arm: int) -> float:
        """Draw the next reward of ``arm`` only. No future, no other arms."""
        idx = int(arm)
        if idx < 0 or idx >= self.k:
            raise IndexError(arm)
        if self.eps > 0.0 and self.rng.random() < self.eps:
            reward = self._levy()
        else:
            reward = self._clean(idx)
        if not np.isfinite(reward):
            reward = float(self.means[idx])
        self.n_generated[idx] += 1
        self.history[idx].append(reward)
        return reward

    def assert_no_extra_generation(self, pulls: np.ndarray) -> None:
        pulls_arr = np.asarray(pulls, dtype=int).reshape(-1)
        if pulls_arr.shape != self.n_generated.shape:
            raise AssertionError("pull vector shape mismatch")
        if np.any(self.n_generated != pulls_arr):
            raise AssertionError(
                f"peeking or dropped pulls: generated={self.n_generated} pulls={pulls_arr}"
            )


def make_gaussian_instance(
    k: int,
    n_nonnull: int | str,
    rng: np.random.Generator,
    *,
    mu_null: float = 0.0,
    mu_nonnull: float = 1.0,
    eps: float = DEFAULT_EPS,
) -> ContaminatedBandit:
    n1 = n_nonnull_from_spec(n_nonnull, k)
    mask = assign_nonnulls(k, n1, rng)
    means = np.full(k, float(mu_null), dtype=float)
    means[mask] = float(mu_nonnull)
    return ContaminatedBandit(means=means, eps=eps, family="gaussian", rng=rng)


def make_heavytail_instance(
    k: int = 50,
    n_nonnull: int = 2,
    rng: np.random.Generator | None = None,
    *,
    mu_null: float = 0.0,
    mu_nonnull: float = 5.0,
    eps: float = DEFAULT_EPS,
    student_df: float = 2.0,
) -> ContaminatedBandit:
    rng = rng if rng is not None else np.random.default_rng(0)
    n1 = n_nonnull_from_spec(n_nonnull, k)
    mask = assign_nonnulls(k, n1, rng)
    means = np.full(k, float(mu_null), dtype=float)
    means[mask] = float(mu_nonnull)
    return ContaminatedBandit(
        means=means,
        eps=eps,
        family="student_t",
        student_df=student_df,
        rng=rng,
    )
