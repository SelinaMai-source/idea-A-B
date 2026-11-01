"""e-PS sampling rules (arXiv:2609.26651 Algorithm 1 and §6 baselines).

Uniform
    Sample uniformly at random from currently unrejected hypotheses.

Greedy (Xu et al. 2021, Appendix C)
    Sample the unrejected arm with the largest current e-value.

e-PS (Algorithm 1)
    First ``K`` pulls: ``A_t = t`` (each arm once).
    Thereafter, for every unrejected ``k``,
        ``m̃_{k,t} ~ N( m̂_{k,t-1}, v_{k,t-1} / (n_{k,t-1} ∨ 1) )``
        ``m̂ = log(E) / (n ∨ 1)``
        ``v_{k,t} = 1.1 σ̂_{k,t}^2``
    where ``σ̂²`` is the sample variance of *observations* on that arm
    up to time ``t−1`` (§6 Implementation details). Then
    ``A_t = argmax_{k ∉ R} m̃_{k,t}``.

When ``n < 2``, sample variance is undefined. Deviation: use
``σ̂² = 1`` (the Gaussian design variance) so the posterior is proper.
Variance is floored at ``1e-12`` so draws stay finite.
"""

from __future__ import annotations

import numpy as np

VARIANCE_PROXY_MULTIPLIER = 1.1
DEFAULT_VAR = 1.0
VAR_FLOOR = 1e-12


def sample_variance(history: list[float] | np.ndarray) -> float:
    arr = np.asarray(history, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return DEFAULT_VAR
    var = float(np.var(arr, ddof=1))
    if not np.isfinite(var) or var <= 0.0:
        return DEFAULT_VAR
    return var


def variance_proxy(history: list[float] | np.ndarray) -> float:
    """``v_{k,t} = 1.1 σ̂_{k,t}^2`` with the n<2 fallback above."""
    return max(VARIANCE_PROXY_MULTIPLIER * sample_variance(history), VAR_FLOOR)


def uniform_unrejected(rejected: np.ndarray, rng: np.random.Generator) -> int:
    alive = np.flatnonzero(~np.asarray(rejected, dtype=bool))
    if alive.size == 0:
        raise RuntimeError("no unrejected arms left")
    return int(rng.choice(alive))


def greedy_unrejected(evalues: np.ndarray, rejected: np.ndarray) -> int:
    scores = np.asarray(evalues, dtype=float).copy()
    scores[np.asarray(rejected, dtype=bool)] = -np.inf
    if not np.any(np.isfinite(scores)):
        raise RuntimeError("no unrejected arms left")
    return int(np.argmax(scores))


def eps_posterior_sample(
    log_e: np.ndarray,
    n_pulls: np.ndarray,
    histories: list[list[float]],
    rejected: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Algorithm 1 lines 6–7. Uses past-only ``v`` and ``m̂``."""
    alive = np.flatnonzero(~np.asarray(rejected, dtype=bool))
    if alive.size == 0:
        raise RuntimeError("no unrejected arms left")
    draws = np.empty(alive.size, dtype=float)
    for j, arm in enumerate(alive):
        n = max(int(n_pulls[arm]), 1)
        mean = float(log_e[arm]) / n
        var = variance_proxy(histories[arm]) / n
        draws[j] = float(rng.normal(loc=mean, scale=math_sqrt(var)))
    return int(alive[int(np.argmax(draws))])


def math_sqrt(x: float) -> float:
    return float(np.sqrt(max(float(x), VAR_FLOOR)))
