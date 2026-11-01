"""Sampling rules for AAAI 2026 robust bandit multiple testing.

Uniform
    Cycle through arms that are not yet rejected. Paper §4: “uniform
    sampling across each arm.” We skip rejected arms so e-BH rejection
    sets stay nested (increments can be < 1 after Catoni / denom).

Mean-based (Algorithm 1)
    Epoch schedule
        ``t(s) = ceil( (4 s / ε²) log(K π² s² / (3 δ)) )``
        (natural log). Each selected arm is pulled
        ``t(N+1) − t(N)`` times.
    Arm choice (OCR of Algorithm 1 is ``argmin μ̂ + bonus``; the prose
    says pull the arm believed to be the best). We take
        ``argmax_{i ∉ S} [ μ̂_i − σ / ((1−ε) √(2 N_i)) ]``
    i.e. the largest lower confidence bound on the α-trimmed mean,
    with ``α = ε/2`` (Lemma 3). Documented as an OCR resolution, not a
    new method.

E-process-based (Algorithm 2)
    Burn-in ``B(δ, ε) = 4 ε^{-1} log(4 K / δ²)`` uniform cycles
    (``K B`` pulls), then ``argmax_{i ∉ S} e_{i,t−1}``.
    Deviation: the typeset Algorithm 2 nests the e-process update and
    e-BH under ``t > K B``. We update and run e-BH **every** pull,
    including burn-in; otherwise Figure 1 TPR would be identically 0
    until ~``K B`` samples.

Algorithm 3 (Appendix D not in the HTML extract)
    Same sampling as Algorithm 2, but the evidence is Eq. (13).
"""

from __future__ import annotations

import math

import numpy as np


def mean_based_schedule_t(s: int, k: int, eps: float, delta: float) -> int:
    """Algorithm 1 ``t(s)``. ``s`` is the 1-based epoch index."""
    if s < 1:
        raise ValueError("s must be >= 1")
    if eps <= 0.0:
        # Paper schedule is undefined at ε=0; sequential unit batches.
        return int(s)
    inner = (k * (math.pi**2) * (s**2)) / (3.0 * delta)
    if inner <= 1.0:
        log_term = 0.0
    else:
        log_term = math.log(inner)
    raw = (4.0 * s / (eps**2)) * log_term
    return max(int(math.ceil(raw)), s)


def trimmed_mean(samples: np.ndarray, alpha: float) -> float:
    """α-trimmed mean (Eq. 7): drop ``r = floor(n α)`` each tail."""
    arr = np.asarray(samples, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return 0.0
    r = int(np.floor(n * float(alpha)))
    if 2 * r >= n:
        return float(np.median(arr))
    ordered = np.sort(arr)
    kept = ordered[r : n - r]
    return float(np.mean(kept))


def mean_based_lcb(
    samples: np.ndarray,
    n_epoch: int,
    *,
    eps: float,
    sigma: float = 1.0,
    trim_alpha: float | None = None,
) -> float:
    """LCB of the trimmed mean used as “believed best” score."""
    alpha = float(eps) / 2.0 if trim_alpha is None else float(trim_alpha)
    mu_hat = trimmed_mean(samples, alpha)
    n = max(int(n_epoch), 1)
    denom = max(1.0 - float(eps), 1e-12) * math.sqrt(2.0 * n)
    bonus = float(sigma) / denom
    return mu_hat - bonus


def eprocess_burnin_length(k: int, delta: float, eps: float) -> int:
    """``B(δ, ε) = 4 ε^{-1} log(4 K / δ²)`` (Algorithm 2). Returns pulls *per arm*."""
    if eps <= 0.0:
        return 1
    return int(math.ceil(4.0 / float(eps) * math.log(4.0 * k / (delta**2))))


def next_uniform_unrejected(
    t_one_based: int,
    rejected: np.ndarray,
    cursor: int | None = None,
) -> int:
    """Round-robin among unrejected arms. ``cursor`` is the last chosen index."""
    alive = np.flatnonzero(~np.asarray(rejected, dtype=bool))
    if alive.size == 0:
        raise RuntimeError("no unrejected arms left")
    if cursor is None:
        return int(alive[int(t_one_based - 1) % alive.size])
    nxt = (int(cursor) + 1) % int(rejected.size)
    for _ in range(int(rejected.size)):
        if not rejected[nxt]:
            return int(nxt)
        nxt = (nxt + 1) % int(rejected.size)
    return int(alive[0])


def argmax_eprocess(evalues: np.ndarray, rejected: np.ndarray) -> int:
    """Greedy / Algorithm 2 post-burn-in: largest e-value among unrejected."""
    scores = np.asarray(evalues, dtype=float).copy()
    scores[np.asarray(rejected, dtype=bool)] = -np.inf
    if not np.any(np.isfinite(scores)):
        raise RuntimeError("no unrejected arms left")
    return int(np.argmax(scores))


def argmax_mean_lcb(
    histories: list[list[float]],
    n_epochs: np.ndarray,
    rejected: np.ndarray,
    *,
    eps: float,
    sigma: float = 1.0,
) -> int:
    k = len(histories)
    scores = np.full(k, -np.inf, dtype=float)
    for i in range(k):
        if rejected[i]:
            continue
        scores[i] = mean_based_lcb(np.asarray(histories[i], dtype=float), int(n_epochs[i]), eps=eps, sigma=sigma)
    if not np.any(np.isfinite(scores)):
        raise RuntimeError("no unrejected arms left")
    return int(np.argmax(scores))
