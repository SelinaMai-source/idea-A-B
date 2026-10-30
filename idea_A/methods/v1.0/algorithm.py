"""Classical stratified Neyman allocation with an exploration floor.

This is textbook two-phase sampling. It is not a novel algorithm. Stratum
standard deviations are estimated from labels that are already visible
(natural observations, and previously realized audits if any). Hidden labels
are not read: callers must pass a revealed outcome vector that is NaN where
``O_t = 0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_v1_config() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def stratum_sigma(
    stratum: np.ndarray,
    y_visible: np.ndarray,
    observed_mask: np.ndarray,
    n_strata: int,
) -> np.ndarray:
    """Sample SD per stratum from currently visible labels only."""
    z = np.asarray(stratum, dtype=int)
    y = np.asarray(y_visible, dtype=float)
    obs = np.asarray(observed_mask).astype(bool, copy=False)
    sigma = np.ones(int(n_strata), dtype=float)
    for h in range(int(n_strata)):
        values = y[(z == h) & obs]
        values = values[np.isfinite(values)]
        if values.size >= 2:
            sigma[h] = float(np.std(values, ddof=1))
        elif values.size == 1:
            sigma[h] = 1.0
        else:
            sigma[h] = 1.0
        if not np.isfinite(sigma[h]) or sigma[h] <= 0.0:
            sigma[h] = 1.0
    return sigma


def fit_mean_q(n_h: np.ndarray, q_h: np.ndarray, budget: float, n: int, cap: float = 1.0) -> np.ndarray:
    """Rescale per-stratum ``q_h`` so ``sum(n_h q_h)/n`` matches ``budget`` under a cap."""
    q = np.clip(np.asarray(q_h, dtype=float), 0.0, cap)
    n_h = np.asarray(n_h, dtype=float)
    n_f = float(n)
    target = float(budget)
    for _ in range(25):
        current = float(np.dot(n_h, q) / n_f)
        if abs(current - target) < 1e-12:
            break
        if current <= 0.0:
            q[:] = min(target, cap)
            break
        q = np.minimum(q * (target / current), cap)
    return q


def neyman_stratum_q(
    n_h: np.ndarray,
    sigma_h: np.ndarray,
    budget: float,
    n: int,
    exploration_mix: float = 0.15,
    q_floor: float = 1e-6,
) -> np.ndarray:
    """Per-stratum audit probabilities. ``q_h ∝ σ_h`` then mixed with uniform ``b``.

    Neyman sample sizes are ``n_h^{audit} ∝ n_h σ_h``, hence ``q_h ∝ σ_h``.
    The exploration floor is a mixture with a constant propensity ``budget``,
    not a claimed new allocation rule.
    """
    n_h = np.asarray(n_h, dtype=float)
    sigma = np.maximum(np.asarray(sigma_h, dtype=float), 1e-8)
    mix = float(exploration_mix)
    mix = min(max(mix, 0.0), 1.0)
    denom = float(np.dot(n_h, sigma))
    if denom <= 0.0:
        q_h = np.full(n_h.shape[0], float(budget), dtype=float)
    else:
        q_h = (float(budget) * float(n)) * sigma / denom
    q_h = np.minimum(q_h, 1.0)
    q_h = (1.0 - mix) * q_h + mix * float(budget)
    q_h = np.maximum(q_h, float(q_floor))
    q_h = np.minimum(q_h, 1.0)
    return fit_mean_q(n_h, q_h, budget, n)


def allocate_neyman(
    stratum: np.ndarray,
    y_visible: np.ndarray,
    observed_mask: np.ndarray,
    budget: float,
    n_strata: int,
    exploration_mix: float = 0.15,
    q_floor: float = 1e-6,
) -> np.ndarray:
    """Unit-level known audit propensity from estimated-stratum Neyman + floor."""
    z = np.asarray(stratum, dtype=int)
    n = int(z.shape[0])
    n_h = np.bincount(z, minlength=int(n_strata)).astype(float)
    sigma = stratum_sigma(z, y_visible, observed_mask, n_strata)
    q_h = neyman_stratum_q(
        n_h,
        sigma,
        float(budget),
        n,
        exploration_mix=exploration_mix,
        q_floor=q_floor,
    )
    return q_h[z]


def allocate_uniform(n: int, budget: float) -> np.ndarray:
    return np.full(int(n), float(budget), dtype=float)


def allocate_fixed_stratified(stratum: np.ndarray, n_strata: int, budget: float) -> np.ndarray:
    """Equal expected audits per stratum: ``n_h q_h = b n / K``.

    This is classical equal allocation, not Neyman. Unequal stratum sizes
    produce unequal ``q_h``.
    """
    z = np.asarray(stratum, dtype=int)
    n = int(z.shape[0])
    k = int(n_strata)
    n_h = np.bincount(z, minlength=k).astype(float)
    q_h = np.zeros(k, dtype=float)
    target_each = float(budget) * n / k
    for h in range(k):
        q_h[h] = 1.0 if n_h[h] <= 0 else min(1.0, target_each / n_h[h])
    q_h = np.maximum(q_h, 1e-6)
    q_h = fit_mean_q(n_h, q_h, budget, n)
    return q_h[z]


def allocate_equal_mix_neyman(
    stratum: np.ndarray,
    y_visible: np.ndarray,
    observed_mask: np.ndarray,
    budget: float,
    n_strata: int,
    exploration_mix: float,
    q_floor: float,
) -> np.ndarray:
    """Alias kept so ablations can import a single allocation entry point."""
    return allocate_neyman(
        stratum,
        y_visible,
        observed_mask,
        budget,
        n_strata,
        exploration_mix=exploration_mix,
        q_floor=q_floor,
    )
