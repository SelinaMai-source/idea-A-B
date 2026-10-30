"""Paired percentile bootstrap for per-seed differences.

The pairs are seeds (or other matched replicates), not individual units inside
one seed. Resampling draws seed indices with replacement and keeps each pair
together.

n_seeds=3 is preliminary statistical evidence. An interval computed from three
seeds is not a confirmatory claim that one method is superior. The numeric
result is still returned so a later report can show it with that caveat.
``preliminary`` is true whenever ``n_seeds <= 3``.

The estimand is the mean of ``d_i = x_i - y_i``. This helper does not decide
whether a larger value is better; callers who compare lower-is-better metrics
choose the order of ``x`` and ``y``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _per_seed(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = np.reshape(arr, -1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector of per-seed values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be >= 1")
    return out


def paired_bootstrap_ci(
    x: Any,
    y: Any,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile confidence interval for the mean paired difference ``x - y``.

    Parameters
    ----------
    x, y:
        Equal-length per-seed outcomes. Index ``i`` is the same seed in both.
    n_boot:
        Number of bootstrap resamples of the seed index. Default 10000.
    alpha:
        Two-sided error rate in ``(0, 1)``. The interval uses the
        ``alpha/2`` and ``1 - alpha/2`` percentiles (NumPy linear quantile).
    seed:
        Non-negative seed for a local ``numpy.random.Generator``. The global
        NumPy RNG is not modified.

    Returns
    -------
    dict
        ``n_seeds``, ``mean_difference``, ``ci_low``, ``ci_high``, ``alpha``,
        ``n_boot``, ``seed``, ``preliminary``, and ``note``.
    """
    x_arr = _per_seed("x", x)
    y_arr = _per_seed("y", y)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same length, got {x_arr.shape} and {y_arr.shape}")
    n_boot_int = _positive_int("n_boot", n_boot)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an int")
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("seed must be >= 0")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)):
        raise TypeError("alpha must be a float in (0, 1)")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    difference = x_arr - y_arr
    n_seeds = int(difference.shape[0])
    generator = np.random.default_rng(seed_int)
    indices = generator.integers(0, n_seeds, size=(n_boot_int, n_seeds))
    boot_means = np.mean(difference[indices], axis=1)
    if n_seeds <= 3:
        note = (
            f"n_seeds=3 is preliminary statistical evidence. This interval uses "
            f"n_seeds={n_seeds} and is not a confirmatory superiority claim."
        )
    else:
        note = (
            "Percentile bootstrap of the mean paired difference. "
            f"n_seeds=3 would be preliminary; this interval uses n_seeds={n_seeds}."
        )
    return {
        "n_seeds": n_seeds,
        "mean_difference": float(np.mean(difference)),
        "ci_low": float(np.quantile(boot_means, alpha_f / 2.0, method="linear")),
        "ci_high": float(np.quantile(boot_means, 1.0 - alpha_f / 2.0, method="linear")),
        "alpha": alpha_f,
        "n_boot": n_boot_int,
        "seed": seed_int,
        "preliminary": n_seeds <= 3,
        "note": note,
    }
