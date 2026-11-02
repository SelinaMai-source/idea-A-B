"""Seed aggregates. A one-seed smoke test does not get a confidence interval."""

from __future__ import annotations

import math


def summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("no values")
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {
            "mean": mean,
            "std": "N/A",
            "ci95": "N/A",
            "n": n,
            "ci_justified": False,
            "interval": "not_reported_for_n_lt_2",
        }
    variance = sum((item - mean) ** 2 for item in values) / (n - 1)
    std = math.sqrt(variance)
    half = 1.96 * std / math.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "n": n,
        "ci_justified": True,
        "interval": "normal_approx_z_1.96",
    }
