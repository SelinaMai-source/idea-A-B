"""Paired bootstrap on per-seed differences."""

import math

import numpy as np
import pytest

from shared.statistical_tests.paired_bootstrap import paired_bootstrap_ci


def test_constant_paired_difference_is_exact_and_preliminary():
    result = paired_bootstrap_ci(
        [5.0, 5.0, 5.0],
        [2.0, 2.0, 2.0],
        n_boot=100,
        alpha=0.05,
        seed=0,
    )
    assert result["n_seeds"] == 3
    assert result["mean_difference"] == pytest.approx(3.0)
    assert result["ci_low"] == pytest.approx(3.0)
    assert result["ci_high"] == pytest.approx(3.0)
    assert result["preliminary"] is True
    assert "preliminary" in result["note"].lower()
    assert "n_seeds=3" in result["note"]


def test_bootstrap_mean_matches_paired_differences_and_is_deterministic():
    x = np.array([0.0, 1.0, 4.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    first = paired_bootstrap_ci(x, y, n_boot=400, alpha=0.1, seed=1)
    second = paired_bootstrap_ci(x, y, n_boot=400, alpha=0.1, seed=1)
    assert first == second
    assert first["mean_difference"] == pytest.approx(float(np.mean(x - y)))
    assert first["ci_low"] <= first["mean_difference"] <= first["ci_high"]
    assert first["preliminary"] is False
    assert first["n_seeds"] == 4
    assert math.isfinite(first["ci_low"]) and math.isfinite(first["ci_high"])
