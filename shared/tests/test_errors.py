"""Scalar-truth error summaries."""

import numpy as np
import pytest

from shared.metrics.errors import bias, mae, rmse


def test_bias_mae_rmse_against_scalar_truth():
    estimates = np.array([2.0, 0.0])
    assert bias(estimates, 1.0) == pytest.approx(0.0)
    assert mae(estimates, 1.0) == pytest.approx(1.0)
    assert rmse(estimates, 1.0) == pytest.approx(1.0)
    assert bias(3.0, 1.0) == pytest.approx(2.0)
    assert rmse(np.array([3.0, 1.0]), 1.0) == pytest.approx(np.sqrt(2.0))


def test_truth_must_be_scalar():
    with pytest.raises(ValueError):
        bias(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
