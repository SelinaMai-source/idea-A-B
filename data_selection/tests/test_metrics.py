import math

import pytest

from evaluation.metrics import average_success, backward_transfer, final_success_rate, forgetting, success_auc
from evaluation.statistics import summarize


def test_success_rate_and_auc_use_the_declared_grid():
    assert final_success_rate([True, False, True]) == pytest.approx(2 / 3)
    curve = [(0, 0.0), (10, 1.0)]
    assert success_auc(curve, [0, 10]) == pytest.approx(5.0)
    assert success_auc(curve, [0, 10], horizon=10) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="predeclared"):
        success_auc([(0, 0.0), (5, 1.0)], [0, 10])


def test_forgetting_and_bwt_on_a_known_matrix():
    matrix = [
        [1.0, None, None],
        [0.8, 1.0, None],
        [0.5, 0.7, 0.9],
    ]
    assert forgetting(matrix) == pytest.approx(0.4)
    assert backward_transfer(matrix) == pytest.approx(-0.4)
    assert average_success(matrix) == pytest.approx(0.7)


def test_missing_cells_are_not_zeros():
    matrix = [
        [1.0, None],
        [None, 1.0],
    ]
    with pytest.raises(ValueError):
        forgetting(matrix)


def test_one_seed_summary_has_no_interval():
    summary = summarize([0.5])
    assert summary["ci_justified"] is False
    assert summary["ci95"] == "N/A"
    two = summarize([0.0, 1.0])
    assert two["n"] == 2
    assert two["mean"] == pytest.approx(0.5)
    assert two["std"] == pytest.approx(math.sqrt(0.5))
