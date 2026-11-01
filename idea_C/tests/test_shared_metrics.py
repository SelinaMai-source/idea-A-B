"""Shared metrics, budget, seeds, serialization."""

from __future__ import annotations

import json

import numpy as np
import pytest

from idea_C.shared.budget import SampleBudget
from idea_C.shared.metrics import fdp, power, samples_at_target_tpr, sup_fdp, tpp, tpr
from idea_C.shared.seeds import seed_everything
from idea_C.shared.serialize import TABLE_METRIC_COLUMNS, write_run
from idea_C.shared.stats import mean_std_ci


def test_fdp_tpp_power_definitions() -> None:
    rejected = np.array([True, True, False, False])
    is_null = np.array([True, False, True, False])
    assert fdp(rejected, is_null) == pytest.approx(0.5)
    assert tpp(rejected, is_null) == pytest.approx(0.5)
    assert tpr(rejected, is_null) == pytest.approx(0.5)
    assert power(rejected, is_null) == pytest.approx(0.5)


def test_empty_rejection_fdp_zero() -> None:
    rejected = np.zeros(4, dtype=bool)
    is_null = np.array([True, True, False, False])
    assert fdp(rejected, is_null) == 0.0
    assert tpp(rejected, is_null) == 0.0


def test_all_null_tpp_zero() -> None:
    rejected = np.array([True, False])
    is_null = np.array([True, True])
    assert tpp(rejected, is_null) == 0.0


def test_sup_fdp_is_path_maximum() -> None:
    path = np.array(
        [
            [True, False, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    is_null = np.array([True, False, True])
    assert sup_fdp(path, is_null) == pytest.approx(1.0)


def test_samples_at_target_nan_if_never_reached() -> None:
    assert np.isnan(samples_at_target_tpr(np.array([0.1, 0.2]), delta=0.05))
    assert samples_at_target_tpr(np.array([0.1, 0.96]), delta=0.05) == 2.0


def test_budget_cannot_overspend() -> None:
    purse = SampleBudget(3)
    purse.spend(2)
    assert purse.remaining == 1
    with pytest.raises(RuntimeError, match="overspend"):
        purse.spend(2)
    assert purse.spent == 2
    assert purse.spend_up_to(5) == 1
    purse.assert_not_overspent()


def test_seed_everything_reproducible() -> None:
    g1 = seed_everything(42)
    a = g1.normal(size=5)
    g2 = seed_everything(42)
    b = g2.normal(size=5)
    np.testing.assert_array_equal(a, b)
    g3 = seed_everything(43)
    assert not np.allclose(g3.normal(size=5), a)


def test_mean_std_ci_finite() -> None:
    out = mean_std_ci([1.0, 2.0, 3.0])
    assert out["mean"] == pytest.approx(2.0)
    assert out["ci_low"] < out["mean"] < out["ci_high"]
    assert np.isnan(mean_std_ci([1.0])["std"])


def test_serialize_omits_nan_metrics(tmp_path) -> None:
    path = write_run(
        "demo_run",
        table_id="aaai_robust_gaussian",
        version="v0_test",
        content="unit",
        data="toy",
        config={"K": 4, "alpha": 0.05},
        seed=0,
        metrics={"FDR↓": float("nan"), "TPR↑": 0.5},
        raw_dir=tmp_path,
    )
    payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    assert "FDR↓" not in payload["metrics"]
    assert payload["metrics"]["TPR↑"] == 0.5
    assert payload["seed"] == 0
    assert "aaai_robust_gaussian" in TABLE_METRIC_COLUMNS
    assert "eps_gaussian" in TABLE_METRIC_COLUMNS
