"""Right-censoring for Samples@Target. Unreached is CENSORED, never 0/NOT_RUN."""

from __future__ import annotations

import json

import numpy as np
import pytest

from idea_C.shared.metrics import (
    CENSORED,
    NOT_IDENTIFIABLE,
    NOT_RUN,
    UNDEFINED,
    can_rank_by_conditional_mean,
    samples_at_target_metric_cells,
    samples_at_target_tpr,
    summarize_samples_at_target,
)
from idea_C.shared.serialize import attach_censoring_metrics, write_run
from idea_C.shared.stats import mean_std_ci


def test_unreached_is_nan_not_zero() -> None:
    assert np.isnan(samples_at_target_tpr(np.array([0.1, 0.2]), delta=0.05))
    assert samples_at_target_tpr(np.array([0.1, 0.96]), delta=0.05) == 2.0


def test_summary_keeps_censored_in_denominator() -> None:
    summary = summarize_samples_at_target(
        [100.0, 200.0, float("nan"), None],
        target_tpr=0.95,
        max_horizon=4000,
        method="uniform",
    )
    assert summary["target_tpr"] == pytest.approx(0.95)
    assert summary["n_trials"] == 4
    assert summary["n_reached"] == 2
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["conditional_mean_among_successes"] == pytest.approx(150.0)
    assert summary["median_reaching_time"] == pytest.approx(200.0)
    assert summary["right_censoring_rate"] == pytest.approx(0.5)
    assert summary["max_horizon"] == 4000
    assert summary["unreached_status"] == CENSORED
    assert summary["per_trial_status"] == [100.0, 200.0, CENSORED, CENSORED]
    assert summary["rank_by_conditional_mean_alone"] is False


def test_zero_and_not_run_are_not_successes() -> None:
    summary = summarize_samples_at_target(
        [0.0, "NOT_RUN", CENSORED],
        target_tpr=0.95,
        max_horizon=100,
    )
    assert summary["n_reached"] == 0
    assert summary["conditional_mean_among_successes"] == UNDEFINED
    assert summary["median_reaching_time"] == NOT_IDENTIFIABLE
    assert summary["per_trial_status"] == [CENSORED, CENSORED, CENSORED]


def test_median_not_identifiable_when_majority_censored() -> None:
    summary = summarize_samples_at_target(
        [10.0, float("nan"), float("nan")],
        target_tpr=0.95,
        max_horizon=50,
    )
    assert summary["median_reaching_time"] == NOT_IDENTIFIABLE
    assert summary["success_rate"] == pytest.approx(1.0 / 3.0)


def test_all_reached_scalar_is_mean() -> None:
    summary = summarize_samples_at_target([10.0, 20.0], target_tpr=0.95, max_horizon=50)
    cells = samples_at_target_metric_cells(summary)
    assert cells["Samples@Target TPR↓"] == pytest.approx(15.0)


def test_partial_success_scalar_is_censored_not_conditional_mean() -> None:
    summary = summarize_samples_at_target(
        [10.0, float("nan")],
        target_tpr=0.95,
        max_horizon=50,
    )
    cells = samples_at_target_metric_cells(summary)
    assert cells["Samples@Target TPR↓"] == CENSORED
    assert cells["conditional_mean_among_successes"] == pytest.approx(10.0)
    assert cells["n_reached"] == 1


def test_empty_trials_are_not_run() -> None:
    summary = summarize_samples_at_target([], target_tpr=0.95, max_horizon=50)
    cells = samples_at_target_metric_cells(summary)
    assert cells["Samples@Target TPR↓"] == NOT_RUN


def test_do_not_rank_by_conditional_mean_when_success_rates_differ() -> None:
    a = summarize_samples_at_target([100.0, 120.0], target_tpr=0.95, max_horizon=200)
    b = summarize_samples_at_target([50.0, float("nan")], target_tpr=0.95, max_horizon=200)
    assert can_rank_by_conditional_mean([a, a]) is True
    assert can_rank_by_conditional_mean([a, b]) is False


def test_mean_std_ci_is_not_the_official_aggregator() -> None:
    times = [3520.5, 3520.5, float("nan")]
    dropped = mean_std_ci(times)
    summary = summarize_samples_at_target(times, target_tpr=0.95, max_horizon=4000)
    assert dropped["n"] == 2.0
    assert dropped["mean"] == pytest.approx(3520.5)
    assert summary["n_trials"] == 3
    assert summary["n_reached"] == 2
    assert samples_at_target_metric_cells(summary)["Samples@Target TPR↓"] == CENSORED


def test_serialize_writes_censored_token(tmp_path) -> None:
    summary = summarize_samples_at_target(
        [12.0, float("nan")],
        target_tpr=0.95,
        max_horizon=40,
    )
    path = write_run(
        "censor_demo",
        table_id="aaai_robust_gaussian",
        version="v0_1_test",
        content="censoring",
        data="toy",
        config={"K": 2},
        seed=0,
        metrics=attach_censoring_metrics({"TPR↑": 0.5}, summary),
        extra={"samples_at_target": summary},
        raw_dir=tmp_path,
    )
    payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["Samples@Target TPR↓"] == CENSORED
    assert payload["metrics"]["n_reached"] == 1
    assert payload["metrics"]["success_rate"] == pytest.approx(0.5)
    assert payload["samples_at_target"]["unreached_status"] == CENSORED
    assert "Samples@Target TPR↓" in (path / "metrics.csv").read_text(encoding="utf-8")
