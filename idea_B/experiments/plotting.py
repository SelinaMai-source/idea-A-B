"""Plots from stored trajectories. Values are caller-supplied; nothing is invented."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.plotting.figures import plot_bars, plot_series


def _close(fig: Any) -> None:
    plt.close(fig)


def _rounds(trajectory: Mapping[str, Sequence[Any]]) -> np.ndarray:
    return np.asarray(trajectory["round"], dtype=float)


def _arr(trajectory: Mapping[str, Sequence[Any]], key: str) -> np.ndarray:
    return np.asarray(trajectory[key], dtype=float)


def save_run_plots(plot_dir: Path, trajectory: Mapping[str, Sequence[Any]], n_skills: int) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = _rounds(trajectory)
    written: list[str] = []

    fig = plot_series(
        x,
        {
            "true population performance": _arr(trajectory, "true_performance"),
            "observed performance": _arr(trajectory, "observed_performance"),
        },
        title="True vs observed performance",
        xlabel="round",
        ylabel="performance 1/(1+MSE)",
        n_seeds=1,
        path=plot_dir / "true_vs_observed_performance.png",
    )
    _close(fig)
    written.append("true_vs_observed_performance.png")

    skill_acc = {
        f"skill {k} accuracy": 1.0 / (1.0 + np.nan_to_num(_arr(trajectory, f"skill_mse_{k}"), nan=1.0))
        for k in range(n_skills)
        if f"skill_mse_{k}" in trajectory
    }
    fig = plot_series(
        x,
        skill_acc,
        title="Skill-specific accuracy",
        xlabel="round",
        ylabel="1/(1+skill MSE)",
        n_seeds=1,
        path=plot_dir / "skill_specific_accuracy.png",
    )
    _close(fig)
    written.append("skill_specific_accuracy.png")

    skill_obs = {
        f"skill {k} observation rate": _arr(trajectory, f"skill_observation_rate_{k}")
        for k in range(n_skills)
        if f"skill_observation_rate_{k}" in trajectory
    }
    fig = plot_series(
        x,
        skill_obs,
        title="Skill-specific observation rates",
        xlabel="round",
        ylabel="fraction observed",
        n_seeds=1,
        path=plot_dir / "skill_specific_observation_rates.png",
    )
    _close(fig)
    written.append("skill_specific_observation_rates.png")

    eigs = {
        f"eigenvalue {k}": _arr(trajectory, f"eigenvalue_{k}")
        for k in range(n_skills)
        if f"eigenvalue_{k}" in trajectory
    }
    fig = plot_series(
        x,
        eigs,
        title="Information eigenvalues of W_T/T",
        xlabel="round",
        ylabel="eigenvalue of observed design Gramian / T",
        n_seeds=1,
        path=plot_dir / "information_eigenvalues.png",
    )
    _close(fig)
    written.append("information_eigenvalues.png")

    fig = plot_series(
        x,
        {"cumulative regret": _arr(trajectory, "cumulative_regret")},
        title="Regret vs rounds",
        xlabel="round",
        ylabel="cumulative excess MSE",
        n_seeds=1,
        path=plot_dir / "regret_vs_rounds.png",
    )
    _close(fig)
    written.append("regret_vs_rounds.png")
    return written


def save_aggregate_plots(
    plot_dir: Path,
    *,
    budget_labels: Sequence[str],
    performance_by_method: Mapping[str, Sequence[float]],
    performance_err: Mapping[str, Sequence[float]],
    collapse_x: Sequence[float],
    collapse_series: Mapping[str, Sequence[float]],
    baseline_labels: Sequence[str],
    baseline_series: Mapping[str, Sequence[float]],
    baseline_err: Mapping[str, Sequence[float]] | None,
    n_seeds: int,
) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    fig = plot_bars(
        budget_labels,
        performance_by_method,
        title="True performance vs feedback budget",
        xlabel="feedback budget",
        ylabel="late true performance 1/(1+MSE)",
        n_seeds=n_seeds,
        yerr=performance_err,
        path=plot_dir / "performance_vs_budget.png",
    )
    _close(fig)
    written.append("performance_vs_budget.png")

    fig = plot_series(
        collapse_x,
        collapse_series,
        title="Collapse and recovery trajectories",
        xlabel="round",
        ylabel="value",
        n_seeds=n_seeds,
        path=plot_dir / "collapse_recovery_trajectories.png",
    )
    _close(fig)
    written.append("collapse_recovery_trajectories.png")

    fig = plot_bars(
        baseline_labels,
        baseline_series,
        title="Controller vs generic control baselines",
        xlabel="method",
        ylabel="seed-mean late metric",
        n_seeds=n_seeds,
        yerr=baseline_err,
        path=plot_dir / "method_vs_generic_baselines.png",
    )
    _close(fig)
    written.append("method_vs_generic_baselines.png")
    return written
