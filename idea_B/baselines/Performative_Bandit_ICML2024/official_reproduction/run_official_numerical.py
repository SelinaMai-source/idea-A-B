#!/usr/bin/env python3
"""Convert and run the official numerical cells from bandit-icml24.ipynb.

This script concatenates the official notebook's numerical experiment cells
(Appendix H / Figures 4-6) and executes them with a non-interactive backend.
Algorithm code is copied from the notebook; the only local controls are:
  - matplotlib Agg backend
  - numpy seed=0 recorded for local reproducibility (notebook has none)
  - JSON/PNG artifacts written under this directory
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy  # noqa: F401  # imported in official notebook

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "metrics.json"
OUT_PNG = ROOT / "pr_comparison.png"
OUT_LOG = ROOT / "run.log"


def as_float(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    return float(arr[0])


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main() -> int:
    np.random.seed(0)
    t0 = time.time()
    log_f = OUT_LOG.open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    print("official_source: bandit-icml24.ipynb")
    print("numpy_seed: 0  # local reproducibility only; notebook did not set a seed")
    print("entropy_imported:", entropy is not None)

    # === official cell: objective functions and Bernoulli KL ===
    def f(y):
        return -y**2 * (y - 1) ** 2 - (1 - y**2) * (y) ** 2

    def g(x):
        return -x * (np.sqrt(x) - 1) ** 2 - (1 - x) * x

    def kl_divergence(p, q):
        if p == 0 or q == 0 or p == 1 or q == 1:
            return np.inf
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def two_point_feedback_zeroth_order_optimization(
        func, dim, x0, alpha=0.01, beta=0.1, num_iterations=1000
    ):
        x = np.array(x0)
        f_values = []
        for i in range(num_iterations):
            d = np.random.randn(dim)
            d = d / np.linalg.norm(d)
            f_plus = func(x + beta * d)
            f_minus = func(x - beta * d)
            grad_est = (f_plus - f_minus) / (2 * beta) * d
            x = x - alpha * grad_est
            x = np.clip(x, 0.01, 0.95)
            f_values.append(func(x))
        return x, f_values

    def LearnModel(x_target, initial_y, alpha=0.01, beta=0.1, num_iterations=10):
        y_current = initial_y
        for _ in range(num_iterations):
            d = np.random.randn(dim)
            d = d / np.linalg.norm(d)
            y_plus = y_current + beta * d
            y_minus = y_current - beta * d
            kl_plus = kl_divergence(x_target, y_plus**2)
            kl_minus = kl_divergence(x_target, y_minus**2)
            grad_kl = (kl_plus - kl_minus) / (2.0 * beta) * d
            y_current = y_current - alpha * grad_kl
            y_current = np.clip(y_current, 0.01, 0.95)
            if np.abs(grad_kl) <= 10e-5:
                print("here")
                break
        return y_current

    def modified_two_point_feedback_optimization(
        func, dim, x0, alpha=0.001, beta=0.1, num_iterations=100
    ):
        x = np.array(x0)
        f_values = []
        for i in range(num_iterations):
            d = np.random.randn(dim)
            d = d / np.linalg.norm(d)
            y_plus = LearnModel(x + beta * d, x)
            y_minus = LearnModel(x - beta * d, x)
            grad_est = (func(y_plus) - func(y_minus)) / (2 * beta) * d
            x = x - alpha * grad_est
            x = np.clip(x, 0.1, 0.9)
            f_values.append(func(x))
        return x, f_values

    dim = 1
    x0 = [0.3]

    print("=== single-run: reparameterized (phi) ===")
    x_opt_g, g_values = modified_two_point_feedback_optimization(
        func=g, dim=dim, x0=x0, alpha=0.01, beta=0.1, num_iterations=500
    )
    print("x_opt_g", x_opt_g, "final_g", as_float(g_values[-1]))

    print("=== single-run: vanilla two-point (theta) ===")
    x_opt_f, f_values = two_point_feedback_zeroth_order_optimization(
        func=f, dim=dim, x0=x0[0] ** 2, alpha=0.01, beta=0.1, num_iterations=500
    )
    print("x_opt_f", x_opt_f, "final_f", as_float(f_values[-1]))

    initial_points_x = np.linspace(0.25, 0.8, 10)
    print("initial_points_x", initial_points_x.tolist())

    g_iteration_values = []
    f_iteration_values = []
    num_iterations_g = 100
    num_iterations_f = 100
    for _ in range(num_iterations_g):
        g_iteration_values.append([])
    for _ in range(num_iterations_f):
        f_iteration_values.append([])

    for x0_i in initial_points_x:
        x0_list = [x0_i]
        y0_list = [x0_i**2]
        _, g_vals_i = modified_two_point_feedback_optimization(
            func=g, dim=1, x0=x0_list, alpha=0.01, beta=0.1, num_iterations=num_iterations_g
        )
        for i, val in enumerate(g_vals_i):
            g_iteration_values[i].append(val)
        _, f_vals_i = two_point_feedback_zeroth_order_optimization(
            func=f, dim=1, x0=y0_list, alpha=0.01, beta=0.1, num_iterations=num_iterations_f
        )
        for i, val in enumerate(f_vals_i):
            f_iteration_values[i].append(val)

    g_avg = np.array([np.mean(vals) for vals in g_iteration_values])
    g_std = np.array([np.std(vals) for vals in g_iteration_values])
    f_avg = np.array([np.mean(vals) for vals in f_iteration_values])
    f_std = np.array([np.std(vals) for vals in f_iteration_values])

    plt.figure(figsize=(10, 6), facecolor="white")
    iterations_g = range(num_iterations_g)
    iterations_f = range(num_iterations_f)
    plt.plot(iterations_g, g_avg, label=r"$PR(\phi)$")
    plt.fill_between(iterations_g, g_avg - g_std, g_avg + g_std, alpha=0.2)
    plt.plot(iterations_f, f_avg, label=r"$PR(\theta)$")
    plt.fill_between(iterations_f, f_avg - f_std, f_avg + f_std, alpha=0.2)
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("PR Value", fontsize=15)
    plt.title(r"PR($\phi$) vs PR($\theta$)", fontsize=15, fontweight="bold")
    plt.xticks(fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    legend = plt.legend(frameon=True, fontsize=12)
    legend.get_frame().set_alpha(0.7)
    legend.get_frame().set_edgecolor("black")
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - t0
    metrics = {
        "paper": "Performative Prediction with Bandit Feedback: Learning through Reparameterization",
        "venue": "ICML 2024",
        "official_notebook": "bandit-icml24.ipynb",
        "numpy_seed": 0,
        "elapsed_sec": elapsed,
        "single_run": {
            "num_iterations": 500,
            "x0": 0.3,
            "x_opt_phi": np.asarray(x_opt_g).tolist(),
            "final_PR_phi": as_float(g_values[-1]),
            "x_opt_theta": np.asarray(x_opt_f).tolist(),
            "final_PR_theta": as_float(f_values[-1]),
        },
        "multi_run": {
            "n_initial_points": 10,
            "initial_points_x": initial_points_x.tolist(),
            "num_iterations": 100,
            "PR_phi_final_mean": as_float(g_avg[-1]),
            "PR_phi_final_std": as_float(g_std[-1]),
            "PR_theta_final_mean": as_float(f_avg[-1]),
            "PR_theta_final_std": as_float(f_std[-1]),
            "PR_phi_mean_curve": np.asarray(g_avg, dtype=float).reshape(-1).tolist(),
            "PR_phi_std_curve": np.asarray(g_std, dtype=float).reshape(-1).tolist(),
            "PR_theta_mean_curve": np.asarray(f_avg, dtype=float).reshape(-1).tolist(),
            "PR_theta_std_curve": np.asarray(f_std, dtype=float).reshape(-1).tolist(),
        },
        "plot": str(OUT_PNG),
    }
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", OUT_JSON)
    print("wrote", OUT_PNG)
    print(f"elapsed_sec={elapsed:.3f}")
    print(
        "multi_run_final",
        "PR_phi",
        metrics["multi_run"]["PR_phi_final_mean"],
        "+/-",
        metrics["multi_run"]["PR_phi_final_std"],
        "PR_theta",
        metrics["multi_run"]["PR_theta_final_mean"],
        "+/-",
        metrics["multi_run"]["PR_theta_final_std"],
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
