"""Idea B v1.1 experiment runner (importable; CLI lives in experiments/v1.1/)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from idea_B.benchmarks.B_Env1_Endogenous_Selection.environment import (
    make_environment as make_env1,
)
from idea_B.benchmarks.B_Env1_Endogenous_Selection.environment import protocol_skill_defaults
from idea_B.benchmarks.B_Env2_Observational_Equivalence.environment import (
    make_environment as make_env2,
)
from idea_B.benchmarks.B_Env3_Recovery_After_Latent_Change.environment import (
    make_environment as make_env3,
)
from idea_B.experiments.plotting import save_run_plots
from idea_B.methods.v1_1 import load_config as load_b_config
from idea_B.methods.v1_1 import run_closed_loop
from shared.evaluation.results import write_run
from shared.hardware.probe import collect_environment
from shared.plotting.figures import plot_bars, plot_series
from shared.statistical_tests.paired_bootstrap import paired_bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "idea_B" / "results" / "v1.1"
ABLATION_ROOT = REPO_ROOT / "idea_B" / "ablations" / "v1.1"

TIER1_METHODS = (
    "Uniform",
    "EqualSkill",
    "E-optimal",
    "D-optimal",
    "B-v1.0",
    "B-v1.1c",
    "Greedy",
)
TIER2_PRIMARY_METHODS = TIER1_METHODS
SKILL_COUNT_METHODS = ("Uniform", "E-optimal", "B-v1.1c")
ENV3_METHODS = ("Uniform", "E-optimal", "B-v1.1c")

ABLATIONS = {
    "B1_No_Observability_Control": {
        "component": "allocation",
        "methods": ("B-v1.1c", "Uniform"),
        "note": "Replace v1.1c allocation with uniform q=b.",
    },
    "B2_EqualSkill_Instead": {
        "component": "equal_skill",
        "methods": ("B-v1.1c", "EqualSkill"),
        "note": "Corrected EqualSkill (equal expected audit count) vs v1.1c.",
    },
    "B3_No_Information_Discount": {
        "component": "information_discount",
        "methods": ("B-v1.1c",),
        "overrides": {"information_discount": 1.0},
        "label": "B-v1.1c_no_discount",
        "note": "v1.1c with information_discount=1 (no discounting of stale Gramian).",
    },
    "B4_E_Optimal": {
        "component": "e_optimal_control",
        "methods": ("B-v1.1c", "E-optimal"),
        "note": "Classical E-optimal vs full v1.1c controller.",
    },
    "B5_D_Optimal": {
        "component": "d_optimal_control",
        "methods": ("B-v1.1c", "D-optimal"),
        "note": "Classical D-optimal vs full v1.1c controller.",
    },
    "B6_No_Endogenous_Feedback": {
        "component": "endogenous_feedback",
        "methods": ("B-v1.1c",),
        "overrides": {"freeze_theta": True},
        "label": "B-v1.1c_frozen_theta",
        "note": "Turn off endogenous feedback by freezing theta on Env1.",
    },
    "B7_Greedy": {
        "component": "greedy",
        "methods": ("B-v1.1c", "Greedy"),
        "note": "Greedy vs v1.1c. Greedy may coincide with natural serving on Env1.",
    },
    "B8_V10_Controller": {
        "component": "v1_0_controller",
        "methods": ("B-v1.1c", "B-v1.0"),
        "note": "Unchanged v1.0 observability controller vs v1.1c.",
    },
    "B9_Proportional_Frequency": {
        "component": "proportional_frequency",
        "methods": ("B-v1.1c", "Proportional"),
        "note": "Frequency-proportional allocation vs v1.1c.",
    },
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > 128:
            return {"shape": list(value.shape), "omitted": True}
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def interval_side(ci_low: float, ci_high: float) -> str:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "undefined"
    if ci_high < 0.0:
        return "entirely_negative"
    if ci_low > 0.0:
        return "entirely_positive"
    return "covers_zero"


def budget_band_ok(mean_realized: float, budget: float, tol: float = 0.01) -> bool:
    if not np.isfinite(mean_realized):
        return False
    return abs(float(mean_realized) - float(budget)) <= float(tol)


def make_env(env_key: str, *, seed: int, d: int, population: int, horizon: int, feature_noise: float) -> Any:
    defaults = protocol_skill_defaults(int(d))
    kwargs: dict[str, Any] = {
        "d": int(d),
        "population_per_round": int(population),
        "feature_noise": float(feature_noise),
        "mix": defaults["mix"],
        "theta_star": defaults["theta_star"],
        "weak_skill": defaults["weak_skill"],
    }
    if env_key == "Env1":
        return make_env1(**kwargs)
    if env_key == "Env2":
        return make_env2(unrecoverable=False, **kwargs)
    if env_key == "Env3_recoverable":
        return make_env3(unrecoverable=False, seed=int(seed), horizon=int(horizon), **kwargs)
    if env_key == "Env3_unrecoverable":
        return make_env3(unrecoverable=True, seed=int(seed), horizon=int(horizon), **kwargs)
    raise ValueError(f"unknown env {env_key!r}")


def experiment_id(prefix: str, env_key: str, method: str, budget: float, seed: int, d: int) -> str:
    method_tag = str(method).replace("/", "-")
    return f"{prefix}_{env_key}_d{int(d)}_{method_tag}_b{float(budget):.2f}_s{int(seed)}"


def environment_record() -> dict[str, Any]:
    env = dict(collect_environment(REPO_ROOT))
    env["experiments_run"] = True
    env["idea"] = "B"
    env["version"] = "v1.1"
    env["python_executable"] = sys.executable
    env["working_copy"] = str(REPO_ROOT)
    return env


def _method_overrides(cfg: dict[str, Any], method: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = {
        "ridge_lambda": float(cfg.get("ridge_lambda", 0.5)),
        "smoothing": float(cfg.get("v1_0_smoothing", 0.6)),
        "mix_uniform": float(cfg.get("v1_0_mix_uniform", 0.1)),
        "sgd_learning_rate": float(cfg.get("sgd_learning_rate", 0.15)),
        "lazy_period": int(cfg.get("lazy_period", 5)),
        "information_discount": float(cfg.get("information_discount", 0.90)),
        "sliding_window": int(cfg.get("sliding_window", 8)),
        "e_optimal_iterations": int(cfg.get("e_optimal_iterations", 12)),
        "d_optimal_iterations": int(cfg.get("d_optimal_iterations", 12)),
        "information_metric": str(cfg.get("v1_0_information_metric", "lambda_min")),
    }
    if extra:
        overrides.update(extra)
    _ = method
    return overrides


def save_run(
    dest: Path,
    env_obj: Any,
    result: dict[str, Any],
    *,
    env_key: str,
    method: str,
    seed: int,
    budget: float,
    d: int,
    prefix: str,
    wall_s: float,
    hardware: dict[str, Any],
    write_plots: bool = True,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    config = dict(result["config"])
    config.update(
        {
            "experiment_id": dest.name,
            "environment": env_key,
            "env_name": env_obj.name,
            "d": int(d),
            "prefix": prefix,
            "paired_world_seed": int(seed),
            "novelty_claim": False,
            "higher_eigenvalue_is_not_success": True,
        }
    )
    summary = dict(result["summary"])
    trajectory = result["trajectory"]
    compact = {
        k: trajectory[k]
        for k in (
            "round",
            "true_mse",
            "excess_mse",
            "worst_skill_mse",
            "population_weighted_mse",
            "parameter_error",
            "cumulative_regret",
            "realized_audit_fraction",
            "expected_audit_fraction",
            "lambda_min_realized",
            "lambda_min_conditional",
            "lambda_min_candidates",
            "weak_skill_mse",
            "weak_skill_error",
        )
        if k in trajectory
    }
    metrics = {
        "experiment_id": dest.name,
        "env": env_key,
        "method": method,
        "seed": int(seed),
        "budget": float(budget),
        "d": int(d),
        "summary": json_safe(summary),
        "collapse": json_safe(result["collapse"]),
        "recovery": json_safe(result.get("recovery")),
        "detection": json_safe(result.get("detection")),
        "greedy_audit_subset_of_natural": summary.get("audit_subset_of_natural_fraction"),
        "runtime_s": float(wall_s),
        "trajectory": json_safe(compact),
    }
    log = (
        f"utc={datetime.now(timezone.utc).isoformat()}\n"
        f"experiment_id={dest.name}\n"
        f"env={env_key} method={method} seed={seed} budget={budget} d={d}\n"
        f"late_worst_skill_mse={summary.get('late_worst_skill_mse')}\n"
        f"cumulative_regret={summary.get('cumulative_regret')}\n"
        f"realized_audit_fraction={summary.get('realized_audit_fraction')}\n"
        f"recovery_time={summary.get('recovery_time')}\n"
        f"runtime_s={wall_s:.4f}\n"
        f"higher_eigenvalue_is_not_success=True\n"
    )
    hardware_payload = dict(hardware)
    hardware_payload["env_metadata"] = json_safe(env_obj.metadata())
    write_run(dest, config, metrics, hardware_payload, log)
    if write_plots:
        save_run_plots(dest / "plots", trajectory, int(d))


def run_one(
    env_key: str,
    method: str,
    seed: int,
    budget: float,
    *,
    prefix: str,
    d: int = 3,
    horizon: int | None = None,
    population: int | None = None,
    skip_existing: bool = True,
    method_overrides: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    cfg = load_b_config()
    hzn = int(horizon if horizon is not None else cfg.get("horizon", 32))
    pop = int(population if population is not None else cfg.get("population_per_round", 40))
    feat = float(cfg.get("feature_noise", 0.05))
    method_name = str(method)
    display = label or method_name
    exp = experiment_id(prefix, env_key, display, float(budget), int(seed), int(d))
    dest = RESULTS_ROOT / exp
    if skip_existing and (dest / "metrics.json").is_file():
        payload = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
        return {
            "experiment_id": exp,
            "path": str(dest),
            "env": env_key,
            "method": display,
            "base_method": method_name,
            "seed": int(seed),
            "budget": float(budget),
            "d": int(d),
            "summary": payload.get("summary") or {},
            "recovery": payload.get("recovery"),
            "skipped_existing": True,
        }
    env_obj = make_env(env_key, seed=int(seed), d=int(d), population=pop, horizon=hzn, feature_noise=feat)
    overrides = _method_overrides(cfg, method_name, method_overrides)
    t0 = time.perf_counter()
    result = run_closed_loop(
        env_obj,
        method_name,
        seed=int(seed),
        budget=float(budget),
        horizon=hzn,
        **overrides,
    )
    wall = time.perf_counter() - t0
    hw = hardware if hardware is not None else environment_record()
    save_run(
        dest,
        env_obj,
        result,
        env_key=env_key,
        method=display,
        seed=int(seed),
        budget=float(budget),
        d=int(d),
        prefix=prefix,
        wall_s=wall,
        hardware=hw,
        write_plots=True,
    )
    return {
        "experiment_id": exp,
        "path": str(dest),
        "env": env_key,
        "method": display,
        "base_method": method_name,
        "seed": int(seed),
        "budget": float(budget),
        "d": int(d),
        "summary": result["summary"],
        "recovery": result.get("recovery"),
        "skipped_existing": False,
        "runtime_s": wall,
    }


def _mean_std(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "values": values}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1) if finite.size > 1 else 0.0),
        "n": int(finite.size),
        "values": [float(v) if np.isfinite(v) else None for v in arr],
    }


def _paired_field(records: list[dict[str, Any]], cand: str, base: str, field: str) -> dict[str, Any]:
    by_seed_c = {int(r["seed"]): r for r in records if r["method"] == cand}
    by_seed_b = {int(r["seed"]): r for r in records if r["method"] == base}
    seeds = sorted(set(by_seed_c) & set(by_seed_b))
    if not seeds:
        return {"reason": "no paired seeds", "field": field}
    a_vals = []
    b_vals = []
    used = []
    for seed in seeds:
        av = by_seed_c[seed]["summary"].get(field)
        bv = by_seed_b[seed]["summary"].get(field)
        if av is None or bv is None:
            continue
        if not np.isfinite(float(av)) or not np.isfinite(float(bv)):
            continue
        a_vals.append(float(av))
        b_vals.append(float(bv))
        used.append(int(seed))
    if len(used) < 2:
        return {"reason": "not enough finite paired seeds", "field": field, "seeds": used}
    boot = paired_bootstrap_ci(a_vals, b_vals, n_boot=10000, alpha=0.05, seed=0)
    boot["interval_side"] = interval_side(boot["ci_low"], boot["ci_high"])
    boot["field"] = field
    boot["candidate"] = cand
    boot["baseline"] = base
    boot["candidate_values"] = a_vals
    boot["baseline_values"] = b_vals
    boot["seeds"] = used
    return boot


def pack_cell(records: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    methods = []
    for r in records:
        if r["method"] not in methods:
            methods.append(r["method"])
    payload: dict[str, Any] = {"methods": {}, "paired_vs_E_optimal": {}, "paired_vs_D_optimal": {}}
    for method in methods:
        subset = [r for r in records if r["method"] == method]
        worst = [float(r["summary"].get("late_worst_skill_mse", float("nan"))) for r in subset]
        regret = [float(r["summary"].get("cumulative_regret", float("nan"))) for r in subset]
        realized = [float(r["summary"].get("realized_audit_fraction", float("nan"))) for r in subset]
        recovery = [r["summary"].get("recovery_time") for r in subset]
        censored = [bool(r["summary"].get("recovery_censored")) for r in subset]
        n_seeds = len(subset)
        rec_vals = [float(v) for v in recovery if v is not None and np.isfinite(float(v))]
        payload["methods"][method] = {
            "n_seeds": n_seeds,
            "seeds": [int(r["seed"]) for r in subset],
            "preliminary": n_seeds <= 3,
            "late_worst_skill_mse": _mean_std(worst),
            "cumulative_regret": _mean_std(regret),
            "realized_audit_fraction": _mean_std(realized),
            "budget_band_ok": budget_band_ok(float(np.nanmean(realized)) if realized else float("nan"), budget),
            "recovery_time": _mean_std(rec_vals) if rec_vals else {"mean": None, "n": 0},
            "recovery_censored_fraction": float(np.mean(censored)) if censored else None,
            "noise_floor": 0.0225,
            "paths": [r["path"] for r in subset],
        }
    for cand in methods:
        if cand in {"E-optimal", "D-optimal"}:
            continue
        payload["paired_vs_E_optimal"][cand] = {
            "late_worst_skill_mse": _paired_field(records, cand, "E-optimal", "late_worst_skill_mse"),
            "cumulative_regret": _paired_field(records, cand, "E-optimal", "cumulative_regret"),
        }
        payload["paired_vs_D_optimal"][cand] = {
            "late_worst_skill_mse": _paired_field(records, cand, "D-optimal", "late_worst_skill_mse"),
            "cumulative_regret": _paired_field(records, cand, "D-optimal", "cumulative_regret"),
        }
    return payload


def write_cell_plots(records: list[dict[str, Any]], dest: Path, title: str, n_seeds: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    methods = []
    for r in records:
        if r["method"] not in methods:
            methods.append(r["method"])
    if not methods:
        return
    worst_mean, worst_se, regret_mean, regret_se = [], [], [], []
    for method in methods:
        subset = [r for r in records if r["method"] == method]
        w = np.array([float(r["summary"]["late_worst_skill_mse"]) for r in subset], dtype=float)
        g = np.array([float(r["summary"]["cumulative_regret"]) for r in subset], dtype=float)
        worst_mean.append(float(np.nanmean(w)))
        worst_se.append(float(np.nanstd(w, ddof=1) / np.sqrt(max(np.sum(np.isfinite(w)), 1))) if w.size > 1 else 0.0)
        regret_mean.append(float(np.nanmean(g)))
        regret_se.append(float(np.nanstd(g, ddof=1) / np.sqrt(max(np.sum(np.isfinite(g)), 1))) if g.size > 1 else 0.0)
    fig = plot_bars(
        methods,
        {"late worst-skill MSE": worst_mean},
        yerr={"late worst-skill MSE": worst_se},
        title=f"{title}: late worst-skill MSE",
        xlabel="method",
        ylabel="worst-skill MSE (late window)",
        n_seeds=n_seeds,
        path=dest / "worst_skill_mse.png",
    )
    plt.close(fig)
    fig = plot_bars(
        methods,
        {"cumulative regret": regret_mean},
        yerr={"cumulative regret": regret_se},
        title=f"{title}: cumulative regret vs noise floor 0.0225",
        xlabel="method",
        ylabel="sum_t max(true_mse_t - 0.0225, 0)",
        n_seeds=n_seeds,
        path=dest / "cumulative_regret.png",
    )
    plt.close(fig)
    # Trajectory mean ± SE if trajectories exist.
    series: dict[str, list[float]] = {}
    yerr: dict[str, list[float]] = {}
    x = None
    for method in methods:
        mats = []
        for r in records:
            if r["method"] != method:
                continue
            traj_path = Path(r["path"]) / "metrics.json"
            if not traj_path.is_file():
                continue
            payload = json.loads(traj_path.read_text(encoding="utf-8"))
            traj = (payload.get("trajectory") or {}).get("worst_skill_mse")
            if traj:
                mats.append(np.asarray(traj, dtype=float))
        if not mats:
            continue
        arr = np.vstack(mats)
        series[method] = np.mean(arr, axis=0).tolist()
        se = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
        yerr[method] = se.tolist()
        x = list(range(arr.shape[1]))
    if series and x is not None:
        fig = plot_series(
            x,
            series,
            yerr=yerr,
            title=f"{title}: worst-skill MSE vs round",
            xlabel="round",
            ylabel="worst-skill MSE",
            n_seeds=n_seeds,
            path=dest / "worst_skill_mse_vs_round.png",
        )
        plt.close(fig)


def _load_prefix(prefix: str) -> list[dict[str, Any]]:
    records = []
    if not RESULTS_ROOT.is_dir():
        return records
    for path in sorted(RESULTS_ROOT.glob(f"{prefix}_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "experiment_id": payload.get("experiment_id", path.parent.name),
                "path": str(path.parent),
                "env": payload.get("env"),
                "method": payload.get("method"),
                "seed": int(payload.get("seed")),
                "budget": float(payload.get("budget")),
                "d": int(payload.get("d", 3)),
                "summary": payload.get("summary") or {},
                "recovery": payload.get("recovery"),
            }
        )
    return records


def run_tier1(*, skip_existing: bool = True) -> dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    hardware = environment_record()
    jobs = []
    for env_key in ("Env1", "Env2", "Env3_recoverable", "Env3_unrecoverable"):
        for method in TIER1_METHODS:
            for seed in (0, 1, 2):
                rec = run_one(
                    env_key,
                    method,
                    seed,
                    0.05,
                    prefix="t1",
                    d=3,
                    skip_existing=skip_existing,
                    hardware=hardware,
                )
                jobs.append(rec)
                print(f"[B TIER1] {rec['experiment_id']}", flush=True)
    records = _load_prefix("t1")
    by_env: dict[str, Any] = {}
    for env_key in ("Env1", "Env2", "Env3_recoverable", "Env3_unrecoverable"):
        subset = [r for r in records if r["env"] == env_key]
        by_env[env_key] = pack_cell(subset, 0.05)
        write_cell_plots(subset, RESULTS_ROOT / "plots" / f"tier1_{env_key}", f"Idea B Tier 1 {env_key}", 3)
    summary = {
        "idea": "B",
        "tier": 1,
        "preliminary": True,
        "n_seeds": 3,
        "label": "PRELIMINARY",
        "n_runs": len(jobs),
        "cells": by_env,
        "noise_floor": 0.0225,
        "promising_not_claimed": True,
        "note": "Tier 1 is a correctness smoke. Eigenvalues are not used to decide success.",
    }
    (RESULTS_ROOT / "TIER1_SUMMARY.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def run_tier2(*, skip_existing: bool = True, extra_seeds: list[int] | None = None) -> dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    hardware = environment_record()
    seeds30 = list(range(30))
    if extra_seeds:
        seeds30 = sorted(set(seeds30) | set(extra_seeds))
    seeds10 = list(range(10))
    n_runs = 0
    for env_key in ("Env1", "Env2"):
        for method in TIER2_PRIMARY_METHODS:
            for seed in seeds30:
                rec = run_one(env_key, method, seed, 0.05, prefix="t2_primary", d=3, skip_existing=skip_existing, hardware=hardware)
                n_runs += 1
                if seed % 10 == 0:
                    print(f"[B TIER2 primary] {rec['experiment_id']}", flush=True)
    for method in TIER2_PRIMARY_METHODS:
        for seed in seeds10:
            rec = run_one("Env1", method, seed, 0.10, prefix="t2_budget", d=3, skip_existing=skip_existing, hardware=hardware)
            n_runs += 1
            print(f"[B TIER2 budget] {rec['experiment_id']}", flush=True)
    for d in (2, 5):
        for method in SKILL_COUNT_METHODS:
            for seed in seeds10:
                rec = run_one("Env1", method, seed, 0.05, prefix="t2_skills", d=d, skip_existing=skip_existing, hardware=hardware)
                n_runs += 1
                print(f"[B TIER2 skills d={d}] {rec['experiment_id']}", flush=True)
    for env_key in ("Env3_recoverable", "Env3_unrecoverable"):
        for method in ENV3_METHODS:
            for seed in seeds10:
                rec = run_one(env_key, method, seed, 0.05, prefix="t2_env3", d=3, skip_existing=skip_existing, hardware=hardware)
                n_runs += 1
                print(f"[B TIER2 Env3] {rec['experiment_id']}", flush=True)

    primary = _load_prefix("t2_primary")
    budget = _load_prefix("t2_budget")
    skills = _load_prefix("t2_skills")
    env3 = _load_prefix("t2_env3")
    cells: dict[str, Any] = {}
    for env_key in ("Env1", "Env2"):
        subset = [r for r in primary if r["env"] == env_key]
        cells[env_key] = pack_cell(subset, 0.05)
        write_cell_plots(subset, RESULTS_ROOT / "plots" / f"tier2_{env_key}_b0.05", f"Idea B Tier 2 {env_key} b=0.05 d=3", 30)
    cells["Env1_b0.10"] = pack_cell(budget, 0.10)
    write_cell_plots(budget, RESULTS_ROOT / "plots" / "tier2_Env1_b0.10", "Idea B Tier 2 Env1 b=0.10 d=3", 10)
    for d in (2, 5):
        subset = [r for r in skills if r["d"] == d]
        cells[f"Env1_d{d}"] = pack_cell(subset, 0.05)
        write_cell_plots(subset, RESULTS_ROOT / "plots" / f"tier2_Env1_d{d}", f"Idea B Tier 2 Env1 d={d} b=0.05", 10)
    for env_key in ("Env3_recoverable", "Env3_unrecoverable"):
        subset = [r for r in env3 if r["env"] == env_key]
        packed = pack_cell(subset, 0.05)
        packed["paired_recovery_v11c_vs_E_optimal"] = _paired_field(subset, "B-v1.1c", "E-optimal", "recovery_time")
        packed["paired_recovery_v11c_vs_Uniform"] = _paired_field(subset, "B-v1.1c", "Uniform", "recovery_time")
        cells[env_key] = packed
        write_cell_plots(subset, RESULTS_ROOT / "plots" / f"tier2_{env_key}", f"Idea B Tier 2 {env_key}", 10)
    summary = {
        "idea": "B",
        "tier": 2,
        "n_runs_attempted": n_runs,
        "cells": cells,
        "noise_floor": 0.0225,
        "promising_not_claimed": True,
        "higher_eigenvalue_is_not_success": True,
        "skipped": ["b=0.01 and b=0.20 leftover-time only; not in main conclusion"],
        "note": "Numbers only. PROMISING is not written here. Main H_B1 is Env1/Env2, 3 skills, b=0.05, 30 seeds.",
    }
    (RESULTS_ROOT / "TIER2_SUMMARY.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def run_ablations(*, skip_existing: bool = True, seeds: tuple[int, ...] = (0, 1, 2), only: str | None = None) -> dict[str, Any]:
    hardware = environment_record()
    index: dict[str, Any] = {}
    selected = ABLATIONS.items() if only is None else ((only, ABLATIONS[only]),)
    for ablation_id, spec in selected:
        dest_root = ABLATION_ROOT / ablation_id / "results"
        dest_root.mkdir(parents=True, exist_ok=True)
        ids = []
        overrides = spec.get("overrides")
        label = spec.get("label")
        for method in spec["methods"]:
            extra = overrides if method == "B-v1.1c" else None
            disp = label if (label and method == "B-v1.1c") else None
            for seed in seeds:
                rec = run_one(
                    "Env1",
                    method,
                    int(seed),
                    0.05,
                    prefix=f"ablation_{ablation_id}",
                    d=3,
                    skip_existing=skip_existing,
                    method_overrides=extra,
                    hardware=hardware,
                    label=disp,
                )
                src = RESULTS_ROOT / rec["experiment_id"]
                dst = dest_root / rec["experiment_id"]
                if src.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                    for name in ("config.yaml", "metrics.json", "metrics.csv", "run.log", "environment.json"):
                        f = src / name
                        if f.is_file():
                            (dst / name).write_bytes(f.read_bytes())
                    if (src / "plots").is_dir():
                        (dst / "plots").mkdir(parents=True, exist_ok=True)
                        for png in (src / "plots").glob("*.png"):
                            (dst / "plots" / png.name).write_bytes(png.read_bytes())
                ids.append(rec["experiment_id"])
                print(f"[B ablation {ablation_id}] {rec['experiment_id']}", flush=True)
        index[ablation_id] = {
            "component": spec["component"],
            "note": spec["note"],
            "seeds": list(seeds),
            "preliminary": len(seeds) <= 3,
            "experiment_ids": ids,
        }
    (RESULTS_ROOT / "ABLATION_INDEX.json").write_text(
        json.dumps(json_safe(index), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return index


def borderline_primary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    cells = summary.get("cells") or {}
    for env_key in ("Env1", "Env2"):
        block = cells.get(env_key) or {}
        for baseline_key in ("paired_vs_E_optimal", "paired_vs_D_optimal"):
            paired = block.get(baseline_key) or {}
            cand = paired.get("B-v1.1c") or {}
            for field, boot in cand.items():
                if not isinstance(boot, dict) or boot.get("interval_side") != "covers_zero":
                    continue
                low, high = float(boot.get("ci_low", float("nan"))), float(boot.get("ci_high", float("nan")))
                width = high - low
                if not np.isfinite(width) or width <= 0:
                    continue
                if min(abs(low), abs(high)) <= 0.20 * width:
                    out.append(
                        {
                            "env": env_key,
                            "baseline": baseline_key,
                            "field": field,
                            "ci_low": low,
                            "ci_high": high,
                            "cheap": True,
                        }
                    )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Idea B v1.1 experiments")
    p.add_argument("--tier", choices=["1", "2", "ablations", "all", "summarize"], default="all")
    p.add_argument("--no-skip", action="store_true")
    p.add_argument("--extend-seeds", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skip = not args.no_skip
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if args.tier in {"1", "all", "summarize"}:
        if args.tier != "summarize":
            run_tier1(skip_existing=skip)
        elif list(RESULTS_ROOT.glob("t1_*/metrics.json")):
            run_tier1(skip_existing=True)
    if args.tier in {"2", "all"}:
        extra = list(range(30, args.extend_seeds)) if args.extend_seeds > 30 else None
        summary = run_tier2(skip_existing=skip, extra_seeds=extra)
        borderline = borderline_primary(summary)
        (RESULTS_ROOT / "BORDERLINE.json").write_text(
            json.dumps(json_safe({"cells": borderline}), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if borderline and args.extend_seeds <= 30:
            print(f"[B] borderline cells: {borderline}. Extend with --extend-seeds 50 if cheap.", flush=True)
    if args.tier == "summarize" and list(RESULTS_ROOT.glob("t2_*/metrics.json")):
        run_tier2(skip_existing=True)
    if args.tier in {"ablations", "all"}:
        run_ablations(skip_existing=skip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
