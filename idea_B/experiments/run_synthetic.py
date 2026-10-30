"""Run Idea B synthetic closed-loop smoke experiments and write SMOKE_SUMMARY."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from idea_B.benchmarks.Synthetic_Closed_Loop.environments import make_environment
from idea_B.experiments.plotting import save_aggregate_plots
from idea_B.experiments.save_run import experiment_id, save_closed_loop_run
from idea_B.methods import run_closed_loop
from shared.statistical_tests.paired_bootstrap import paired_bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "idea_B" / "results"
DEFAULT_CONFIG = REPO_ROOT / "idea_B" / "configs" / "synthetic_closed_loop.yaml"

METHODS = [
    "ObservabilityController",
    "UniformExploration",
    "GreedyDeployment",
    "LazyDeployment",
    "StandardOnlineSGD",
    "FixedExplorationSchedule",
    "UncertaintyExploration",
    "NoFeedbackController",
    "FullFeedbackOracle",
]
ENVIRONMENTS = ["B1", "B2", "B3_recoverable", "B3_unrecoverable"]
SEEDS = [0, 1, 2]
BUDGETS = [0.01, 0.05, 0.10, 0.20]
PRIMARY_BUDGET = 0.05
HORIZON = 32
POPULATION = 40


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _env_kwargs(name: str, population: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"population_per_round": int(population), "d": 3}
    return kwargs


def run_one(
    env_name: str,
    method: str,
    seed: int,
    budget: float,
    *,
    horizon: int = HORIZON,
    population: int = POPULATION,
    results_root: Path = DEFAULT_RESULTS,
    write_plots: bool = True,
    prefix: str = "synthetic",
    env_overrides: dict[str, Any] | None = None,
    method_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env_kw = _env_kwargs(env_name, population)
    if env_overrides:
        env_kw.update(env_overrides)
    env = make_environment(env_name, **env_kw)
    overrides = dict(method_overrides or {})
    started = time.perf_counter()
    result = run_closed_loop(
        env,
        method,
        seed=int(seed),
        budget=float(budget),
        horizon=int(horizon),
        **overrides,
    )
    elapsed = time.perf_counter() - started
    run_id = experiment_id(env.name, method, float(budget), int(seed), prefix=prefix)
    dest = Path(results_root) / run_id
    log_text = (
        f"experiment_id={run_id}\n"
        f"environment={env.name} method={method} seed={seed} budget={budget}\n"
        f"policy_affects_feedback={env.policy_affects_feedback}\n"
        f"collapse={result['collapse']}\n"
        f"final_true_performance={result['summary']['final_true_performance']}\n"
        f"final_lambda_min={result['summary']['final_lambda_min']}\n"
        f"runtime_s={elapsed:.4f}\n"
        f"generic_control={result['config']['generic_control']}\n"
        f"official_reproduction=False\n"
    )
    save_closed_loop_run(dest, env, result, log_text, write_plots=write_plots)
    return {
        "experiment_id": run_id,
        "path": str(dest),
        "env": env.name,
        "method": method,
        "seed": int(seed),
        "budget": float(budget),
        "summary": result["summary"],
        "collapse": result["collapse"],
        "runtime_s": elapsed,
    }


def planned_jobs(
    *,
    environments: list[str],
    methods: list[str],
    seeds: list[int],
    budgets: list[float],
    primary_budget: float,
    time_cap_s: float,
) -> tuple[list[tuple[str, str, int, float]], list[dict[str, Any]]]:
    """All envs at primary budget, B2 at every budget, others if time allows."""
    jobs: list[tuple[str, str, int, float]] = []
    for env in environments:
        for method in methods:
            for seed in seeds:
                jobs.append((env, method, seed, float(primary_budget)))
    extra_budgets = [b for b in budgets if abs(b - primary_budget) > 1e-12]
    for method in methods:
        for seed in seeds:
            for budget in extra_budgets:
                jobs.append(("B2", method, seed, float(budget)))
    optional: list[tuple[str, str, int, float]] = []
    for env in environments:
        if env == "B2":
            continue
        for method in methods:
            for seed in seeds:
                for budget in extra_budgets:
                    optional.append((env, method, seed, float(budget)))
    return jobs, [
        {
            "optional": True,
            "jobs": optional,
            "time_cap_s": float(time_cap_s),
            "reason": "non-B2 extra budgets are optional if runtime is tight",
        }
    ]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if arr.size > 1 else 0.0)


def _index_records(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {(r["env"], r["method"], r["seed"], float(r["budget"])): r for r in records}


def _load_trajectory(path: str | Path) -> dict[str, Any] | None:
    traj_path = Path(path) / "trajectories.json"
    if not traj_path.is_file():
        return None
    return json.loads(traj_path.read_text(encoding="utf-8"))


def aggregate(
    records: list[dict[str, Any]],
    skips: list[dict[str, Any]],
    *,
    results_root: Path,
    n_seeds: int,
) -> dict[str, Any]:
    by_key: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    collapse_flags: dict[tuple[str, str, float], list[bool]] = defaultdict(list)
    for rec in records:
        key = (rec["env"], rec["method"], float(rec["budget"]))
        summary = rec["summary"]
        for field in (
            "late_true_performance",
            "late_observed_performance",
            "late_lambda_min",
            "late_weak_skill_error",
            "final_true_performance",
            "final_observed_performance",
            "final_lambda_min",
            "mean_feedback_cost",
            "cumulative_regret",
            "runtime_s",
            "trace_W_over_T",
        ):
            by_key[key][field].append(float(summary[field]))
        collapse_flags[key].append(bool(rec["collapse"]["endogenous_collapse"]))

    def pack(env: str, method: str, budget: float) -> dict[str, Any]:
        key = (env, method, float(budget))
        payload: dict[str, Any] = {"n": len(by_key.get(key, {}).get("late_true_performance", []))}
        for field, values in by_key.get(key, {}).items():
            mean, std = _mean_std(values)
            payload[field] = {"mean": mean, "std": std, "values": values}
        flags = collapse_flags.get(key, [])
        payload["endogenous_collapse_flags"] = flags
        payload["endogenous_collapse_any"] = any(flags)
        payload["endogenous_collapse_all"] = bool(flags) and all(flags)
        return payload

    b1 = {
        "note": "B1 is an exogenous-missing diagnostic. Do not call this endogenous collapse.",
        "ObservabilityController": pack("B1", "ObservabilityController", PRIMARY_BUDGET),
        "UniformExploration": pack("B1", "UniformExploration", PRIMARY_BUDGET),
        "GreedyDeployment": pack("B1", "GreedyDeployment", PRIMARY_BUDGET),
        "NoFeedbackController": pack("B1", "NoFeedbackController", PRIMARY_BUDGET),
        "FullFeedbackOracle": pack("B1", "FullFeedbackOracle", PRIMARY_BUDGET),
    }
    b2_by_budget = {
        f"{budget:.2f}": {
            method: pack("B2", method, budget) for method in METHODS
        }
        for budget in BUDGETS
        if any(r["env"] == "B2" and abs(r["budget"] - budget) < 1e-12 for r in records)
    }
    greedy_b2 = pack("B2", "GreedyDeployment", PRIMARY_BUDGET)
    none_b2 = pack("B2", "NoFeedbackController", PRIMARY_BUDGET)
    b2_showed = bool(
        greedy_b2.get("endogenous_collapse_any") or none_b2.get("endogenous_collapse_any")
    )
    b2_evidence = {
        "GreedyDeployment": greedy_b2,
        "NoFeedbackController": none_b2,
        "observed_performance_stable_while_lambda_min_and_weak_skill_error_worsen": b2_showed,
        "evaluated_on": "Greedy Deployment and No Feedback Controller at budget 0.05, not on a fixed policy",
    }

    controller_lmin = []
    uniform_lmin = []
    controller_perf = []
    uniform_perf = []
    paired_seeds = []
    for seed in SEEDS:
        c = next(
            (
                r
                for r in records
                if r["env"] == "B2"
                and r["method"] == "ObservabilityController"
                and r["seed"] == seed
                and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
            ),
            None,
        )
        u = next(
            (
                r
                for r in records
                if r["env"] == "B2"
                and r["method"] == "UniformExploration"
                and r["seed"] == seed
                and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
            ),
            None,
        )
        if c is None or u is None:
            continue
        controller_lmin.append(float(c["summary"]["late_lambda_min"]))
        uniform_lmin.append(float(u["summary"]["late_lambda_min"]))
        controller_perf.append(float(c["summary"]["late_true_performance"]))
        uniform_perf.append(float(u["summary"]["late_true_performance"]))
        paired_seeds.append(seed)

    comparison: dict[str, Any]
    if len(paired_seeds) >= 2:
        info_ci = paired_bootstrap_ci(controller_lmin, uniform_lmin, n_boot=5000, seed=0)
        perf_ci = paired_bootstrap_ci(controller_perf, uniform_perf, n_boot=5000, seed=0)
        info_better = float(np.mean(controller_lmin)) > float(np.mean(uniform_lmin))
        perf_better = float(np.mean(controller_perf)) > float(np.mean(uniform_perf))
        comparison = {
            "budget": PRIMARY_BUDGET,
            "paired_seeds": paired_seeds,
            "controller_late_lambda_min": controller_lmin,
            "uniform_late_lambda_min": uniform_lmin,
            "controller_late_true_performance": controller_perf,
            "uniform_late_true_performance": uniform_perf,
            "information_paired_bootstrap": info_ci,
            "true_performance_paired_bootstrap": perf_ci,
            "controller_mean_lambda_min_higher": info_better,
            "controller_mean_true_performance_higher": perf_better,
            "beats_uniform_on_both": bool(info_better and perf_better),
            "note": (
                "n_seeds=3 is preliminary. Information and long-horizon true "
                "performance are tested separately; better W_T is not assumed "
                "to imply better reward."
            ),
        }
    else:
        comparison = {
            "beats_uniform_on_both": False,
            "reason": "missing paired controller/uniform runs on B2",
        }

    b3 = {
        "recoverable": {
            "ObservabilityController": pack("B3_recoverable", "ObservabilityController", PRIMARY_BUDGET),
            "UniformExploration": pack("B3_recoverable", "UniformExploration", PRIMARY_BUDGET),
            "GreedyDeployment": pack("B3_recoverable", "GreedyDeployment", PRIMARY_BUDGET),
            "NoFeedbackController": pack("B3_recoverable", "NoFeedbackController", PRIMARY_BUDGET),
            "FullFeedbackOracle": pack("B3_recoverable", "FullFeedbackOracle", PRIMARY_BUDGET),
        },
        "unrecoverable": {
            "ObservabilityController": pack("B3_unrecoverable", "ObservabilityController", PRIMARY_BUDGET),
            "UniformExploration": pack("B3_unrecoverable", "UniformExploration", PRIMARY_BUDGET),
            "GreedyDeployment": pack("B3_unrecoverable", "GreedyDeployment", PRIMARY_BUDGET),
            "NoFeedbackController": pack("B3_unrecoverable", "NoFeedbackController", PRIMARY_BUDGET),
            "FullFeedbackOracle": pack("B3_unrecoverable", "FullFeedbackOracle", PRIMARY_BUDGET),
        },
    }

    try:
        _write_aggregate_plots(records, results_root / "plots", n_seeds=n_seeds)
        plot_status = "written"
    except Exception as exc:
        plot_status = f"failed: {type(exc).__name__}: {exc}"

    return {
        "name": "Feedback-Induced Observability Collapse",
        "information_matrix": "W_T = sum_t S_t phi_t phi_t^T (linear-model observed design Gramian)",
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "primary_budget": PRIMARY_BUDGET,
        "horizon": HORIZON,
        "population_per_round": POPULATION,
        "n_runs": len(records),
        "skips": skips,
        "B1": b1,
        "B2": {
            "endogenous_collapse": b2_showed,
            "evidence": b2_evidence,
            "by_budget": b2_by_budget,
        },
        "B3": b3,
        "controller_vs_uniform": comparison,
        "aggregate_plots": plot_status,
        "torch_installed": False,
        "official_baseline_clones": "not modified; another agent owns those directories",
        "fixed_policy_fixed_sampling": {
            "note": "Must not be labeled endogenous collapse.",
            "runs": pack("fixed_policy_fixed_sampling", "FrozenPolicyUniformSampling", PRIMARY_BUDGET),
        },
    }


def _mean_trajectory(paths: list[Path], field: str) -> np.ndarray | None:
    series = []
    for path in paths:
        payload = _load_trajectory(path)
        if payload is None:
            continue
        series.append(np.asarray(payload["trajectory"][field], dtype=float))
    if not series:
        return None
    width = min(item.shape[0] for item in series)
    stacked = np.stack([item[:width] for item in series], axis=0)
    return np.nanmean(stacked, axis=0)


def _write_aggregate_plots(records: list[dict[str, Any]], plot_dir: Path, n_seeds: int) -> None:
    methods_for_budget = [
        "ObservabilityController",
        "UniformExploration",
        "GreedyDeployment",
        "NoFeedbackController",
    ]
    budgets_present = sorted({float(r["budget"]) for r in records if r["env"] == "B2"})
    if not budgets_present:
        budgets_present = [PRIMARY_BUDGET]
    budget_labels = [f"{b:.0%}" for b in budgets_present]
    performance_by_method: dict[str, list[float]] = {}
    performance_err: dict[str, list[float]] = {}
    for method in methods_for_budget:
        means: list[float] = []
        stds: list[float] = []
        for budget in budgets_present:
            values = [
                float(r["summary"]["late_true_performance"])
                for r in records
                if r["env"] == "B2" and r["method"] == method and abs(r["budget"] - budget) < 1e-12
            ]
            mean, std = _mean_std(values)
            means.append(mean)
            stds.append(std)
        performance_by_method[method] = means
        performance_err[method] = stds

    collapse_paths = {
        "B2 greedy true performance": [
            Path(r["path"])
            for r in records
            if r["env"] == "B2"
            and r["method"] == "GreedyDeployment"
            and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
        ],
        "B2 greedy lambda_min": [
            Path(r["path"])
            for r in records
            if r["env"] == "B2"
            and r["method"] == "GreedyDeployment"
            and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
        ],
        "B2 controller true performance": [
            Path(r["path"])
            for r in records
            if r["env"] == "B2"
            and r["method"] == "ObservabilityController"
            and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
        ],
        "B3 recoverable controller true performance": [
            Path(r["path"])
            for r in records
            if r["env"] == "B3_recoverable"
            and r["method"] == "ObservabilityController"
            and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
        ],
        "B3 unrecoverable controller lambda_min": [
            Path(r["path"])
            for r in records
            if r["env"] == "B3_unrecoverable"
            and r["method"] == "ObservabilityController"
            and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
        ],
    }
    field_for = {
        "B2 greedy true performance": "true_performance",
        "B2 greedy lambda_min": "lambda_min_over_t",
        "B2 controller true performance": "true_performance",
        "B3 recoverable controller true performance": "true_performance",
        "B3 unrecoverable controller lambda_min": "lambda_min_over_t",
    }
    collapse_series: dict[str, np.ndarray] = {}
    collapse_x = None
    for label, paths in collapse_paths.items():
        series = _mean_trajectory(paths, field_for[label])
        if series is None:
            continue
        collapse_series[label] = series
        collapse_x = np.arange(series.shape[0], dtype=float)
    if collapse_x is None or not collapse_series:
        collapse_x = np.arange(HORIZON, dtype=float)
        collapse_series = {"no trajectories": np.zeros(HORIZON, dtype=float)}
    width = min(item.shape[0] for item in collapse_series.values())
    collapse_x = np.arange(width, dtype=float)
    collapse_series = {key: np.asarray(val[:width], dtype=float) for key, val in collapse_series.items()}

    baseline_order = [
        "ObservabilityController",
        "UniformExploration",
        "GreedyDeployment",
        "LazyDeployment",
        "UncertaintyExploration",
        "NoFeedbackController",
        "FullFeedbackOracle",
    ]
    perf_means = []
    perf_stds = []
    lmin_means = []
    lmin_stds = []
    labels = []
    for method in baseline_order:
        values_p = [
            float(r["summary"]["late_true_performance"])
            for r in records
            if r["env"] == "B2" and r["method"] == method and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
        ]
        values_l = [
            float(r["summary"]["late_lambda_min"])
            for r in records
            if r["env"] == "B2" and r["method"] == method and abs(r["budget"] - PRIMARY_BUDGET) < 1e-12
        ]
        if not values_p:
            continue
        labels.append(method)
        mean_p, std_p = _mean_std(values_p)
        mean_l, std_l = _mean_std(values_l)
        perf_means.append(mean_p)
        perf_stds.append(std_p)
        lmin_means.append(mean_l)
        lmin_stds.append(std_l)

    save_aggregate_plots(
        plot_dir,
        budget_labels=budget_labels,
        performance_by_method=performance_by_method,
        performance_err=performance_err,
        collapse_x=collapse_x,
        collapse_series=collapse_series,
        baseline_labels=labels,
        baseline_series={
            "late true performance": perf_means,
            "late lambda_min(W_T/T)": lmin_means,
        },
        baseline_err={
            "late true performance": perf_stds,
            "late lambda_min(W_T/T)": lmin_stds,
        },
        n_seeds=n_seeds,
    )


def run_matrix(
    *,
    results_root: Path = DEFAULT_RESULTS,
    environments: list[str] | None = None,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    budgets: list[float] | None = None,
    horizon: int = HORIZON,
    population: int = POPULATION,
    write_plots: bool = True,
    time_cap_s: float = 480.0,
    include_optional_budgets: bool = True,
) -> dict[str, Any]:
    environments = list(environments or ENVIRONMENTS)
    methods = list(methods or METHODS)
    seeds = list(seeds or SEEDS)
    budgets = list(budgets or BUDGETS)
    results_root.mkdir(parents=True, exist_ok=True)
    required, optional_spec = planned_jobs(
        environments=environments,
        methods=methods,
        seeds=seeds,
        budgets=budgets,
        primary_budget=PRIMARY_BUDGET,
        time_cap_s=time_cap_s,
    )
    records: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for env, method, seed, budget in required:
        rec = run_one(
            env,
            method,
            seed,
            budget,
            horizon=horizon,
            population=population,
            results_root=results_root,
            write_plots=write_plots,
        )
        records.append(rec)
    for seed in seeds:
        rec = run_one(
            "fixed_policy_fixed_sampling",
            "FrozenPolicyUniformSampling",
            seed,
            PRIMARY_BUDGET,
            horizon=horizon,
            population=population,
            results_root=results_root,
            write_plots=write_plots,
            prefix="diagnostic",
        )
        records.append(rec)
    optional_jobs = optional_spec[0]["jobs"]
    if not include_optional_budgets:
        skips.append(
            {
                "n_skipped": len(optional_jobs),
                "reason": "optional non-B2 extra budgets disabled by flag; all envs ran at 5% and B2 at all budgets",
            }
        )
    if include_optional_budgets:
        for env, method, seed, budget in optional_jobs:
            elapsed = time.perf_counter() - t0
            if elapsed > time_cap_s:
                skips.append(
                    {
                        "env": env,
                        "method": method,
                        "seed": seed,
                        "budget": budget,
                        "reason": (
                            f"skipped non-B2 extra budget after {elapsed:.1f}s "
                            f"(cap {time_cap_s:.0f}s); all envs at {PRIMARY_BUDGET:.0%} "
                            "and B2 at all requested budgets already finished"
                        ),
                    }
                )
                continue
            rec = run_one(
                env,
                method,
                seed,
                budget,
                horizon=horizon,
                population=population,
                results_root=results_root,
                write_plots=write_plots,
            )
            records.append(rec)
    remaining_optional = [
        item
        for item in skips
        if item.get("reason", "").startswith("skipped non-B2")
    ]
    if remaining_optional:
        n_skip = len(remaining_optional)
        skips = [
            {
                "n_skipped": n_skip,
                "reason": remaining_optional[0]["reason"],
                "example": remaining_optional[0],
            }
        ]
    summary = aggregate(records, skips, results_root=results_root, n_seeds=len(seeds))
    summary["wall_time_s"] = time.perf_counter() - t0
    summary_path = results_root / "SMOKE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    summary["path"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Idea B synthetic closed-loop smoke runner")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--time-cap-s", type=float, default=480.0)
    parser.add_argument("--skip-optional-budgets", action="store_true")
    args = parser.parse_args(argv)
    _ = _load_yaml(DEFAULT_CONFIG)
    try:
        summary = run_matrix(
            results_root=args.results_root,
            horizon=args.horizon,
            population=args.population,
            write_plots=not args.no_plots,
            time_cap_s=args.time_cap_s,
            include_optional_budgets=not args.skip_optional_budgets,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps({"path": summary.get("path"), "n_runs": summary.get("n_runs"), "skips": summary.get("skips")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
