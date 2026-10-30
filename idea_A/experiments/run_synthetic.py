"""Run Idea A synthetic matrix and write per-run artifacts plus SMOKE_SUMMARY.json."""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import MECHANISMS, simulate
from idea_A.experiments.protocol import MethodSpec, default_method_specs, run_world_methods
from idea_A.methods.v1_load import load_v1_module
from shared.evaluation.results import write_run
from shared.hardware.probe import collect_environment
from shared.plotting.figures import plot_bars, plot_series
from shared.statistical_tests.paired_bootstrap import paired_bootstrap_ci

_algorithm = load_v1_module("algorithm")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "idea_A" / "results"
V1_CONFIG = _algorithm.load_v1_config()

# Full matrix is 7 mechanisms × 4 budgets × 3 seeds. NumPy n=2000 is cheap on
# this machine, so the default is the full matrix. If a future operator needs
# the documented subset, set IDEA_A_SUBSET=1: all mechanisms at b=0.05 plus
# all budgets on mnar and zero_overlap.
SUBSET_ENV = "IDEA_A_SUBSET"


def _budget_tag(budget: float) -> str:
    return f"{float(budget):.2f}"


def experiment_id(mechanism: str, budget: float, seed: int) -> str:
    return f"syn_{mechanism}_b{_budget_tag(budget)}_s{int(seed)}"


def matrix_plan(config: dict[str, Any]) -> dict[str, Any]:
    mechanisms = list(config.get("mechanisms", list(MECHANISMS)))
    budgets = [float(b) for b in config.get("budgets", [0.01, 0.05, 0.10, 0.20])]
    seeds = [int(s) for s in config.get("seeds", [0, 1, 2])]
    subset = os.environ.get(SUBSET_ENV, "").strip() in {"1", "true", "True", "yes"}
    selected: list[tuple[str, float, int]] = []
    skipped: list[dict[str, Any]] = []
    if not subset:
        for mechanism in mechanisms:
            for budget in budgets:
                for seed in seeds:
                    selected.append((mechanism, budget, seed))
        return {
            "mode": "full",
            "selected": selected,
            "skipped": skipped,
            "mechanisms": mechanisms,
            "budgets": budgets,
            "seeds": seeds,
            "note": "Full matrix executed: all mechanisms × all budgets × seeds 0,1,2.",
        }
    focus = {"mnar", "zero_overlap"}
    for mechanism in mechanisms:
        for budget in budgets:
            keep = abs(budget - 0.05) < 1e-12 or mechanism in focus
            for seed in seeds:
                if keep:
                    selected.append((mechanism, budget, seed))
                else:
                    skipped.append(
                        {
                            "mechanism": mechanism,
                            "budget": budget,
                            "seed": seed,
                            "reason": "subset mode: kept all mechanisms at b=0.05 and all budgets on mnar/zero_overlap",
                        }
                    )
    return {
        "mode": "subset",
        "selected": selected,
        "skipped": skipped,
        "mechanisms": mechanisms,
        "budgets": budgets,
        "seeds": seeds,
        "note": (
            "IDEA_A_SUBSET=1: all mechanisms at b=0.05 plus all budgets on "
            "mnar and zero_overlap. Other (mechanism, budget) cells were skipped."
        ),
    }


def _log_text(world_meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"utc={datetime.now(timezone.utc).isoformat()}",
        f"idea=Adaptive Identification under Endogenous Feedback Censoring",
        f"version=1.0",
        f"allocation_note=Classical stratified Neyman + exploration floor. Not a novel algorithm.",
        f"mechanism={world_meta['mechanism']}",
        f"seed={world_meta['seed']}",
        f"n={world_meta['n']}",
        f"budget={world_meta['budget']}",
        f"truth_population_mean={world_meta['truth_population_mean']}",
        f"natural_identification={world_meta['natural_identification']}",
        "",
        "methods:",
    ]
    for row in rows:
        pop = row["population_mean_reward"]
        lines.append(
            f"  {row['method']}: estimate={pop['estimate']} truth={pop['truth']} "
            f"bias={pop['bias']} expected_q={row['expected_audit_fraction']} "
            f"realized_q={row['realized_audit_fraction']}"
        )
    return "\n".join(lines) + "\n"


def _write_run_plot(result_dir: Path, rows: list[dict[str, Any]], n_seeds: int = 1) -> None:
    labels = [row["method"] for row in rows]
    estimates = [row["population_mean_reward"]["estimate"] for row in rows]
    truths = [row["population_mean_reward"]["truth"] for row in rows]
    plots = result_dir / "plots"
    fig = plot_bars(
        labels,
        {"estimate": estimates, "finite-population truth": truths},
        title="Population mean reward vs truth (single seed)",
        xlabel="method",
        ylabel="mean reward",
        n_seeds=n_seeds,
        path=plots / "population_mean_vs_truth.png",
    )
    plt.close(fig)
    fig.clf()


def _flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        item = {
            "method": row["method"],
            "allocation": row["allocation"],
            "estimator": row["estimator"],
            "mechanism": row["mechanism"],
            "seed": row["seed"],
            "budget_requested": row["budget_requested"],
            "uses_budget": row["uses_budget"],
            "expected_audit_fraction": row["expected_audit_fraction"],
            "realized_audit_fraction": row["realized_audit_fraction"],
            "min_q": row["min_q"],
            "n_audited": row["n_audited"],
            "notes": row["notes"],
        }
        for estimand in (
            "population_mean_reward",
            "preference_rate",
            "policy_value_target_distribution",
        ):
            payload = row[estimand]
            item[f"{estimand}_estimate"] = payload["estimate"]
            item[f"{estimand}_truth"] = payload["truth"]
            item[f"{estimand}_bias"] = payload["bias"]
            item[f"{estimand}_mae"] = payload["mae"]
            item[f"{estimand}_rmse"] = payload["rmse"]
        item["context_conditioned_mae"] = row["context_conditioned_reward"]["mae"]
        item["n_identified_strata"] = row["context_conditioned_reward"]["n_identified_strata"]
        flat.append(item)
    return flat


def run_one(
    mechanism: str,
    seed: int,
    budget: float,
    config: dict[str, Any],
    environment: dict[str, Any],
    specs: list[MethodSpec],
    world=None,
) -> dict[str, Any]:
    world, rows = run_world_methods(mechanism, seed, budget, config, specs=specs, world=world)
    exp_id = experiment_id(mechanism, budget, seed)
    result_dir = RESULTS_ROOT / exp_id
    run_config = {
        "experiment_id": exp_id,
        "idea": "A",
        "name": "Adaptive Identification under Endogenous Feedback Censoring",
        "version": "1.0",
        "mechanism": mechanism,
        "seed": seed,
        "budget": budget,
        "n": world.n,
        "methods": [spec.method_id for spec in specs],
        "novelty_claim": False,
        "allocation_note": config.get("novelty_note"),
        "paired_world": True,
        "v1_config": config,
    }
    metrics = {
        "experiment_id": exp_id,
        "mechanism": mechanism,
        "seed": seed,
        "budget": budget,
        "truths": {
            "population_mean_reward": world.truths["population_mean_reward"],
            "preference_rate": world.truths["preference_rate"],
            "policy_value_target_distribution": world.truths["policy_value_target_distribution"],
            "context_conditioned_reward": np.asarray(
                world.truths["context_conditioned_reward"], dtype=float
            ).tolist(),
        },
        "methods": rows,
        "rows": _flatten_rows(rows),
    }
    log_text = _log_text(
        {
            "mechanism": mechanism,
            "seed": seed,
            "n": world.n,
            "budget": budget,
            "truth_population_mean": world.truths["population_mean_reward"],
            "natural_identification": world.identification_natural,
        },
        rows,
    )
    write_run(result_dir, run_config, metrics, environment, log_text)
    _write_run_plot(result_dir, rows, n_seeds=1)
    return {"experiment_id": exp_id, "result_dir": str(result_dir), "metrics": metrics, "world_seed": world.seed}


def _collect_flat(run_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in run_payloads:
        rows.extend(payload["metrics"]["rows"])
    return rows


def _agg_mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1) if finite.size > 1 else 0.0),
        "n": int(finite.size),
    }


def _rmse_across_seeds(estimates: list[float], truths: list[float]) -> float:
    est = np.asarray(estimates, dtype=float)
    tru = np.asarray(truths, dtype=float)
    finite = np.isfinite(est) & np.isfinite(tru)
    if not np.any(finite):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(est[finite] - tru[finite]))))


def build_summary(
    run_payloads: list[dict[str, Any]],
    plan: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = _collect_flat(run_payloads)
    methods = sorted({row["method"] for row in rows})
    mechanisms = sorted({row["mechanism"] for row in rows})
    budgets = sorted({float(row["budget_requested"]) for row in rows})
    estimands = [
        "population_mean_reward",
        "preference_rate",
        "policy_value_target_distribution",
    ]
    per_method: dict[str, Any] = {}
    for method in methods:
        per_method[method] = {}
        for mechanism in mechanisms:
            per_method[method][mechanism] = {}
            for budget in budgets:
                cell_rows = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["mechanism"] == mechanism
                    and abs(float(row["budget_requested"]) - budget) < 1e-12
                ]
                if not cell_rows:
                    continue
                cell: dict[str, Any] = {
                    "n_seeds": len(cell_rows),
                    "expected_audit_fraction": _agg_mean_std(
                        [row["expected_audit_fraction"] for row in cell_rows]
                    ),
                    "realized_audit_fraction": _agg_mean_std(
                        [row["realized_audit_fraction"] for row in cell_rows]
                    ),
                }
                for estimand in estimands:
                    biases = [row[f"{estimand}_bias"] for row in cell_rows]
                    maes = [row[f"{estimand}_mae"] for row in cell_rows]
                    estimates = [row[f"{estimand}_estimate"] for row in cell_rows]
                    truths = [row[f"{estimand}_truth"] for row in cell_rows]
                    cell[estimand] = {
                        "bias": _agg_mean_std(biases),
                        "mae": _agg_mean_std(maes),
                        "rmse_across_seeds": _rmse_across_seeds(estimates, truths),
                    }
                cell["context_conditioned_mae"] = _agg_mean_std(
                    [row["context_conditioned_mae"] for row in cell_rows]
                )
                per_method[method][mechanism][f"b={budget:.2f}"] = cell

    paired: dict[str, Any] = {}
    for mechanism in mechanisms:
        paired[mechanism] = {}
        for budget in budgets:
            ney = [
                row
                for row in rows
                if row["method"] == "neyman_ipw"
                and row["mechanism"] == mechanism
                and abs(float(row["budget_requested"]) - budget) < 1e-12
            ]
            uni = [
                row
                for row in rows
                if row["method"] == "uniform_ipw"
                and row["mechanism"] == mechanism
                and abs(float(row["budget_requested"]) - budget) < 1e-12
            ]
            ney = sorted(ney, key=lambda r: int(r["seed"]))
            uni = sorted(uni, key=lambda r: int(r["seed"]))
            if not ney or len(ney) != len(uni):
                continue
            seeds = [int(r["seed"]) for r in ney]
            if seeds != [int(r["seed"]) for r in uni]:
                continue
            ney_err = np.array([r["population_mean_reward_rmse"] for r in ney], dtype=float)
            uni_err = np.array([r["population_mean_reward_rmse"] for r in uni], dtype=float)
            ney_bias = np.array([r["population_mean_reward_bias"] for r in ney], dtype=float)
            uni_bias = np.array([r["population_mean_reward_bias"] for r in uni], dtype=float)
            boot_rmse = paired_bootstrap_ci(ney_err, uni_err, n_boot=10000, alpha=0.05, seed=0)
            boot_abs = paired_bootstrap_ci(np.abs(ney_bias), np.abs(uni_bias), n_boot=10000, alpha=0.05, seed=1)
            paired[mechanism][f"b={budget:.2f}"] = {
                "seeds": seeds,
                "neyman_ipw_rmse": ney_err.tolist(),
                "uniform_ipw_rmse": uni_err.tolist(),
                "mean_rmse_difference_neyman_minus_uniform": float(np.mean(ney_err - uni_err)),
                "paired_bootstrap_rmse": boot_rmse,
                "paired_bootstrap_abs_bias": boot_abs,
                "note": boot_rmse["note"],
            }

    key_findings = {
        "mnar_observed_only_vs_audit_ipw": {},
        "neyman_vs_uniform_rmse": {},
    }
    for budget in budgets:
        tag = f"b={budget:.2f}"
        try:
            obs = per_method["observed_only"]["mnar"][tag]["population_mean_reward"]
            ipw = per_method["neyman_ipw"]["mnar"][tag]["population_mean_reward"]
            uni = per_method["uniform_ipw"]["mnar"][tag]["population_mean_reward"]
            key_findings["mnar_observed_only_vs_audit_ipw"][tag] = {
                "observed_only_bias_mean": obs["bias"]["mean"],
                "observed_only_bias_std": obs["bias"]["std"],
                "neyman_ipw_bias_mean": ipw["bias"]["mean"],
                "neyman_ipw_bias_std": ipw["bias"]["std"],
                "uniform_ipw_bias_mean": uni["bias"]["mean"],
                "uniform_ipw_bias_std": uni["bias"]["std"],
            }
        except KeyError:
            pass
        if "mnar" in paired and tag in paired["mnar"]:
            key_findings["neyman_vs_uniform_rmse"][f"mnar_{tag}"] = paired["mnar"][tag]
        if "zero_overlap" in paired and tag in paired["zero_overlap"]:
            key_findings["neyman_vs_uniform_rmse"][f"zero_overlap_{tag}"] = paired["zero_overlap"][tag]

    return {
        "idea": "A",
        "name": "Adaptive Identification under Endogenous Feedback Censoring",
        "version": "1.0",
        "allocation_note": (
            "v1.0 allocation is classical stratified Neyman allocation plus an "
            "exploration floor, then IPW/Hájek/AIPW with known audit propensities. "
            "This is not a novel algorithm."
        ),
        "n": int(config.get("n", 2000)),
        "seeds": list(plan["seeds"]),
        "n_seeds": len(plan["seeds"]),
        "preliminary": len(plan["seeds"]) <= 3,
        "matrix": {
            "mode": plan["mode"],
            "mechanisms_requested": plan["mechanisms"],
            "budgets_requested": plan["budgets"],
            "n_runs_written": len(run_payloads),
            "n_cells_selected": len(plan["selected"]),
            "n_cells_skipped": len(plan["skipped"]),
            "skipped": plan["skipped"],
            "note": plan["note"],
        },
        "per_method": per_method,
        "paired_neyman_vs_uniform": paired,
        "key_findings": key_findings,
        "result_dirs": [payload["result_dir"] for payload in run_payloads],
    }


def write_aggregate_plots(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    out_dir = RESULTS_ROOT / "aggregate_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = int(summary["n_seeds"])
    paths: list[str] = []

    mechanisms = sorted({row["mechanism"] for row in rows})
    methods_bias = ["observed_only", "naive_ips", "uniform_ipw", "neyman_ipw", "full_feedback_oracle"]
    focus_budget = 0.05
    labels = []
    series: dict[str, list[float]] = {m: [] for m in methods_bias}
    yerr: dict[str, list[float]] = {m: [] for m in methods_bias}
    present_methods = []
    for mechanism in mechanisms:
        labels.append(mechanism)
        for method in methods_bias:
            cell = (
                summary["per_method"]
                .get(method, {})
                .get(mechanism, {})
                .get("b=0.05")
            )
            if cell is None:
                series[method].append(float("nan"))
                yerr[method].append(0.0)
            else:
                series[method].append(cell["population_mean_reward"]["bias"]["mean"])
                yerr[method].append(cell["population_mean_reward"]["bias"]["std"])
                if method not in present_methods:
                    present_methods.append(method)
    series = {k: v for k, v in series.items() if k in present_methods}
    yerr = {k: yerr[k] for k in series}
    if labels and series:
        path = out_dir / "bias_by_mechanism_b0.05.png"
        fig = plot_bars(
            labels,
            series,
            title="Population-mean bias at budget b=0.05",
            xlabel="feedback mechanism",
            ylabel="bias (estimate − truth)",
            n_seeds=n_seeds,
            yerr=yerr,
            path=path,
        )
        plt.close(fig)
        paths.append(str(path))

    for mechanism in ("mnar", "zero_overlap"):
        budgets = sorted(
            {
                float(row["budget_requested"])
                for row in rows
                if row["mechanism"] == mechanism and row["method"] in {"neyman_ipw", "uniform_ipw"}
            }
        )
        if not budgets:
            continue
        series_rmse: dict[str, list[float]] = {"neyman_ipw": [], "uniform_ipw": []}
        yerr_rmse: dict[str, list[float]] = {"neyman_ipw": [], "uniform_ipw": []}
        for budget in budgets:
            for method in ("neyman_ipw", "uniform_ipw"):
                cell = summary["per_method"].get(method, {}).get(mechanism, {}).get(f"b={budget:.2f}")
                if cell is None:
                    series_rmse[method].append(float("nan"))
                    yerr_rmse[method].append(0.0)
                else:
                    series_rmse[method].append(cell["population_mean_reward"]["rmse_across_seeds"])
                    yerr_rmse[method].append(cell["population_mean_reward"]["mae"]["std"])
        path = out_dir / f"rmse_vs_budget_{mechanism}.png"
        fig = plot_series(
            budgets,
            series_rmse,
            title=f"Population-mean RMSE vs audit budget ({mechanism})",
            xlabel="audit budget b",
            ylabel="RMSE across seeds",
            n_seeds=n_seeds,
            yerr=yerr_rmse,
            path=path,
        )
        plt.close(fig)
        paths.append(str(path))

    _ = focus_budget
    return paths


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) or isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    return value


def main() -> int:
    config = dict(V1_CONFIG)
    plan = matrix_plan(config)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(REPO_ROOT)
    environment = dict(environment)
    environment["experiments_run"] = True
    environment["idea_A_synthetic"] = True
    specs = default_method_specs()

    print(
        f"[idea_A] matrix mode={plan['mode']} selected={len(plan['selected'])} "
        f"skipped={len(plan['skipped'])}",
        flush=True,
    )
    worlds: dict[tuple[str, int], Any] = {}
    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for mechanism, budget, seed in plan["selected"]:
        key = (mechanism, int(seed))
        try:
            if key not in worlds:
                worlds[key] = simulate(mechanism, int(seed), n=int(config.get("n", 2000)))
            payload = run_one(
                mechanism,
                int(seed),
                float(budget),
                config,
                environment,
                specs,
                world=worlds[key],
            )
            payloads.append(payload)
            print(f"[idea_A] wrote {payload['experiment_id']}", flush=True)
        except Exception as exc:
            failures.append(
                {
                    "mechanism": mechanism,
                    "budget": budget,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[idea_A] FAILED {mechanism} b={budget} seed={seed}: {exc}", flush=True)

    summary = build_summary(payloads, plan, config)
    summary["failures"] = failures
    rows = _collect_flat(payloads)
    summary["aggregate_plots"] = write_aggregate_plots(summary, rows)
    summary_path = RESULTS_ROOT / "SMOKE_SUMMARY.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    skipped_path = RESULTS_ROOT / "SKIPPED.json"
    skipped_path.write_text(
        json.dumps(
            {"mode": plan["mode"], "note": plan["note"], "skipped": plan["skipped"], "failures": failures},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[idea_A] summary -> {summary_path}", flush=True)
    print(f"[idea_A] failures={len(failures)}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
