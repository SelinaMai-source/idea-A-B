"""Idea A v1.1 experiment runner (importable; CLI lives in experiments/v1.1/)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from idea_A.benchmarks.Sequential_Closed_Loop.environment import build_population, run_loop
from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.experiments.protocol import MethodSpec, audit_rng, errors_against_truth, misspecify_q, run_method
from idea_A.methods.v1_load import load_v1_1_module, load_v1_module
from shared.evaluation.results import write_run
from shared.hardware.probe import collect_environment
from shared.metrics.errors import bias as bias_fn
from shared.metrics.errors import mae, rmse
from shared.plotting.figures import plot_bars
from shared.statistical_tests.paired_bootstrap import paired_bootstrap_ci

_alg = load_v1_1_module("algorithm")
_seq = load_v1_1_module("sequential_estimator")
_v1_est = load_v1_module("estimators")

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE_V11 = Path(__file__).resolve().parent / "v1.1"
RESULTS_ROOT = REPO_ROOT / "idea_A" / "results" / "v1.1"
ABLATION_ROOT = REPO_ROOT / "idea_A" / "ablations" / "v1.1"

STATIC_TIER1_METHODS = (
    "observed_only",
    "uniform_ipw",
    "uniform_dr",
    "neyman_ipw",
    "neyman_dr",
    "v1.1b",
    "v1.1c",
)
STATIC_TIER2_METHODS = (
    "observed_only",
    "uniform_ipw",
    "uniform_dr",
    "neyman_dr",
    "v1.1b",
    "v1.1c",
)
CONTINUAL_TIER1_METHODS = (
    "observed_only",
    "uniform_ipw",
    "uniform_dr",
    "neyman_ipw",
    "neyman_dr",
    "v1.1b",
    "v1.1c",
)
CONTINUAL_TIER2_METHODS = ("uniform_dr", "neyman_dr", "v1.1c")

METHOD_NOTES = {
    "observed_only": "Natural observations only. Reference arm.",
    "uniform_ipw": "Uniform audit + IPW. Reference arm.",
    "uniform_dr": "A-v1.1a: Uniform audit + 2-fold cross-fit AIPW. Required competitor.",
    "neyman_ipw": "A-v1.0: Neyman + exploration floor + IPW. Behavior unchanged.",
    "neyman_dr": "Neyman allocation + 2-fold cross-fit AIPW (or history-respecting DR on sequential).",
    "v1.1b": "A-v1.1b: audit-residual AIPW-variance allocation + cross-fit DR.",
    "v1.1c": "A-v1.1c: remaining-budget + policy-drift on top of v1.1b. Main proposal.",
}

SEQUENTIAL_SPEC = {
    "observed_only": ("observed_only", "observed_only"),
    "uniform_ipw": ("uniform_ipw", "ipw"),
    "uniform_dr": ("A-v1.1a", "dr"),
    "neyman_ipw": ("A-v1.0", "ipw"),
    "neyman_dr": ("neyman_dr", "dr"),
    "v1.1b": ("A-v1.1b", "dr"),
    "v1.1c": ("A-v1.1c", "dr"),
}

ABLATIONS = {
    "A1_No_Adaptive_Allocation": {
        "component": "allocation",
        "methods": ("v1.1c", "uniform_dr"),
        "note": "Replace AIPW-variance / remaining / drift allocation with uniform q=b.",
    },
    "A2_No_Remaining_Budget": {
        "component": "remaining_budget",
        "methods": ("v1.1c", "v1.1b"),
        "note": "Drop remaining-budget spending and drift; keep AIPW-variance allocation.",
    },
    "A3_No_Policy_Drift": {
        "component": "policy_drift",
        "methods": ("v1.1c",),
        "config": {"drift_kappa": 0.0},
        "note": "A-v1.1c with drift_kappa=0 (no policy-drift upweighting).",
        "label": "v1.1c_no_drift",
    },
    "A4_Logistic_Policy": {
        "component": "policy_class",
        "methods": ("v1.1c",),
        "policy_kind": "logistic",
        "note": "A-v1.1c with logistic policy instead of linear.",
        "label": "v1.1c_logistic_policy",
    },
    "A5_Misspecified_Propensity": {
        "component": "propensity",
        "methods": ("v1.1c",),
        "q_transform": "mix_uniform",
        "note": "Estimator uses q mixed 50/50 with its mean. Allocation unchanged.",
        "label": "v1.1c_wrong_q",
    },
    "A6_IPW_Instead_Of_DR": {
        "component": "estimator",
        "methods": ("v1.1c",),
        "estimator_override": "ipw",
        "note": "Keep v1.1c allocation; replace history-respecting DR with IPW.",
        "label": "v1.1c_ipw",
    },
    "A7_Strong_Ridge": {
        "component": "outcome_model",
        "methods": ("v1.1c",),
        "config": {"ridge_lambda": 1.0},
        "note": "A-v1.1c with ridge_lambda=1.0 (stronger shrinkage; not tuned on eval seeds).",
        "label": "v1.1c_strong_ridge",
    },
    "A8_No_Policy_Update": {
        "component": "policy_update",
        "methods": ("v1.1c",),
        "adapt_policy": False,
        "note": "Same MNAR-audit world; freeze the policy (no outcome-driven update).",
        "label": "v1.1c_frozen_policy",
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return {"shape": list(value.shape), "omitted": True, "reason": "array stripped to keep artifacts small"}
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


def wilson_interval(hits: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    if n <= 0:
        return {"center": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0.0}
    p = float(hits) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return {
        "center": float(center),
        "low": float(max(0.0, center - margin)),
        "high": float(min(1.0, center + margin)),
        "n": float(n),
        "hits": float(hits),
    }


def paired_bootstrap_rmse_difference(
    err_cand: Any,
    err_base: Any,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap RMSE_cand - RMSE_base by resampling seeds and recomputing RMSE."""
    x = np.asarray(err_cand, dtype=float)
    y = np.asarray(err_base, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("err_cand and err_base must be equal-length 1-D arrays")
    n_seeds = int(x.shape[0])
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, n_seeds, size=(int(n_boot), n_seeds))
    rmse_c = np.sqrt(np.mean(np.square(x[indices]), axis=1))
    rmse_b = np.sqrt(np.mean(np.square(y[indices]), axis=1))
    diffs = rmse_c - rmse_b
    point_c = float(np.sqrt(np.mean(np.square(x))))
    point_b = float(np.sqrt(np.mean(np.square(y))))
    return {
        "n_seeds": n_seeds,
        "rmse_candidate": point_c,
        "rmse_baseline": point_b,
        "difference_candidate_minus_baseline": float(point_c - point_b),
        "ci_low": float(np.quantile(diffs, alpha / 2.0, method="linear")),
        "ci_high": float(np.quantile(diffs, 1.0 - alpha / 2.0, method="linear")),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "preliminary": n_seeds <= 3,
        "functional": "sqrt(mean squared error) recomputed on resampled seeds",
        "note": (
            "n_seeds<=3 is PRELIMINARY."
            if n_seeds <= 3
            else "Percentile bootstrap of RMSE_cand - RMSE_base; not mean |error|."
        ),
    }


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


def _strip_arrays(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("q", "a", "mu_y", "mu_p", "fold_ids", "influence"):
        out.pop(key, None)
    return out


def _scalar_errors(estimate: float, truth: float) -> dict[str, float]:
    return {
        "estimate": float(estimate) if np.isfinite(estimate) else float("nan"),
        "truth": float(truth),
        "bias": float(bias_fn(estimate, truth)) if np.isfinite(estimate) else float("nan"),
        "mae": float(mae(estimate, truth)) if np.isfinite(estimate) else float("nan"),
        "rmse": float(rmse(estimate, truth)) if np.isfinite(estimate) else float("nan"),
    }


def _attach_ci_coverage(row: dict[str, Any], truth: float) -> None:
    ci = row.get("ci") or row.get("population_mean_reward") or {}
    covers = _seq.interval_covers(ci, truth) if isinstance(ci, dict) else False
    row["ci_coverage"] = bool(covers)
    row["ci_low"] = ci.get("ci_low") if isinstance(ci, dict) else None
    row["ci_high"] = ci.get("ci_high") if isinstance(ci, dict) else None
    row["ci_formula"] = ci.get("formula") if isinstance(ci, dict) else None


def run_static_method(world: Any, method: str, budget: float, rng: np.random.Generator) -> dict[str, Any]:
    truth = float(world.truths["population_mean_reward"])
    if method == "observed_only":
        spec = MethodSpec("observed_only", "none", "observed_only", uses_budget=False, notes=METHOD_NOTES[method])
        row = run_method(world, spec, budget, {"ridge_lambda": 1e-3}, rng)
        y = world.observe(np.zeros(world.n, dtype=int))
        mask = world.s.astype(bool)
        obs = np.asarray(y, dtype=float)[mask]
        if obs.size >= 2:
            row["ci"] = _seq.wald_interval(obs)
        else:
            row["ci"] = {"estimate": row["population_mean_reward"]["estimate"], "ci_low": float("nan"), "ci_high": float("nan"), "formula": "observed_only_wald_95"}
        _attach_ci_coverage(row, truth)
        row["method"] = "observed_only"
        return _strip_arrays(row)
    if method == "uniform_ipw":
        spec = MethodSpec("uniform_ipw", "uniform", "ipw", notes=METHOD_NOTES[method])
        row = run_method(world, spec, budget, {"ridge_lambda": 1e-3}, rng)
        y_vis = world.observe(np.zeros(world.n, dtype=int))
        # Rebuild q,a is not stored. Use IPW influence via the written estimates only if q known.
        # Re-run allocation to get q for the Wald formula (same rng consumed already).
        # Instead: Wald from per-unit IPW scores requires q. Use protocol path again with a clone rng.
        _attach_ci_coverage(row, truth)
        row["method"] = "uniform_ipw"
        return _strip_arrays(row)
    if method == "uniform_dr":
        raw = _alg.run_static_config("A-v1.1a", world, budget, rng)
    elif method == "neyman_ipw":
        raw = _alg.run_static_config("A-v1.0", world, budget, rng)
    elif method == "v1.1b":
        raw = _alg.run_static_config("A-v1.1b", world, budget, rng)
    elif method == "v1.1c":
        raw = _alg.run_static_config("A-v1.1c", world, budget, rng)
    elif method == "neyman_dr":
        cfg = _alg.load_config()
        y_nat = world.observe(np.zeros(world.n, dtype=int))
        q = load_v1_module("algorithm").allocate_neyman(
            world.stratum,
            y_nat,
            world.s,
            float(budget),
            world.n_strata,
            exploration_mix=0.15,
            q_floor=1e-6,
        )
        a = _alg._draw_audit(q, rng)
        y_vis = world.observe(a)
        estimates, ci, _mu, _folds = _alg._crossfit_estimates(world, y_vis, a, q, rng, cfg)
        raw = _alg._static_result(
            "neyman_dr",
            "v1.0_neyman",
            "cross_fit_dr",
            world,
            q,
            a,
            estimates,
            ci,
            METHOD_NOTES["neyman_dr"],
        )
    else:
        raise ValueError(f"unknown static method {method!r}")

    est = float(raw["population_mean_reward"]["estimate"])
    pop = _scalar_errors(est, truth)
    pref_est = raw["estimands"].get("preference_rate")
    pol_est = raw["estimands"].get("policy_value_target_distribution")
    row = {
        "method": method,
        "config_id": raw.get("config_id", method),
        "allocation": raw.get("allocation"),
        "estimator": raw.get("estimator"),
        "mechanism": world.mechanism,
        "seed": world.seed,
        "n": world.n,
        "budget_requested": float(budget),
        "notes": METHOD_NOTES[method],
        "expected_audit_fraction": raw["expected_audit_fraction"],
        "realized_audit_fraction": raw["realized_audit_fraction"],
        "min_q": raw.get("min_q"),
        "max_q": raw.get("max_q"),
        "n_audited": raw.get("n_audited"),
        "population_mean_reward": pop,
        "preference_rate": raw["estimands"].get("preference_rate"),
        "policy_value_target_distribution": raw["estimands"].get("policy_value_target_distribution"),
        "ci": raw.get("ci") or raw.get("population_mean_reward"),
    }
    _attach_ci_coverage(row, truth)
    if method == "uniform_ipw":
        # Add Wald after the fact from IPW scores with mu=0 using stored q if present.
        pass
    if "q" in raw and "a" in raw:
        y_vis = world.observe(raw["a"])
        if method in {"uniform_ipw", "neyman_ipw"}:
            inf = _seq.aipw_influence(y_vis, raw["a"], raw["q"], np.zeros(world.n, dtype=float))
            row["ci"] = _seq.wald_interval(inf)
            _attach_ci_coverage(row, truth)
    return _strip_arrays(row)


def _uniform_ipw_with_ci(world: Any, budget: float, rng: np.random.Generator) -> dict[str, Any]:
    spec = MethodSpec("uniform_ipw", "uniform", "ipw", notes=METHOD_NOTES["uniform_ipw"])
    # run_method draws a internally; rebuild q uniformly and influence from a stored result.
    from idea_A.experiments.protocol import allocate_q, draw_a, estimates_from_audit

    q = allocate_q("uniform", world, budget, {"ridge_lambda": 1e-3})
    a = draw_a(q, rng)
    y_vis = world.observe(a)
    estimates = estimates_from_audit(world, y_vis, a, q, "ipw", {"ridge_lambda": 1e-3})
    err = errors_against_truth(estimates, world.truths)
    inf = _seq.aipw_influence(y_vis, a, q, np.zeros(world.n, dtype=float))
    ci = _seq.wald_interval(inf)
    row = {
        "method": "uniform_ipw",
        "allocation": "uniform",
        "estimator": "ipw",
        "mechanism": world.mechanism,
        "seed": world.seed,
        "n": world.n,
        "budget_requested": float(budget),
        "notes": METHOD_NOTES["uniform_ipw"],
        "expected_audit_fraction": float(np.mean(q)),
        "realized_audit_fraction": float(np.mean(a)),
        "min_q": float(np.min(q)),
        "max_q": float(np.max(q)),
        "n_audited": int(np.sum(a)),
        "population_mean_reward": err["population_mean_reward"],
        "preference_rate": err["preference_rate"],
        "policy_value_target_distribution": err["policy_value_target_distribution"],
        "ci": ci,
    }
    _attach_ci_coverage(row, float(world.truths["population_mean_reward"]))
    return row


def run_static_method_fixed(world: Any, method: str, budget: float, rng: np.random.Generator) -> dict[str, Any]:
    if method == "uniform_ipw":
        return _uniform_ipw_with_ci(world, budget, rng)
    return run_static_method(world, method, budget, rng)


def flatten_static_row(row: dict[str, Any]) -> dict[str, Any]:
    pop = row["population_mean_reward"]
    if not isinstance(pop, dict):
        pop = {"estimate": pop, "truth": None, "bias": None, "mae": None, "rmse": None}
    return {
        "method": row["method"],
        "allocation": row.get("allocation"),
        "estimator": row.get("estimator"),
        "mechanism": row.get("mechanism") or row.get("scenario"),
        "seed": row["seed"],
        "n": row.get("n"),
        "budget_requested": row.get("budget_requested"),
        "expected_audit_fraction": row.get("expected_audit_fraction"),
        "realized_audit_fraction": row.get("realized_audit_fraction"),
        "n_audited": row.get("n_audited"),
        "population_mean_reward_estimate": pop.get("estimate"),
        "population_mean_reward_truth": pop.get("truth"),
        "population_mean_reward_bias": pop.get("bias"),
        "population_mean_reward_mae": pop.get("mae"),
        "population_mean_reward_rmse": pop.get("rmse"),
        "ci_coverage": row.get("ci_coverage"),
        "ci_low": row.get("ci_low"),
        "ci_high": row.get("ci_high"),
        "ci_formula": row.get("ci_formula"),
        "notes": row.get("notes"),
        "track": row.get("track", "static"),
    }


def _log_static(world: Any, budget: float, rows: list[dict[str, Any]], extra: str = "") -> str:
    lines = [
        f"utc={datetime.now(timezone.utc).isoformat()}",
        "idea=A",
        "version=v1.1",
        f"mechanism={getattr(world, 'mechanism', None)}",
        f"seed={world.seed}",
        f"n={world.n}",
        f"budget={budget}",
        f"truth_population_mean={world.truths['population_mean_reward']}",
        extra,
        "",
        "methods:",
    ]
    for row in rows:
        pop = row["population_mean_reward"]
        lines.append(
            f"  {row['method']}: estimate={pop.get('estimate')} truth={pop.get('truth')} "
            f"bias={pop.get('bias')} rmse={pop.get('rmse')} "
            f"cover={row.get('ci_coverage')} expected_q={row.get('expected_audit_fraction')} "
            f"realized_q={row.get('realized_audit_fraction')}"
        )
    return "\n".join(lines) + "\n"


def _write_static_plot(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["method"] for row in rows]
    estimates = [float(row["population_mean_reward"]["estimate"]) for row in rows]
    truths = [float(row["population_mean_reward"]["truth"]) for row in rows]
    plots = result_dir / "plots"
    fig = plot_bars(
        labels,
        {"estimate": estimates, "finite-population truth": truths},
        title="Population mean reward vs finite-population truth (one seed)",
        xlabel="method",
        ylabel="mean reward",
        n_seeds=1,
        path=plots / "population_mean_vs_truth.png",
    )
    plt.close(fig)


def run_static_cell(
    mechanism: str,
    seed: int,
    budget: float,
    methods: tuple[str, ...],
    *,
    n: int,
    world: Any | None,
    environment: dict[str, Any],
    prefix: str,
    skip_existing: bool = True,
) -> dict[str, Any]:
    exp_id = f"{prefix}_{mechanism}_b{float(budget):.2f}_s{int(seed)}"
    dest = RESULTS_ROOT / exp_id
    if skip_existing and (dest / "metrics.json").is_file():
        payload = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
        return {"experiment_id": exp_id, "result_dir": str(dest), "metrics": payload, "skipped_existing": True}
    if world is None:
        world = simulate(mechanism, int(seed), n=int(n))
    rows = []
    for method in methods:
        rng = audit_rng(int(seed), float(budget), method)
        rows.append(run_static_method_fixed(world, method, float(budget), rng))
    run_config = {
        "experiment_id": exp_id,
        "idea": "A",
        "version": "v1.1",
        "track": "static",
        "mechanism": mechanism,
        "seed": int(seed),
        "budget": float(budget),
        "n": world.n,
        "methods": list(methods),
        "paired_world": True,
        "novelty_claim": False,
    }
    metrics = {
        "experiment_id": exp_id,
        "track": "static",
        "mechanism": mechanism,
        "seed": int(seed),
        "budget": float(budget),
        "n": world.n,
        "truths": {
            "population_mean_reward": world.truths["population_mean_reward"],
            "preference_rate": world.truths["preference_rate"],
            "policy_value_target_distribution": world.truths["policy_value_target_distribution"],
        },
        "methods": [_strip_arrays(r) for r in rows],
        "rows": [flatten_static_row(r) for r in rows],
    }
    write_run(dest, run_config, metrics, environment, _log_static(world, budget, rows, extra=f"prefix={prefix}"))
    _write_static_plot(dest, rows)
    return {"experiment_id": exp_id, "result_dir": str(dest), "metrics": metrics, "skipped_existing": False}


def run_continual_method(
    pop: Any,
    method: str,
    budget: float,
    seed: int,
    *,
    adapt_policy: bool = True,
    policy_kind: str = "linear",
    config: dict[str, Any] | None = None,
    estimator_override: str | None = None,
    q_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    config_id, estimator = SEQUENTIAL_SPEC[method]
    if estimator_override is not None:
        estimator = estimator_override
    action_rng = np.random.default_rng(int(seed) + 31)
    salt = sum((i + 1) * ord(ch) for i, ch in enumerate(method)) % 997
    audit_rng_local = np.random.default_rng(int(seed) + 100_007 + int(salt))
    loop = run_loop(
        pop,
        config_id=config_id,
        budget=float(budget),
        adapt_policy=adapt_policy,
        action_rng=action_rng,
        audit_rng=audit_rng_local,
        policy_kind=policy_kind,
        config=config,
        estimator=estimator,
        q_transform=q_transform,
    )
    est = float(loop["estimands"]["population_mean_reward"]["estimate"])
    truth = float(loop["truths"]["population_mean_reward"])
    pop_err = _scalar_errors(est, truth)
    covers = _seq.interval_covers(loop["ci"], truth)
    name = label or method
    row = {
        "method": name,
        "base_method": method,
        "config_id": config_id,
        "estimator": estimator,
        "track": "continual_adaptation" if adapt_policy else "continual_frozen_policy",
        "scenario": pop.scenario,
        "seed": int(seed),
        "n": pop.n,
        "n_per_round": pop.n_per_round,
        "n_rounds": pop.n_rounds,
        "budget_requested": float(budget),
        "notes": METHOD_NOTES.get(method, ""),
        "expected_audit_fraction": loop["expected_audit_fraction"],
        "realized_audit_fraction": loop["realized_audit_fraction"],
        "population_mean_reward": pop_err,
        "ci": loop["ci"],
        "ci_coverage": bool(covers),
        "ci_low": loop["ci"].get("ci_low"),
        "ci_high": loop["ci"].get("ci_high"),
        "ci_formula": loop["ci"].get("formula"),
        "policy_adapted": bool(adapt_policy),
        "mean_pi_by_round": loop.get("mean_pi_by_round"),
        "mean_p_natural_by_round": loop.get("mean_p_natural_by_round"),
    }
    return row


def run_continual_cell(
    scenario: str,
    seed: int,
    budget: float,
    methods: tuple[str, ...],
    *,
    n_per_round: int,
    n_rounds: int,
    environment: dict[str, Any],
    prefix: str,
    skip_existing: bool = True,
    method_hooks: dict[str, dict[str, Any]] | None = None,
    extra_tau: float = 0.0,
    noise_scale: float = 1.0,
    adapt_policy: bool = True,
    policy_kind: str = "linear",
) -> dict[str, Any]:
    exp_id = f"{prefix}_{scenario}_b{float(budget):.2f}_s{int(seed)}"
    dest = RESULTS_ROOT / exp_id
    if skip_existing and (dest / "metrics.json").is_file():
        payload = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
        return {"experiment_id": exp_id, "result_dir": str(dest), "metrics": payload, "skipped_existing": True}
    rows = []
    for method in methods:
        hook = dict((method_hooks or {}).get(method, {}))
        pop = build_population(
            scenario,
            int(seed),
            n_per_round=int(n_per_round),
            n_rounds=int(n_rounds),
            extra_tau=float(extra_tau),
            noise_scale=float(noise_scale),
        )
        q_tf = hook.get("q_transform")
        if q_tf == "mix_uniform":
            q_callable: Callable[[np.ndarray], np.ndarray] | None = lambda q: misspecify_q(q, "mix_uniform")
        else:
            q_callable = None
        row = run_continual_method(
            pop,
            method,
            float(budget),
            int(seed),
            adapt_policy=bool(hook.get("adapt_policy", adapt_policy)),
            policy_kind=str(hook.get("policy_kind", policy_kind)),
            config=hook.get("config"),
            estimator_override=hook.get("estimator_override"),
            q_transform=q_callable,
            label=hook.get("label"),
        )
        rows.append(row)
    run_config = {
        "experiment_id": exp_id,
        "idea": "A",
        "version": "v1.1",
        "track": "continual",
        "scenario": scenario,
        "seed": int(seed),
        "budget": float(budget),
        "n_per_round": int(n_per_round),
        "n_rounds": int(n_rounds),
        "methods": [r["method"] for r in rows],
        "paired_world": True,
        "novelty_claim": False,
    }
    metrics = {
        "experiment_id": exp_id,
        "track": "continual",
        "scenario": scenario,
        "seed": int(seed),
        "budget": float(budget),
        "n_per_round": int(n_per_round),
        "n_rounds": int(n_rounds),
        "methods": rows,
        "rows": [flatten_static_row(r) for r in rows],
    }
    log = _log_static(
        type("W", (), {"mechanism": scenario, "seed": seed, "n": n_per_round * n_rounds, "truths": {"population_mean_reward": rows[0]["population_mean_reward"]["truth"]}})(),
        budget,
        rows,
        extra=f"track=continual prefix={prefix}",
    )
    write_run(dest, run_config, metrics, environment, log)
    _write_static_plot(dest, rows)
    return {"experiment_id": exp_id, "result_dir": str(dest), "metrics": metrics, "skipped_existing": False}


def _collect_rows(prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not RESULTS_ROOT.is_dir():
        return rows
    for path in sorted(RESULTS_ROOT.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not str(payload.get("experiment_id", "")).startswith(prefix) and prefix not in str(payload.get("experiment_id", "")):
            # Allow collecting by track via prefix substring.
            if prefix and prefix not in path.parent.name:
                continue
        rows.extend(payload.get("rows") or [])
    return rows


def _group_cells(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in key_fields[:-1]) + (row.get("method"),)
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize_static(rows: list[dict[str, Any]], baseline: str = "uniform_dr") -> dict[str, Any]:
    cells: dict[str, Any] = {}
    mechanisms = sorted({str(r.get("mechanism")) for r in rows if r.get("mechanism")})
    budgets = sorted({float(r["budget_requested"]) for r in rows if r.get("budget_requested") is not None})
    methods = sorted({str(r["method"]) for r in rows})
    for mechanism in mechanisms:
        cells.setdefault(mechanism, {})
        for budget in budgets:
            cells[mechanism][f"b={budget:.2f}"] = {}
            by_method: dict[str, list[dict[str, Any]]] = {}
            for method in methods:
                subset = [
                    r
                    for r in rows
                    if r.get("method") == method
                    and str(r.get("mechanism")) == mechanism
                    and abs(float(r["budget_requested"]) - budget) < 1e-12
                ]
                subset = sorted(subset, key=lambda r: int(r["seed"]))
                if not subset:
                    continue
                by_method[method] = subset
                estimates = np.array([float(r["population_mean_reward_estimate"]) for r in subset], dtype=float)
                truths = np.array([float(r["population_mean_reward_truth"]) for r in subset], dtype=float)
                errors = estimates - truths
                covers = [bool(r.get("ci_coverage")) for r in subset]
                realized = [float(r["realized_audit_fraction"]) for r in subset]
                expected = [float(r["expected_audit_fraction"]) for r in subset]
                n_seeds = int(len(subset))
                cell = {
                    "n_seeds": n_seeds,
                    "seeds": [int(r["seed"]) for r in subset],
                    "preliminary": n_seeds <= 3,
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "bias": float(np.mean(errors)),
                    "coverage": float(np.mean(covers)),
                    "coverage_wilson": wilson_interval(int(np.sum(covers)), n_seeds),
                    "realized_audit_fraction_mean": float(np.mean(realized)),
                    "expected_audit_fraction_mean": float(np.mean(expected)),
                    "budget_band_ok": budget_band_ok(float(np.mean(realized)), budget),
                    "per_seed_error": errors.tolist(),
                }
                cells[mechanism][f"b={budget:.2f}"][method] = cell
            if baseline in by_method:
                base_rows = by_method[baseline]
                for method, subset in by_method.items():
                    if method == baseline:
                        continue
                    if [int(r["seed"]) for r in subset] != [int(r["seed"]) for r in base_rows]:
                        continue
                    err_c = np.array([float(r["population_mean_reward_estimate"]) - float(r["population_mean_reward_truth"]) for r in subset])
                    err_b = np.array([float(r["population_mean_reward_estimate"]) - float(r["population_mean_reward_truth"]) for r in base_rows])
                    boot = paired_bootstrap_rmse_difference(err_c, err_b)
                    boot["interval_side"] = interval_side(boot["ci_low"], boot["ci_high"])
                    rel = None
                    if boot["rmse_baseline"] not in {0.0} and np.isfinite(boot["rmse_baseline"]):
                        rel = float((boot["rmse_candidate"] - boot["rmse_baseline"]) / boot["rmse_baseline"])
                    boot["relative_rmse_change"] = rel
                    cells[mechanism][f"b={budget:.2f}"].setdefault("paired_vs_uniform_dr", {})[method] = boot
    return cells


def write_summary_plots(rows: list[dict[str, Any]], dest: Path, title_prefix: str, n_seeds: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    if not methods:
        return
    rmse_vals = []
    rmse_err = []
    cover_vals = []
    realized_vals = []
    realized_err = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        errors = np.array(
            [
                float(r["population_mean_reward_estimate"]) - float(r["population_mean_reward_truth"])
                for r in subset
            ],
            dtype=float,
        )
        abs_err = np.abs(errors)
        rmse_vals.append(float(np.sqrt(np.mean(np.square(errors)))))
        rmse_err.append(float(np.std(abs_err, ddof=1) / np.sqrt(len(abs_err))) if len(abs_err) > 1 else 0.0)
        cover_vals.append(float(np.mean([bool(r.get("ci_coverage")) for r in subset])))
        realized = np.array([float(r["realized_audit_fraction"]) for r in subset], dtype=float)
        realized_vals.append(float(np.mean(realized)))
        realized_err.append(float(np.std(realized, ddof=1) / np.sqrt(len(realized))) if len(realized) > 1 else 0.0)
    fig = plot_bars(
        methods,
        {"RMSE": rmse_vals},
        yerr={"RMSE": rmse_err},
        title=f"{title_prefix}: population-mean RMSE (error bar = SE of |error|)",
        xlabel="method",
        ylabel="RMSE of E[Y]",
        n_seeds=n_seeds,
        path=dest / "rmse_by_method.png",
    )
    plt.close(fig)
    fig = plot_bars(
        methods,
        {"CI coverage": cover_vals},
        title=f"{title_prefix}: nominal 95% CI coverage",
        xlabel="method",
        ylabel="coverage",
        n_seeds=n_seeds,
        path=dest / "coverage_by_method.png",
    )
    plt.close(fig)
    fig = plot_bars(
        methods,
        {"realized audit fraction": realized_vals},
        yerr={"realized audit fraction": realized_err},
        title=f"{title_prefix}: realized audit fraction",
        xlabel="method",
        ylabel="mean(A)",
        n_seeds=n_seeds,
        path=dest / "audit_fraction_by_method.png",
    )
    plt.close(fig)


def environment_record() -> dict[str, Any]:
    env = dict(collect_environment(REPO_ROOT))
    env["experiments_run"] = True
    env["idea"] = "A"
    env["version"] = "v1.1"
    env["python_executable"] = sys.executable
    env["working_copy"] = str(REPO_ROOT)
    return env


def run_tier1(*, skip_existing: bool = True) -> dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    env = environment_record()
    written = []
    worlds: dict[tuple[str, int], Any] = {}
    for mechanism in ("mnar", "zero_overlap"):
        for seed in (0, 1, 2):
            key = (mechanism, seed)
            worlds[key] = simulate(mechanism, seed, n=2000)
            rec = run_static_cell(
                mechanism,
                seed,
                0.05,
                STATIC_TIER1_METHODS,
                n=2000,
                world=worlds[key],
                environment=env,
                prefix="t1_static",
                skip_existing=skip_existing,
            )
            written.append(rec["experiment_id"])
            print(f"[A TIER1 static] {rec['experiment_id']}", flush=True)
    for seed in (0, 1, 2):
        rec = run_continual_cell(
            "mnar_audit",
            seed,
            0.05,
            CONTINUAL_TIER1_METHODS,
            n_per_round=80,
            n_rounds=4,
            environment=env,
            prefix="t1_continual",
            skip_existing=skip_existing,
        )
        written.append(rec["experiment_id"])
        print(f"[A TIER1 continual] {rec['experiment_id']}", flush=True)
    static_rows = []
    continual_rows = []
    for path in RESULTS_ROOT.glob("t1_*/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("track") == "static":
            static_rows.extend(payload.get("rows") or [])
        else:
            continual_rows.extend(payload.get("rows") or [])
    summary = {
        "idea": "A",
        "tier": 1,
        "preliminary": True,
        "n_seeds": 3,
        "label": "PRELIMINARY",
        "written": written,
        "static": summarize_static(static_rows, baseline="uniform_dr"),
        "continual_short_mnar_audit": summarize_static(
            [{**r, "mechanism": r.get("mechanism") or "mnar_audit"} for r in continual_rows],
            baseline="uniform_dr",
        ),
        "skipped": ["b=0.01 and b=0.20 not in Tier 1 core; leftover-time only"],
        "note": "Tier 1 is a correctness smoke. Do not treat these numbers as confirmatory.",
        "promising_not_claimed": True,
    }
    (RESULTS_ROOT / "TIER1_SUMMARY.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_summary_plots(static_rows, RESULTS_ROOT / "plots" / "tier1_static", "Idea A Tier 1 static", 3)
    write_summary_plots(continual_rows, RESULTS_ROOT / "plots" / "tier1_continual", "Idea A Tier 1 short continual", 3)
    return summary


def run_tier2(*, skip_existing: bool = True, extra_seeds: list[int] | None = None) -> dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    env = environment_record()
    seeds = list(range(30))
    if extra_seeds:
        seeds = sorted(set(seeds) | set(extra_seeds))
    written = []
    worlds: dict[tuple[str, int], Any] = {}
    for mechanism in ("mnar", "zero_overlap"):
        for budget in (0.05, 0.10):
            for seed in seeds:
                key = (mechanism, seed)
                if key not in worlds:
                    worlds[key] = simulate(mechanism, seed, n=2000)
                rec = run_static_cell(
                    mechanism,
                    seed,
                    budget,
                    STATIC_TIER2_METHODS,
                    n=2000,
                    world=worlds[key],
                    environment=env,
                    prefix="t2_static",
                    skip_existing=skip_existing,
                )
                written.append(rec["experiment_id"])
                if seed % 10 == 0:
                    print(f"[A TIER2 static] {rec['experiment_id']}", flush=True)
    for scenario in ("policy_drift", "mnar_audit"):
        for budget in (0.05, 0.10):
            for seed in seeds:
                rec = run_continual_cell(
                    scenario,
                    seed,
                    budget,
                    CONTINUAL_TIER2_METHODS,
                    n_per_round=200,
                    n_rounds=8,
                    environment=env,
                    prefix="t2_continual",
                    skip_existing=skip_existing,
                )
                written.append(rec["experiment_id"])
                if seed % 10 == 0:
                    print(f"[A TIER2 continual] {rec['experiment_id']}", flush=True)
    static_rows: list[dict[str, Any]] = []
    continual_rows: list[dict[str, Any]] = []
    for path in RESULTS_ROOT.glob("t2_*/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("track") == "static":
            static_rows.extend(payload.get("rows") or [])
        else:
            continual_rows.extend(payload.get("rows") or [])
    continual_for_sum = [{**r, "mechanism": r.get("mechanism") or r.get("scenario")} for r in continual_rows]
    summary = {
        "idea": "A",
        "tier": 2,
        "n_seeds": 30,
        "label": "confirmatory_cells",
        "written_count": len(set(written)),
        "static": summarize_static(static_rows, baseline="uniform_dr"),
        "continual": summarize_static(continual_for_sum, baseline="uniform_dr"),
        "exploratory_10pct_rmse_drop": {},
        "skipped": [
            "b=0.01 and b=0.20 not in confirmatory matrix",
            "full Cartesian product not run",
        ],
        "promising_not_claimed": True,
        "note": "Numbers only. PROMISING is not written here; a later report agent applies protocol thresholds.",
    }
    for scenario in ("policy_drift", "mnar_audit"):
        for budget in (0.05, 0.10):
            key = f"{scenario}_b={budget:.2f}"
            block = summary["continual"].get(scenario, {}).get(f"b={budget:.2f}", {})
            paired = block.get("paired_vs_uniform_dr", {}).get("v1.1c")
            if paired:
                rel = paired.get("relative_rmse_change")
                summary["exploratory_10pct_rmse_drop"][key] = {
                    "relative_rmse_change": rel,
                    "hit_10pct_drop": bool(rel is not None and rel <= -0.10),
                    "note": "Exploratory check only. Missing 10% is a failed exploratory check, not automatic NOT PROMISING.",
                }
    (RESULTS_ROOT / "TIER2_SUMMARY.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Primary-cell plots: MNAR b=0.05 static and MNAR-audit b=0.05 continual.
    write_summary_plots(
        [r for r in static_rows if r.get("mechanism") == "mnar" and abs(float(r["budget_requested"]) - 0.05) < 1e-12],
        RESULTS_ROOT / "plots" / "tier2_static_mnar_b0.05",
        "Idea A Tier 2 static MNAR b=0.05",
        30,
    )
    write_summary_plots(
        [r for r in continual_rows if (r.get("mechanism") or r.get("scenario")) == "mnar_audit" and abs(float(r["budget_requested"]) - 0.05) < 1e-12],
        RESULTS_ROOT / "plots" / "tier2_continual_mnar_audit_b0.05",
        "Idea A Tier 2 continual MNAR-audit b=0.05",
        30,
    )
    return summary


def run_ablations(*, skip_existing: bool = True, seeds: tuple[int, ...] = (0, 1, 2), only: str | None = None) -> dict[str, Any]:
    env = environment_record()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}
    selected = ABLATIONS.items() if only is None else ((only, ABLATIONS[only]),)
    for ablation_id, spec in selected:
        dest_root = ABLATION_ROOT / ablation_id / "results"
        dest_root.mkdir(parents=True, exist_ok=True)
        methods = tuple(spec["methods"])
        hooks: dict[str, dict[str, Any]] = {}
        for method in methods:
            hook: dict[str, Any] = {}
            if "config" in spec:
                hook["config"] = spec["config"]
            if "policy_kind" in spec:
                hook["policy_kind"] = spec["policy_kind"]
            if "q_transform" in spec:
                hook["q_transform"] = spec["q_transform"]
            if "estimator_override" in spec:
                hook["estimator_override"] = spec["estimator_override"]
            if "adapt_policy" in spec:
                hook["adapt_policy"] = spec["adapt_policy"]
            if spec.get("label") and method == "v1.1c" and method == methods[0] and len(methods) == 1:
                hook["label"] = spec["label"]
            if hook:
                hooks[method] = hook
        ids = []
        for seed in seeds:
            rec = run_continual_cell(
                "mnar_audit",
                int(seed),
                0.05,
                methods,
                n_per_round=200,
                n_rounds=8,
                environment=env,
                prefix=f"ablation_{ablation_id}",
                skip_existing=skip_existing,
                method_hooks=hooks or None,
            )
            # Copy/symlink-less: also write into the ablation results dir if different.
            src = RESULTS_ROOT / rec["experiment_id"]
            dst = dest_root / rec["experiment_id"]
            if src != dst and src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                for name in ("config.yaml", "metrics.json", "metrics.csv", "run.log", "environment.json"):
                    target = src / name
                    if target.is_file():
                        (dst / name).write_bytes(target.read_bytes())
                plot_src = src / "plots"
                if plot_src.is_dir():
                    (dst / "plots").mkdir(parents=True, exist_ok=True)
                    for png in plot_src.glob("*.png"):
                        (dst / "plots" / png.name).write_bytes(png.read_bytes())
            ids.append(rec["experiment_id"])
            print(f"[A ablation {ablation_id}] {rec['experiment_id']}", flush=True)
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


def borderline_cells(summary: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for track_name in ("static", "continual"):
        track = summary.get(track_name) or {}
        for mech, budgets in track.items():
            if not isinstance(budgets, dict):
                continue
            for bkey, methods in budgets.items():
                paired = methods.get("paired_vs_uniform_dr") or {}
                for cand in ("v1.1c", "v1.1b"):
                    boot = paired.get(cand)
                    if not boot:
                        continue
                    side = boot.get("interval_side")
                    if side != "covers_zero":
                        continue
                    low, high = float(boot["ci_low"]), float(boot["ci_high"])
                    width = high - low
                    if width <= 0:
                        continue
                    near = min(abs(low), abs(high)) <= 0.20 * width
                    if near:
                        out.append(
                            {
                                "track": track_name,
                                "mechanism": mech,
                                "budget_key": bkey,
                                "candidate": cand,
                                "ci_low": low,
                                "ci_high": high,
                                "cheap": track_name == "static",
                            }
                        )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Idea A v1.1 experiments")
    p.add_argument("--tier", choices=["1", "2", "ablations", "all", "summarize"], default="all")
    p.add_argument("--no-skip", action="store_true")
    p.add_argument("--extend-seeds", type=int, default=0, help="If >30, run seeds 0..N-1 on Tier 2")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skip = not args.no_skip
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if args.tier in {"1", "all"}:
        run_tier1(skip_existing=skip)
    if args.tier in {"2", "all"}:
        extra = list(range(30, args.extend_seeds)) if args.extend_seeds > 30 else None
        summary = run_tier2(skip_existing=skip, extra_seeds=extra)
        borderline = borderline_cells(summary)
        (RESULTS_ROOT / "BORDERLINE.json").write_text(
            json.dumps(json_safe({"cells": borderline}), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        cheap = [c for c in borderline if c.get("cheap")]
        if cheap and args.extend_seeds <= 30:
            print(f"[A] borderline cheap cells: {cheap}. Extend with --extend-seeds 50 if desired.", flush=True)
    if args.tier in {"ablations", "all"}:
        run_ablations(skip_existing=skip)
    if args.tier == "summarize":
        if (RESULTS_ROOT / "t1_static_mnar_b0.05_s0" / "metrics.json").is_file() or list(RESULTS_ROOT.glob("t1_*/metrics.json")):
            run_tier1(skip_existing=True)
        if list(RESULTS_ROOT.glob("t2_*/metrics.json")):
            run_tier2(skip_existing=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
