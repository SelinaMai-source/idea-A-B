"""Idea B common-protocol slice on environment B2.

Comparable methods (same env, 5% budget, seeds 0/1/2): Observability
controller, Uniform exploration, Greedy deployment. Greedy is a generic
control, not official Performative SGD.

The ICML 2024 performative-bandit zeroth-order reparameterization is
**adapted-skipped**: the official core is a 1-D Bernoulli map ``θ ↦ θ²``
plus an inner KL inversion (`LearnModel`) that queries an uncensored
performative-risk oracle. Idea B is R^d linear scoring with skill-wise
feedback censoring and a budgeted audit channel. Porting the 1-D KL
inversion would rewrite the paper's procedure; a two-point ZO-SGD with
uniform audits would be a different generic optimizer, not PP-bandit.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from idea_B.experiments.run_synthetic import HORIZON, POPULATION, run_one
from shared.statistical_tests.paired_bootstrap import paired_bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
SUMMARY_PATH = HERE / "SUMMARY.json"

ENV = "B2"
BUDGET = 0.05
SEEDS = (0, 1, 2)
COMPARABLE_METHODS = (
    "ObservabilityController",
    "UniformExploration",
    "GreedyDeployment",
)

PP_BANDIT_SKIP = {
    "method_id": "pp_bandit_zeroth_order_reparam_adapted",
    "protocol_label": "adapted-skipped",
    "status": "BLOCKED",
    "official_reproduction": False,
    "reason": (
        "Official PP-bandit core (bandit-icml24.ipynb / "
        "official_reproduction/run_official_numerical.py) is a 1-D "
        "reparameterization x = y^2 with two-point zeroth-order updates and "
        "an inner LearnModel that minimizes Bernoulli KL(x_target || y^2). "
        "That procedure queries an uncensored scalar performative-risk "
        "function. Idea B environment B2 is a 3-D linear skill model whose "
        "performative channel is skill-wise observation censoring, not a "
        "Bernoulli mean map, and labels are revealed only under a limited "
        "audit budget. A faithful LearnModel+reparameterization port is not "
        "feasible on this NumPy protocol without rewriting the paper's "
        "1-D Bernoulli experiment. A two-point ZO-SGD with uniform audits "
        "would be a different generic optimizer and must not be relabeled "
        "as adapted PP-bandit. Official numerical demo remains in "
        "idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/ "
        "(COMPLETE, separate from this leaderboard)."
    ),
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
        "values": values,
    }


def _pack_method(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    subset = [r for r in records if r["method"] == method]
    fields = (
        "late_true_performance",
        "late_observed_performance",
        "late_lambda_min",
        "late_weak_skill_error",
        "mean_feedback_cost",
        "realized_audit_fraction",
        "cumulative_regret",
        "runtime_s",
    )
    payload: dict[str, Any] = {
        "n_seeds": len(subset),
        "seeds": [int(r["seed"]) for r in subset],
        "generic_control": method != "ObservabilityController",
        "official_reproduction": False,
        "paths": [r["path"] for r in subset],
    }
    for field in fields:
        payload[field] = _mean_std([float(r["summary"][field]) for r in subset])
    payload["endogenous_collapse_flags"] = [
        bool(r["collapse"]["endogenous_collapse"]) for r in subset
    ]
    return payload


def _paired(records: list[dict[str, Any]], method_a: str, method_b: str, field: str) -> dict[str, Any]:
    a_vals = []
    b_vals = []
    seeds = []
    for seed in SEEDS:
        a = next((r for r in records if r["method"] == method_a and r["seed"] == seed), None)
        b = next((r for r in records if r["method"] == method_b and r["seed"] == seed), None)
        if a is None or b is None:
            continue
        a_vals.append(float(a["summary"][field]))
        b_vals.append(float(b["summary"][field]))
        seeds.append(int(seed))
    if len(seeds) < 2:
        return {"reason": "not enough paired seeds", "seeds": seeds}
    boot = paired_bootstrap_ci(a_vals, b_vals, n_boot=5000, seed=0)
    return {
        "field": field,
        "seeds": seeds,
        f"{method_a}": a_vals,
        f"{method_b}": b_vals,
        "mean_a_minus_b": float(np.mean(np.asarray(a_vals) - np.asarray(b_vals))),
        "paired_bootstrap": boot,
    }


def build_summary(records: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    by_method = {method: _pack_method(records, method) for method in COMPARABLE_METHODS}
    return {
        "idea": "B",
        "name": "Feedback-Induced Observability Collapse",
        "protocol_label": "adapted_to_common_protocol",
        "environment": ENV,
        "budget": BUDGET,
        "seeds": list(SEEDS),
        "horizon": HORIZON,
        "population_per_round": POPULATION,
        "n_runs": len(records),
        "wall_time_s": wall_s,
        "leaderboard": {
            "name": "idea_B_B2_budget0.05_common_protocol",
            "methods_on_this_board": list(COMPARABLE_METHODS),
            "excluded": [
                "PP-bandit adapted (skipped; see adapted_skipped)",
                "PP-bandit official 1-D numerical demo (different protocol)",
                "ncvxpp / Performative SGD synthetic-iter (different protocol)",
                "Greedy/Lazy must not be relabeled as official SGD-GD",
            ],
            "rule": "Do not mix incomparable methods into one leaderboard.",
        },
        "methods": by_method,
        "paired_controller_vs_uniform": {
            "late_lambda_min": _paired(
                records, "ObservabilityController", "UniformExploration", "late_lambda_min"
            ),
            "late_true_performance": _paired(
                records,
                "ObservabilityController",
                "UniformExploration",
                "late_true_performance",
            ),
        },
        "adapted_skipped": [PP_BANDIT_SKIP],
        "greedy_label": (
            "GreedyDeployment is a generic control already used in Idea B smoke. "
            "It is not an official Performative SGD / SGD-GD reproduction."
        ),
        "information_matrix": "W_T = sum_t S_t phi_t phi_t^T (linear-model observed design Gramian)",
        "result_dirs": [r["path"] for r in records],
    }


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for method in COMPARABLE_METHODS:
        for seed in SEEDS:
            rec = run_one(
                ENV,
                method,
                int(seed),
                BUDGET,
                horizon=HORIZON,
                population=POPULATION,
                results_root=RESULTS_ROOT,
                write_plots=True,
                prefix="common_protocol",
            )
            records.append(rec)
            print(
                f"[idea_B common_protocol] wrote {rec['experiment_id']}",
                flush=True,
            )
    summary = build_summary(records, time.perf_counter() - t0)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"[idea_B common_protocol] summary -> {SUMMARY_PATH}", flush=True)
    print("[idea_B common_protocol] PP-bandit adapted: SKIPPED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
