"""Run Idea A common-protocol adapters on the existing synthetic world.

Paired methods on each simulated world: Uniform IPW, Neyman IPW, and the
adapted volume heuristic. Official AMPLe is not executed here.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.experiments.common_protocol.volume_adapter import (
    METHOD_ID,
    NOTES,
    run_volume_adapted_ipw,
)
from idea_A.experiments.protocol import MethodSpec, audit_rng, run_world_methods
from idea_A.experiments.run_synthetic import (
    _collect_flat,
    _flatten_rows,
    _json_safe,
    build_summary,
)
from idea_A.methods.v1_load import load_v1_module
from shared.evaluation.results import write_run
from shared.hardware.probe import collect_environment
from shared.statistical_tests.paired_bootstrap import paired_bootstrap_ci

_algorithm = load_v1_module("algorithm")

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
SUMMARY_PATH = Path(__file__).resolve().parent / "SUMMARY.json"

MECHANISMS = ("mnar", "zero_overlap")
BUDGET = 0.05
SEEDS = (0, 1, 2)

COMPARATOR_SPECS = [
    MethodSpec("uniform_ipw", "uniform", "ipw", notes="Uniform random audit + audit IPW."),
    MethodSpec(
        "neyman_ipw",
        "neyman",
        "ipw",
        notes="Classical Neyman + exploration floor + IPW. Not a novel algorithm.",
    ),
]


def experiment_id(mechanism: str, seed: int, budget: float = BUDGET) -> str:
    return f"common_protocol_{mechanism}_b{float(budget):.2f}_s{int(seed)}"


def _log_text(world: Any, budget: float, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"utc={datetime.now(timezone.utc).isoformat()}",
        "idea=A",
        "protocol=adapted_to_common_protocol",
        "volume_note=" + NOTES,
        f"mechanism={world.mechanism}",
        f"seed={world.seed}",
        f"n={world.n}",
        f"budget={budget}",
        f"truth_population_mean={world.truths['population_mean_reward']}",
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


def _paired_errors(rows: list[dict[str, Any]], method_a: str, method_b: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        a_rows = sorted(
            [r for r in rows if r["method"] == method_a and r["mechanism"] == mechanism],
            key=lambda r: int(r["seed"]),
        )
        b_rows = sorted(
            [r for r in rows if r["method"] == method_b and r["mechanism"] == mechanism],
            key=lambda r: int(r["seed"]),
        )
        if not a_rows or len(a_rows) != len(b_rows):
            continue
        if [int(r["seed"]) for r in a_rows] != [int(r["seed"]) for r in b_rows]:
            continue
        a_err = np.array([r["population_mean_reward_rmse"] for r in a_rows], dtype=float)
        b_err = np.array([r["population_mean_reward_rmse"] for r in b_rows], dtype=float)
        boot = paired_bootstrap_ci(a_err, b_err, n_boot=10000, alpha=0.05, seed=0)
        out[mechanism] = {
            "seeds": [int(r["seed"]) for r in a_rows],
            f"{method_a}_rmse": a_err.tolist(),
            f"{method_b}_rmse": b_err.tolist(),
            "mean_rmse_difference_a_minus_b": float(np.mean(a_err - b_err)),
            "paired_bootstrap_rmse": boot,
        }
    return out


def run_one(
    mechanism: str,
    seed: int,
    budget: float,
    config: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    world = simulate(mechanism, int(seed), n=int(config.get("n", 2000)))
    _, rows = run_world_methods(
        mechanism, int(seed), float(budget), config, specs=COMPARATOR_SPECS, world=world
    )
    volume_rng = audit_rng(int(seed), float(budget), "volume_adapted")
    rows.append(run_volume_adapted_ipw(world, float(budget), config, volume_rng))
    exp_id = experiment_id(mechanism, seed, budget)
    result_dir = RESULTS_ROOT / exp_id
    run_config = {
        "experiment_id": exp_id,
        "idea": "A",
        "protocol": "adapted_to_common_protocol",
        "official_reproduction": False,
        "volume_note": NOTES,
        "mechanism": mechanism,
        "seed": int(seed),
        "budget": float(budget),
        "n": world.n,
        "methods": [row["method"] for row in rows],
        "paired_world": True,
        "novelty_claim": False,
    }
    metrics = {
        "experiment_id": exp_id,
        "mechanism": mechanism,
        "seed": int(seed),
        "budget": float(budget),
        "protocol_label": "adapted_to_common_protocol",
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
    write_run(result_dir, run_config, metrics, environment, _log_text(world, budget, rows))
    return {"experiment_id": exp_id, "result_dir": str(result_dir), "metrics": metrics}


def build_common_protocol_summary(
    payloads: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    plan = {
        "mode": "common_protocol",
        "selected": [
            (mechanism, BUDGET, seed) for mechanism in MECHANISMS for seed in SEEDS
        ],
        "skipped": [],
        "mechanisms": list(MECHANISMS),
        "budgets": [BUDGET],
        "seeds": list(SEEDS),
        "note": (
            "Adapted volume heuristic vs Uniform IPW vs Neyman IPW on the "
            "existing synthetic selective-feedback world. MNAR and zero-overlap "
            "only, b=0.05, seeds 0,1,2. Not AMPLe official."
        ),
    }
    summary = build_summary(payloads, plan, config)
    rows = _collect_flat(payloads)
    summary["protocol_label"] = "adapted_to_common_protocol"
    summary["official_reproduction"] = False
    summary["volume_note"] = NOTES
    summary["leaderboard"] = {
        "name": "idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap",
        "methods_on_this_board": ["uniform_ipw", "neyman_ipw", METHOD_ID],
        "excluded": [
            "AMPLe official (different target: multi-dimensional preference identification)",
            "HyPER (blocked; RewardBench routing)",
            "Drago 2026 (theory only)",
        ],
        "rule": "Do not mix incomparable methods into one leaderboard.",
    }
    summary["paired_volume_vs_uniform"] = _paired_errors(rows, METHOD_ID, "uniform_ipw")
    summary["paired_volume_vs_neyman"] = _paired_errors(rows, METHOD_ID, "neyman_ipw")
    return summary


def main() -> int:
    config = dict(_algorithm.load_v1_config())
    config["volume_prior_count"] = 1.0
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    environment = dict(collect_environment(REPO_ROOT))
    environment["experiments_run"] = True
    environment["idea_A_common_protocol"] = True
    environment["protocol_label"] = "adapted_to_common_protocol"

    payloads: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        for seed in SEEDS:
            payload = run_one(mechanism, seed, BUDGET, config, environment)
            payloads.append(payload)
            print(f"[idea_A common_protocol] wrote {payload['experiment_id']}", flush=True)

    summary = build_common_protocol_summary(payloads, config)
    SUMMARY_PATH.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[idea_A common_protocol] summary -> {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
