"""Static Estimation track for the sequential closed loop.

The estimand is the realized finite-population mean of factual ``Y``. The
policy is not updated from outcomes. Metrics are estimation error, interval
coverage, and audit cost. This is not the Continual Adaptation experiment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from idea_A.benchmarks.Sequential_Closed_Loop.environment import build_population, run_loop
from idea_A.methods.v1_load import load_v1_1_module

_seq = load_v1_1_module("sequential_estimator")

TRACK_NAME = "static_estimation"
TRACK_METRICS = (
    "population_mean_rmse",
    "bias",
    "ci_coverage",
    "expected_audit_fraction",
    "realized_audit_fraction",
)


def run_static_estimation(
    scenario: str,
    seed: int,
    budget: float,
    config_id: str,
    *,
    n_per_round: int = 200,
    n_rounds: int = 8,
    extra_tau: float = 0.0,
    noise_scale: float = 1.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pop = build_population(
        scenario,
        seed,
        n_per_round=n_per_round,
        n_rounds=n_rounds,
        extra_tau=extra_tau,
        noise_scale=noise_scale,
    )
    action_rng = np.random.default_rng(int(seed) + 17)
    audit_rng = np.random.default_rng(int(seed) + 100_003)
    loop = run_loop(
        pop,
        config_id=config_id,
        budget=budget,
        adapt_policy=False,
        action_rng=action_rng,
        audit_rng=audit_rng,
        config=config,
    )
    est = float(loop["estimands"]["population_mean_reward"]["estimate"])
    truth = float(loop["truths"]["population_mean_reward"])
    covers = _seq.interval_covers(loop["ci"], truth)
    return {
        "track": TRACK_NAME,
        "track_metrics": list(TRACK_METRICS),
        "policy_adapted": False,
        "population_mean_rmse": float(np.sqrt((est - truth) ** 2)),
        "bias": float(est - truth),
        "ci_coverage": bool(covers),
        **loop,
    }
