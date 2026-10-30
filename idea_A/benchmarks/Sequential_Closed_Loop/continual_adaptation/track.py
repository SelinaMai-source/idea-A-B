"""Continual Adaptation track for the sequential closed loop.

Each round updates a linear or logistic policy from observed labels. The
next round's natural observation probability depends on the new policy.
Metrics include policy-path summaries in addition to estimation error.
This is not the Static Estimation experiment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from idea_A.benchmarks.Sequential_Closed_Loop.environment import build_population, run_loop
from idea_A.methods.v1_load import load_v1_1_module

_seq = load_v1_1_module("sequential_estimator")

TRACK_NAME = "continual_adaptation"
TRACK_METRICS = (
    "population_mean_rmse",
    "bias",
    "ci_coverage",
    "expected_audit_fraction",
    "realized_audit_fraction",
    "mean_pi_by_round",
    "mean_p_natural_by_round",
    "pi_changed",
    "p_natural_changed",
)


def run_continual_adaptation(
    scenario: str,
    seed: int,
    budget: float,
    config_id: str,
    *,
    n_per_round: int = 200,
    n_rounds: int = 8,
    extra_tau: float = 0.0,
    noise_scale: float = 1.0,
    policy_kind: str = "linear",
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
    action_rng = np.random.default_rng(int(seed) + 31)
    audit_rng = np.random.default_rng(int(seed) + 100_007)
    loop = run_loop(
        pop,
        config_id=config_id,
        budget=budget,
        adapt_policy=True,
        action_rng=action_rng,
        audit_rng=audit_rng,
        policy_kind=policy_kind,
        config=config,
    )
    est = float(loop["estimands"]["population_mean_reward"]["estimate"])
    truth = float(loop["truths"]["population_mean_reward"])
    covers = _seq.interval_covers(loop["ci"], truth)
    pi_path = np.asarray(loop["mean_pi_by_round"], dtype=float)
    p_path = np.asarray(loop["mean_p_natural_by_round"], dtype=float)
    return {
        "track": TRACK_NAME,
        "track_metrics": list(TRACK_METRICS),
        "policy_adapted": True,
        "population_mean_rmse": float(np.sqrt((est - truth) ** 2)),
        "bias": float(est - truth),
        "ci_coverage": bool(covers),
        "pi_changed": bool(pi_path.size >= 2 and abs(float(pi_path[-1] - pi_path[0])) > 1e-12),
        "p_natural_changed": bool(p_path.size >= 2 and abs(float(p_path[-1] - p_path[0])) > 1e-12),
        **loop,
    }
