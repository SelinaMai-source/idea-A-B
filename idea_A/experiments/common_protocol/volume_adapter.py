"""Adapted AMPLe-style volume / max-variance allocation for Idea A's simulator.

This is ``adapted_to_common_protocol``, not AMPLe official.

AMPLe's official ``volume`` acquisition (``idea_A/baselines/AMPLe/main.py``,
branch ``args.acquisition.startswith("volume")``) scores RM-labeled preference
pairs under Metropolis–Hastings posterior samples of a simplex preference
weight. That path needs torch and the HuggingFace preference dataset. It is
not reused here.

The heuristic below is the mathematical analogue that *does* fit the existing
synthetic selective-feedback world: a diagonal-Gaussian posterior over
stratum means, using only currently visible labels. Audit propensity is
tilted toward strata with large posterior variance (max-variance / volume
of that posterior). Hidden ``Y`` is never read.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from idea_A.experiments.protocol import (
    draw_a,
    errors_against_truth,
    estimates_from_audit,
)
from idea_A.methods.v1_load import load_v1_module
from shared.evaluation.budget import expected_audit_fraction, realized_audit_fraction

_algorithm = load_v1_module("algorithm")
_identification = load_v1_module("identification")

METHOD_ID = "volume_adapted_ipw"
ALLOCATION_NAME = "volume_adapted"
NOTES = (
    "Adapted max-variance / posterior-volume allocation on strata, then "
    "audit IPW with known q. Not AMPLe official; AMPLe volume needs RM-scored "
    "preference pairs and an MH posterior on the simplex."
)


def stratum_visible_counts(
    stratum: np.ndarray,
    y_visible: np.ndarray,
    observed_mask: np.ndarray,
    n_strata: int,
) -> np.ndarray:
    """Count currently visible finite labels per stratum."""
    z = np.asarray(stratum, dtype=int)
    y = np.asarray(y_visible, dtype=float)
    obs = np.asarray(observed_mask).astype(bool, copy=False)
    finite = obs & np.isfinite(y)
    counts = np.zeros(int(n_strata), dtype=float)
    for h in range(int(n_strata)):
        counts[h] = float(np.sum((z == h) & finite))
    return counts


def posterior_variance(
    sigma_h: np.ndarray,
    n_visible_h: np.ndarray,
    prior_count: float = 1.0,
) -> np.ndarray:
    """Independent Gaussian posterior variance of each stratum mean.

    ``v_h = σ_h² / (m_h + λ)``. The generalized variance (squared volume) of
    the K-dimensional posterior of ``μ`` is ``∏ v_h`` when the covariance is
    diagonal. Allocating more audits to large ``v_h`` is a volume-reduction /
    max-variance heuristic.
    """
    sigma = np.maximum(np.asarray(sigma_h, dtype=float), 1e-8)
    counts = np.maximum(np.asarray(n_visible_h, dtype=float), 0.0)
    lam = max(float(prior_count), 1e-8)
    return (sigma**2) / (counts + lam)


def allocate_volume(
    stratum: np.ndarray,
    y_visible: np.ndarray,
    observed_mask: np.ndarray,
    budget: float,
    n_strata: int,
    exploration_mix: float = 0.15,
    q_floor: float = 1e-6,
    prior_count: float = 1.0,
) -> np.ndarray:
    """Unit-level known audit propensity ``q_h ∝ v_h``, mixed with uniform ``b``.

    Neyman uses ``q_h ∝ σ_h``. This rule uses ``q_h ∝ σ_h² / (m_h + λ)``, so
    rarely observed noisy strata get more of the same budget. The exploration
    floor is the same mixture used by v1.0 Neyman, not a claimed new rule.
    """
    z = np.asarray(stratum, dtype=int)
    n = int(z.shape[0])
    k = int(n_strata)
    n_h = np.bincount(z, minlength=k).astype(float)
    sigma = _algorithm.stratum_sigma(z, y_visible, observed_mask, k)
    m_h = stratum_visible_counts(z, y_visible, observed_mask, k)
    v_h = posterior_variance(sigma, m_h, prior_count=prior_count)
    scores = np.maximum(v_h, 1e-12)
    mix = min(max(float(exploration_mix), 0.0), 1.0)
    denom = float(np.dot(n_h, scores))
    if denom <= 0.0:
        q_h = np.full(k, float(budget), dtype=float)
    else:
        q_h = (float(budget) * float(n)) * scores / denom
    q_h = np.minimum(q_h, 1.0)
    q_h = (1.0 - mix) * q_h + mix * float(budget)
    q_h = np.maximum(q_h, float(q_floor))
    q_h = np.minimum(q_h, 1.0)
    q_h = _algorithm.fit_mean_q(n_h, q_h, budget, n)
    return q_h[z]


def run_volume_adapted_ipw(
    world: Any,
    budget: float,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Draw audits from the adapted volume heuristic and estimate with IPW."""
    y_nat = world.observe(np.zeros(world.n, dtype=int))
    q = allocate_volume(
        world.stratum,
        y_nat,
        world.s,
        float(budget),
        world.n_strata,
        exploration_mix=float(config.get("exploration_mix", 0.15)),
        q_floor=float(config.get("q_floor", 1e-6)),
        prior_count=float(config.get("volume_prior_count", 1.0)),
    )
    a = draw_a(q, rng)
    y_vis = world.observe(a)
    estimates = estimates_from_audit(world, y_vis, a, q, "ipw", config)
    err = errors_against_truth(estimates, world.truths)
    ident = _identification.combine_status(
        world.identification_natural,
        _identification.audit_channel_status(q),
    )
    return {
        "method": METHOD_ID,
        "allocation": ALLOCATION_NAME,
        "estimator": "ipw",
        "mechanism": world.mechanism,
        "seed": world.seed,
        "n": world.n,
        "budget_requested": float(budget),
        "uses_budget": True,
        "notes": NOTES,
        "protocol_label": "adapted_to_common_protocol",
        "official_reproduction": False,
        "expected_audit_fraction": expected_audit_fraction(q),
        "realized_audit_fraction": realized_audit_fraction(a),
        "min_q": float(np.min(q)),
        "max_q": float(np.max(q)),
        "min_q_for_estimator": float(np.min(q)),
        "n_audited": int(np.sum(a)),
        "n_natural": int(np.sum(world.s)),
        "identification": ident,
        "estimands": err,
        "population_mean_reward": err["population_mean_reward"],
        "preference_rate": err["preference_rate"],
        "policy_value_target_distribution": err["policy_value_target_distribution"],
        "context_conditioned_reward": err["context_conditioned_reward"],
        "volume_diagnostics": {
            "stratum_visible_counts": stratum_visible_counts(
                world.stratum, y_nat, world.s, world.n_strata
            ).tolist(),
            "posterior_variance": posterior_variance(
                _algorithm.stratum_sigma(world.stratum, y_nat, world.s, world.n_strata),
                stratum_visible_counts(world.stratum, y_nat, world.s, world.n_strata),
                prior_count=float(config.get("volume_prior_count", 1.0)),
            ).tolist(),
        },
    }
