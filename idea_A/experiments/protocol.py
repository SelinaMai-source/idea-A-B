"""Shared protocol: one simulated world, many paired audit designs.

Allocation rules never receive hidden labels. Callers pass ``World.observe``
output, which is NaN unless ``S=1`` or ``A=1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from idea_A.baselines.Doubly_Robust.adapter import estimate as dr_adapter_estimate
from idea_A.baselines.IPS.adapter import estimate as ips_adapter_estimate
from idea_A.baselines.SNIPS.adapter import estimate as snips_adapter_estimate
from idea_A.baselines.Uniform_Audit.adapter import allocate as allocate_uniform_adapter
from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import World, simulate
from idea_A.methods.v1_load import load_v1_module
from shared.evaluation.budget import expected_audit_fraction, realized_audit_fraction
from shared.metrics.errors import bias, mae, rmse

_algorithm = load_v1_module("algorithm")
_intervention = load_v1_module("intervention")
_identification = load_v1_module("identification")
_estimators = load_v1_module("estimators")

SCALAR_ESTIMANDS = (
    "population_mean_reward",
    "preference_rate",
    "policy_value_target_distribution",
)


def _stable_salt(text: str) -> int:
    acc = 2166136261
    for byte in str(text).encode("utf-8"):
        acc ^= byte
        acc = (acc * 16777619) % 2**32
    return int(acc)


def audit_rng(seed: int, budget: float, allocation: str) -> np.random.Generator:
    salt = (_stable_salt(allocation) % 10007) + int(round(float(budget) * 1000.0)) * 17
    return np.random.default_rng(int(seed) + 100_003 + salt)


def _finite_sigma_from_natural(world: World) -> tuple[np.ndarray, np.ndarray]:
    y_nat = world.observe(np.zeros(world.n, dtype=int))
    mask = world.s.astype(bool)
    return y_nat, mask


def allocate_uncertainty(
    world: World,
    y_visible: np.ndarray,
    observed_mask: np.ndarray,
    budget: float,
    exploration_mix: float,
    ridge_lambda: float,
) -> np.ndarray:
    """``q ∝`` predictive SD of a ridge model fit on currently visible labels."""
    y = np.asarray(y_visible, dtype=float)
    obs = np.asarray(observed_mask).astype(bool, copy=False)
    finite = obs & np.isfinite(y)
    n = world.n
    if int(np.sum(finite)) < 8:
        return allocate_uniform_adapter(n, budget)
    phi = _estimators.design_matrix(world.x, world.pi, world.stratum, world.n_strata)
    p = phi.shape[1]
    phi_o = phi[finite]
    y_o = y[finite]
    xtx = phi_o.T @ phi_o + float(ridge_lambda) * np.eye(p)
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return allocate_uniform_adapter(n, budget)
    beta = xtx_inv @ (phi_o.T @ y_o)
    resid = y_o - phi_o @ beta
    dof = max(int(y_o.shape[0] - p), 1)
    sigma2 = float(np.dot(resid, resid) / dof)
    quad = np.sum((phi @ xtx_inv) * phi, axis=1)
    scores = np.sqrt(np.maximum(sigma2 * (1.0 + quad), 1e-12))
    scores = np.maximum(scores, 1e-8)
    q = float(budget) * scores / float(np.mean(scores))
    mix = min(max(float(exploration_mix), 0.0), 1.0)
    q = (1.0 - mix) * q + mix * float(budget)
    q = np.clip(q, 1e-6, 1.0)
    mean_q = float(np.mean(q))
    if mean_q > 0.0:
        q = np.minimum(q * (float(budget) / mean_q), 1.0)
        mean_q = float(np.mean(q))
        if mean_q > 0.0:
            q = np.minimum(q * (float(budget) / mean_q), 1.0)
    return q


def allocate_q(
    name: str,
    world: World,
    budget: float,
    config: dict[str, Any],
    y_visible: np.ndarray | None = None,
    observed_mask: np.ndarray | None = None,
) -> np.ndarray:
    y_nat, mask_nat = _finite_sigma_from_natural(world)
    if y_visible is None:
        y_visible = y_nat
    if observed_mask is None:
        observed_mask = mask_nat
    mix = float(config.get("exploration_mix", 0.15))
    floor = float(config.get("q_floor", 1e-6))
    ridge = float(config.get("ridge_lambda", 1e-3))
    if name in {"uniform", "uniform_audit"}:
        return allocate_uniform_adapter(world.n, budget)
    if name in {"neyman", "neyman_oneshot"}:
        return _algorithm.allocate_neyman(
            world.stratum,
            y_visible,
            observed_mask,
            budget,
            world.n_strata,
            exploration_mix=mix,
            q_floor=floor,
        )
    if name == "fixed_stratified":
        return _algorithm.allocate_fixed_stratified(world.stratum, world.n_strata, budget)
    if name == "uncertainty":
        return allocate_uncertainty(world, y_visible, observed_mask, budget, mix, ridge)
    if name in {"none", "observed_only", "naive_ips", "no_audit"}:
        q, _a = _intervention.no_audit(world.n)
        return q
    if name in {"full_oracle", "full_feedback"}:
        q, _a = _intervention.full_feedback(world.n)
        return q
    raise ValueError(f"unknown allocation {name!r}")


def draw_a(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _intervention.draw_audit(q, rng)


def _mu_for_dr(world: World, y_visible: np.ndarray, a: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return _estimators.fit_mu_hat(
        y_visible,
        a,
        world.x,
        world.pi,
        world.stratum,
        world.n_strata,
        ridge_lambda=float(config.get("ridge_lambda", 1e-3)),
    )


def estimates_from_audit(
    world: World,
    y_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    estimator: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    pref = world.observe_pref(a)
    mu_hat = None
    if estimator in {"dr", "aipw", "doubly_robust"}:
        mu_hat = _mu_for_dr(world, y_visible, a, config)
        # Touch the adapter so the baseline module is part of the run path.
        _ = dr_adapter_estimate(
            y_visible,
            a,
            q,
            world.x,
            world.pi,
            world.stratum,
            world.n_strata,
            ridge_lambda=float(config.get("ridge_lambda", 1e-3)),
        )
    elif estimator == "ipw":
        _ = ips_adapter_estimate(y_visible, a, q)
    elif estimator in {"snips", "hajek", "hajek_snips"}:
        _ = snips_adapter_estimate(y_visible, a, q)
    return _estimators.estimate_estimands(
        y_visible,
        pref,
        a,
        q,
        world.stratum,
        world.target_weights,
        world.n_strata,
        estimator,
        mu_hat=mu_hat,
        clip=None,
    )


def _stratum_mae(estimate: np.ndarray, truth: np.ndarray) -> float:
    est = np.asarray(estimate, dtype=float)
    tru = np.asarray(truth, dtype=float)
    finite = np.isfinite(est) & np.isfinite(tru)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.abs(est[finite] - tru[finite])))


def errors_against_truth(estimates: dict[str, Any], truths: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in SCALAR_ESTIMANDS:
        est = estimates[name]
        tru = truths[name]
        out[name] = {
            "estimate": None if est is None else float(est),
            "truth": float(tru),
            "bias": float(bias(est, tru)) if np.isfinite(est) else float("nan"),
            "mae": float(mae(est, tru)) if np.isfinite(est) else float("nan"),
            "rmse": float(rmse(est, tru)) if np.isfinite(est) else float("nan"),
        }
    cond_est = np.asarray(estimates["context_conditioned_reward"], dtype=float)
    cond_tru = np.asarray(truths["context_conditioned_reward"], dtype=float)
    out["context_conditioned_reward"] = {
        "estimate": cond_est.tolist(),
        "truth": cond_tru.tolist(),
        "mae": _stratum_mae(cond_est, cond_tru),
        "n_identified_strata": int(np.sum(np.isfinite(cond_est))),
    }
    return out


def misspecify_q(q: np.ndarray, mode: str = "mix_uniform") -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    if mode == "mix_uniform":
        mean_q = float(np.mean(q_arr))
        wrong = 0.5 * q_arr + 0.5 * mean_q
        wrong = np.clip(wrong, 1e-6, 1.0)
        return wrong * (mean_q / max(float(np.mean(wrong)), 1e-12))
    if mode == "half":
        mean_q = float(np.mean(q_arr))
        wrong = np.clip(0.5 * q_arr, 1e-6, 1.0)
        return wrong * (mean_q / max(float(np.mean(wrong)), 1e-12))
    raise ValueError(f"unknown misspecification {mode!r}")


def batched_neyman_q(
    world: World,
    budget: float,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Spend the same total budget in sequential batches, updating stratum SDs.

    Used by ablation A4. The main v1.0 run uses one-shot Neyman instead.
    """
    n_batches = int(config.get("n_batches_adaptive_budget", 4))
    n_batches = max(n_batches, 1)
    n = world.n
    index = np.arange(n)
    # Use time order so nonstationary worlds remain sequential.
    q = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=int)
    y_vis = world.observe(a)
    obs = _intervention.observed_mask(world.s, a).astype(bool)
    remaining = float(budget) * n
    remaining_units = n
    edges = np.linspace(0, n, n_batches + 1, dtype=int)
    for b_idx in range(n_batches):
        lo, hi = int(edges[b_idx]), int(edges[b_idx + 1])
        if hi <= lo:
            continue
        batch = index[lo:hi]
        n_left_batches = n_batches - b_idx
        share = remaining / max(remaining_units, 1) if remaining_units else 0.0
        local_budget = min(1.0, share)
        q_all = _algorithm.allocate_neyman(
            world.stratum,
            y_vis,
            obs,
            local_budget,
            world.n_strata,
            exploration_mix=float(config.get("exploration_mix", 0.15)),
            q_floor=float(config.get("q_floor", 1e-6)),
        )
        q[batch] = q_all[batch]
        a[batch] = (rng.random(batch.shape[0]) < q[batch]).astype(int)
        remaining -= float(np.sum(q[batch]))
        remaining_units -= int(batch.shape[0])
        y_vis = world.observe(a)
        obs = _intervention.observed_mask(world.s, a).astype(bool)
    return q, a


@dataclass
class MethodSpec:
    method_id: str
    allocation: str
    estimator: str
    uses_budget: bool = True
    q_transform: Callable[[np.ndarray], np.ndarray] | None = None
    estimator_q_transform: Callable[[np.ndarray], np.ndarray] | None = None
    notes: str = ""


def default_method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("observed_only", "none", "observed_only", uses_budget=False, notes="Natural observations only."),
        MethodSpec("naive_ips", "none", "naive_ips", uses_budget=False, notes="Natural IPS pretending P(S=1|stratum) is known."),
        MethodSpec("uniform_ipw", "uniform", "ipw", notes="Uniform random audit + audit IPW."),
        MethodSpec("uniform_snips", "uniform", "snips", notes="Uniform random audit + Hájek/SNIPS."),
        MethodSpec("uniform_dr", "uniform", "dr", notes="Uniform random audit + AIPW."),
        MethodSpec("fixed_stratified_ipw", "fixed_stratified", "ipw", notes="Equal expected audits per stratum + IPW."),
        MethodSpec("neyman_ipw", "neyman", "ipw", notes="Classical Neyman + exploration floor + IPW. Not a novel algorithm."),
        MethodSpec("neyman_snips", "neyman", "snips", notes="Classical Neyman + Hájek/SNIPS."),
        MethodSpec("neyman_dr", "neyman", "dr", notes="Classical Neyman + AIPW."),
        MethodSpec("uncertainty_ipw", "uncertainty", "ipw", notes="Uncertainty sampling with known q ∝ predictive SD."),
        MethodSpec("full_feedback_oracle", "full_oracle", "ipw", uses_budget=False, notes="q=1 oracle. Does not respect the audit budget."),
    ]


def run_method(
    world: World,
    spec: MethodSpec,
    budget: float,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    if spec.allocation == "neyman_batched":
        q, a = batched_neyman_q(world, budget, config, rng)
    elif spec.allocation == "full_oracle":
        q, a = _intervention.full_feedback(world.n)
    elif spec.allocation in {"none", "no_audit"}:
        q, a = _intervention.no_audit(world.n)
    else:
        q = allocate_q(spec.allocation, world, budget, config)
        if spec.q_transform is not None:
            q = spec.q_transform(q)
        a = draw_a(q, rng)
    q_for_estimator = q
    if spec.estimator_q_transform is not None:
        q_for_estimator = spec.estimator_q_transform(q)

    y_vis = world.observe(a)
    if spec.estimator == "observed_only":
        if spec.allocation in {"none", "no_audit"}:
            mask = world.s
            y_use = world.observe(np.zeros(world.n, dtype=int))
            pref_use = world.observe_pref(np.zeros(world.n, dtype=int))
        else:
            # Unweighted mean of the audit sample (A=1), not S∨A. Used by A2.
            mask = a
            y_use = y_vis
            pref_use = world.observe_pref(a)
        estimates = _estimators.observed_only_estimands(
            y_use,
            pref_use,
            mask,
            world.stratum,
            world.target_weights,
            world.n_strata,
        )
    elif spec.estimator == "naive_ips":
        estimates = _estimators.naive_ips_estimands(
            world.observe(np.zeros(world.n, dtype=int)),
            world.observe_pref(np.zeros(world.n, dtype=int)),
            world.s,
            world.p_natural_mar,
            world.stratum,
            world.target_weights,
            world.n_strata,
            clip=None,
        )
    else:
        estimates = estimates_from_audit(world, y_vis, a, q_for_estimator, spec.estimator, config)

    err = errors_against_truth(estimates, world.truths)
    expected = expected_audit_fraction(q)
    realized = realized_audit_fraction(a)
    ident = _identification.combine_status(
        world.identification_natural,
        _identification.audit_channel_status(q_for_estimator),
    )
    return {
        "method": spec.method_id,
        "allocation": spec.allocation,
        "estimator": spec.estimator,
        "mechanism": world.mechanism,
        "seed": world.seed,
        "n": world.n,
        "budget_requested": float(budget),
        "uses_budget": spec.uses_budget,
        "notes": spec.notes,
        "expected_audit_fraction": expected,
        "realized_audit_fraction": realized,
        "min_q": float(np.min(q)),
        "max_q": float(np.max(q)),
        "min_q_for_estimator": float(np.min(q_for_estimator)),
        "n_audited": int(np.sum(a)),
        "n_natural": int(np.sum(world.s)),
        "identification": ident,
        "estimands": err,
        "population_mean_reward": err["population_mean_reward"],
        "preference_rate": err["preference_rate"],
        "policy_value_target_distribution": err["policy_value_target_distribution"],
        "context_conditioned_reward": err["context_conditioned_reward"],
    }


def run_world_methods(
    mechanism: str,
    seed: int,
    budget: float,
    config: dict[str, Any],
    specs: list[MethodSpec] | None = None,
    world: World | None = None,
) -> tuple[World, list[dict[str, Any]]]:
    n = int(config.get("n", 2000))
    if world is None:
        world = simulate(mechanism, seed, n=n)
    if specs is None:
        specs = default_method_specs()
    rows = []
    for spec in specs:
        rng = audit_rng(seed, budget, spec.allocation)
        rows.append(run_method(world, spec, budget, config, rng))
    return world, rows
