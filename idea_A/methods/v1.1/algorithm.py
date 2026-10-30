"""Idea A v1.1 entry points: A-v1.0, A-v1.1a, A-v1.1b, A-v1.1c.

A-v1.0 calls the existing ``idea_A/methods/v1.0`` Neyman+IPW path and does
not change v1.0 behavior. A-v1.1a is Uniform audit + 2-fold cross-fit AIPW
(the required competitor). A-v1.1b allocates from audit-residual AIPW
variance. A-v1.1c is A-v1.1b plus remaining budget and policy-drift
upweighting (the main proposal).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from idea_A.methods.v1_load import load_v1_module
from shared.evaluation.budget import expected_audit_fraction, realized_audit_fraction

from .adaptive_allocation import (
    allocate_aipw_variance,
    allocate_aipw_variance_remaining_drift,
    allocate_uniform,
    remaining_unit_budget,
    residual_sigma_by_stratum,
    stratum_policy_drift,
    two_phase_indices,
)
from .cross_fitting import cross_fit_nuisances
from .outcome_model import estimate_estimands, fit_mu_y, load_v11_config
from .propensity import audit_channel_status, identification_notes
from .sequential_estimator import (
    RoundSnapshot,
    SequentialNuisanceState,
    aipw_influence,
    combine_round_influences,
    estimate_current_round,
    wald_interval,
)

CONFIG_IDS = ("A-v1.0", "A-v1.1a", "A-v1.1b", "A-v1.1c")
MAIN_PROPOSAL = "A-v1.1c"

_v1_algorithm = load_v1_module("algorithm")
_v1_estimators = load_v1_module("estimators")
_v1_intervention = load_v1_module("intervention")


def load_config() -> dict[str, Any]:
    return load_v11_config()


def _cfg_scalars(config: dict[str, Any] | None) -> dict[str, Any]:
    base = load_v11_config()
    if config:
        merged = dict(base)
        merged.update(config)
        return merged
    return base


def _pref_from_y(y_visible: np.ndarray) -> np.ndarray:
    y = np.asarray(y_visible, dtype=float)
    pref = np.full(y.shape[0], np.nan, dtype=float)
    finite = np.isfinite(y)
    pref[finite] = (y[finite] > 0.0).astype(float)
    return pref


def _draw_audit(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _v1_intervention.draw_audit(q, rng)


def _static_result(
    config_id: str,
    allocation: str,
    estimator: str,
    world: Any,
    q: np.ndarray,
    a: np.ndarray,
    estimates: dict[str, Any],
    ci: dict[str, Any] | None,
    notes: str,
) -> dict[str, Any]:
    ident = {
        "natural": getattr(world, "identification_natural", {}),
        "audit": audit_channel_status(q),
        "notes": identification_notes(),
    }
    pop = estimates["population_mean_reward"]
    out = {
        "config_id": config_id,
        "allocation": allocation,
        "estimator": estimator,
        "mechanism": getattr(world, "mechanism", None),
        "seed": getattr(world, "seed", None),
        "n": int(getattr(world, "n", q.shape[0])),
        "expected_audit_fraction": expected_audit_fraction(q),
        "realized_audit_fraction": realized_audit_fraction(a),
        "min_q": float(np.min(q)),
        "max_q": float(np.max(q)),
        "n_audited": int(np.sum(a)),
        "identification": ident,
        "estimands": estimates,
        "population_mean_reward": {
            "estimate": float(pop) if pop is not None and np.isfinite(pop) else float("nan"),
        },
        "q": np.asarray(q, dtype=float),
        "a": np.asarray(a, dtype=int),
        "notes": notes,
        "is_main_proposal": config_id == MAIN_PROPOSAL,
    }
    if ci is not None:
        out["population_mean_reward"].update(ci)
        out["ci"] = ci
    return out


def run_a_v10(world: Any, budget: float, rng: np.random.Generator, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """A-v1.0: existing v1.0 Neyman allocation + IPW. v1.0 files are not modified."""
    cfg = _v1_algorithm.load_v1_config()
    if config:
        cfg = dict(cfg)
        cfg.update(config)
    y_nat = world.observe(np.zeros(world.n, dtype=int))
    q = _v1_algorithm.allocate_neyman(
        world.stratum,
        y_nat,
        world.s,
        float(budget),
        world.n_strata,
        exploration_mix=float(cfg.get("exploration_mix", 0.15)),
        q_floor=float(cfg.get("q_floor", 1e-6)),
    )
    a = _draw_audit(q, rng)
    y_vis = world.observe(a)
    pref = world.observe_pref(a)
    estimates = _v1_estimators.estimate_estimands(
        y_vis,
        pref,
        a,
        q,
        world.stratum,
        world.target_weights,
        world.n_strata,
        "ipw",
        mu_hat=None,
        clip=None,
    )
    # v1.0 has no CI. Report IPW influence scores with the frozen Wald formula
    # so coverage is defined for the same interval family as v1.1, without
    # changing the v1.0 point estimator.
    mu0 = np.zeros(world.n, dtype=float)
    ci = wald_interval(aipw_influence(y_vis, a, q, mu0))
    return _static_result(
        "A-v1.0",
        "v1.0_neyman",
        "ipw",
        world,
        q,
        a,
        estimates,
        ci,
        "Existing v1.0 Neyman + exploration floor + IPW. Point estimate from v1.0.",
    )


def _crossfit_estimates(world: Any, y_vis: np.ndarray, a: np.ndarray, q: np.ndarray, rng: np.random.Generator, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    mu_y, mu_p, fold_ids = cross_fit_nuisances(
        y_vis,
        a,
        world.x,
        world.pi,
        world.stratum,
        world.n_strata,
        rng=rng,
        n_folds=int(cfg.get("n_folds", 2)),
        ridge_lambda=float(cfg.get("ridge_lambda", 1e-3)),
    )
    pref = _pref_from_y(y_vis)
    estimates = estimate_estimands(
        y_vis,
        pref,
        a,
        q,
        mu_y,
        mu_p,
        world.stratum,
        world.target_weights,
        world.n_strata,
    )
    ci = wald_interval(aipw_influence(y_vis, a, q, mu_y))
    return estimates, ci, mu_y, fold_ids


def run_a_v11a(world: Any, budget: float, rng: np.random.Generator, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """A-v1.1a: Uniform audit + 2-fold cross-fit AIPW."""
    cfg = _cfg_scalars(config)
    q = allocate_uniform(world.n, float(budget))
    a = _draw_audit(q, rng)
    y_vis = world.observe(a)
    estimates, ci, _mu, _folds = _crossfit_estimates(world, y_vis, a, q, rng, cfg)
    return _static_result(
        "A-v1.1a",
        "uniform",
        "cross_fit_dr",
        world,
        q,
        a,
        estimates,
        ci,
        "Uniform audit + 2-fold cross-fit AIPW. Required competitor.",
    )


def _pilot_sigma(
    world: Any,
    y_vis: np.ndarray,
    a: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    audited = np.asarray(a).astype(bool)
    if int(np.sum(audited)) < 2:
        return None
    model = fit_mu_y(
        y_vis,
        a,
        world.x,
        world.pi,
        world.stratum,
        world.n_strata,
        ridge_lambda=float(cfg.get("ridge_lambda", 1e-3)),
    )
    mu = model.predict(world.x, world.pi, world.stratum)
    y = np.asarray(y_vis, dtype=float)
    resid = np.full(world.n, np.nan, dtype=float)
    resid[audited] = y[audited] - mu[audited]
    return residual_sigma_by_stratum(
        world.stratum,
        resid,
        audited,
        world.n_strata,
        sigma_floor=float(cfg.get("sigma_floor", 1e-8)),
    )


def run_a_v11b(world: Any, budget: float, rng: np.random.Generator, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """A-v1.1b: two-phase allocation from audit-residual AIPW variance, then CF DR."""
    cfg = _cfg_scalars(config)
    n = int(world.n)
    b = float(budget)
    q_floor = float(cfg.get("q_floor", 1e-6))
    pilot, rest = two_phase_indices(n, float(cfg.get("pilot_fraction", 0.5)), rng)
    q = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=int)
    q[pilot] = b
    a[pilot] = _draw_audit(q[pilot], rng)
    y_vis = world.observe(a)
    sigma_h = _pilot_sigma(world, y_vis, a, cfg)
    if rest.size:
        q_rest = allocate_aipw_variance(
            world.stratum[rest],
            world.n_strata,
            b,
            None if sigma_h is None else sigma_h,
            q_floor=q_floor,
        )
        q[rest] = q_rest
        a[rest] = _draw_audit(q[rest], rng)
    y_vis = world.observe(a)
    estimates, ci, _mu, _folds = _crossfit_estimates(world, y_vis, a, q, rng, cfg)
    return _static_result(
        "A-v1.1b",
        "aipw_variance",
        "cross_fit_dr",
        world,
        q,
        a,
        estimates,
        ci,
        "Two-phase q_h from audit-residual AIPW variance approximation; E[q] targeted at b.",
    )


def run_a_v11c(world: Any, budget: float, rng: np.random.Generator, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """A-v1.1c: remaining-budget phase-2 spend; drift is zero on a static world."""
    cfg = _cfg_scalars(config)
    n = int(world.n)
    b = float(budget)
    q_floor = float(cfg.get("q_floor", 1e-6))
    pilot, rest = two_phase_indices(n, float(cfg.get("pilot_fraction", 0.5)), rng)
    q = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=int)
    remaining_expected = b * n
    remaining_units = n
    local_b = remaining_unit_budget(remaining_expected, remaining_units)
    q[pilot] = local_b
    a[pilot] = _draw_audit(q[pilot], rng)
    remaining_expected -= float(np.sum(q[pilot]))
    remaining_units -= int(pilot.shape[0])
    y_vis = world.observe(a)
    sigma_h = _pilot_sigma(world, y_vis, a, cfg)
    drift_h = stratum_policy_drift(world.pi, world.stratum, None, None, world.n_strata)
    if rest.size:
        q_rest = allocate_aipw_variance_remaining_drift(
            world.stratum[rest],
            world.n_strata,
            remaining_expected,
            remaining_units,
            None if sigma_h is None else sigma_h,
            drift_h,
            q_floor=q_floor,
            drift_kappa=float(cfg.get("drift_kappa", 1.0)),
            drift_scale=float(cfg.get("drift_scale", 0.10)),
            drift_factor_cap=float(cfg.get("drift_factor_cap", 2.0)),
        )
        q[rest] = q_rest
        a[rest] = _draw_audit(q[rest], rng)
    y_vis = world.observe(a)
    estimates, ci, _mu, _folds = _crossfit_estimates(world, y_vis, a, q, rng, cfg)
    return _static_result(
        "A-v1.1c",
        "aipw_variance_remaining_drift",
        "cross_fit_dr",
        world,
        q,
        a,
        estimates,
        ci,
        "Main proposal: A-v1.1b plus remaining-budget spending and policy-drift upweighting.",
    )


def run_static_config(
    config_id: str,
    world: Any,
    budget: float,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(config_id)
    if name == "A-v1.0":
        return run_a_v10(world, budget, rng, config=config)
    if name == "A-v1.1a":
        return run_a_v11a(world, budget, rng, config=config)
    if name == "A-v1.1b":
        return run_a_v11b(world, budget, rng, config=config)
    if name == "A-v1.1c":
        return run_a_v11c(world, budget, rng, config=config)
    raise ValueError(f"unknown config_id {name!r}; expected one of {CONFIG_IDS}")


def allocate_round(
    config_id: str,
    stratum: np.ndarray,
    n_strata: int,
    *,
    budget: float,
    remaining_expected_audits: float,
    remaining_units: int,
    sigma_h: np.ndarray | None,
    drift_h: np.ndarray | None,
    y_visible: np.ndarray | None = None,
    observed_mask: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Per-round allocation used by the sequential closed loop."""
    cfg = _cfg_scalars(config)
    q_floor = float(cfg.get("q_floor", 1e-6))
    name = str(config_id)
    n = int(np.asarray(stratum).shape[0])
    if name in {"observed_only", "none"}:
        return np.zeros(n, dtype=float)
    if name in {"uniform_ipw"}:
        return allocate_uniform(n, float(budget))
    if name in {"A-v1.0", "neyman_ipw", "neyman_dr", "Neyman+DR"}:
        if y_visible is None or observed_mask is None:
            raise ValueError("Neyman sequential allocation needs visible labels and the natural mask")
        return _v1_algorithm.allocate_neyman(
            stratum,
            y_visible,
            observed_mask,
            float(budget),
            int(n_strata),
            exploration_mix=float(cfg.get("exploration_mix", 0.15)),
            q_floor=q_floor,
        )
    if name in {"A-v1.1a", "uniform_dr"}:
        local_b = float(budget)
        return allocate_uniform(n, local_b)
    if name == "A-v1.1b":
        return allocate_aipw_variance(stratum, n_strata, float(budget), sigma_h, q_floor=q_floor)
    if name == "A-v1.1c":
        return allocate_aipw_variance_remaining_drift(
            stratum,
            n_strata,
            remaining_expected_audits,
            remaining_units,
            sigma_h,
            drift_h,
            q_floor=q_floor,
            drift_kappa=float(cfg.get("drift_kappa", 1.0)),
            drift_scale=float(cfg.get("drift_scale", 0.10)),
            drift_factor_cap=float(cfg.get("drift_factor_cap", 2.0)),
        )
    raise ValueError(f"unknown config_id {name!r}")


def sequential_round_estimate(
    round_data: RoundSnapshot,
    state: SequentialNuisanceState,
) -> dict[str, Any]:
    return estimate_current_round(round_data, state)


def sequential_combined_ci(influences: list[np.ndarray]) -> dict[str, Any]:
    return combine_round_influences(influences)
