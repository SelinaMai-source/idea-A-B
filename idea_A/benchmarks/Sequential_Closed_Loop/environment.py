"""Sequential closed-loop population for Idea A v1.1.

Covariates and potential outcomes are drawn once. Each round chooses an
action ``π``, factual ``Y = Y_π``, and a natural observation ``S`` whose
law depends on the scenario. Total expected audit count is ``b * n``.

Scenarios:

- ``policy_drift``: ``P(S=1 | π)`` is high when ``π=1`` and low when ``π=0``.
- ``mnar_audit``: ``P(S=1 | Y)`` depends on the unobserved outcome. Known
  audit ``q`` is the identification channel.

Static Estimation and Continual Adaptation are separate tracks that call
this environment with different policy rules. They are not one experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import numpy as np

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import (
    N_STRATA,
    STRATUM_EFFECTS,
    STRATUM_NOISE,
    STRATUM_PROBS,
)
from idea_A.methods.v1_load import load_v1_1_module, load_v1_module
from shared.data_utils.seeds import seed_everything
from shared.evaluation.budget import expected_audit_fraction, realized_audit_fraction

SCENARIOS = ("policy_drift", "mnar_audit")

_algorithm = load_v1_1_module("algorithm")
_seq = load_v1_1_module("sequential_estimator")
_policy = load_v1_1_module("policy_update")
_alloc = load_v1_1_module("adaptive_allocation")
_v1_int = load_v1_module("intervention")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _require_scenario(scenario: str) -> str:
    name = str(scenario)
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")
    return name


@dataclass
class SequentialPopulation:
    """Finite population of covariates and potential outcomes.

    Factual ``Y``, ``π``, and ``S`` are filled when a track runs. Algorithms
    may read ``Y`` only through ``observe``.
    """

    scenario: str
    seed: int
    n_per_round: int
    n_rounds: int
    n: int
    n_strata: int
    extra_tau: float
    noise_scale: float
    round_id: np.ndarray
    stratum: np.ndarray
    x: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    p_stratum: np.ndarray
    p_target_stratum: np.ndarray
    target_weights: np.ndarray
    pi: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    y: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    pref: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    s: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    p_natural: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    q: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    a: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def slice_round(self, t: int) -> np.ndarray:
        return np.asarray(self.round_id, dtype=int) == int(t)

    def observe(self, a: np.ndarray | None = None) -> np.ndarray:
        if a is None:
            audited = np.zeros(self.n, dtype=bool)
        else:
            audited = np.asarray(a).astype(bool, copy=False)
        visible = np.full(self.n, np.nan, dtype=float)
        revealed = self.s.astype(bool) | audited
        visible[revealed] = self.y[revealed]
        return visible

    def truths(self) -> dict[str, Any]:
        y = np.asarray(self.y, dtype=float)
        pref = np.asarray(self.pref, dtype=float)
        z = np.asarray(self.stratum, dtype=int)
        w = np.asarray(self.target_weights, dtype=float)
        stratum_means = np.array(
            [float(np.mean(y[z == h])) if np.any(z == h) else float("nan") for h in range(self.n_strata)],
            dtype=float,
        )
        return {
            "population_mean_reward": float(np.mean(y)),
            "preference_rate": float(np.mean(pref)),
            "policy_value_target_distribution": float(np.mean(w * y)),
            "context_conditioned_reward": stratum_means,
            "n": int(self.n),
            "n_strata": int(self.n_strata),
        }


def natural_propensity(scenario: str, pi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bernoulli parameter for ``S``. Callers must not use hidden ``Y`` except here."""
    name = _require_scenario(scenario)
    pi_arr = np.asarray(pi, dtype=int)
    y_arr = np.asarray(y, dtype=float)
    if name == "policy_drift":
        return np.where(pi_arr >= 1, 0.72, 0.08).astype(float)
    return np.clip(_sigmoid(-0.55 + 1.25 * y_arr), 0.02, 0.95)


def build_population(
    scenario: str,
    seed: int,
    n_per_round: int = 200,
    n_rounds: int = 8,
    extra_tau: float = 0.0,
    noise_scale: float = 1.0,
) -> SequentialPopulation:
    """Draw covariates and potential outcomes. Factual ``Y`` is not yet assigned."""
    name = _require_scenario(scenario)
    n_r = int(n_per_round)
    t_n = int(n_rounds)
    if n_r < 8 or t_n < 1:
        raise ValueError("n_per_round must be >= 8 and n_rounds >= 1")
    n = n_r * t_n
    rng = seed_everything(int(seed))
    stratum = rng.choice(N_STRATA, size=n, p=STRATUM_PROBS)
    x = np.column_stack(
        [
            rng.normal(loc=(stratum - 3.5) / 3.5, scale=0.85, size=n),
            rng.normal(loc=0.15 * STRATUM_EFFECTS[stratum], scale=1.0, size=n),
        ]
    )
    round_id = np.repeat(np.arange(t_n, dtype=int), n_r)
    eps = rng.normal(loc=0.0, scale=float(noise_scale) * STRATUM_NOISE[stratum], size=n)
    mu0 = 0.15 + 0.55 * x[:, 0] - 0.25 * x[:, 1] + STRATUM_EFFECTS[stratum]
    tau = 0.70 + 0.35 * x[:, 0] + float(extra_tau)
    y0 = mu0 + eps
    y1 = mu0 + tau + eps
    p_stratum = np.array([(stratum == h).mean() for h in range(N_STRATA)], dtype=float)
    p_target = np.full(N_STRATA, 1.0 / N_STRATA, dtype=float)
    safe_p = np.maximum(p_stratum, 1e-12)
    target_weights = p_target[stratum] / safe_p[stratum]
    return SequentialPopulation(
        scenario=name,
        seed=int(seed),
        n_per_round=n_r,
        n_rounds=t_n,
        n=n,
        n_strata=N_STRATA,
        extra_tau=float(extra_tau),
        noise_scale=float(noise_scale),
        round_id=round_id,
        stratum=np.asarray(stratum, dtype=int),
        x=np.asarray(x, dtype=float),
        y0=np.asarray(y0, dtype=float),
        y1=np.asarray(y1, dtype=float),
        p_stratum=p_stratum,
        p_target_stratum=p_target,
        target_weights=np.asarray(target_weights, dtype=float),
    )


def _exogenous_static_pi(
    pop: SequentialPopulation,
    t: int,
    idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Static track: no outcome-driven update. policy_drift flips at the midpoint."""
    x = pop.x[idx]
    z = pop.stratum[idx]
    base = _policy.initial_simulator_logit_policy(x, z, rng)
    if pop.scenario == "policy_drift" and t >= pop.n_rounds // 2:
        return (1 - base).astype(int)
    return base


def run_loop(
    pop: SequentialPopulation,
    *,
    config_id: str,
    budget: float,
    adapt_policy: bool,
    action_rng: np.random.Generator,
    audit_rng: np.random.Generator,
    policy_kind: str = "linear",
    config: dict[str, Any] | None = None,
    estimator: str = "dr",
    q_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fill factual outcomes round by round. ``adapt_policy`` selects the track.

    Static Estimation passes ``adapt_policy=False``. Continual Adaptation
    passes ``adapt_policy=True``. They must be invoked from their own track
    modules, not mixed into one result object.

    ``estimator`` is ``dr`` (history-respecting AIPW), ``ipw`` (mu=0), or
    ``observed_only`` (mean of naturally observed labels). Default remains DR
    so existing tests are unchanged.
    """
    cfg = _algorithm.load_config()
    if config:
        cfg = dict(cfg)
        cfg.update(config)
    n = pop.n
    b = float(budget)
    pop.pi = np.zeros(n, dtype=int)
    pop.y = np.full(n, np.nan, dtype=float)
    pop.pref = np.full(n, np.nan, dtype=float)
    pop.s = np.zeros(n, dtype=int)
    pop.p_natural = np.zeros(n, dtype=float)
    pop.q = np.zeros(n, dtype=float)
    pop.a = np.zeros(n, dtype=int)

    remaining_expected = b * n
    remaining_units = n
    state = _seq.SequentialNuisanceState(
        n_strata=pop.n_strata,
        ridge_lambda=float(cfg.get("ridge_lambda", 1e-3)),
    )
    policy = None
    prev_pi = None
    prev_stratum = None
    influences: list[np.ndarray] = []
    mean_pi: list[float] = []
    mean_p_s: list[float] = []
    mean_s: list[float] = []
    observed_x: list[np.ndarray] = []
    observed_pi: list[np.ndarray] = []
    observed_y: list[np.ndarray] = []

    for t in range(pop.n_rounds):
        idx = pop.slice_round(t)
        x_t = pop.x[idx]
        z_t = pop.stratum[idx]
        if adapt_policy:
            if policy is None:
                pi_t = _policy.FixedBernoulliPolicy(0.25).act(x_t, action_rng)
            else:
                pi_t = policy.act(x_t, action_rng)
        else:
            pi_t = _exogenous_static_pi(pop, t, idx, action_rng)
        y_t = np.where(pi_t == 1, pop.y1[idx], pop.y0[idx])
        p_s = natural_propensity(pop.scenario, pi_t, y_t)
        s_t = (action_rng.random(int(np.sum(idx))) < p_s).astype(int)
        pop.pi[idx] = pi_t
        pop.y[idx] = y_t
        pop.pref[idx] = (y_t > 0.0).astype(float)
        pop.p_natural[idx] = p_s
        pop.s[idx] = s_t

        sigma_h = state.audit_residual_sigma(
            pop.n_strata, sigma_floor=float(cfg.get("sigma_floor", 1e-8))
        )
        drift_h = _alloc.stratum_policy_drift(pi_t, z_t, prev_pi, prev_stratum, pop.n_strata)
        y_nat = np.full(int(np.sum(idx)), np.nan, dtype=float)
        y_nat[s_t.astype(bool)] = y_t[s_t.astype(bool)]
        q_t = _algorithm.allocate_round(
            config_id,
            z_t,
            pop.n_strata,
            budget=b,
            remaining_expected_audits=remaining_expected,
            remaining_units=remaining_units,
            sigma_h=sigma_h,
            drift_h=drift_h,
            y_visible=y_nat,
            observed_mask=s_t,
            config=cfg,
        )
        if q_transform is not None:
            q_t = np.asarray(q_transform(q_t), dtype=float)
        a_t = _v1_int.draw_audit(q_t, audit_rng)
        pop.q[idx] = q_t
        pop.a[idx] = a_t
        remaining_expected -= float(np.sum(q_t))
        remaining_units -= int(q_t.shape[0])

        y_vis_t = np.full(y_t.shape[0], np.nan, dtype=float)
        revealed = s_t.astype(bool) | a_t.astype(bool)
        y_vis_t[revealed] = y_t[revealed]
        pref_t = np.full(y_t.shape[0], np.nan, dtype=float)
        pref_t[revealed] = (y_t[revealed] > 0.0).astype(float)
        snap = _seq.RoundSnapshot(
            y_visible=y_vis_t,
            pref_visible=pref_t,
            a=a_t,
            q=q_t,
            x=x_t,
            pi=pi_t,
            stratum=z_t,
            target_weights=pop.target_weights[idx],
            n_strata=pop.n_strata,
        )
        est_name = str(estimator)
        if est_name == "ipw":
            inf = _seq.aipw_influence(y_vis_t, a_t, q_t, np.zeros(y_t.shape[0], dtype=float))
            influences.append(inf)
        elif est_name == "observed_only":
            inf = np.full(y_t.shape[0], np.nan, dtype=float)
            nat = s_t.astype(bool)
            inf[nat] = y_t[nat]
            influences.append(inf)
        else:
            est = _algorithm.sequential_round_estimate(snap, state)
            influences.append(est["influence"])
        state.append(snap)

        mean_pi.append(float(np.mean(pi_t)))
        mean_p_s.append(float(np.mean(p_s)))
        mean_s.append(float(np.mean(s_t)))
        prev_pi = pi_t.copy()
        prev_stratum = z_t.copy()

        if np.any(revealed):
            observed_x.append(x_t[revealed])
            observed_pi.append(pi_t[revealed])
            observed_y.append(y_vis_t[revealed])

        if adapt_policy and observed_x:
            x_obs = np.concatenate(observed_x, axis=0)
            pi_obs = np.concatenate(observed_pi, axis=0)
            y_obs = np.concatenate(observed_y, axis=0)
            m_obs = np.ones(y_obs.shape[0], dtype=int)
            if str(policy_kind) == "logistic":
                policy = _policy.fit_logistic_policy(
                    x_obs,
                    pi_obs,
                    y_obs,
                    m_obs,
                    ridge_lambda=float(cfg.get("policy_ridge_lambda", 1e-3)),
                    explore_mix=float(cfg.get("policy_explore_mix", 0.1)),
                )
            else:
                policy = _policy.fit_linear_policy(
                    x_obs,
                    pi_obs,
                    y_obs,
                    m_obs,
                    ridge_lambda=float(cfg.get("policy_ridge_lambda", 1e-3)),
                    explore_mix=float(cfg.get("policy_explore_mix", 0.1)),
                )

    est_name = str(estimator)
    if est_name == "observed_only":
        y_all = np.asarray(pop.y, dtype=float)
        nat = np.asarray(pop.s).astype(bool)
        obs = y_all[nat]
        n_obs = int(obs.shape[0])
        estimate = float(np.mean(obs)) if n_obs else float("nan")
        if n_obs >= 2:
            se = float(np.std(obs, ddof=1) / np.sqrt(n_obs))
            z = float(_seq.WALD_Z)
            ci = {
                "estimate": estimate,
                "se": se,
                "ci_low": estimate - z * se,
                "ci_high": estimate + z * se,
                "z": z,
                "formula": "observed_only_wald_95",
                "n": float(n_obs),
            }
        else:
            ci = {
                "estimate": estimate,
                "se": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "z": float(_seq.WALD_Z),
                "formula": "observed_only_wald_95",
                "n": float(n_obs),
            }
    else:
        ci = _algorithm.sequential_combined_ci(influences)
    truths = pop.truths()
    return {
        "config_id": str(config_id),
        "estimator": est_name,
        "scenario": pop.scenario,
        "seed": pop.seed,
        "n": pop.n,
        "n_per_round": pop.n_per_round,
        "n_rounds": pop.n_rounds,
        "budget_requested": b,
        "expected_audit_fraction": expected_audit_fraction(pop.q),
        "realized_audit_fraction": realized_audit_fraction(pop.a),
        "mean_pi_by_round": mean_pi,
        "mean_p_natural_by_round": mean_p_s,
        "mean_s_by_round": mean_s,
        "policy_adapted": bool(adapt_policy),
        "estimands": {
            "population_mean_reward": {
                "estimate": ci["estimate"],
                "truth": truths["population_mean_reward"],
                "se": ci["se"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "formula": ci["formula"],
            }
        },
        "truths": truths,
        "ci": ci,
        "identification_notes": (
            "Known audit q is the MNAR identification channel. "
            "policy_drift uses P(S=1|π); mnar_audit uses P(S=1|Y). "
            "An audit does not convert MNAR into MAR."
        ),
    }
