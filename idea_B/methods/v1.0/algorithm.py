"""Closed-loop runner for Idea B v1.0.

The loop is: parameters score the batch, the environment maps those scores to
a natural feedback distribution, the method spends a limited audit budget,
the observed design Gramian ``W_T = sum S phi phi^T`` is updated, and the
parameters are updated from observed labels.

A fixed policy with fixed sampling is never reported as endogenous collapse.
Better information is recorded separately from true population performance;
this runner does not treat one as a proxy for the other.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from feedback_controller import allocate_audit, make_controller_state
from observability import (
    accumulate_conditional_gramian,
    accumulate_observed_gramian,
    condition_number,
    eigenspectrum,
    lambda_min,
    skill_information,
)
from policy_update import accumulate_xy, ridge_update, sgd_update

GENERIC_CONTROLS = {
    "UniformExploration",
    "GreedyDeployment",
    "LazyDeployment",
    "StandardOnlineSGD",
    "FixedExplorationSchedule",
    "UncertaintyExploration",
    "NoFeedbackController",
    "FullFeedbackOracle",
}

METHOD_LABELS = {
    "ObservabilityController": "Observability controller v1.0",
    "UniformExploration": "Uniform Exploration (generic control)",
    "GreedyDeployment": "Greedy Deployment (generic control)",
    "LazyDeployment": "Lazy Deployment (generic control)",
    "StandardOnlineSGD": "Standard Online SGD (generic control)",
    "FixedExplorationSchedule": "Fixed Exploration Schedule (generic control)",
    "UncertaintyExploration": "Uncertainty-based Exploration (generic control)",
    "NoFeedbackController": "No Feedback Controller (generic control)",
    "FullFeedbackOracle": "Full Feedback Oracle (generic control)",
    "EqualSkillAllocation": "Equal-skill allocation (ablation)",
    "FrozenPolicyUniformSampling": "Frozen policy, fixed sampling (diagnostic)",
}


def classify_collapse(
    *,
    policy_affects_feedback: bool,
    theta_is_fixed: bool,
    sampling_is_fixed: bool,
    observed_performance: np.ndarray,
    lambda_min_over_t: np.ndarray,
    weak_skill_error: np.ndarray,
    observed_stable_drop: float = 0.05,
    lambda_ratio: float = 0.85,
    error_ratio: float = 1.10,
) -> dict[str, Any]:
    """Label endogenous collapse from trajectories and mechanism flags.

    Endogenous collapse requires a policy-dependent feedback channel AND
    moving parameters AND moving sampling. A drop in ``lambda_min`` under a
    frozen policy and frozen sampling is not endogenous collapse.
    """
    if theta_is_fixed and sampling_is_fixed:
        return {
            "endogenous_collapse": False,
            "reason": "fixed policy with fixed sampling is not endogenous collapse",
        }
    if not policy_affects_feedback:
        return {
            "endogenous_collapse": False,
            "reason": "feedback distribution does not depend on the policy",
        }

    obs = np.asarray(observed_performance, dtype=float)
    lmin = np.asarray(lambda_min_over_t, dtype=float)
    weak = np.asarray(weak_skill_error, dtype=float)
    if obs.size < 4 or lmin.size != obs.size or weak.size != obs.size:
        return {
            "endogenous_collapse": False,
            "reason": "trajectory too short to assess collapse",
        }

    split = max(1, obs.size // 4)
    early_obs = float(np.nanmean(obs[:split]))
    late_obs = float(np.nanmean(obs[-split:]))
    early_lmin = float(np.nanmean(lmin[:split]))
    late_lmin = float(np.nanmean(lmin[-split:]))
    early_weak = float(np.nanmean(weak[:split]))
    late_weak = float(np.nanmean(weak[-split:]))

    observed_stable = np.isfinite(early_obs) and np.isfinite(late_obs) and (
        late_obs >= early_obs - float(observed_stable_drop)
    )
    lambda_worsens = np.isfinite(early_lmin) and np.isfinite(late_lmin) and (
        late_lmin < early_lmin * float(lambda_ratio)
    )
    weak_worsens = np.isfinite(early_weak) and np.isfinite(late_weak) and (
        late_weak > early_weak * float(error_ratio)
    )
    collapsed = bool(observed_stable and lambda_worsens and weak_worsens)
    return {
        "endogenous_collapse": collapsed,
        "reason": (
            "observed performance stayed high while lambda_min(W_T/T) and "
            "weak-skill error worsened"
            if collapsed
            else "policy-dependent loop did not jointly show stable observed "
            "performance, shrinking lambda_min, and rising weak-skill error"
        ),
        "early_observed_performance": early_obs,
        "late_observed_performance": late_obs,
        "early_lambda_min": early_lmin,
        "late_lambda_min": late_lmin,
        "early_weak_skill_error": early_weak,
        "late_weak_skill_error": late_weak,
        "observed_stable": observed_stable,
        "lambda_min_worsened": lambda_worsens,
        "weak_skill_error_worsened": weak_worsens,
    }


def _performance(mse: float) -> float:
    if not np.isfinite(mse):
        return float("nan")
    return float(1.0 / (1.0 + max(mse, 0.0)))


def _batch_mse(phi: np.ndarray, theta: np.ndarray, y: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None:
        selected = np.ones(phi.shape[0], dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        if selected.shape[0] != phi.shape[0]:
            raise ValueError("mask length must match batch")
    if not np.any(selected):
        return float("nan")
    residual = phi[selected] @ theta - y[selected]
    return float(np.mean(residual**2))


def run_closed_loop(
    env: Any,
    method_name: str,
    *,
    seed: int,
    budget: float,
    horizon: int,
    ridge_lambda: float = 0.5,
    smoothing: float = 0.6,
    mix_uniform: float = 0.1,
    sgd_learning_rate: float = 0.15,
    lazy_period: int = 5,
    theta0: np.ndarray | None = None,
    freeze_theta: bool = False,
    freeze_sampling: bool = False,
    information_metric: str = "lambda_min",
    update_rule: str | None = None,
    rng_factory: Callable[[int], np.random.Generator] | None = None,
) -> dict[str, Any]:
    """Run one synthetic closed loop and return trajectories plus summaries."""
    from shared.data_utils.seeds import seed_everything

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if rng_factory is None:
        root_rng = seed_everything(int(seed))
    else:
        root_rng = rng_factory(int(seed))
    rng_pop = np.random.default_rng(int(root_rng.integers(0, 2**31 - 1)))
    rng_obs = np.random.default_rng(int(root_rng.integers(0, 2**31 - 1)))
    rng_method = np.random.default_rng(int(root_rng.integers(0, 2**31 - 1)))

    env.reset(rng_pop)
    d = int(env.d)
    n_skills = d
    theta_star = np.asarray(env.theta_star_array, dtype=float)
    if theta0 is None:
        theta = np.zeros(d, dtype=float)
    else:
        theta = np.asarray(theta0, dtype=float).reshape(-1).copy()
        if theta.shape[0] != d:
            raise ValueError("theta0 length must equal d")

    gramian = np.zeros((d, d), dtype=float)
    gramian_cond = np.zeros((d, d), dtype=float)
    xy_vector = np.zeros(d, dtype=float)
    state = make_controller_state(n_skills)

    method = str(method_name)
    theta_for_sampling = theta.copy()
    allocation_name = method
    if method == "FrozenPolicyUniformSampling":
        freeze_theta = True
        freeze_sampling = True
        allocation_name = "UniformExploration"
    if method == "LazyDeployment":
        update_every = int(lazy_period)
    else:
        update_every = 1
    if update_rule is None:
        update_rule = "sgd" if method == "StandardOnlineSGD" else "ridge"

    theta_is_fixed = bool(freeze_theta)
    sampling_is_fixed = bool(freeze_sampling)

    records: dict[str, list[Any]] = {
        "round": [],
        "true_mse": [],
        "observed_mse": [],
        "true_performance": [],
        "observed_performance": [],
        "gap": [],
        "lambda_min": [],
        "lambda_min_over_t": [],
        "condition_number": [],
        "feedback_cost": [],
        "cumulative_feedback_cost": [],
        "regret": [],
        "cumulative_regret": [],
        "n_observed": [],
        "n_audited": [],
        "realized_audit_fraction": [],
        "expected_audit_fraction": [],
        "weak_skill_observation_rate": [],
        "weak_skill_error": [],
        "runtime_s": [],
    }
    for k in range(n_skills):
        records[f"skill_error_{k}"] = []
        records[f"skill_mse_{k}"] = []
        records[f"skill_observation_rate_{k}"] = []
        records[f"eigenvalue_{k}"] = []
        records[f"theta_{k}"] = []

    cumulative_cost = 0.0
    cumulative_regret = 0.0
    irreducible = float(env.noise_std**2)
    t0 = time.perf_counter()
    audit_flags_all: list[np.ndarray] = []
    expected_q_all: list[np.ndarray] = []

    for t in range(int(horizon)):
        batch = env.sample_batch(rng_pop, t)
        phi = np.asarray(batch["phi"], dtype=float)
        y = np.asarray(batch["y"], dtype=float)
        skills = np.asarray(batch["skills"], dtype=int)
        n = phi.shape[0]

        sampling_theta = theta_for_sampling if freeze_sampling else theta
        sampling_t = 0 if freeze_sampling else t
        q_nat = np.asarray(
            env.natural_observation_prob(phi, skills, sampling_theta, sampling_t),
            dtype=float,
        ).reshape(-1)

        q_audit, state = allocate_audit(
            allocation_name,
            phi,
            skills,
            theta,
            gramian,
            float(budget) if method != "NoFeedbackController" else 0.0,
            state,
            rng_method,
            smoothing=smoothing,
            mix_uniform=mix_uniform,
            ridge_lambda=ridge_lambda,
            information_metric=information_metric,
            n_skills=n_skills,
        )
        if method == "FullFeedbackOracle":
            q_audit = np.ones(n, dtype=float)
        if method == "NoFeedbackController":
            q_audit = np.zeros(n, dtype=float)
        if freeze_sampling and method == "FrozenPolicyUniformSampling":
            q_audit = np.full(n, float(budget), dtype=float)

        u_nat = rng_obs.random(n)
        u_audit = rng_obs.random(n)
        natural = u_nat < q_nat
        audit_draw = u_audit < q_audit
        audit = env.apply_audit_identifiability(batch, audit_draw)
        observed = natural | audit
        q_obs = 1.0 - (1.0 - np.clip(q_nat, 0.0, 1.0)) * (1.0 - np.clip(q_audit, 0.0, 1.0))
        if not bool(env.audits_identify_weak_skill):
            identifiable = np.asarray(batch["identifiable"], dtype=float).reshape(-1)
            q_obs = 1.0 - (1.0 - np.clip(q_nat, 0.0, 1.0)) * (
                1.0 - np.clip(q_audit, 0.0, 1.0) * identifiable
            )

        gramian = accumulate_observed_gramian(gramian, phi, observed.astype(float))
        gramian_cond = accumulate_conditional_gramian(gramian_cond, phi, q_obs)
        xy_vector = accumulate_xy(xy_vector, phi, y, observed.astype(float))

        do_update = (not freeze_theta) and ((t + 1) % update_every == 0)
        if do_update:
            if update_rule == "sgd":
                eta = float(sgd_learning_rate) / np.sqrt(t + 1.0)
                theta = sgd_update(theta, phi, y, observed.astype(float), eta)
            else:
                theta = ridge_update(gramian, xy_vector, ridge_lambda)

        t_index = t + 1
        w_over_t = gramian / float(t_index)
        eigvals, _vecs = eigenspectrum(w_over_t)
        lmin = lambda_min(w_over_t)
        cond = condition_number(w_over_t, ridge=1e-9)
        true_mse = _batch_mse(phi, theta, y, None)
        observed_mse = _batch_mse(phi, theta, y, observed)
        true_perf = _performance(true_mse)
        obs_perf = _performance(observed_mse)
        gap = float(obs_perf - true_perf) if np.isfinite(obs_perf) and np.isfinite(true_perf) else float("nan")
        regret = float(max(true_mse - irreducible, 0.0))
        cumulative_regret += regret
        cost = float(np.mean(audit.astype(float)))
        cumulative_cost += cost

        records["round"].append(t)
        records["true_mse"].append(true_mse)
        records["observed_mse"].append(observed_mse)
        records["true_performance"].append(true_perf)
        records["observed_performance"].append(obs_perf)
        records["gap"].append(gap)
        records["lambda_min"].append(float(lmin))
        records["lambda_min_over_t"].append(float(lmin))
        records["condition_number"].append(float(cond) if np.isfinite(cond) else 1e12)
        records["feedback_cost"].append(cost)
        records["cumulative_feedback_cost"].append(cumulative_cost)
        records["regret"].append(regret)
        records["cumulative_regret"].append(cumulative_regret)
        records["n_observed"].append(int(np.sum(observed)))
        records["n_audited"].append(int(np.sum(audit)))
        records["realized_audit_fraction"].append(cost)
        records["expected_audit_fraction"].append(float(np.mean(q_audit)))
        skill_info = skill_information(w_over_t)
        _ = skill_info
        for k in range(n_skills):
            skill_mask = skills == k
            err_k = float(abs(theta[k] - theta_star[k]))
            mse_k = _batch_mse(phi, theta, y, skill_mask)
            rate_k = float(np.mean(observed[skill_mask])) if np.any(skill_mask) else 0.0
            records[f"skill_error_{k}"].append(err_k)
            records[f"skill_mse_{k}"].append(mse_k)
            records[f"skill_observation_rate_{k}"].append(rate_k)
            records[f"eigenvalue_{k}"].append(float(eigvals[k]) if k < eigvals.shape[0] else float("nan"))
            records[f"theta_{k}"].append(float(theta[k]))
        weak = int(env.weak_skill)
        records["weak_skill_observation_rate"].append(records[f"skill_observation_rate_{weak}"][-1])
        records["weak_skill_error"].append(records[f"skill_error_{weak}"][-1])
        records["runtime_s"].append(time.perf_counter() - t0)
        audit_flags_all.append(audit.astype(float))
        expected_q_all.append(np.asarray(q_audit, dtype=float))

    elapsed = time.perf_counter() - t0
    collapse = classify_collapse(
        policy_affects_feedback=bool(env.policy_affects_feedback),
        theta_is_fixed=theta_is_fixed,
        sampling_is_fixed=sampling_is_fixed,
        observed_performance=np.asarray(records["observed_performance"], dtype=float),
        lambda_min_over_t=np.asarray(records["lambda_min_over_t"], dtype=float),
        weak_skill_error=np.asarray(records["weak_skill_error"], dtype=float),
    )
    realized = float(np.mean(np.concatenate(audit_flags_all))) if audit_flags_all else 0.0
    expected = float(np.mean(np.concatenate(expected_q_all))) if expected_q_all else 0.0

    config = {
        "idea": "Feedback-Induced Observability Collapse",
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "generic_control": method in GENERIC_CONTROLS,
        "official_reproduction": False,
        "seed": int(seed),
        "budget": float(budget),
        "horizon": int(horizon),
        "ridge_lambda": float(ridge_lambda),
        "smoothing": float(smoothing),
        "mix_uniform": float(mix_uniform),
        "sgd_learning_rate": float(sgd_learning_rate),
        "lazy_period": int(lazy_period),
        "freeze_theta": theta_is_fixed,
        "freeze_sampling": sampling_is_fixed,
        "information_metric": str(information_metric),
        "update_rule": str(update_rule),
        "information_matrix": "observed_design_gramian W_T = sum_t S_t phi_t phi_t^T",
        "conditional_information_matrix": "W_T_cond = sum_t q_t phi_t phi_t^T",
    }
    summary = {
        "final_true_performance": records["true_performance"][-1],
        "final_observed_performance": records["observed_performance"][-1],
        "final_gap": records["gap"][-1],
        "final_true_mse": records["true_mse"][-1],
        "final_observed_mse": records["observed_mse"][-1],
        "final_lambda_min": records["lambda_min"][-1],
        "final_condition_number": records["condition_number"][-1],
        "late_true_performance": float(np.nanmean(records["true_performance"][-max(1, horizon // 4) :])),
        "late_observed_performance": float(np.nanmean(records["observed_performance"][-max(1, horizon // 4) :])),
        "late_lambda_min": float(np.nanmean(records["lambda_min"][-max(1, horizon // 4) :])),
        "late_weak_skill_error": float(np.nanmean(records["weak_skill_error"][-max(1, horizon // 4) :])),
        "mean_true_performance": float(np.nanmean(records["true_performance"])),
        "mean_observed_performance": float(np.nanmean(records["observed_performance"])),
        "cumulative_regret": records["cumulative_regret"][-1],
        "mean_feedback_cost": float(np.mean(records["feedback_cost"])),
        "realized_audit_fraction": realized,
        "expected_audit_fraction": expected,
        "runtime_s": elapsed,
        "final_skill_error": [records[f"skill_error_{k}"][-1] for k in range(n_skills)],
        "final_skill_observation_rate": [
            records[f"skill_observation_rate_{k}"][-1] for k in range(n_skills)
        ],
        "trace_W_over_T": float(np.trace(gramian / float(horizon))),
        "lambda_min_conditional": float(lambda_min(gramian_cond / float(horizon))),
    }
    return {
        "config": config,
        "trajectory": records,
        "summary": summary,
        "collapse": collapse,
        "theta_final": theta.tolist(),
        "W_T": gramian.tolist(),
        "W_T_cond": gramian_cond.tolist(),
        "method_meta": {
            "theta_is_fixed": theta_is_fixed,
            "sampling_is_fixed": sampling_is_fixed,
            "generic_control": method in GENERIC_CONTROLS,
            "official_reproduction": False,
        },
    }


def make_method_spec(name: str, **overrides: Any) -> dict[str, Any]:
    spec = {"method_name": name}
    spec.update(overrides)
    return spec


METHOD_FACTORIES = {
    name: (lambda n=name: make_method_spec(n))
    for name in list(GENERIC_CONTROLS)
    + ["ObservabilityController", "EqualSkillAllocation", "FrozenPolicyUniformSampling"]
}
