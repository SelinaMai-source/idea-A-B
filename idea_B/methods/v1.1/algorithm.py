"""Closed-loop runner for Idea B v1.1.

The loop is: parameters score the batch, the environment maps those scores
to a natural feedback distribution, the method spends a limited audit
budget, three information matrices are updated, and the parameters are
updated from observed labels.

Primary metrics: excess MSE versus the noise floor, worst-skill MSE,
population-weighted MSE, parameter error, cumulative regret, recovery
time. ``1/(1+MSE)`` is secondary. A larger eigenvalue is not success.

A fixed policy with fixed sampling is never reported as endogenous collapse.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from .feedback_dynamics import (
    NOISE_FLOOR,
    batch_mse,
    classify_collapse,
    cusum_detection_delay,
    excess_mse,
    late_window_mean,
    observation_probability,
    parameter_error,
    per_skill_parameter_error,
    population_weighted_mse,
    recovery_time,
    regret_increment,
    secondary_performance,
    worst_skill_mse,
)
from .information_controller import (
    D_OPTIMAL_NAMES,
    E_OPTIMAL_NAMES,
    EQUAL_SKILL_NAMES,
    GREEDY_NAMES,
    LAZY_NAMES,
    NONE_NAMES,
    ORACLE_NAMES,
    PROPORTIONAL_NAMES,
    UNCERTAINTY_NAMES,
    UNIFORM_NAMES,
    V10_NAMES,
    V11C_NAMES,
    allocate_audit,
    make_controller_state,
)
from .observability import (
    SlidingWindowGramian,
    accumulate_candidates_gramian,
    accumulate_conditional_gramian,
    accumulate_observed_gramian,
    condition_number,
    discount_gramian,
    eigenspectrum,
    information_bundle,
    lambda_min,
    numerical_rank,
    skill_information,
)
from .policy_update import accumulate_xy, discount_xy, ridge_update, sgd_update

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


GENERIC_CONTROLS = (
    UNIFORM_NAMES
    | GREEDY_NAMES
    | LAZY_NAMES
    | NONE_NAMES
    | ORACLE_NAMES
    | UNCERTAINTY_NAMES
    | PROPORTIONAL_NAMES
    | {"StandardOnlineSGD", "FixedExplorationSchedule"}
)

METHOD_LABELS = {
    "B-v1.0": "B-v1.0 original observability controller (reference; unchanged)",
    "ObservabilityController": "B-v1.0 original observability controller (reference; unchanged)",
    "B-v1.1a": "B-v1.1a equal expected audit count per skill",
    "EqualSkill": "B-v1.1a equal expected audit count per skill",
    "EqualSkillAllocation": "B-v1.1a equal expected audit count per skill",
    "B-v1.1b": "B-v1.1b E-optimal (classical control; not novel)",
    "E-optimal": "B-v1.1b E-optimal (classical control; not novel)",
    "D-optimal": "D-optimal (classical control; not novel)",
    "B-v1.1c": "B-v1.1c discounted information + one-step worst-skill MSE (proposal)",
    "UniformExploration": "Uniform Exploration (generic control)",
    "Uniform": "Uniform Exploration (generic control)",
    "GreedyDeployment": "Greedy Deployment (generic control)",
    "Greedy": "Greedy Deployment (generic control)",
    "LazyDeployment": "Lazy Deployment (generic control)",
    "Lazy": "Lazy Deployment (generic control)",
    "StandardOnlineSGD": "Standard Online SGD (generic control)",
    "FixedExplorationSchedule": "Fixed Exploration Schedule (generic control)",
    "UncertaintyExploration": "Uncertainty-based Exploration (generic control)",
    "NoFeedbackController": "No Feedback Controller (generic control)",
    "FullFeedbackOracle": "Full Feedback Oracle (generic control)",
    "Oracle": "Full Feedback Oracle (generic control)",
    "Proportional": "Proportional-to-frequency allocation (reference rule)",
    "ProportionalFrequency": "Proportional-to-frequency allocation (reference rule)",
    "FrozenPolicyUniformSampling": "Frozen policy, fixed sampling (diagnostic)",
}


def _is_generic(method: str) -> bool:
    if method in EQUAL_SKILL_NAMES or method in V11C_NAMES or method in V10_NAMES:
        return False
    if method in E_OPTIMAL_NAMES or method in D_OPTIMAL_NAMES:
        return False
    return method in GENERIC_CONTROLS or method in {
        "Uniform",
        "Greedy",
        "Lazy",
        "Oracle",
        "Proportional",
        "ProportionalFrequency",
    }


def _default_theta0(env: Any, d: int) -> np.ndarray:
    default = getattr(env, "default_theta0", None)
    if callable(default):
        theta = np.asarray(default(), dtype=float).reshape(-1)
        if theta.shape[0] != d:
            raise ValueError("env.default_theta0 length must equal d")
        return theta
    return np.zeros(d, dtype=float)


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
    information_discount: float = 0.90,
    sliding_window: int = 8,
    e_optimal_iterations: int = 12,
    d_optimal_iterations: int = 12,
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
    noise_std = float(getattr(env, "noise_std", 0.15))
    noise_floor = float(noise_std**2)
    if theta0 is None:
        theta = _default_theta0(env, d)
    else:
        theta = np.asarray(theta0, dtype=float).reshape(-1).copy()
        if theta.shape[0] != d:
            raise ValueError("theta0 length must equal d")

    gramian = np.zeros((d, d), dtype=float)
    gramian_cond = np.zeros((d, d), dtype=float)
    gramian_candidates = np.zeros((d, d), dtype=float)
    gramian_disc = np.zeros((d, d), dtype=float)
    xy_vector = np.zeros(d, dtype=float)
    xy_disc = np.zeros(d, dtype=float)
    window = SlidingWindowGramian(d, int(sliding_window))
    state = make_controller_state(n_skills)

    method = str(method_name)
    theta_for_sampling = theta.copy()
    allocation_name = method
    if method == "FrozenPolicyUniformSampling":
        freeze_theta = True
        freeze_sampling = True
        allocation_name = "UniformExploration"
    if method in LAZY_NAMES:
        update_every = int(lazy_period)
    else:
        update_every = 1
    if update_rule is None:
        update_rule = "sgd" if method == "StandardOnlineSGD" else "ridge"

    theta_is_fixed = bool(freeze_theta)
    sampling_is_fixed = bool(freeze_sampling)
    rho = float(information_discount)

    records: dict[str, list[Any]] = {
        "round": [],
        "true_mse": [],
        "observed_mse": [],
        "excess_mse": [],
        "worst_skill_mse": [],
        "population_weighted_mse": [],
        "parameter_error": [],
        "true_performance_secondary": [],
        "observed_performance_secondary": [],
        "gap_secondary": [],
        "lambda_min_realized": [],
        "lambda_min_conditional": [],
        "lambda_min_candidates": [],
        "lambda_min_window": [],
        "lambda_min_over_t": [],
        "condition_number": [],
        "condition_number_window": [],
        "rank_realized": [],
        "rank_window": [],
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
        "weak_skill_mse": [],
        "audit_subset_of_natural": [],
        "runtime_s": [],
    }
    for k in range(n_skills):
        records[f"skill_error_{k}"] = []
        records[f"skill_mse_{k}"] = []
        records[f"skill_observation_rate_{k}"] = []
        records[f"skill_information_{k}"] = []
        records[f"eigenvalue_{k}"] = []
        records[f"theta_{k}"] = []

    cumulative_cost = 0.0
    cumulative_regret = 0.0
    t0 = time.perf_counter()
    audit_flags_all: list[np.ndarray] = []
    expected_q_all: list[np.ndarray] = []

    for t in range(int(horizon)):
        batch = env.sample_batch(rng_pop, t)
        phi = np.asarray(batch["phi"], dtype=float)
        y = np.asarray(batch["y"], dtype=float)
        skills = np.asarray(batch["skills"], dtype=int)
        n = phi.shape[0]
        theta_star_t = np.asarray(getattr(env, "theta_star_array", env.theta_star), dtype=float)

        sampling_theta = theta_for_sampling if freeze_sampling else theta
        sampling_t = 0 if freeze_sampling else t
        q_nat = np.asarray(
            env.natural_observation_prob(phi, skills, sampling_theta, sampling_t),
            dtype=float,
        ).reshape(-1)

        gramian_for_alloc = gramian_disc if allocation_name in V11C_NAMES else gramian
        q_audit, state = allocate_audit(
            allocation_name,
            phi,
            skills,
            theta,
            gramian_for_alloc,
            float(budget) if method not in NONE_NAMES else 0.0,
            state,
            rng_method,
            smoothing=smoothing,
            mix_uniform=mix_uniform,
            ridge_lambda=ridge_lambda,
            information_metric=information_metric,
            n_skills=n_skills,
            q_nat=q_nat,
            gramian_discounted=gramian_disc,
            information_discount=rho,
            noise_std=noise_std,
            e_optimal_iterations=e_optimal_iterations,
            d_optimal_iterations=d_optimal_iterations,
        )
        if method in ORACLE_NAMES:
            q_audit = np.ones(n, dtype=float)
        if method in NONE_NAMES:
            q_audit = np.zeros(n, dtype=float)
        if freeze_sampling and method == "FrozenPolicyUniformSampling":
            q_audit = np.full(n, float(budget), dtype=float)

        u_nat = rng_obs.random(n)
        u_audit = rng_obs.random(n)
        natural = u_nat < q_nat
        audit_draw = u_audit < q_audit
        audit = env.apply_audit_identifiability(batch, audit_draw)
        observed = natural | audit
        q_obs = observation_probability(q_nat, q_audit)
        if not bool(env.audits_identify_weak_skill):
            identifiable = np.asarray(batch["identifiable"], dtype=float).reshape(-1)
            q_obs = observation_probability(q_nat, np.clip(q_audit, 0.0, 1.0) * identifiable)

        gramian_disc = discount_gramian(gramian_disc, rho)
        xy_disc = discount_xy(xy_disc, rho)
        gramian = accumulate_observed_gramian(gramian, phi, observed.astype(float))
        gramian_cond = accumulate_conditional_gramian(gramian_cond, phi, q_obs)
        gramian_candidates = accumulate_candidates_gramian(gramian_candidates, phi)
        gramian_disc = accumulate_observed_gramian(gramian_disc, phi, observed.astype(float))
        xy_vector = accumulate_xy(xy_vector, phi, y, observed.astype(float))
        xy_disc = accumulate_xy(xy_disc, phi, y, observed.astype(float))
        window.push(accumulate_observed_gramian(np.zeros((d, d)), phi, observed.astype(float)))

        do_update = (not freeze_theta) and ((t + 1) % update_every == 0)
        if do_update:
            if update_rule == "sgd":
                eta = float(sgd_learning_rate) / np.sqrt(t + 1.0)
                theta = sgd_update(theta, phi, y, observed.astype(float), eta)
            else:
                use_disc = allocation_name in V11C_NAMES
                theta = ridge_update(
                    gramian_disc if use_disc else gramian,
                    xy_disc if use_disc else xy_vector,
                    ridge_lambda,
                )

        t_index = t + 1
        w_over_t = gramian / float(t_index)
        eigvals, _vecs = eigenspectrum(w_over_t)
        window_stats = window.summaries(ridge=1e-9)
        true_mse = batch_mse(phi, theta, y, None)
        observed_mse = batch_mse(phi, theta, y, observed)
        true_perf = secondary_performance(true_mse)
        obs_perf = secondary_performance(observed_mse)
        gap = (
            float(obs_perf - true_perf)
            if np.isfinite(obs_perf) and np.isfinite(true_perf)
            else float("nan")
        )
        regret = regret_increment(true_mse, noise_floor=noise_floor)
        cumulative_regret += 0.0 if not np.isfinite(regret) else regret
        cost = float(np.mean(audit.astype(float)))
        cumulative_cost += cost
        skill_mses = []
        mix_weights = []
        for k in range(n_skills):
            skill_mask = skills == k
            mse_k = batch_mse(phi, theta, y, skill_mask)
            skill_mses.append(mse_k)
            mix_weights.append(float(np.mean(skill_mask)))
        worst = worst_skill_mse(skill_mses)
        weighted = population_weighted_mse(skill_mses, mix_weights)
        param_err = parameter_error(theta, theta_star_t)
        skill_err = per_skill_parameter_error(theta, theta_star_t)
        skill_info = skill_information(w_over_t)
        audit_inside_natural = bool(np.all((~audit) | natural)) if n > 0 else True

        records["round"].append(t)
        records["true_mse"].append(true_mse)
        records["observed_mse"].append(observed_mse)
        records["excess_mse"].append(excess_mse(true_mse, noise_floor=noise_floor))
        records["worst_skill_mse"].append(worst)
        records["population_weighted_mse"].append(weighted)
        records["parameter_error"].append(param_err)
        records["true_performance_secondary"].append(true_perf)
        records["observed_performance_secondary"].append(obs_perf)
        records["gap_secondary"].append(gap)
        records["lambda_min_realized"].append(float(lambda_min(w_over_t)))
        records["lambda_min_conditional"].append(float(lambda_min(gramian_cond / float(t_index))))
        records["lambda_min_candidates"].append(
            float(lambda_min(gramian_candidates / float(t_index)))
        )
        records["lambda_min_window"].append(float(window_stats["lambda_min"]))
        records["lambda_min_over_t"].append(float(lambda_min(w_over_t)))
        records["condition_number"].append(
            float(condition_number(w_over_t, ridge=1e-9))
            if np.isfinite(condition_number(w_over_t, ridge=1e-9))
            else 1e12
        )
        cond_w = window_stats["condition_number"]
        records["condition_number_window"].append(float(cond_w) if np.isfinite(cond_w) else 1e12)
        records["rank_realized"].append(int(numerical_rank(w_over_t)))
        records["rank_window"].append(int(window_stats["rank"]))
        records["feedback_cost"].append(cost)
        records["cumulative_feedback_cost"].append(cumulative_cost)
        records["regret"].append(regret)
        records["cumulative_regret"].append(cumulative_regret)
        records["n_observed"].append(int(np.sum(observed)))
        records["n_audited"].append(int(np.sum(audit)))
        records["realized_audit_fraction"].append(cost)
        records["expected_audit_fraction"].append(float(np.mean(q_audit)))
        for k in range(n_skills):
            skill_mask = skills == k
            rate_k = float(np.mean(observed[skill_mask])) if np.any(skill_mask) else 0.0
            records[f"skill_error_{k}"].append(float(skill_err[k]))
            records[f"skill_mse_{k}"].append(skill_mses[k])
            records[f"skill_observation_rate_{k}"].append(rate_k)
            records[f"skill_information_{k}"].append(
                float(skill_info[k]) if k < skill_info.shape[0] else float("nan")
            )
            records[f"eigenvalue_{k}"].append(
                float(eigvals[k]) if k < eigvals.shape[0] else float("nan")
            )
            records[f"theta_{k}"].append(float(theta[k]))
        weak = int(env.weak_skill)
        records["weak_skill_observation_rate"].append(records[f"skill_observation_rate_{weak}"][-1])
        records["weak_skill_error"].append(records[f"skill_error_{weak}"][-1])
        records["weak_skill_mse"].append(records[f"skill_mse_{weak}"][-1])
        records["audit_subset_of_natural"].append(audit_inside_natural)
        records["runtime_s"].append(time.perf_counter() - t0)
        audit_flags_all.append(audit.astype(float))
        expected_q_all.append(np.asarray(q_audit, dtype=float))

    elapsed = time.perf_counter() - t0
    collapse = classify_collapse(
        policy_affects_feedback=bool(env.policy_affects_feedback),
        theta_is_fixed=theta_is_fixed,
        sampling_is_fixed=sampling_is_fixed,
        observed_performance=np.asarray(records["observed_performance_secondary"], dtype=float),
        lambda_min_over_t=np.asarray(records["lambda_min_over_t"], dtype=float),
        weak_skill_error=np.asarray(records["weak_skill_error"], dtype=float),
    )
    realized = float(np.mean(np.concatenate(audit_flags_all))) if audit_flags_all else 0.0
    expected = float(np.mean(np.concatenate(expected_q_all))) if expected_q_all else 0.0

    change_round = getattr(env, "change_round", None)
    if change_round is None:
        recovery = {
            "recovery_time": float("nan"),
            "recovery_censored": False,
            "reason": "no pre-specified change point; recovery time not defined",
        }
        detection = {
            "detected": False,
            "detection_delay": "undefined",
            "detection_censored": False,
            "reason": "no pre-specified change point; detection delay not defined",
        }
    else:
        recovery = recovery_time(
            records["worst_skill_mse"],
            int(change_round),
            horizon=int(horizon),
        )
        detection = cusum_detection_delay(
            records["observed_mse"],
            int(change_round),
            noise_floor=noise_floor,
            horizon=int(horizon),
        )

    info = information_bundle(
        realized=gramian,
        conditional=gramian_cond,
        candidates=gramian_candidates,
        ridge=1e-9,
    )

    config = {
        "idea": "Feedback-Induced Observability Collapse",
        "version": "v1.1",
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "generic_control": _is_generic(method),
        "classical_control": method in E_OPTIMAL_NAMES or method in D_OPTIMAL_NAMES
        or method in {"B-v1.1b", "E-optimal", "D-optimal"},
        "proposal": method in V11C_NAMES,
        "official_reproduction": False,
        "novelty_claim_e_or_d_optimal": False,
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
        "information_discount": rho,
        "sliding_window": int(sliding_window),
        "e_optimal_iterations": int(e_optimal_iterations),
        "d_optimal_iterations": int(d_optimal_iterations),
        "noise_std": noise_std,
        "noise_floor": noise_floor,
        "information_matrix": "observed_design_gramian W_T = sum_t S_t phi_t phi_t^T",
        "conditional_information_matrix": "W_T_cond = sum_t q_t phi_t phi_t^T",
        "candidates_information_matrix": "W_T_cand = sum_t phi_t phi_t^T",
        "higher_eigenvalue_is_not_success": True,
        "primary_metrics": [
            "excess_mse",
            "worst_skill_mse",
            "population_weighted_mse",
            "parameter_error",
            "cumulative_regret",
            "recovery_time",
        ],
        "secondary_metrics": ["true_performance_secondary", "lambda_min_realized"],
    }
    late = max(1, int(horizon) // 4)
    summary = {
        "noise_floor": noise_floor,
        "final_true_mse": records["true_mse"][-1],
        "final_observed_mse": records["observed_mse"][-1],
        "final_excess_mse": records["excess_mse"][-1],
        "final_worst_skill_mse": records["worst_skill_mse"][-1],
        "final_population_weighted_mse": records["population_weighted_mse"][-1],
        "final_parameter_error": records["parameter_error"][-1],
        "final_true_performance_secondary": records["true_performance_secondary"][-1],
        "final_observed_performance_secondary": records["observed_performance_secondary"][-1],
        "final_gap_secondary": records["gap_secondary"][-1],
        "final_lambda_min_realized": records["lambda_min_realized"][-1],
        "final_lambda_min_conditional": records["lambda_min_conditional"][-1],
        "final_lambda_min_candidates": records["lambda_min_candidates"][-1],
        "final_condition_number": records["condition_number"][-1],
        "late_true_mse": late_window_mean(records["true_mse"], horizon),
        "late_excess_mse": late_window_mean(records["excess_mse"], horizon),
        "late_worst_skill_mse": late_window_mean(records["worst_skill_mse"], horizon),
        "late_population_weighted_mse": late_window_mean(
            records["population_weighted_mse"], horizon
        ),
        "late_parameter_error": late_window_mean(records["parameter_error"], horizon),
        "late_true_performance_secondary": late_window_mean(
            records["true_performance_secondary"], horizon
        ),
        "late_observed_performance_secondary": late_window_mean(
            records["observed_performance_secondary"], horizon
        ),
        "late_lambda_min": late_window_mean(records["lambda_min_realized"], horizon),
        "late_weak_skill_error": late_window_mean(records["weak_skill_error"], horizon),
        "late_weak_skill_mse": late_window_mean(records["weak_skill_mse"], horizon),
        "mean_true_mse": float(np.nanmean(records["true_mse"])),
        "cumulative_regret": records["cumulative_regret"][-1],
        "mean_feedback_cost": float(np.mean(records["feedback_cost"])),
        "realized_audit_fraction": realized,
        "expected_audit_fraction": expected,
        "runtime_s": elapsed,
        "final_skill_error": [records[f"skill_error_{k}"][-1] for k in range(n_skills)],
        "final_skill_mse": [records[f"skill_mse_{k}"][-1] for k in range(n_skills)],
        "final_skill_observation_rate": [
            records[f"skill_observation_rate_{k}"][-1] for k in range(n_skills)
        ],
        "trace_W_over_T": float(np.trace(gramian / float(horizon))),
        "lambda_min_conditional": float(lambda_min(gramian_cond / float(horizon))),
        "audit_subset_of_natural_fraction": float(np.mean(records["audit_subset_of_natural"])),
        "recovery_time": recovery.get("recovery_time"),
        "recovery_censored": recovery.get("recovery_censored"),
        "detection_delay": detection.get("detection_delay"),
        "detected": detection.get("detected"),
        "detection_censored": detection.get("detection_censored"),
        "late_window": late,
        "higher_eigenvalue_is_not_success": True,
        **{f"info_{key}": value for key, value in info.items()},
    }
    # Keep v1 field names as secondary aliases so older plotters do not crash.
    summary["final_true_performance"] = summary["final_true_performance_secondary"]
    summary["final_observed_performance"] = summary["final_observed_performance_secondary"]
    summary["final_gap"] = summary["final_gap_secondary"]
    summary["late_true_performance"] = summary["late_true_performance_secondary"]
    summary["late_observed_performance"] = summary["late_observed_performance_secondary"]
    summary["final_lambda_min"] = summary["final_lambda_min_realized"]
    records["true_performance"] = records["true_performance_secondary"]
    records["observed_performance"] = records["observed_performance_secondary"]
    records["gap"] = records["gap_secondary"]
    records["lambda_min"] = records["lambda_min_realized"]

    return {
        "config": config,
        "trajectory": records,
        "summary": summary,
        "collapse": collapse,
        "recovery": recovery,
        "detection": detection,
        "theta_final": theta.tolist(),
        "W_T": gramian.tolist(),
        "W_T_cond": gramian_cond.tolist(),
        "W_T_candidates": gramian_candidates.tolist(),
        "W_T_discounted": gramian_disc.tolist(),
        "method_meta": {
            "theta_is_fixed": theta_is_fixed,
            "sampling_is_fixed": sampling_is_fixed,
            "generic_control": _is_generic(method),
            "classical_control": method in E_OPTIMAL_NAMES or method in D_OPTIMAL_NAMES,
            "proposal": method in V11C_NAMES,
            "official_reproduction": False,
            "v1_0_controller_wrapped": method in V10_NAMES,
        },
    }


def make_method_spec(name: str, **overrides: Any) -> dict[str, Any]:
    spec = {"method_name": name}
    spec.update(overrides)
    return spec


METHOD_FACTORIES = {
    name: (lambda n=name: make_method_spec(n))
    for name in [
        "B-v1.0",
        "B-v1.1a",
        "B-v1.1b",
        "B-v1.1c",
        "D-optimal",
        "E-optimal",
        "EqualSkill",
        "Proportional",
        "UniformExploration",
        "GreedyDeployment",
        "LazyDeployment",
        "StandardOnlineSGD",
        "FixedExplorationSchedule",
        "UncertaintyExploration",
        "NoFeedbackController",
        "FullFeedbackOracle",
        "FrozenPolicyUniformSampling",
    ]
}
