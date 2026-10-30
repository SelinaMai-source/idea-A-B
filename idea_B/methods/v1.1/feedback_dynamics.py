"""Closed-loop observation composition, primary metrics, recovery, detection.

Primary metrics in v1.1 are excess MSE versus the irreducible noise floor,
worst-skill MSE, population-weighted MSE, parameter error, cumulative
regret, and (Env3) recovery time. ``1/(1+MSE)`` is recorded only as a
secondary descriptor. A larger Gramian eigenvalue is not success.

Collapse labeling copies the v1.0 rule: a fixed policy with fixed sampling
is never endogenous collapse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NOISE_STD = 0.15
NOISE_FLOOR = float(NOISE_STD**2)  # 0.0225
RECOVERY_CONSECUTIVE = 4
RECOVERY_PRE_WINDOW = 8
DETECTION_CUSUM_THRESHOLD = 3.0
DETECTION_CUSUM_ALLOWANCE = 0.05


def observation_probability(q_nat: Any, q_audit: Any) -> np.ndarray:
    """``P(S=1) = 1 - (1-q_nat)(1-q_audit)`` with probabilities clipped to ``[0,1]``."""
    natural = np.clip(np.asarray(q_nat, dtype=float).reshape(-1), 0.0, 1.0)
    audit = np.clip(np.asarray(q_audit, dtype=float).reshape(-1), 0.0, 1.0)
    if natural.shape[0] != audit.shape[0]:
        raise ValueError("q_nat and q_audit must have the same length")
    return 1.0 - (1.0 - natural) * (1.0 - audit)


def batch_mse(
    phi: Any,
    theta: Any,
    y: Any,
    mask: Any | None,
) -> float:
    features = np.asarray(phi, dtype=float)
    param = np.asarray(theta, dtype=float).reshape(-1)
    outcomes = np.asarray(y, dtype=float).reshape(-1)
    if mask is None:
        selected = np.ones(features.shape[0], dtype=bool)
    else:
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        if selected.shape[0] != features.shape[0]:
            raise ValueError("mask length must match batch")
    if not np.any(selected):
        return float("nan")
    residual = features[selected] @ param - outcomes[selected]
    return float(np.mean(residual**2))


def excess_mse(mse: float, noise_floor: float = NOISE_FLOOR) -> float:
    """Signed excess over the irreducible noise floor. Regret clips this at 0."""
    if not np.isfinite(mse):
        return float("nan")
    return float(mse - float(noise_floor))


def regret_increment(mse: float, noise_floor: float = NOISE_FLOOR) -> float:
    extra = excess_mse(mse, noise_floor=noise_floor)
    if not np.isfinite(extra):
        return float("nan")
    return float(max(extra, 0.0))


def secondary_performance(mse: float) -> float:
    """``1/(1+MSE)``. Secondary descriptor only; not a PROMISING criterion."""
    if not np.isfinite(mse):
        return float("nan")
    return float(1.0 / (1.0 + max(mse, 0.0)))


def worst_skill_mse(skill_mses: Any) -> float:
    values = np.asarray(skill_mses, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def population_weighted_mse(skill_mses: Any, weights: Any) -> float:
    values = np.asarray(skill_mses, dtype=float).reshape(-1)
    mix = np.asarray(weights, dtype=float).reshape(-1)
    if values.shape[0] != mix.shape[0]:
        raise ValueError("skill_mses and weights must have the same length")
    finite = np.isfinite(values) & np.isfinite(mix) & (mix >= 0.0)
    if not np.any(finite):
        return float("nan")
    mass = float(np.sum(mix[finite]))
    if mass <= 0.0:
        return float("nan")
    return float(np.sum(values[finite] * mix[finite]) / mass)


def parameter_error(theta: Any, theta_star: Any) -> float:
    param = np.asarray(theta, dtype=float).reshape(-1)
    truth = np.asarray(theta_star, dtype=float).reshape(-1)
    if param.shape[0] != truth.shape[0]:
        raise ValueError("theta and theta_star must have the same length")
    return float(np.linalg.norm(param - truth))


def per_skill_parameter_error(theta: Any, theta_star: Any) -> np.ndarray:
    param = np.asarray(theta, dtype=float).reshape(-1)
    truth = np.asarray(theta_star, dtype=float).reshape(-1)
    if param.shape[0] != truth.shape[0]:
        raise ValueError("theta and theta_star must have the same length")
    return np.abs(param - truth)


def late_window_mean(values: Any, horizon: int) -> float:
    series = np.asarray(values, dtype=float).reshape(-1)
    width = max(1, int(horizon) // 4)
    window = series[-width:]
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def recovery_time(
    worst_skill_mse_series: Any,
    change_round: int,
    *,
    consecutive: int = RECOVERY_CONSECUTIVE,
    pre_window: int = RECOVERY_PRE_WINDOW,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Env3 recovery time as frozen in the evaluation protocol.

    After the change round, recovery is the first index at which worst-skill
    MSE has been no higher than the pre-change median for ``consecutive``
    rounds. The pre-change median uses the last ``pre_window`` rounds before
    the change. If the condition is never met, the value is ``horizon``
    (right-censored). Indices are 0-based; ``horizon`` is the run length.
    """
    series = np.asarray(worst_skill_mse_series, dtype=float).reshape(-1)
    change = int(change_round)
    consec = int(consecutive)
    pre_n = int(pre_window)
    total = int(horizon if horizon is not None else series.shape[0])
    if consec < 1:
        raise ValueError("consecutive must be >= 1")
    if change < 0:
        raise ValueError("change_round out of range")
    if change >= series.shape[0]:
        return {
            "recovery_time": total,
            "recovery_censored": True,
            "pre_change_median": float("nan"),
            "reason": "change point not reached in this trajectory",
        }

    pre = series[max(0, change - pre_n) : change]
    finite_pre = pre[np.isfinite(pre)]
    if finite_pre.size == 0:
        return {
            "recovery_time": total,
            "recovery_censored": True,
            "pre_change_median": float("nan"),
            "reason": "no finite pre-change worst-skill MSE",
        }
    threshold = float(np.median(finite_pre))
    post = series[change:]
    for offset in range(0, post.shape[0] - consec + 1):
        window = post[offset : offset + consec]
        if np.all(np.isfinite(window) & (window <= threshold)):
            recovered_at = change + offset + consec - 1
            return {
                "recovery_time": int(recovered_at),
                "recovery_censored": False,
                "pre_change_median": threshold,
                "reason": "worst-skill MSE stayed at or below the pre-change median",
            }
    return {
        "recovery_time": total,
        "recovery_censored": True,
        "pre_change_median": threshold,
        "reason": "right-censored at horizon",
    }


def cusum_detection_delay(
    observed_mse: Any,
    change_round: int,
    *,
    threshold: float = DETECTION_CUSUM_THRESHOLD,
    allowance: float = DETECTION_CUSUM_ALLOWANCE,
    noise_floor: float = NOISE_FLOOR,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Pre-specified CUSUM on observed MSE. Not a substitute for recovery time.

    Statistic: ``S_t = max(0, S_{t-1} + (mse_t - noise_floor - allowance))``.
    Detection is the first ``t`` with ``S_t > threshold``. Delay is
    ``t - change_round`` (negative if a false alarm fires before the change).
    If nothing fires, detection is undefined.
    """
    series = np.asarray(observed_mse, dtype=float).reshape(-1)
    change = int(change_round)
    total = int(horizon if horizon is not None else series.shape[0])
    score = 0.0
    detect_at: int | None = None
    for t, mse in enumerate(series):
        if not np.isfinite(mse):
            continue
        score = max(0.0, score + (float(mse) - float(noise_floor) - float(allowance)))
        if score > float(threshold):
            detect_at = int(t)
            break
    if detect_at is None:
        return {
            "detected": False,
            "detect_round": None,
            "detection_delay": int(total),
            "detection_censored": True,
            "cusum_final": float(score),
            "horizon": total,
        }
    return {
        "detected": True,
        "detect_round": detect_at,
        "detection_delay": int(detect_at - change),
        "detection_censored": False,
        "cusum_final": float(score),
        "horizon": total,
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
