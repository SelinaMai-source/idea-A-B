"""Audit allocation rules for Idea B v1.1.

Three non-adaptive rules are distinct when skill frequencies are unequal:

- Uniform: ``q_i = b`` (equal per-sample probability).
- EqualSkill / B-v1.1a: equal expected audit *count* per skill,
  ``q_i ∝ 1/n_{skill(i)}``, clipped to ``[0,1]``, then matched to budget.
- Proportional-to-frequency: ``q_i ∝ n_{skill(i)}``, then matched to budget.

Classical design controls (not claimed as novel):

- B-v1.1b E-optimal: raise ``λ_min`` of the conditional design Gramian.
- D-optimal: raise ``logdet`` / predictive variance of the same Gramian.

B-v1.1c is the proposal: discounted information plus a one-step look-ahead
on expected worst-skill MSE. If it does not beat E-optimal on the frozen
protocol, that is a negative result.

B-v1.0 wraps the original v1.0 observability controller unchanged, including
its smoothing and ``mix_uniform``. That wrapper is a reference arm, not an
E-optimal control.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .observability import EPS, eigenspectrum

UNIFORM_NAMES = {
    "Uniform",
    "UniformExploration",
    "uniform",
    "StandardOnlineSGD",
    "FixedExplorationSchedule",
    "fixed_schedule",
}
EQUAL_SKILL_NAMES = {
    "B-v1.1a",
    "EqualSkill",
    "EqualSkillAllocation",
    "equal_expected_count",
    "equal_skill",
}
PROPORTIONAL_NAMES = {
    "Proportional",
    "ProportionalFrequency",
    "proportional",
    "proportional_to_frequency",
}
E_OPTIMAL_NAMES = {"B-v1.1b", "E-optimal", "E_optimal", "e_optimal"}
D_OPTIMAL_NAMES = {"D-optimal", "D_optimal", "d_optimal"}
V11C_NAMES = {
    "B-v1.1c",
    "v1.1c",
    "InformationDiscountWorstSkill",
    "information_discount_worst_skill",
}
V10_NAMES = {
    "B-v1.0",
    "ObservabilityController",
    "ObservabilityController_v1.0",
    "controller",
}
GREEDY_NAMES = {"Greedy", "GreedyDeployment", "greedy"}
LAZY_NAMES = {"Lazy", "LazyDeployment", "lazy"}
ORACLE_NAMES = {"Oracle", "FullFeedbackOracle", "oracle"}
NONE_NAMES = {"NoFeedbackController", "none"}
UNCERTAINTY_NAMES = {"UncertaintyExploration", "uncertainty"}


def make_controller_state(n_skills: int) -> dict[str, Any]:
    return {
        "prev_skill_alloc": np.ones(int(n_skills), dtype=float) / float(n_skills),
        "rounds_since_update": 0,
    }


def match_mean_budget(weights: Any, budget: float) -> np.ndarray:
    """Clip ``α w`` to ``[0,1]`` and choose ``α`` so the mean matches ``budget``.

    If the box constraints make the target infeasible, the closest feasible
    mean is returned (all zeros or all ones, or ones on the support of ``w``).
    """
    values = np.asarray(weights, dtype=float).reshape(-1)
    n = int(values.shape[0])
    target = float(np.clip(budget, 0.0, 1.0))
    if n == 0:
        return values
    if target <= 0.0:
        return np.zeros(n, dtype=float)
    if target >= 1.0:
        return np.ones(n, dtype=float)
    raw = np.maximum(values, 0.0)
    if float(np.max(raw)) <= EPS:
        return np.full(n, target, dtype=float)

    def mean_at(scale: float) -> float:
        return float(np.mean(np.clip(scale * raw, 0.0, 1.0)))

    lo = 0.0
    hi = max(target / (float(np.mean(raw)) + EPS), 1.0)
    for _ in range(80):
        if mean_at(hi) >= target - 1e-15:
            break
        hi *= 2.0
        if hi > 1e16:
            break
    if mean_at(hi) < target - 1e-12:
        return np.clip(hi * raw, 0.0, 1.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return np.clip(hi * raw, 0.0, 1.0)


def allocate_uniform(n: int, budget: float) -> np.ndarray:
    """Per-sample audit probability ``q_i = b``."""
    target = float(np.clip(budget, 0.0, 1.0))
    size = int(n)
    if size < 0:
        raise ValueError("n must be >= 0")
    if size == 0:
        return np.zeros(0, dtype=float)
    return np.full(size, target, dtype=float)


def _skill_counts(skills: Any, n_skills: int) -> np.ndarray:
    skill_ids = np.asarray(skills, dtype=int).reshape(-1)
    if np.any(skill_ids < 0) or np.any(skill_ids >= int(n_skills)):
        raise ValueError("skill ids must lie in 0..n_skills-1")
    return np.bincount(skill_ids, minlength=int(n_skills)).astype(float)


def allocate_equal_expected_count(
    skills: Any,
    budget: float,
    n_skills: int,
) -> np.ndarray:
    """Equal expected audit count per skill that appears in the batch.

    Raw weights are ``1/n_{skill(i)}``, clipped to ``[0,1]``, then scaled so
    the mean audit probability matches the budget. Rare skills therefore
    receive a *higher* per-sample probability than common skills when
    frequencies are unequal.
    """
    skill_ids = np.asarray(skills, dtype=int).reshape(-1)
    n = int(skill_ids.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    counts = _skill_counts(skill_ids, n_skills)
    raw = np.zeros(n, dtype=float)
    present = counts[skill_ids]
    positive = present > 0.0
    raw[positive] = 1.0 / present[positive]
    return match_mean_budget(raw, budget)


def allocate_proportional_frequency(
    skills: Any,
    budget: float,
    n_skills: int,
) -> np.ndarray:
    """Per-sample probability proportional to the batch frequency of the skill.

    Raw weights are ``n_{skill(i)}``. Common skills receive a *higher*
    per-sample probability than rare skills. This is a third rule, distinct
    from Uniform (constant ``q_i``) and from equal expected counts.
    """
    skill_ids = np.asarray(skills, dtype=int).reshape(-1)
    n = int(skill_ids.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    counts = _skill_counts(skill_ids, n_skills)
    raw = counts[skill_ids]
    return match_mean_budget(raw, budget)


def expected_skill_audit_counts(q: Any, skills: Any, n_skills: int) -> np.ndarray:
    probs = np.asarray(q, dtype=float).reshape(-1)
    skill_ids = np.asarray(skills, dtype=int).reshape(-1)
    if probs.shape[0] != skill_ids.shape[0]:
        raise ValueError("q and skills must have the same length")
    counts = np.zeros(int(n_skills), dtype=float)
    for k in range(int(n_skills)):
        mask = skill_ids == k
        if np.any(mask):
            counts[k] = float(np.sum(probs[mask]))
    return counts


def _frank_wolfe_optimal(
    phi: np.ndarray,
    gramian: np.ndarray,
    budget: float,
    ridge_lambda: float,
    n_iter: int,
    criterion: str,
) -> np.ndarray:
    """Classical sequential E- or D-optimal allocation on the current batch.

    No ``mix_uniform`` and no skill-space smoothing: those belong to the v1.0
    controller, not to this control.
    """
    features = np.asarray(phi, dtype=float)
    n, dim = features.shape
    target = float(np.clip(budget, 0.0, 1.0))
    if n == 0:
        return np.zeros(0, dtype=float)
    q = np.full(n, target, dtype=float)
    base = np.asarray(gramian, dtype=float) + float(ridge_lambda) * np.eye(dim)
    steps = max(1, int(n_iter))
    for iteration in range(steps):
        gram = base + features.T @ (q[:, None] * features)
        regularized = 0.5 * (gram + gram.T)
        if criterion == "e":
            _values, vectors = eigenspectrum(regularized)
            vmin = vectors[:, 0]
            scores = (features @ vmin) ** 2
        elif criterion == "d":
            try:
                precision = np.linalg.inv(regularized)
            except np.linalg.LinAlgError:
                precision = np.linalg.pinv(regularized)
            scores = np.einsum("ij,ji->i", features @ precision, features.T)
        else:
            raise ValueError(f"unknown optimality criterion {criterion!r}")
        vertex = match_mean_budget(np.maximum(scores, 0.0), target)
        step = 2.0 / (float(iteration) + 2.0)
        q = (1.0 - step) * q + step * vertex
    return np.clip(q, 0.0, 1.0)


def allocate_e_optimal(
    phi: Any,
    gramian: Any,
    budget: float,
    *,
    ridge_lambda: float = 0.5,
    n_iter: int = 12,
) -> np.ndarray:
    """One-step E-optimal: raise ``λ_min`` of ``W + Σ q_i φ_i φ_i^T``."""
    return _frank_wolfe_optimal(
        np.asarray(phi, dtype=float),
        np.asarray(gramian, dtype=float),
        budget,
        ridge_lambda,
        n_iter,
        "e",
    )


def allocate_d_optimal(
    phi: Any,
    gramian: Any,
    budget: float,
    *,
    ridge_lambda: float = 0.5,
    n_iter: int = 12,
) -> np.ndarray:
    """One-step D-optimal: raise ``logdet`` via predictive-variance scores."""
    return _frank_wolfe_optimal(
        np.asarray(phi, dtype=float),
        np.asarray(gramian, dtype=float),
        budget,
        ridge_lambda,
        n_iter,
        "d",
    )


def _observation_probability(q_nat: np.ndarray, q_audit: np.ndarray) -> np.ndarray:
    natural = np.clip(q_nat, 0.0, 1.0)
    audit = np.clip(q_audit, 0.0, 1.0)
    return 1.0 - (1.0 - natural) * (1.0 - audit)


def _worst_skill_mse_proxy(
    gramian: np.ndarray,
    phi: np.ndarray,
    skills: np.ndarray,
    ridge_lambda: float,
    noise_var: float,
    n_skills: int,
) -> float:
    dim = gramian.shape[0]
    regularized = gramian + float(ridge_lambda) * np.eye(dim)
    try:
        cov = np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(regularized)
    pred_var = np.einsum("ij,ji->i", phi @ cov, phi.T)
    worst = -np.inf
    for k in range(int(n_skills)):
        mask = skills == k
        if np.any(mask):
            skill_var = float(np.mean(pred_var[mask]))
        else:
            skill_var = float(cov[k, k]) if k < dim else float("inf")
        worst = max(worst, float(noise_var) * (1.0 + skill_var))
    if not np.isfinite(worst):
        return float("inf")
    return float(worst)


def allocate_v11c(
    phi: Any,
    skills: Any,
    gramian_discounted: Any,
    q_nat: Any,
    budget: float,
    n_skills: int,
    *,
    ridge_lambda: float = 0.5,
    information_discount: float = 0.90,
    noise_std: float = 0.15,
    e_optimal_iterations: int = 12,
) -> np.ndarray:
    """Discounted-information allocation targeting one-step worst-skill MSE.

    Candidate designs: uniform, E-optimal on the discounted Gramian, equal
    expected counts, and putting the whole budget on each present skill.
    The chosen design is the one with the smallest expected worst-skill MSE
    after one expected information increment. Ties prefer E-optimal.
    """
    features = np.asarray(phi, dtype=float)
    skill_ids = np.asarray(skills, dtype=int).reshape(-1)
    n = features.shape[0]
    target = float(np.clip(budget, 0.0, 1.0))
    if n == 0:
        return np.zeros(0, dtype=float)
    discounted = np.asarray(gramian_discounted, dtype=float)
    natural = np.asarray(q_nat, dtype=float).reshape(-1)
    if natural.shape[0] != n:
        natural = np.zeros(n, dtype=float)
    rho = float(information_discount)
    noise_var = float(noise_std) ** 2
    ridge = float(ridge_lambda)

    candidates: list[tuple[str, np.ndarray, int]] = [
        ("e_optimal", allocate_e_optimal(
            features, discounted, target, ridge_lambda=ridge, n_iter=e_optimal_iterations
        ), 0),
        ("uniform", allocate_uniform(n, target), 1),
        ("equal_count", allocate_equal_expected_count(skill_ids, target, n_skills), 2),
    ]
    for k in range(int(n_skills)):
        mask = (skill_ids == k).astype(float)
        if float(np.sum(mask)) <= 0.0:
            continue
        candidates.append((f"skill_{k}", match_mean_budget(mask, target), 10 + k))

    def expected_gramian(q_audit: np.ndarray) -> np.ndarray:
        q_obs = _observation_probability(natural, q_audit)
        increment = features.T @ (q_obs[:, None] * features)
        return rho * discounted + increment

    best_q = candidates[0][1]
    best_mse = np.inf
    best_rank = 10**9
    for _name, q_audit, rank in candidates:
        mse = _worst_skill_mse_proxy(
            expected_gramian(q_audit),
            features,
            skill_ids,
            ridge,
            noise_var,
            n_skills,
        )
        if (mse < best_mse - 1e-15) or (abs(mse - best_mse) <= 1e-15 and rank < best_rank):
            best_mse = mse
            best_q = q_audit
            best_rank = rank
    return np.clip(best_q, 0.0, 1.0)


def _allocate_greedy(phi: np.ndarray, theta: np.ndarray, budget: float) -> np.ndarray:
    n = phi.shape[0]
    target = float(np.clip(budget, 0.0, 1.0))
    if n == 0:
        return np.zeros(0, dtype=float)
    if target <= 0.0:
        return np.zeros(n, dtype=float)
    yhat = phi @ np.asarray(theta, dtype=float).reshape(-1)
    k = max(1, int(round(target * n)))
    k = min(k, n)
    chosen = np.argsort(yhat)[-k:]
    probs = np.zeros(n, dtype=float)
    probs[chosen] = 1.0
    return probs


def _allocate_uncertainty(
    phi: np.ndarray,
    gramian: np.ndarray,
    budget: float,
    ridge_lambda: float,
) -> np.ndarray:
    n, dim = phi.shape
    regularized = np.asarray(gramian, dtype=float) + float(ridge_lambda) * np.eye(dim)
    try:
        precision = np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(regularized)
    scores = np.einsum("ij,ji->i", phi @ precision, phi.T)
    return match_mean_budget(scores, budget)


def _allocate_v10(
    phi: np.ndarray,
    skills: np.ndarray,
    theta: np.ndarray,
    gramian: np.ndarray,
    budget: float,
    state: dict[str, Any],
    rng: np.random.Generator,
    *,
    smoothing: float,
    mix_uniform: float,
    ridge_lambda: float,
    information_metric: str,
    n_skills: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Call the original v1.0 controller without modifying it."""
    from idea_B.methods import allocate_audit as allocate_audit_v10

    return allocate_audit_v10(
        "ObservabilityController",
        phi,
        skills,
        theta,
        gramian,
        budget,
        state,
        rng,
        smoothing=smoothing,
        mix_uniform=mix_uniform,
        ridge_lambda=ridge_lambda,
        information_metric=information_metric,
        n_skills=n_skills,
    )


def allocate_audit(
    method: str,
    phi: np.ndarray,
    skills: np.ndarray,
    theta: np.ndarray,
    gramian: np.ndarray,
    budget: float,
    state: dict[str, Any],
    rng: np.random.Generator,
    *,
    smoothing: float = 0.6,
    mix_uniform: float = 0.1,
    ridge_lambda: float = 0.5,
    information_metric: str = "lambda_min",
    n_skills: int | None = None,
    q_nat: np.ndarray | None = None,
    gramian_discounted: np.ndarray | None = None,
    information_discount: float = 0.90,
    noise_std: float = 0.15,
    e_optimal_iterations: int = 12,
    d_optimal_iterations: int = 12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-query audit probabilities with mean approximately ``budget``."""
    features = np.asarray(phi, dtype=float)
    n, dim = features.shape
    skill_ids = np.asarray(skills, dtype=int).reshape(-1)
    n_sk = int(n_skills if n_skills is not None else dim)
    target = float(np.clip(budget, 0.0, 1.0))
    name = str(method)
    _ = rng

    if name in ORACLE_NAMES:
        return np.ones(n, dtype=float), state
    if name in NONE_NAMES:
        return np.zeros(n, dtype=float), state
    if target <= 0.0:
        return np.zeros(n, dtype=float), state

    if name in UNIFORM_NAMES:
        return allocate_uniform(n, target), state
    if name in EQUAL_SKILL_NAMES:
        return allocate_equal_expected_count(skill_ids, target, n_sk), state
    if name in PROPORTIONAL_NAMES:
        return allocate_proportional_frequency(skill_ids, target, n_sk), state
    if name in GREEDY_NAMES or name in LAZY_NAMES:
        return _allocate_greedy(features, theta, target), state
    if name in UNCERTAINTY_NAMES:
        return _allocate_uncertainty(features, gramian, target, ridge_lambda), state
    if name in E_OPTIMAL_NAMES:
        return (
            allocate_e_optimal(
                features,
                gramian,
                target,
                ridge_lambda=ridge_lambda,
                n_iter=e_optimal_iterations,
            ),
            state,
        )
    if name in D_OPTIMAL_NAMES:
        return (
            allocate_d_optimal(
                features,
                gramian,
                target,
                ridge_lambda=ridge_lambda,
                n_iter=d_optimal_iterations,
            ),
            state,
        )
    if name in V11C_NAMES:
        discounted = gramian if gramian_discounted is None else gramian_discounted
        natural = np.zeros(n, dtype=float) if q_nat is None else np.asarray(q_nat, dtype=float)
        return (
            allocate_v11c(
                features,
                skill_ids,
                discounted,
                natural,
                target,
                n_sk,
                ridge_lambda=ridge_lambda,
                information_discount=information_discount,
                noise_std=noise_std,
                e_optimal_iterations=e_optimal_iterations,
            ),
            state,
        )
    if name in V10_NAMES:
        return _allocate_v10(
            features,
            skill_ids,
            theta,
            gramian,
            target,
            state,
            rng,
            smoothing=smoothing,
            mix_uniform=mix_uniform,
            ridge_lambda=ridge_lambda,
            information_metric=information_metric,
            n_skills=n_sk,
        )
    raise ValueError(f"unknown allocation method {name!r}")
