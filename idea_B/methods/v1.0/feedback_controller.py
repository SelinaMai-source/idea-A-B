"""Limited-budget observation allocation.

The observability controller estimates skill-specific information from the
observed design Gramian ``W_T``, identifies weakly observed directions, and
allocates a limited audit/exploration budget to those directions. Allocation
is smoothed in skill space so the controller does not switch every round.

Generic controls (uniform, greedy, lazy, uncertainty, fixed schedule, none,
oracle) live here as allocation rules. They are not official reproductions of
Performative Bandit (ICML 2024) or Performative SGD (NeurIPS 2024).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from observability import eigenspectrum

EPS = 1e-12


def make_controller_state(n_skills: int) -> dict[str, Any]:
    return {
        "prev_skill_alloc": np.ones(int(n_skills), dtype=float) / float(n_skills),
        "rounds_since_update": 0,
    }


def _clip_mean(values: np.ndarray, mean_target: float) -> np.ndarray:
    """Clip to ``[0, 1]`` and rescale so the mean matches ``mean_target`` if possible."""
    target = float(np.clip(mean_target, 0.0, 1.0))
    if values.size == 0:
        return values
    if target <= 0.0:
        return np.zeros_like(values)
    if target >= 1.0:
        return np.ones_like(values)
    scaled = np.clip(values, 0.0, 1.0)
    current = float(np.mean(scaled))
    if current <= EPS:
        return np.full_like(scaled, target)
    scaled = scaled * (target / current)
    scaled = np.clip(scaled, 0.0, 1.0)
    current = float(np.mean(scaled))
    if current > target + 1e-12:
        overflow = scaled > target
        if np.any(~overflow):
            scaled = np.minimum(scaled, 1.0)
            # Binary search a multiplier in [0, 1] after clipping is messy;
            # a second proportional shrink of the excess is enough for smoke tests.
            if current > EPS:
                scaled = np.clip(scaled * (target / current), 0.0, 1.0)
    elif current < target - 1e-12:
        room = 1.0 - scaled
        need = target * scaled.size - float(np.sum(scaled))
        available = float(np.sum(room))
        if available > EPS:
            scaled = scaled + room * (need / available)
            scaled = np.clip(scaled, 0.0, 1.0)
    return scaled


def _skill_scores_from_gramian(
    gramian: np.ndarray,
    n_skills: int,
    metric: str,
    ridge_lambda: float,
) -> np.ndarray:
    dim = gramian.shape[0]
    regularized = gramian + float(ridge_lambda) * np.eye(dim)
    values, vectors = eigenspectrum(regularized)
    if metric == "lambda_min":
        # E-optimal: mass on the current weakest eigenvector, read out per axis.
        weakest = vectors[:, 0]
        scores = weakest**2 / (float(values[0]) + EPS)
        if scores.shape[0] != n_skills:
            scores = scores[:n_skills]
        return np.maximum(scores, 0.0)
    if metric == "trace_inverse":
        inv = np.linalg.pinv(regularized)
        return np.maximum(np.diag(inv)[:n_skills], 0.0)
    if metric == "logdet":
        # D-optimal proxy: 1 / (lambda_k + eps) along eigenvectors, mapped to axes.
        weights = 1.0 / (values + EPS)
        axis_scores = (vectors**2) @ weights
        return np.maximum(axis_scores[:n_skills], 0.0)
    if metric == "predictive_variance":
        inv = np.linalg.pinv(regularized)
        return np.maximum(np.diag(inv)[:n_skills], 0.0)
    raise ValueError(f"unknown information metric {metric!r}")


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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-query audit probabilities with mean approximately ``budget``.

    ``method`` selects a generic control or the observability controller.
    """
    features = np.asarray(phi, dtype=float)
    n, dim = features.shape
    skill_ids = np.asarray(skills, dtype=int).reshape(-1)
    n_sk = int(n_skills if n_skills is not None else dim)
    target = float(np.clip(budget, 0.0, 1.0))
    name = str(method)

    if name in {"FullFeedbackOracle", "oracle"}:
        return np.ones(n, dtype=float), state
    if name in {"NoFeedbackController", "none"}:
        return np.zeros(n, dtype=float), state
    if target <= 0.0:
        return np.zeros(n, dtype=float), state

    yhat = features @ np.asarray(theta, dtype=float).reshape(-1)

    if name in {"UniformExploration", "uniform", "StandardOnlineSGD"}:
        return np.full(n, target, dtype=float), state

    if name in {"FixedExplorationSchedule", "fixed_schedule"}:
        return np.full(n, target, dtype=float), state

    if name in {"GreedyDeployment", "greedy", "LazyDeployment", "lazy"}:
        k = max(1, int(round(target * n)))
        k = min(k, n)
        chosen = np.argsort(yhat)[-k:]
        probs = np.zeros(n, dtype=float)
        probs[chosen] = 1.0
        return probs, state

    if name in {"UncertaintyExploration", "uncertainty"}:
        regularized = gramian + float(ridge_lambda) * np.eye(dim)
        try:
            inv = np.linalg.inv(regularized)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(regularized)
        scores = np.einsum("ij,ji->i", features @ inv, features.T)
        probs = _clip_mean(scores, target)
        return probs, state

    if name in {"EqualSkillAllocation", "fixed_allocation"}:
        # Same expected audit mass on each skill that appears in the batch.
        present = np.unique(skill_ids)
        skill_p = np.zeros(n_sk, dtype=float)
        if present.size > 0:
            skill_p[present] = target
        probs = skill_p[np.clip(skill_ids, 0, n_sk - 1)]
        probs = _clip_mean(probs, target)
        return probs, state

    if name not in {"ObservabilityController", "controller"}:
        raise ValueError(f"unknown allocation method {name!r}")

    metric_scores = _skill_scores_from_gramian(
        np.asarray(gramian, dtype=float),
        n_sk,
        information_metric,
        ridge_lambda,
    )
    raw = metric_scores / (float(np.sum(metric_scores)) + EPS)
    uniform = np.ones(n_sk, dtype=float) / float(n_sk)
    mix = float(np.clip(mix_uniform, 0.0, 1.0))
    blended = (1.0 - mix) * raw + mix * uniform
    prev = np.asarray(state.get("prev_skill_alloc", uniform), dtype=float).reshape(-1)
    if prev.shape[0] != n_sk:
        prev = uniform
    alpha = float(np.clip(smoothing, 0.0, 1.0))
    skill_alloc = alpha * prev + (1.0 - alpha) * blended
    skill_alloc = skill_alloc / (float(np.sum(skill_alloc)) + EPS)
    state = dict(state)
    state["prev_skill_alloc"] = skill_alloc

    # Map skill weights to per-query probabilities, then match the budget mean.
    query_weight = skill_alloc[np.clip(skill_ids, 0, n_sk - 1)]
    # Queries aligned with weak directions get higher audit probability.
    probs = _clip_mean(query_weight, target)
    _ = rng  # allocation is deterministic given W; rng reserved for methods that need it
    return probs, state
