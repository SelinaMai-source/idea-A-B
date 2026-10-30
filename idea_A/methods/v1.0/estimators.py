"""Idea A v1.0 estimators: IPW, Hájek/SNIPS, AIPW on the known audit propensity.

All three call the shared functions in ``shared.metrics.estimators``. Hidden
labels must already be NaN: ``ipw_mean`` / ``snips_mean`` / ``dr_mean`` do not
read ``Y`` where ``A=0``. The outcome model for AIPW is fit only on audited
units.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.metrics.estimators import dr_mean, ipw_mean, naive_ips_mean, observed_only_mean, snips_mean


def design_matrix(x: np.ndarray, pi: np.ndarray, stratum: np.ndarray, n_strata: int) -> np.ndarray:
    z = np.asarray(stratum, dtype=int)
    dummies = np.eye(int(n_strata), dtype=float)[z]
    return np.column_stack(
        [
            np.ones(z.shape[0], dtype=float),
            np.asarray(x, dtype=float),
            np.asarray(pi, dtype=float).reshape(-1, 1),
            dummies[:, 1:],
        ]
    )


def fit_mu_hat(
    y_visible: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    pi: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    ridge_lambda: float = 1e-3,
) -> np.ndarray:
    """Ridge linear outcome model fit on ``A=1`` units only."""
    y = np.asarray(y_visible, dtype=float)
    audited = np.asarray(a).astype(bool, copy=False)
    phi = design_matrix(x, pi, stratum, n_strata)
    n, p = phi.shape
    y_a = y[audited]
    finite = audited.copy()
    finite[audited] = np.isfinite(y_a)
    if int(np.sum(finite)) < max(p, 2):
        if int(np.sum(finite)) == 0:
            return np.zeros(n, dtype=float)
        return np.full(n, float(np.mean(y[finite])), dtype=float)
    phi_a = phi[finite]
    y_a = y[finite]
    xtx = phi_a.T @ phi_a + float(ridge_lambda) * np.eye(p)
    beta = np.linalg.solve(xtx, phi_a.T @ y_a)
    return phi @ beta


def _scalar_or_nan(value: float) -> float:
    return float(value)


def estimate_population_mean(
    y_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    estimator: str,
    mu_hat: np.ndarray | None = None,
    clip: float | None = None,
) -> float:
    name = str(estimator)
    if name == "ipw":
        return _scalar_or_nan(ipw_mean(y_visible, a, q, clip=clip))
    if name in {"snips", "hajek", "hajek_snips"}:
        return _scalar_or_nan(snips_mean(y_visible, a, q, clip=clip))
    if name in {"dr", "aipw", "doubly_robust"}:
        if mu_hat is None:
            raise ValueError("mu_hat is required for DR/AIPW")
        return _scalar_or_nan(dr_mean(y_visible, a, q, mu_hat, clip=clip))
    if name == "observed_only":
        return _scalar_or_nan(observed_only_mean(y_visible, a))
    raise ValueError(f"unknown estimator {name!r}")


def stratum_means(
    y_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    estimator: str,
    mu_hat: np.ndarray | None = None,
    clip: float | None = None,
) -> np.ndarray:
    z = np.asarray(stratum, dtype=int)
    estimates = np.full(int(n_strata), np.nan, dtype=float)
    for h in range(int(n_strata)):
        mask = z == h
        if not np.any(mask):
            continue
        a_h = np.asarray(a)[mask]
        if estimator == "observed_only":
            if not np.any(np.asarray(a_h).astype(bool)):
                estimates[h] = float("nan")
                continue
        elif not np.any(np.asarray(a_h).astype(bool)):
            estimates[h] = float("nan")
            continue
        mu_h = None if mu_hat is None else np.asarray(mu_hat)[mask]
        estimates[h] = estimate_population_mean(
            np.asarray(y_visible)[mask],
            a_h,
            np.asarray(q)[mask],
            estimator,
            mu_hat=mu_h,
            clip=clip,
        )
    return estimates


def estimate_estimands(
    y_visible: np.ndarray,
    pref_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    stratum: np.ndarray,
    target_weights: np.ndarray,
    n_strata: int,
    estimator: str,
    mu_hat: np.ndarray | None = None,
    clip: float | None = None,
) -> dict[str, Any]:
    """Estimates of the four v1.0 estimands using the audit indicator ``a``."""
    pop = estimate_population_mean(y_visible, a, q, estimator, mu_hat=mu_hat, clip=clip)
    pref = estimate_population_mean(pref_visible, a, q, estimator, mu_hat=mu_hat, clip=clip)
    weighted = np.asarray(target_weights, dtype=float) * np.asarray(y_visible, dtype=float)
    policy = estimate_population_mean(weighted, a, q, estimator, mu_hat=mu_hat, clip=clip)
    cond = stratum_means(
        y_visible, a, q, stratum, n_strata, estimator, mu_hat=mu_hat, clip=clip
    )
    return {
        "population_mean_reward": pop,
        "preference_rate": pref,
        "policy_value_target_distribution": policy,
        "context_conditioned_reward": cond,
    }


def observed_only_estimands(
    y_visible: np.ndarray,
    pref_visible: np.ndarray,
    observed_mask: np.ndarray,
    stratum: np.ndarray,
    target_weights: np.ndarray,
    n_strata: int,
) -> dict[str, Any]:
    mask = np.asarray(observed_mask)
    pop = observed_only_mean(y_visible, mask)
    pref = observed_only_mean(pref_visible, mask)
    weighted = np.asarray(target_weights, dtype=float) * np.asarray(y_visible, dtype=float)
    policy = observed_only_mean(weighted, mask)
    cond = stratum_means(
        y_visible, mask, np.ones(mask.shape[0]), stratum, n_strata, "observed_only"
    )
    return {
        "population_mean_reward": pop,
        "preference_rate": pref,
        "policy_value_target_distribution": policy,
        "context_conditioned_reward": cond,
    }


def naive_ips_estimands(
    y_visible: np.ndarray,
    pref_visible: np.ndarray,
    observed_mask: np.ndarray,
    p_hat: np.ndarray,
    stratum: np.ndarray,
    target_weights: np.ndarray,
    n_strata: int,
    clip: float | None = None,
) -> dict[str, Any]:
    """Horvitz–Thompson on the natural mask with a claimed natural propensity.

    This is not audit IPW. Under MNAR the claimed ``p_hat = P(S=1|stratum)``
    does not make ``S ⊥ Y | stratum``, so the estimator is not identifying.
    """
    pop = naive_ips_mean(y_visible, observed_mask, p_hat, clip=clip)
    pref = naive_ips_mean(pref_visible, observed_mask, p_hat, clip=clip)
    weighted = np.asarray(target_weights, dtype=float) * np.asarray(y_visible, dtype=float)
    policy = naive_ips_mean(weighted, observed_mask, p_hat, clip=clip)
    z = np.asarray(stratum, dtype=int)
    cond = np.full(int(n_strata), np.nan, dtype=float)
    for h in range(int(n_strata)):
        m = z == h
        if not np.any(m) or not np.any(np.asarray(observed_mask)[m].astype(bool)):
            continue
        cond[h] = naive_ips_mean(np.asarray(y_visible)[m], np.asarray(observed_mask)[m], np.asarray(p_hat)[m], clip=clip)
    return {
        "population_mean_reward": pop,
        "preference_rate": pref,
        "policy_value_target_distribution": policy,
        "context_conditioned_reward": cond,
    }
