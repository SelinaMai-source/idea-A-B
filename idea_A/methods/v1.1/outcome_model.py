"""Estimand-specific outcome models for Idea A v1.1.

- ``mu_Y(x) = E[Y|X]``: ridge linear regression on audited units only.
- ``mu_P(x) = P(Y>0|X)``: a separate logistic regression. The continuous
  ``mu_Y`` is never used as a probability.
- ``mu_V(x) = E[w(X)Y|X]``: when ``w`` is a known stratum density ratio,
  ``mu_V(x) = w(x) mu_Y(x)``. Do not plug ``mu_Y`` into the DR residual of
  the weighted outcome ``wY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pathlib import Path

import yaml

from shared.metrics.estimators import dr_mean

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_v11_config() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def _defaults() -> dict[str, Any]:
    cfg = load_v11_config()
    return {
        "ridge_lambda": float(cfg.get("ridge_lambda", 1e-3)),
        "logistic_max_iter": int(cfg.get("logistic_max_iter", 40)),
        "logistic_tol": float(cfg.get("logistic_tol", 1e-8)),
    }


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def design_matrix(x: np.ndarray, pi: np.ndarray, stratum: np.ndarray, n_strata: int) -> np.ndarray:
    """Intercept, covariates, action, and stratum dummies (drop the first)."""
    z = np.asarray(stratum, dtype=int)
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    dummies = np.eye(int(n_strata), dtype=float)[z]
    return np.column_stack(
        [
            np.ones(z.shape[0], dtype=float),
            x_arr,
            np.asarray(pi, dtype=float).reshape(-1, 1),
            dummies[:, 1:],
        ]
    )


def _audited_finite(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    audited = np.asarray(a).astype(bool, copy=False)
    y_arr = np.asarray(y, dtype=float)
    return audited & np.isfinite(y_arr)


def fit_ridge_beta(
    phi: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray | None, float]:
    """Return ``(beta, constant)``. ``beta is None`` means predict ``constant``."""
    phi_a = np.asarray(phi, dtype=float)
    y_a = np.asarray(y, dtype=float)
    n, p = phi_a.shape
    if n == 0:
        return None, 0.0
    if n < max(p, 2):
        return None, float(np.mean(y_a))
    xtx = phi_a.T @ phi_a + float(ridge_lambda) * np.eye(p)
    try:
        beta = np.linalg.solve(xtx, phi_a.T @ y_a)
    except np.linalg.LinAlgError:
        return None, float(np.mean(y_a))
    return np.asarray(beta, dtype=float), 0.0


def predict_ridge(phi: np.ndarray, beta: np.ndarray | None, constant: float) -> np.ndarray:
    phi_a = np.asarray(phi, dtype=float)
    if beta is None:
        return np.full(phi_a.shape[0], float(constant), dtype=float)
    return phi_a @ np.asarray(beta, dtype=float)


def fit_logistic_beta(
    phi: np.ndarray,
    y_bin: np.ndarray,
    ridge_lambda: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray | None, float]:
    """Ridge logistic IRLS. ``y_bin`` is in ``{0,1}``."""
    phi_a = np.asarray(phi, dtype=float)
    y_a = np.asarray(y_bin, dtype=float)
    n, p = phi_a.shape
    if n == 0:
        return None, 0.5
    rate = float(np.clip(np.mean(y_a), 1e-4, 1.0 - 1e-4))
    if n < max(p, 2) or np.unique(y_a).size < 2:
        return None, rate
    beta = np.zeros(p, dtype=float)
    for _ in range(int(max_iter)):
        eta = np.clip(phi_a @ beta, -30.0, 30.0)
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p_hat * (1.0 - p_hat), 1e-8)
        z = eta + (y_a - p_hat) / w
        xtw = phi_a.T * w
        hess = xtw @ phi_a + float(ridge_lambda) * np.eye(p)
        try:
            beta_new = np.linalg.solve(hess, xtw @ z)
        except np.linalg.LinAlgError:
            return None, rate
        if float(np.max(np.abs(beta_new - beta))) < float(tol):
            beta = beta_new
            break
        beta = beta_new
    return np.asarray(beta, dtype=float), rate


def predict_logistic(phi: np.ndarray, beta: np.ndarray | None, rate: float) -> np.ndarray:
    phi_a = np.asarray(phi, dtype=float)
    if beta is None:
        return np.full(phi_a.shape[0], float(rate), dtype=float)
    return sigmoid(phi_a @ np.asarray(beta, dtype=float))


@dataclass
class FittedRidge:
    beta: np.ndarray | None
    constant: float
    n_strata: int

    def predict(self, x: np.ndarray, pi: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        phi = design_matrix(x, pi, stratum, self.n_strata)
        return predict_ridge(phi, self.beta, self.constant)


@dataclass
class FittedLogistic:
    beta: np.ndarray | None
    rate: float
    n_strata: int

    def predict(self, x: np.ndarray, pi: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        phi = design_matrix(x, pi, stratum, self.n_strata)
        return predict_logistic(phi, self.beta, self.rate)


def fit_mu_y(
    y_visible: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    pi: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    ridge_lambda: float | None = None,
) -> FittedRidge:
    """Ridge ``E[Y|X]`` on ``A=1`` units only. Natural ``S=1, A=0`` labels are ignored."""
    cfg = _defaults()
    lam = float(cfg["ridge_lambda"] if ridge_lambda is None else ridge_lambda)
    mask = _audited_finite(y_visible, a)
    phi = design_matrix(x, pi, stratum, n_strata)
    beta, constant = fit_ridge_beta(phi[mask], np.asarray(y_visible, dtype=float)[mask], lam)
    return FittedRidge(beta=beta, constant=constant, n_strata=int(n_strata))


def fit_mu_p(
    y_visible: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    pi: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    ridge_lambda: float | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
) -> FittedLogistic:
    """Separate logistic ``P(Y>0|X)`` on audited units. Does not reuse ``mu_Y``."""
    cfg = _defaults()
    lam = float(cfg["ridge_lambda"] if ridge_lambda is None else ridge_lambda)
    iters = int(cfg["logistic_max_iter"] if max_iter is None else max_iter)
    tol_v = float(cfg["logistic_tol"] if tol is None else tol)
    mask = _audited_finite(y_visible, a)
    y = np.asarray(y_visible, dtype=float)
    y_bin = (y[mask] > 0.0).astype(float)
    phi = design_matrix(x, pi, stratum, n_strata)
    beta, rate = fit_logistic_beta(phi[mask], y_bin, lam, iters, tol_v)
    return FittedLogistic(beta=beta, rate=rate, n_strata=int(n_strata))


def mu_v_from_mu_y(mu_y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``mu_V(x) = w(x) mu_Y(x)`` for a known density ratio ``w``."""
    return np.asarray(weights, dtype=float) * np.asarray(mu_y, dtype=float)


def aipw_population_mean(
    y_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    mu_y: np.ndarray,
    clip: float | None = None,
) -> float:
    return float(dr_mean(y_visible, a, q, mu_y, clip=clip))


def aipw_preference_rate(
    pref_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    mu_p: np.ndarray,
    clip: float | None = None,
) -> float:
    """AIPW for ``P(Y>0)`` using logistic ``mu_P``, never continuous ``mu_Y``."""
    return float(dr_mean(pref_visible, a, q, mu_p, clip=clip))


def aipw_policy_value(
    y_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    mu_y: np.ndarray,
    weights: np.ndarray,
    clip: float | None = None,
) -> float:
    """AIPW for ``E[w(X)Y]`` with ``mu_V = w mu_Y``.

    The residual is ``w (Y - mu_Y)``, not ``wY - mu_Y``.
    """
    w = np.asarray(weights, dtype=float)
    y = np.asarray(y_visible, dtype=float)
    mu_v = mu_v_from_mu_y(mu_y, w)
    y_w = w * y
    return float(dr_mean(y_w, a, q, mu_v, clip=clip))


def aipw_stratum_means(
    y_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    mu_y: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    clip: float | None = None,
) -> np.ndarray:
    z = np.asarray(stratum, dtype=int)
    out = np.full(int(n_strata), np.nan, dtype=float)
    for h in range(int(n_strata)):
        mask = z == h
        if not np.any(mask):
            continue
        a_h = np.asarray(a)[mask]
        if not np.any(np.asarray(a_h).astype(bool)):
            continue
        out[h] = aipw_population_mean(
            np.asarray(y_visible)[mask],
            a_h,
            np.asarray(q)[mask],
            np.asarray(mu_y)[mask],
            clip=clip,
        )
    return out


def estimate_estimands(
    y_visible: np.ndarray,
    pref_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    mu_y: np.ndarray,
    mu_p: np.ndarray,
    stratum: np.ndarray,
    target_weights: np.ndarray,
    n_strata: int,
    clip: float | None = None,
) -> dict[str, Any]:
    """Four estimands with estimand-specific nuisances."""
    return {
        "population_mean_reward": aipw_population_mean(y_visible, a, q, mu_y, clip=clip),
        "preference_rate": aipw_preference_rate(pref_visible, a, q, mu_p, clip=clip),
        "policy_value_target_distribution": aipw_policy_value(
            y_visible, a, q, mu_y, target_weights, clip=clip
        ),
        "context_conditioned_reward": aipw_stratum_means(
            y_visible, a, q, mu_y, stratum, n_strata, clip=clip
        ),
    }
