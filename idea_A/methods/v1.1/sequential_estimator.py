"""History-respecting AIPW for sequential Idea A v1.1 logs.

Fit nuisances on *past* rounds and estimate the *current* round. Ordinary
shuffled cross-validation that mixes future rounds into the training fold
leaks under adaptive sampling: future audit probabilities and the law of
``Y`` (via an updated policy) depend on past estimates.

The first round has no history. Its outcome regression is the constant 0,
so AIPW reduces to IPW on that round (correct propensity, misspecified
outcome). Subsequent rounds use ridge / logistic models fit on past
``A=1`` units only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shared.metrics.estimators import dr_mean

from .cross_fitting import SHUFFLED_CV_LEAKAGE_NOTE
from .outcome_model import (
    aipw_policy_value,
    aipw_preference_rate,
    aipw_stratum_means,
    fit_mu_p,
    fit_mu_y,
    load_v11_config,
)

# Frozen 97.5% standard-normal quantile. Influence-function Wald intervals
# use this constant for every v1.1 method; it is not tuned on evaluation seeds.
WALD_Z = float(load_v11_config().get("ci_z", 1.959963984540054))


def aipw_influence(
    y: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    mu: np.ndarray,
    clip: float | None = None,
) -> np.ndarray:
    """Per-unit AIPW scores. ``Y`` is not read where ``A=0``."""
    y_arr = np.asarray(y, dtype=float)
    audited = np.asarray(a).astype(bool, copy=False)
    propensity = np.asarray(q, dtype=float)
    prediction = np.asarray(mu, dtype=float).copy()
    if clip is not None:
        propensity = np.maximum(propensity, float(clip))
    contribution = prediction.copy()
    if np.any(audited):
        q_a = propensity[audited]
        if np.any(~np.isfinite(q_a) | (q_a <= 0.0)):
            raise ValueError("q must be finite and > 0 where A==1")
        contribution[audited] = prediction[audited] + (
            (y_arr[audited] - prediction[audited]) / q_a
        )
    return contribution


def wald_interval(influence: np.ndarray, z: float = WALD_Z) -> dict[str, float | str]:
    """Nominal 95% Wald interval from i.i.d. influence scores. Frozen formula."""
    psi = np.asarray(influence, dtype=float)
    n = int(psi.shape[0])
    estimate = float(np.mean(psi)) if n else float("nan")
    if n < 2:
        se = float("nan")
        lo = float("nan")
        hi = float("nan")
    else:
        se = float(np.std(psi, ddof=1) / np.sqrt(n))
        lo = estimate - float(z) * se
        hi = estimate + float(z) * se
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "z": float(z),
        "formula": "influence_wald_95",
        "n": float(n),
    }


def interval_covers(ci: dict[str, Any], truth: float) -> bool:
    low = ci.get("ci_low")
    high = ci.get("ci_high")
    if low is None or high is None:
        return False
    if not np.isfinite(low) or not np.isfinite(high):
        return False
    return float(low) <= float(truth) <= float(high)


@dataclass
class RoundSnapshot:
    y_visible: np.ndarray
    pref_visible: np.ndarray
    a: np.ndarray
    q: np.ndarray
    x: np.ndarray
    pi: np.ndarray
    stratum: np.ndarray
    target_weights: np.ndarray
    n_strata: int


@dataclass
class SequentialNuisanceState:
    """Past-round audited features used to fit the next-round nuisances."""

    y: list[np.ndarray] = field(default_factory=list)
    a: list[np.ndarray] = field(default_factory=list)
    x: list[np.ndarray] = field(default_factory=list)
    pi: list[np.ndarray] = field(default_factory=list)
    stratum: list[np.ndarray] = field(default_factory=list)
    n_strata: int = 8
    ridge_lambda: float = 1e-3

    def has_history(self) -> bool:
        return any(int(np.sum(np.asarray(a).astype(bool))) > 0 for a in self.a)

    def append(self, round_data: RoundSnapshot) -> None:
        self.y.append(np.asarray(round_data.y_visible, dtype=float))
        self.a.append(np.asarray(round_data.a))
        self.x.append(np.asarray(round_data.x, dtype=float))
        self.pi.append(np.asarray(round_data.pi))
        self.stratum.append(np.asarray(round_data.stratum, dtype=int))
        self.n_strata = int(round_data.n_strata)

    def _stack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(self.y, axis=0),
            np.concatenate(self.a, axis=0),
            np.concatenate(self.x, axis=0),
            np.concatenate(self.pi, axis=0),
            np.concatenate(self.stratum, axis=0),
        )

    def predict_current(self, x: np.ndarray, pi: np.ndarray, stratum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(np.asarray(x).shape[0])
        if not self.has_history():
            return np.zeros(n, dtype=float), np.full(n, 0.5, dtype=float)
        y, a, x_h, pi_h, z_h = self._stack()
        ridge = fit_mu_y(y, a, x_h, pi_h, z_h, self.n_strata, ridge_lambda=self.ridge_lambda)
        logistic = fit_mu_p(y, a, x_h, pi_h, z_h, self.n_strata, ridge_lambda=self.ridge_lambda)
        return ridge.predict(x, pi, stratum), logistic.predict(x, pi, stratum)

    def audit_residual_sigma(self, n_strata: int, sigma_floor: float = 1e-8) -> np.ndarray | None:
        if not self.has_history():
            return None
        from .adaptive_allocation import residual_sigma_by_stratum

        y, a, x_h, pi_h, z_h = self._stack()
        ridge = fit_mu_y(y, a, x_h, pi_h, z_h, n_strata, ridge_lambda=self.ridge_lambda)
        mu = ridge.predict(x_h, pi_h, z_h)
        y_arr = np.asarray(y, dtype=float)
        resid = np.full(y_arr.shape[0], np.nan, dtype=float)
        audited = np.asarray(a).astype(bool)
        resid[audited] = y_arr[audited] - mu[audited]
        return residual_sigma_by_stratum(z_h, resid, audited, n_strata, sigma_floor=sigma_floor)


def estimate_current_round(
    round_data: RoundSnapshot,
    state: SequentialNuisanceState,
    clip: float | None = None,
) -> dict[str, Any]:
    """Estimate the current round with nuisances fit on past rounds only."""
    mu_y, mu_p = state.predict_current(round_data.x, round_data.pi, round_data.stratum)
    pop_inf = aipw_influence(round_data.y_visible, round_data.a, round_data.q, mu_y, clip=clip)
    pref = aipw_preference_rate(round_data.pref_visible, round_data.a, round_data.q, mu_p, clip=clip)
    policy = aipw_policy_value(
        round_data.y_visible,
        round_data.a,
        round_data.q,
        mu_y,
        round_data.target_weights,
        clip=clip,
    )
    cond = aipw_stratum_means(
        round_data.y_visible,
        round_data.a,
        round_data.q,
        mu_y,
        round_data.stratum,
        round_data.n_strata,
        clip=clip,
    )
    return {
        "population_mean_reward": float(np.mean(pop_inf)),
        "preference_rate": pref,
        "policy_value_target_distribution": policy,
        "context_conditioned_reward": cond,
        "influence": pop_inf,
        "mu_y": mu_y,
        "mu_p": mu_p,
        "dr_mean_check": float(dr_mean(round_data.y_visible, round_data.a, round_data.q, mu_y, clip=clip)),
    }


def combine_round_influences(influences: list[np.ndarray], z: float = WALD_Z) -> dict[str, Any]:
    if not influences:
        return wald_interval(np.array([], dtype=float), z=z)
    return wald_interval(np.concatenate([np.asarray(v, dtype=float) for v in influences], axis=0), z=z)


def sequential_leakage_note() -> str:
    return SHUFFLED_CV_LEAKAGE_NOTE
