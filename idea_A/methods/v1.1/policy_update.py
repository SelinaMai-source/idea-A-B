"""Logistic / linear policy updates from observed data.

The updater may read ``Y`` only on units with ``O = S ∨ A``. Hidden labels
stay unused. A linear greedy rule compares ``μ̂(x, π=1)`` and ``μ̂(x, π=0)``.
A logistic rule compares ``P̂(Y>0 | x, π)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .outcome_model import fit_logistic_beta, fit_ridge_beta, predict_logistic, predict_ridge, sigmoid


def policy_features(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    return np.column_stack([np.ones(x_arr.shape[0], dtype=float), x_arr])


def _observed_finite(y_visible: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    obs = np.asarray(observed_mask).astype(bool, copy=False)
    y = np.asarray(y_visible, dtype=float)
    return obs & np.isfinite(y)


@dataclass
class LinearGreedyPolicy:
    beta_control: np.ndarray | None
    constant_control: float
    beta_treated: np.ndarray | None
    constant_treated: float
    explore_mix: float

    def predict_mu(self, x: np.ndarray, pi: np.ndarray) -> np.ndarray:
        phi = policy_features(x)
        mu0 = predict_ridge(phi, self.beta_control, self.constant_control)
        mu1 = predict_ridge(phi, self.beta_treated, self.constant_treated)
        pi_arr = np.asarray(pi, dtype=int)
        return np.where(pi_arr == 1, mu1, mu0)

    def greedy_action(self, x: np.ndarray) -> np.ndarray:
        phi = policy_features(x)
        mu0 = predict_ridge(phi, self.beta_control, self.constant_control)
        mu1 = predict_ridge(phi, self.beta_treated, self.constant_treated)
        return (mu1 > mu0).astype(int)

    def act(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = int(np.asarray(x).shape[0])
        greedy = self.greedy_action(x)
        mix = min(max(float(self.explore_mix), 0.0), 1.0)
        if mix <= 0.0:
            return greedy
        explore = rng.random(n) < mix
        random_pi = rng.integers(0, 2, size=n)
        out = greedy.copy()
        out[explore] = random_pi[explore]
        return out.astype(int)


@dataclass
class LogisticGreedyPolicy:
    beta_control: np.ndarray | None
    rate_control: float
    beta_treated: np.ndarray | None
    rate_treated: float
    explore_mix: float

    def greedy_action(self, x: np.ndarray) -> np.ndarray:
        phi = policy_features(x)
        p0 = predict_logistic(phi, self.beta_control, self.rate_control)
        p1 = predict_logistic(phi, self.beta_treated, self.rate_treated)
        return (p1 > p0).astype(int)

    def act(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = int(np.asarray(x).shape[0])
        greedy = self.greedy_action(x)
        mix = min(max(float(self.explore_mix), 0.0), 1.0)
        if mix <= 0.0:
            return greedy
        explore = rng.random(n) < mix
        random_pi = rng.integers(0, 2, size=n)
        out = greedy.copy()
        out[explore] = random_pi[explore]
        return out.astype(int)


@dataclass
class FixedBernoulliPolicy:
    probability: float

    def greedy_action(self, x: np.ndarray) -> np.ndarray:
        n = int(np.asarray(x).shape[0])
        return np.zeros(n, dtype=int) if self.probability <= 0.0 else np.ones(n, dtype=int)

    def act(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = int(np.asarray(x).shape[0])
        p = min(max(float(self.probability), 0.0), 1.0)
        return (rng.random(n) < p).astype(int)


def initial_simulator_logit_policy(x: np.ndarray, stratum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same logit used by the v1 static simulator, without reading Y."""
    x_arr = np.asarray(x, dtype=float)
    z = np.asarray(stratum, dtype=float)
    logit = 0.35 * x_arr[:, 0] - 0.15 * x_arr[:, 1] - 0.08 * (z - 3.5)
    return (rng.random(x_arr.shape[0]) < sigmoid(logit)).astype(int)


def fit_linear_policy(
    x: np.ndarray,
    pi: np.ndarray,
    y_visible: np.ndarray,
    observed_mask: np.ndarray,
    ridge_lambda: float = 1e-3,
    explore_mix: float = 0.1,
) -> LinearGreedyPolicy:
    """Fit separate ridge models on observed ``π=0`` and ``π=1`` units."""
    mask = _observed_finite(y_visible, observed_mask)
    x_arr = np.asarray(x, dtype=float)
    pi_arr = np.asarray(pi, dtype=int)
    y = np.asarray(y_visible, dtype=float)
    phi = policy_features(x_arr)
    m0 = mask & (pi_arr == 0)
    m1 = mask & (pi_arr == 1)
    b0, c0 = fit_ridge_beta(phi[m0], y[m0], ridge_lambda)
    b1, c1 = fit_ridge_beta(phi[m1], y[m1], ridge_lambda)
    return LinearGreedyPolicy(
        beta_control=b0,
        constant_control=c0,
        beta_treated=b1,
        constant_treated=c1,
        explore_mix=float(explore_mix),
    )


def fit_logistic_policy(
    x: np.ndarray,
    pi: np.ndarray,
    y_visible: np.ndarray,
    observed_mask: np.ndarray,
    ridge_lambda: float = 1e-3,
    explore_mix: float = 0.1,
    max_iter: int = 40,
    tol: float = 1e-8,
) -> LogisticGreedyPolicy:
    """Fit separate logistics for ``P(Y>0 | X, π)`` on observed units."""
    mask = _observed_finite(y_visible, observed_mask)
    x_arr = np.asarray(x, dtype=float)
    pi_arr = np.asarray(pi, dtype=int)
    y = np.asarray(y_visible, dtype=float)
    y_bin = (y > 0.0).astype(float)
    phi = policy_features(x_arr)
    m0 = mask & (pi_arr == 0)
    m1 = mask & (pi_arr == 1)
    b0, r0 = fit_logistic_beta(phi[m0], y_bin[m0], ridge_lambda, max_iter, tol)
    b1, r1 = fit_logistic_beta(phi[m1], y_bin[m1], ridge_lambda, max_iter, tol)
    return LogisticGreedyPolicy(
        beta_control=b0,
        rate_control=r0,
        beta_treated=b1,
        rate_treated=r1,
        explore_mix=float(explore_mix),
    )
