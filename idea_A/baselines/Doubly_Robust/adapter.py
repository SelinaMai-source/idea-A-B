"""AIPW / doubly robust mean with the known audit propensity ``q``.

Fits the outcome model on ``A=1`` only, then calls
``shared.metrics.estimators.dr_mean``, which does not read ``Y`` where
``A=0``. Statistical baseline, not an official clone.
"""

from __future__ import annotations

import numpy as np

from shared.metrics.estimators import dr_mean

from idea_A.methods.v1_load import load_v1_module

_estimators = load_v1_module("estimators")


def estimate(
    y_visible: np.ndarray,
    a: np.ndarray,
    q: np.ndarray,
    x: np.ndarray,
    pi: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    ridge_lambda: float = 1e-3,
    clip: float | None = None,
) -> float:
    mu_hat = _estimators.fit_mu_hat(
        y_visible, a, x, pi, stratum, n_strata, ridge_lambda=ridge_lambda
    )
    return float(dr_mean(y_visible, a, q, mu_hat, clip=clip))
