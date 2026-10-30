"""2-fold cross-fitting for static Idea A v1.1 data.

Out-of-fold predictions only. Hidden labels are never used. Ordinary shuffled
K-fold that mixes *future rounds* into a training fold leaks under adaptive
sampling: future audit probabilities and the outcome law (via an updated
policy) depend on past estimates. Sequential data must use
``sequential_estimator.py`` instead of this module.
"""

from __future__ import annotations

import numpy as np

from .outcome_model import fit_mu_p, fit_mu_y

SHUFFLED_CV_LEAKAGE_NOTE = (
    "Ordinary shuffled K-fold CV that pools future rounds into a training "
    "fold leaks under adaptive sampling. Future q and the distribution of Y "
    "(through an updated policy) depend on past estimates. Static 2-fold "
    "cross-fit on a single fixed sample does not have that time leak; "
    "sequential estimation must fit nuisances on past rounds only."
)


def assign_folds(n: int, n_folds: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Fold ids in ``{0, ..., n_folds-1}``. Does not read outcomes."""
    n_int = int(n)
    k = int(n_folds)
    if n_int <= 0 or k < 2:
        raise ValueError("n must be positive and n_folds >= 2")
    generator = np.random.default_rng(0) if rng is None else rng
    perm = generator.permutation(n_int)
    folds = np.empty(n_int, dtype=int)
    for i, idx in enumerate(perm):
        folds[idx] = i % k
    return folds


def cross_fit_mu_y(
    y_visible: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    pi: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    fold_ids: np.ndarray,
    ridge_lambda: float | None = None,
) -> np.ndarray:
    """Out-of-fold ridge ``mu_Y``. Each unit is predicted from the other fold."""
    folds = np.asarray(fold_ids, dtype=int)
    n = int(folds.shape[0])
    mu = np.zeros(n, dtype=float)
    for k in np.unique(folds):
        train = folds != k
        test = folds == k
        if not np.any(test):
            continue
        model = fit_mu_y(
            np.asarray(y_visible)[train],
            np.asarray(a)[train],
            np.asarray(x)[train],
            np.asarray(pi)[train],
            np.asarray(stratum)[train],
            n_strata,
            ridge_lambda=ridge_lambda,
        )
        mu[test] = model.predict(np.asarray(x)[test], np.asarray(pi)[test], np.asarray(stratum)[test])
    return mu


def cross_fit_mu_p(
    y_visible: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    pi: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    fold_ids: np.ndarray,
    ridge_lambda: float | None = None,
) -> np.ndarray:
    """Out-of-fold logistic ``mu_P``. Not a transform of ``mu_Y``."""
    folds = np.asarray(fold_ids, dtype=int)
    n = int(folds.shape[0])
    mu = np.full(n, 0.5, dtype=float)
    for k in np.unique(folds):
        train = folds != k
        test = folds == k
        if not np.any(test):
            continue
        model = fit_mu_p(
            np.asarray(y_visible)[train],
            np.asarray(a)[train],
            np.asarray(x)[train],
            np.asarray(pi)[train],
            np.asarray(stratum)[train],
            n_strata,
            ridge_lambda=ridge_lambda,
        )
        mu[test] = model.predict(np.asarray(x)[test], np.asarray(pi)[test], np.asarray(stratum)[test])
    return mu


def cross_fit_nuisances(
    y_visible: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    pi: np.ndarray,
    stratum: np.ndarray,
    n_strata: int,
    rng: np.random.Generator | None = None,
    n_folds: int = 2,
    ridge_lambda: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mu_y, mu_p, fold_ids)`` from 2-fold out-of-fold fits."""
    n = int(np.asarray(a).shape[0])
    fold_ids = assign_folds(n, n_folds=n_folds, rng=rng)
    mu_y = cross_fit_mu_y(
        y_visible, a, x, pi, stratum, n_strata, fold_ids, ridge_lambda=ridge_lambda
    )
    mu_p = cross_fit_mu_p(
        y_visible, a, x, pi, stratum, n_strata, fold_ids, ridge_lambda=ridge_lambda
    )
    return mu_y, mu_p, fold_ids
