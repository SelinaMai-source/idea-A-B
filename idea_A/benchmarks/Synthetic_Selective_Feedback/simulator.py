"""Finite-population simulator for endogenous feedback censoring.

The hidden outcome ``Y`` is generated for every unit. An algorithm may read
``Y`` only through ``World.observe``, which reveals a label iff the natural
observation bit ``S`` or the audit bit ``A`` is 1. Ground-truth estimands are
exact finite-population quantities computed from the full ``Y``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shared.data_utils.seeds import seed_everything

MECHANISMS = (
    "mcar",
    "mar",
    "mnar",
    "policy_dependent",
    "low_overlap",
    "zero_overlap",
    "nonstationary",
)

N_DEFAULT = 2000
N_STRATA = 8
STRATUM_PROBS = np.array([0.18, 0.16, 0.14, 0.13, 0.12, 0.11, 0.09, 0.07], dtype=float)
STRATUM_PROBS = STRATUM_PROBS / STRATUM_PROBS.sum()
STRATUM_EFFECTS = np.array([-1.20, -0.70, -0.30, 0.00, 0.25, 0.65, 1.05, 1.50], dtype=float)
STRATUM_NOISE = np.array([0.35, 0.45, 0.55, 0.75, 1.00, 1.30, 1.65, 2.10], dtype=float)
ZERO_OVERLAP_STRATA = (6, 7)
LOW_OVERLAP_STRATA = (5, 6, 7)


def _require_mechanism(mechanism: str) -> str:
    name = str(mechanism)
    if name not in MECHANISMS:
        raise ValueError(f"unknown mechanism {name!r}; expected one of {MECHANISMS}")
    return name


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _natural_propensity(
    mechanism: str,
    *,
    x: np.ndarray,
    stratum: np.ndarray,
    pi: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n: int,
) -> np.ndarray:
    """Bernoulli parameter used to draw the natural observation ``S``.

    Under MNAR this parameter depends on the unobserved outcome ``Y``. That
    dependence is an identification failure for the natural channel, not an
    estimator bug.
    """
    if mechanism == "mcar":
        return np.full(n, 0.28, dtype=float)
    if mechanism == "mar":
        logit = -0.45 + 1.10 * x[:, 0] - 0.35 * (stratum / (N_STRATA - 1.0))
        return np.clip(_sigmoid(logit), 0.05, 0.90)
    if mechanism == "mnar":
        return np.clip(_sigmoid(-0.55 + 1.25 * y), 0.02, 0.95)
    if mechanism == "policy_dependent":
        return np.where(pi >= 0.5, 0.72, 0.08).astype(float)
    if mechanism == "low_overlap":
        p = np.full(n, 0.55, dtype=float)
        for h in LOW_OVERLAP_STRATA:
            p[stratum == h] = 0.025
        return p
    if mechanism == "zero_overlap":
        p = np.full(n, 0.42, dtype=float)
        for h in ZERO_OVERLAP_STRATA:
            p[stratum == h] = 0.0
        return p
    if mechanism == "nonstationary":
        time_frac = t / max(n - 1, 1)
        base = 0.62 - 0.52 * time_frac
        return np.clip(base, 0.04, 0.85)
    raise ValueError(f"unknown mechanism {mechanism!r}")


def _truths(
    y: np.ndarray,
    pref: np.ndarray,
    stratum: np.ndarray,
    target_weights: np.ndarray,
    k: int,
) -> dict[str, Any]:
    stratum_means = np.array(
        [float(np.mean(y[stratum == h])) if np.any(stratum == h) else float("nan") for h in range(k)],
        dtype=float,
    )
    return {
        "population_mean_reward": float(np.mean(y)),
        "preference_rate": float(np.mean(pref)),
        "policy_value_target_distribution": float(np.mean(target_weights * y)),
        "context_conditioned_reward": stratum_means,
        "n": int(y.shape[0]),
        "n_strata": int(k),
    }


@dataclass
class World:
    """One finite population. Hidden labels live in ``y`` / ``y0`` / ``y1``.

    Call ``observe(a)`` to obtain the only labels an algorithm is allowed to
    read. Direct use of ``y`` is reserved for computing truth and for tests
    that construct a revealed copy.
    """

    mechanism: str
    seed: int
    n: int
    n_strata: int
    stratum: np.ndarray
    x: np.ndarray
    t: np.ndarray
    pi: np.ndarray
    pi_star: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    y: np.ndarray
    pref: np.ndarray
    s: np.ndarray
    p_natural_true: np.ndarray
    p_natural_mar: np.ndarray
    p_stratum: np.ndarray
    p_target_stratum: np.ndarray
    target_weights: np.ndarray
    truths: dict[str, Any]
    identification_natural: dict[str, Any] = field(default_factory=dict)

    def observe(self, a: np.ndarray | None = None) -> np.ndarray:
        """Reveal ``Y`` on ``S=1`` or ``A=1``. All other entries are NaN."""
        if a is None:
            audited = np.zeros(self.n, dtype=bool)
        else:
            audited = np.asarray(a).astype(bool, copy=False)
            if audited.shape != (self.n,):
                raise ValueError(f"a must have shape {(self.n,)}, got {audited.shape}")
        visible = np.full(self.n, np.nan, dtype=float)
        revealed = self.s.astype(bool) | audited
        visible[revealed] = self.y[revealed]
        return visible

    def observe_pref(self, a: np.ndarray | None = None) -> np.ndarray:
        y_vis = self.observe(a)
        pref = np.full(self.n, np.nan, dtype=float)
        mask = np.isfinite(y_vis)
        pref[mask] = (y_vis[mask] > 0.0).astype(float)
        return pref


def simulate(mechanism: str, seed: int, n: int = N_DEFAULT) -> World:
    """Draw one finite world. Ground truth is the realized finite population."""
    name = _require_mechanism(mechanism)
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or int(n) < 8:
        raise ValueError("n must be an int >= 8")
    n_int = int(n)
    rng = seed_everything(int(seed))

    stratum = rng.choice(N_STRATA, size=n_int, p=STRATUM_PROBS)
    x = np.column_stack(
        [
            rng.normal(loc=(stratum - 3.5) / 3.5, scale=0.85, size=n_int),
            rng.normal(loc=0.15 * STRATUM_EFFECTS[stratum], scale=1.0, size=n_int),
        ]
    )
    t = np.arange(n_int, dtype=int)
    pi_logit = 0.35 * x[:, 0] - 0.15 * x[:, 1] - 0.08 * (stratum - 3.5)
    pi = (rng.random(n_int) < _sigmoid(pi_logit)).astype(int)
    pi_star = (x[:, 0] > 0.0).astype(int)

    eps = rng.normal(loc=0.0, scale=STRATUM_NOISE[stratum], size=n_int)
    mu0 = 0.15 + 0.55 * x[:, 0] - 0.25 * x[:, 1] + STRATUM_EFFECTS[stratum]
    tau = 0.70 + 0.35 * x[:, 0]
    y0 = mu0 + eps
    y1 = mu0 + tau + eps
    y = np.where(pi == 1, y1, y0)
    pref = (y > 0.0).astype(float)

    p_true = _natural_propensity(
        name, x=x, stratum=stratum, pi=pi, y=y, t=t, n=n_int
    )
    s = (rng.random(n_int) < p_true).astype(int)

    p_stratum = np.array(
        [(stratum == h).mean() for h in range(N_STRATA)], dtype=float
    )
    p_target = np.full(N_STRATA, 1.0 / N_STRATA, dtype=float)
    # Density ratio w(x) = p*(stratum) / p(stratum) for the stated target
    # distribution: uniform over the K discrete strata.
    safe_p = np.maximum(p_stratum, 1e-12)
    target_weights = p_target[stratum] / safe_p[stratum]
    p_mar = np.zeros(n_int, dtype=float)
    for h in range(N_STRATA):
        mask = stratum == h
        if np.any(mask):
            p_mar[mask] = float(np.mean(p_true[mask]))

    truths = _truths(y, pref, stratum, target_weights, N_STRATA)
    identification = _natural_identification(name, p_true, stratum)

    return World(
        mechanism=name,
        seed=int(seed),
        n=n_int,
        n_strata=N_STRATA,
        stratum=np.asarray(stratum, dtype=int),
        x=np.asarray(x, dtype=float),
        t=t,
        pi=np.asarray(pi, dtype=int),
        pi_star=np.asarray(pi_star, dtype=int),
        y0=np.asarray(y0, dtype=float),
        y1=np.asarray(y1, dtype=float),
        y=np.asarray(y, dtype=float),
        pref=pref,
        s=np.asarray(s, dtype=int),
        p_natural_true=np.asarray(p_true, dtype=float),
        p_natural_mar=p_mar,
        p_stratum=p_stratum,
        p_target_stratum=p_target,
        target_weights=np.asarray(target_weights, dtype=float),
        truths=truths,
        identification_natural=identification,
    )


def _natural_identification(mechanism: str, p_true: np.ndarray, stratum: np.ndarray) -> dict[str, Any]:
    min_p = float(np.min(p_true))
    if mechanism == "mnar":
        return {
            "status": "identification_failure",
            "channel": "natural",
            "reason": (
                "Natural observation depends on unobserved Y (MNAR). "
                "The natural propensity is not identified from (X, S) without "
                "extra assumptions. This is identification failure, not an "
                "estimator bug."
            ),
        }
    if mechanism == "zero_overlap" or min_p == 0.0:
        empty = [int(h) for h in range(N_STRATA) if not np.any((stratum == h) & (p_true > 0.0))]
        return {
            "status": "identification_failure",
            "channel": "natural",
            "reason": (
                "Zero overlap: P(S=1|stratum)=0 for some strata "
                f"{empty}. Those stratum means are not identified from the "
                "natural channel."
            ),
            "zero_overlap_strata": empty,
        }
    if mechanism == "low_overlap":
        return {
            "status": "weak_overlap",
            "channel": "natural",
            "reason": "Some strata have P(S=1) near 0. Natural IPW has high variance; this is weak overlap, not a coding error.",
        }
    if mechanism == "policy_dependent":
        return {
            "status": "identified_if_condition_on_action_and_overlap",
            "channel": "natural",
            "reason": "S depends on the observed action. A propensity that ignores the action is misspecified.",
        }
    return {
        "status": "identified_if_overlap",
        "channel": "natural",
        "reason": "Natural observation does not depend on unobserved Y.",
    }
