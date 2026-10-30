"""Estimands and identification status for Idea A v1.0.

The independent audit channel with known ``q_t > 0`` identifies the finite-
population mean even when the natural channel is MNAR. Zero audit overlap
(``q=0`` on a subpopulation that is also unobserved naturally) is
identification failure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ESTIMANDS: dict[str, str] = {
    "population_mean_reward": (
        "Finite-population mean of the factual reward Y under the behavior policy."
    ),
    "preference_rate": (
        "Finite-population preference rate P(Y > 0) under the behavior policy."
    ),
    "context_conditioned_reward": (
        "Finite-population mean E[Y | stratum = h] for each discrete stratum h."
    ),
    "policy_value_target_distribution": (
        "Finite-population policy value E_{x ~ p*}[Y] where p* is uniform over "
        "strata (density ratio w = p*(stratum)/p(stratum)). The target "
        "distribution is this stated p*, not a different action policy."
    ),
}


def audit_channel_status(q: np.ndarray) -> dict[str, Any]:
    q_arr = np.asarray(q, dtype=float)
    min_q = float(np.min(q_arr)) if q_arr.size else float("nan")
    if q_arr.size == 0 or min_q <= 0.0:
        return {
            "status": "identification_failure",
            "channel": "audit",
            "reason": (
                "Audit propensity is zero on at least one unit. Those units are "
                "not identified by the audit channel. If they are also missing "
                "from the natural channel, this is identification failure."
            ),
            "min_q": min_q,
        }
    return {
        "status": "identified",
        "channel": "audit",
        "reason": (
            "Independent audits with known q_t > 0 identify the finite-population "
            "mean by Horvitz–Thompson / Hájek / AIPW, including under MNAR of the "
            "natural observation S."
        ),
        "min_q": min_q,
    }


def combine_status(natural: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "natural": natural,
        "audit": audit,
        "note": (
            "Natural-channel MNAR or zero overlap is identification failure for "
            "estimators that use only S. It is not an estimator bug. The audit "
            "channel is a separate design with a known propensity."
        ),
    }


def density_ratio_target_weights(
    stratum: np.ndarray,
    p_stratum: np.ndarray,
    p_target_stratum: np.ndarray,
) -> np.ndarray:
    """``w_i = p*(z_i) / p(z_i)`` for the stated target distribution."""
    z = np.asarray(stratum, dtype=int)
    p = np.asarray(p_stratum, dtype=float)
    p_star = np.asarray(p_target_stratum, dtype=float)
    return p_star[z] / np.maximum(p[z], 1e-12)
