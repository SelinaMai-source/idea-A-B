"""Audit propensity rules for Idea A v1.1.

Known audit ``q`` is the MNAR identification channel. Combined observation
probability ``P(O=1)=1-(1-p_S)(1-q)`` is valid only when ``p_S`` does not
depend on unobserved ``Y``. Ordinary AIPW double robustness is only for
audit missingness with known ``q`` and ``A ⊥ Y | X, history``. An audit
does not convert MNAR of the natural channel into MAR.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DR_VALIDITY_SCOPE = (
    "Ordinary AIPW double robustness holds only for audit missingness with "
    "known q and A independent of Y given X and history. It is not claimed "
    "for arbitrary MNAR of the natural observation S."
)

COMBINED_O_FORMULA_SCOPE = (
    "P(O=1)=1-(1-p_S)(1-q) is the probability of the union S∨A when A is "
    "drawn independently of S given the logged history and p_S does not "
    "depend on unobserved Y. It is not an identification formula under MNAR."
)

MNAR_NOT_MAR_NOTE = (
    "The existence of an independent audit with known q does not make the "
    "natural channel MAR. Natural-channel MNAR remains identification "
    "failure for estimators that use only S."
)


def _as_prob(name: str, value: Any, n: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        if n is None:
            return arr.reshape(1)
        arr = np.full(int(n), float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D vector")
    if n is not None and arr.shape[0] != int(n):
        raise ValueError(f"{name} length {arr.shape[0]} does not match {n}")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return arr


def audit_channel_status(q: np.ndarray) -> dict[str, Any]:
    """Identification status of the known-audit channel."""
    q_arr = _as_prob("q", q)
    min_q = float(np.min(q_arr)) if q_arr.size else float("nan")
    if q_arr.size == 0 or min_q <= 0.0:
        return {
            "status": "identification_failure",
            "channel": "audit",
            "reason": (
                "Audit propensity is zero on at least one unit. Those units are "
                "not identified by the audit channel."
            ),
            "min_q": min_q,
        }
    return {
        "status": "identified",
        "channel": "audit",
        "reason": (
            "Independent audits with known q>0 identify finite-population "
            "means by Horvitz–Thompson / AIPW, including when the natural "
            "channel S is MNAR. This does not convert MNAR into MAR."
        ),
        "min_q": min_q,
    }


def combined_observation_probability(
    p_s: np.ndarray,
    q: np.ndarray,
    *,
    p_s_depends_on_unobserved_y: bool,
) -> np.ndarray:
    """``P(O=1)=1-(1-p_S)(1-q)`` when that product formula is valid.

    Raises ``ValueError`` if ``p_s_depends_on_unobserved_y`` is true: the
    product formula is then not an identification device, and callers must
    use the known audit propensity ``q`` instead of a combined ``P(O=1)``.
    """
    if p_s_depends_on_unobserved_y:
        raise ValueError(
            "P(O=1)=1-(1-p_S)(1-q) is not an identification formula when "
            "p_S depends on unobserved Y. Known audit q is the MNAR "
            "identification channel. An audit does not convert MNAR into MAR."
        )
    p_arr = _as_prob("p_s", p_s)
    q_arr = _as_prob("q", q, n=int(p_arr.shape[0]))
    return 1.0 - (1.0 - p_arr) * (1.0 - q_arr)


def dr_double_robustness_applies(
    *,
    missingness: str,
    q_known: bool,
    audit_independent_of_y_given_history: bool,
) -> bool:
    """True only for audit missingness with known q and A ⊥ Y | X, history."""
    return (
        str(missingness) == "audit"
        and bool(q_known)
        and bool(audit_independent_of_y_given_history)
    )


def identification_notes() -> dict[str, str]:
    return {
        "dr_validity_scope": DR_VALIDITY_SCOPE,
        "combined_o_formula_scope": COMBINED_O_FORMULA_SCOPE,
        "mnar_not_mar": MNAR_NOT_MAR_NOTE,
    }
