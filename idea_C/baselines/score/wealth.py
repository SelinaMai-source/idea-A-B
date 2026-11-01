"""Overshoot refund identities for SCORE (Kuang, Gang, Xia; ICML 2026 Main).

independent_reimplementation: no official code.

Paper §3–4:
    O_j = (α_j e_j - 1)_+
    C^{SCORE}_j = (C_j - O_j)_+

For LOND/LORD the base cost is C_j = α_j, so
    C^{S-LOND/LORD}_j = (α_j - O_j)_+
and the LOND refund that re-enters wealth is min(O_j, α_j).

These quantities are computed from the *current* e-value only after the
threshold α_j has already been fixed from F_{j-1}. Ground-truth labels
never enter wealth updates.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "overshoot",
    "score_cost",
    "lond_refund",
    "saffron_tight_cost",
    "saffron_score_cost",
    "rejects",
]


def overshoot(alpha: float, evalue: float) -> float:
    """O = (α e - 1)_+."""
    if alpha <= 0.0 or not np.isfinite(alpha):
        return 0.0
    return float(max(alpha * evalue - 1.0, 0.0))


def score_cost(base_cost: float, overshoot_value: float) -> float:
    """C^{SCORE} = (C - O)_+."""
    return float(max(base_cost - overshoot_value, 0.0))


def lond_refund(alpha: float, overshoot_value: float) -> float:
    """min(O, α) = α - (α - O)_+  (used in SCORE-LOND wealth, Prop. 4.1)."""
    return float(min(max(overshoot_value, 0.0), max(alpha, 0.0)))


def saffron_tight_cost(alpha: float, evalue: float, lam: float) -> float:
    """Candidate-tightened cost C' before subtracting overshoot (paper §4.3 step 1).

    C' = α (1 - λ e + (λ e - 1)_+) / (1 - λ) = α (1 - min(λ e, 1)) / (1 - λ).
    """
    if lam >= 1.0 or lam <= 0.0 or alpha <= 0.0:
        return 0.0
    return float(alpha * (1.0 - min(lam * evalue, 1.0)) / (1.0 - lam))


def saffron_score_cost(alpha: float, evalue: float, lam: float) -> float:
    """C^{S-SAF} = (α (1 - λ e) / (1 - λ) - O)_+  (paper §4.3)."""
    o = overshoot(alpha, evalue)
    if lam >= 1.0 or lam <= 0.0:
        return score_cost(alpha, o)
    raw = alpha * (1.0 - lam * evalue) / (1.0 - lam)
    return float(max(raw - o, 0.0))


def rejects(alpha: float, evalue: float) -> bool:
    """Reject iff e ≥ 1/α, evaluated as e·α ≥ 1 to avoid overflow."""
    if alpha <= 0.0 or not np.isfinite(alpha):
        return False
    return bool(evalue * alpha >= 1.0)
