"""Audit allocation for A-v1.1a/b/c.

A-v1.1a: uniform ``q_i = b``.
A-v1.1b: ``q_h`` from estimated AIPW variance contribution using *audit
residuals only*, then rescale so ``E[q] = b``.
A-v1.1c: A-v1.1b plus remaining-budget spending and policy-drift
upweighting.

The variance formula below is a computable approximation of the stratified
Horvitz–Thompson / AIPW variance. It is not a new optimality theorem.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Variance formula (also written in README.md)
#
# Treat the cross-fitted outcome regression as fixed. For known per-unit
# audit propensity q_i and residual e_i = Y_i - mu_Y(X_i),
#
#   ψ_i = mu_Y(X_i) + (A_i / q_i) e_i
#   Var(ψ_i | X_i) = σ²(X_i) (1 - q_i) / q_i
#
# Stratifying with q_h constant on stratum h and σ̂_h² estimated from
# audited residuals in h,
#
#   σ̂_h² = (1 / n_{h,A}) Σ_{i ∈ h, A_i=1} e_i²
#   n Var(μ̂) ≈ Σ_h W_h σ̂_h² (1/q_h - 1)     where W_h = n_h / n
#
# The q-dependent piece is Σ_h n_h σ̂_h² / q_h. Minimizing that subject to
# Σ_h n_h q_h = b n and 0 ≤ q_h ≤ 1 yields the Neyman rule q_h ∝ σ̂_h,
# then a rescale so mean(q) = b. This is textbook stratified allocation
# applied to AIPW residuals, not a new design theorem.
# ---------------------------------------------------------------------------


def rescale_to_budget(n_h: np.ndarray, q_h: np.ndarray, budget: float, n: int, cap: float = 1.0) -> np.ndarray:
    """Rescale per-stratum ``q_h`` so ``sum(n_h q_h)/n`` equals ``budget``."""
    q = np.clip(np.asarray(q_h, dtype=float), 0.0, cap)
    n_h_arr = np.asarray(n_h, dtype=float)
    n_f = float(n)
    target = float(budget)
    for _ in range(25):
        current = float(np.dot(n_h_arr, q) / n_f) if n_f else 0.0
        if abs(current - target) < 1e-12:
            break
        if current <= 0.0:
            q[:] = min(target, cap)
            break
        q = np.minimum(q * (target / current), cap)
    return q


def remaining_unit_budget(remaining_expected_audits: float, remaining_units: int, cap: float = 1.0) -> float:
    """Local per-unit budget from leftover expected audits."""
    if int(remaining_units) <= 0:
        return 0.0
    return float(np.clip(float(remaining_expected_audits) / float(remaining_units), 0.0, cap))


def allocate_uniform(n: int, budget: float) -> np.ndarray:
    return np.full(int(n), float(budget), dtype=float)


def residual_sigma_by_stratum(
    stratum: np.ndarray,
    residuals: np.ndarray,
    audit_mask: np.ndarray,
    n_strata: int,
    sigma_floor: float = 1e-8,
) -> np.ndarray:
    """Per-stratum residual SD from audited units only. Hidden labels are not used."""
    z = np.asarray(stratum, dtype=int)
    e = np.asarray(residuals, dtype=float)
    audited = np.asarray(audit_mask).astype(bool, copy=False)
    sigma = np.ones(int(n_strata), dtype=float)
    for h in range(int(n_strata)):
        values = e[(z == h) & audited]
        values = values[np.isfinite(values)]
        if values.size >= 2:
            sigma[h] = float(np.sqrt(np.mean(np.square(values))))
        elif values.size == 1:
            sigma[h] = float(max(abs(values[0]), float(sigma_floor)))
        else:
            sigma[h] = 1.0
        if not np.isfinite(sigma[h]) or sigma[h] <= 0.0:
            sigma[h] = 1.0
        sigma[h] = max(sigma[h], float(sigma_floor))
    return sigma


def aipw_variance_stratum_q(
    n_h: np.ndarray,
    sigma_h: np.ndarray,
    budget: float,
    n: int,
    q_floor: float = 1e-6,
) -> np.ndarray:
    """``q_h ∝ σ_h``, clipped, then matched to ``E[q]=b``.

    This is the Neyman solution of the stratified AIPW variance
    approximation ``Σ_h n_h σ_h² / q_h`` under ``Σ_h n_h q_h = b n``.
    """
    n_h_arr = np.asarray(n_h, dtype=float)
    sigma = np.maximum(np.asarray(sigma_h, dtype=float), 1e-8)
    denom = float(np.dot(n_h_arr, sigma))
    if denom <= 0.0:
        q_h = np.full(n_h_arr.shape[0], float(budget), dtype=float)
    else:
        q_h = (float(budget) * float(n)) * sigma / denom
    q_h = np.minimum(q_h, 1.0)
    q_h = np.maximum(q_h, float(q_floor))
    q_h = np.minimum(q_h, 1.0)
    return rescale_to_budget(n_h_arr, q_h, budget, n)


def stratum_policy_drift(
    pi_current: np.ndarray,
    stratum_current: np.ndarray,
    pi_previous: np.ndarray | None,
    stratum_previous: np.ndarray | None,
    n_strata: int,
) -> np.ndarray:
    """Absolute change in mean action per stratum. Zeros if no previous round."""
    drift = np.zeros(int(n_strata), dtype=float)
    if pi_previous is None or stratum_previous is None:
        return drift
    z_c = np.asarray(stratum_current, dtype=int)
    z_p = np.asarray(stratum_previous, dtype=int)
    pi_c = np.asarray(pi_current, dtype=float)
    pi_p = np.asarray(pi_previous, dtype=float)
    for h in range(int(n_strata)):
        cur = pi_c[z_c == h]
        prev = pi_p[z_p == h]
        if cur.size == 0 or prev.size == 0:
            continue
        drift[h] = abs(float(np.mean(cur)) - float(np.mean(prev)))
    return drift


def apply_drift_upweight(
    sigma_h: np.ndarray,
    drift_h: np.ndarray,
    kappa: float,
    scale: float,
    cap: float,
) -> np.ndarray:
    """Upweight residual SD where policy drift is large.

    ``factor_h = 1 + kappa * clip(|Δπ_h| / scale, 0, cap)``.
    """
    scale_v = max(float(scale), 1e-12)
    factor = 1.0 + float(kappa) * np.clip(
        np.asarray(drift_h, dtype=float) / scale_v,
        0.0,
        float(cap),
    )
    return np.asarray(sigma_h, dtype=float) * factor


def allocate_aipw_variance(
    stratum: np.ndarray,
    n_strata: int,
    budget: float,
    sigma_h: np.ndarray | None,
    q_floor: float = 1e-6,
) -> np.ndarray:
    """A-v1.1b unit-level ``q``. Falls back to uniform when ``sigma_h`` is missing."""
    z = np.asarray(stratum, dtype=int)
    n = int(z.shape[0])
    n_h = np.bincount(z, minlength=int(n_strata)).astype(float)
    if sigma_h is None:
        return allocate_uniform(n, budget)
    q_h = aipw_variance_stratum_q(n_h, sigma_h, float(budget), n, q_floor=q_floor)
    return q_h[z]


def allocate_aipw_variance_remaining_drift(
    stratum: np.ndarray,
    n_strata: int,
    remaining_expected_audits: float,
    remaining_units: int,
    sigma_h: np.ndarray | None,
    drift_h: np.ndarray | None,
    *,
    q_floor: float = 1e-6,
    drift_kappa: float = 1.0,
    drift_scale: float = 0.10,
    drift_factor_cap: float = 2.0,
) -> np.ndarray:
    """A-v1.1c: remaining-budget local ``b`` plus optional drift upweighting."""
    z = np.asarray(stratum, dtype=int)
    n = int(z.shape[0])
    local_b = remaining_unit_budget(remaining_expected_audits, remaining_units)
    n_h = np.bincount(z, minlength=int(n_strata)).astype(float)
    if sigma_h is None:
        return allocate_uniform(n, local_b)
    sigma = np.asarray(sigma_h, dtype=float)
    if drift_h is not None:
        sigma = apply_drift_upweight(sigma, drift_h, drift_kappa, drift_scale, drift_factor_cap)
    q_h = aipw_variance_stratum_q(n_h, sigma, local_b, n, q_floor=q_floor)
    return q_h[z]


def two_phase_indices(n: int, pilot_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split units into a pilot phase and a remainder. Does not read labels."""
    n_int = int(n)
    frac = min(max(float(pilot_fraction), 0.05), 0.95)
    n_pilot = int(round(frac * n_int))
    n_pilot = min(max(n_pilot, 1), n_int - 1) if n_int >= 2 else n_int
    perm = rng.permutation(n_int)
    return perm[:n_pilot], perm[n_pilot:]


def allocation_notes() -> dict[str, Any]:
    return {
        "formula": (
            "n Var(μ̂) ≈ Σ_h W_h σ̂_h² (1/q_h - 1), σ̂_h² from audited AIPW "
            "residuals, q_h ∝ σ̂_h, E[q]=b"
        ),
        "status": "computable_approximation_not_a_new_optimality_theorem",
        "residuals": "audit_only",
    }
