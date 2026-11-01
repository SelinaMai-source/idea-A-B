"""r-LOND family (p-value, arbitrary dependence). UAI 2026, NOT CCF-A Main.

    r-LOND: α_t = δ γ_t β_t(|R_{t-1}|+1),  β_t(r) = (⌊r⌋ ∧ t) / ℓ_t

Donation r-LOND inflates that level by the Eq. (8)/(14) wealth using
calibrated e-values E_i = 1 / max(P_i, ε). Closed r-LOND uses the same
DP as closed e-LOND on those calibrated e-values but with γ_k / ℓ_k in
the numerator (BY-style). Both are independent_reimplementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .online_elond import NEG_INF, gamma_inv_t_tplus1
from .result import ONLINE_CODE_STATUS, ProcedureResult


def harmonic_number(t: int) -> float:
    if t < 1:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, t + 1)))


def calibrate_p_to_e(p: float, eps: float = 1e-300) -> float:
    """Heuristic calibrator E = 1/P used to match e-LOND correspondence.

    Not a finite-mean e-value under Uniform(0,1). Documented in DEVIATION.md.
    """
    p = min(max(float(p), eps), 1.0)
    return 1.0 / p


class RLOND:
    method_name = "r-LOND"
    code_status = ONLINE_CODE_STATUS

    def __init__(
        self,
        delta: float,
        gamma: Callable[[int], float] | None = None,
    ) -> None:
        self.delta = float(delta)
        self.gamma = gamma or gamma_inv_t_tplus1
        self.n_rej = 0
        self.alphas: list[float] = []
        self.rejected: list[bool] = []

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        ell = harmonic_number(t)
        level = min(self.n_rej + 1, t) / ell
        return min(1.0, self.delta * self.gamma(t) * level)

    def process_next(self, pvalue: float) -> bool:
        pvalue = float(pvalue)
        alpha = float(self.next_alpha())
        rejected = pvalue <= alpha
        self.alphas.append(alpha)
        self.rejected.append(rejected)
        if rejected:
            self.n_rej += 1
        return rejected

    def run(self, pvalues: Sequence[float]) -> ProcedureResult:
        for p in pvalues:
            self.process_next(p)
        return ProcedureResult.ok(
            self.method_name,
            np.asarray(self.rejected, dtype=bool),
            np.asarray(self.alphas, dtype=float),
            self.code_status,
        )


class DonationRLOND:
    method_name = "donation-r-LOND"
    code_status = ONLINE_CODE_STATUS

    def __init__(
        self,
        delta: float,
        gamma: Callable[[int], float] | None = None,
    ) -> None:
        self.delta = float(delta)
        self.gamma = gamma or gamma_inv_t_tplus1
        self.alphas: list[float] = []
        self.rejected: list[bool] = []
        self.pvalues: list[float] = []
        self.evalues: list[float] = []
        self.rej_mask: list[bool] = []

    def donation_wealth(self) -> float:
        n_past = len(self.evalues)
        if n_past == 0:
            return 0.0
        n_rej = int(sum(self.rej_mask))
        r_plus = n_rej + 1
        wealth = 0.0
        for i, (e_i, rej) in enumerate(zip(self.evalues, self.rej_mask), start=1):
            g = self.gamma(i)
            if rej:
                required = 1.0 / (self.delta * g * r_plus) if self.delta * g * r_plus > 0 else np.inf
                wealth += g * min(e_i - required, 1.0)
            else:
                wealth += g * min(e_i, 1.0)
        return float(max(wealth, 0.0))

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        n_rej = int(sum(self.rej_mask))
        r_plus = n_rej + 1
        ell = harmonic_number(t)
        wbar = self.donation_wealth()
        cap = min(self.delta * r_plus * wbar, 1.0)
        base = self.delta * self.gamma(t) * min(r_plus, t) / ell
        if cap >= 1.0 - 1e-15:
            return 1.0
        return float(min(1.0, base / (1.0 - cap)))

    def process_next(self, pvalue: float) -> bool:
        pvalue = float(pvalue)
        alpha = float(self.next_alpha())
        rejected = pvalue <= alpha
        self.alphas.append(alpha)
        self.rejected.append(rejected)
        self.pvalues.append(pvalue)
        self.evalues.append(calibrate_p_to_e(pvalue))
        self.rej_mask.append(rejected)
        return rejected

    def run(self, pvalues: Sequence[float]) -> ProcedureResult:
        for p in pvalues:
            self.process_next(p)
        return ProcedureResult.ok(
            self.method_name,
            np.asarray(self.rejected, dtype=bool),
            np.asarray(self.alphas, dtype=float),
            self.code_status,
            extra={"calibrator": "E=1/P (see DEVIATION.md)"},
        )


class ClosedRLOND:
    """Closed r-LOND, independent_reimplementation.

    DP of closed e-LOND on E=1/P, numerator uses γ_{k+1}/ℓ_{k+1}.
    """

    method_name = "closed-r-LOND"
    code_status = ONLINE_CODE_STATUS

    def __init__(
        self,
        delta: float,
        gamma: Callable[[int], float] | None = None,
    ) -> None:
        self.delta = float(delta)
        self.gamma = gamma or gamma_inv_t_tplus1
        self.alphas: list[float] = []
        self.rejected: list[bool] = []
        self.evalues: list[float] = []
        self.rej_mask: list[bool] = []

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        n_past = t - 1
        n_rej = int(sum(self.rej_mask))
        r_plus = n_rej + 1
        ell_t = harmonic_number
        if n_past == 0:
            return min(1.0, self.delta * self.gamma(1) / ell_t(1) * r_plus)

        v_prev = np.full(n_past + 1, NEG_INF, dtype=float)
        v_prev[0] = 0.0
        for i in range(1, n_past + 1):
            e_i = self.evalues[i - 1]
            in_r = 1.0 if self.rej_mask[i - 1] else 0.0
            v_cur = v_prev.copy()
            for k in range(1, i + 1):
                take = v_prev[k - 1] + in_r - self.delta * self.gamma(k) * e_i * r_plus
                v_cur[k] = max(v_prev[k], take)
            v_prev = v_cur

        best = np.inf
        for k in range(0, n_past + 1):
            denom = 1.0 + v_prev[k]
            if denom > 0.0 and np.isfinite(denom):
                ell = ell_t(k + 1)
                cand = self.delta * (self.gamma(k + 1) / ell) * r_plus / denom
                if cand < best:
                    best = cand
        if not np.isfinite(best):
            return min(1.0, self.delta * self.gamma(t) * min(r_plus, t) / ell_t(t))
        return float(min(1.0, best))

    def process_next(self, pvalue: float) -> bool:
        pvalue = float(pvalue)
        alpha = float(self.next_alpha())
        rejected = pvalue <= alpha
        self.alphas.append(alpha)
        self.rejected.append(rejected)
        self.evalues.append(calibrate_p_to_e(pvalue))
        self.rej_mask.append(rejected)
        return rejected

    def run(self, pvalues: Sequence[float]) -> ProcedureResult:
        for p in pvalues:
            self.process_next(p)
        return ProcedureResult.ok(
            self.method_name,
            np.asarray(self.rejected, dtype=bool),
            np.asarray(self.alphas, dtype=float),
            self.code_status,
            extra={"calibrator": "E=1/P; DP analog of Theorem 2 with γ/ℓ"},
        )


def r_lond(pvalues: Sequence[float], delta: float = 0.1, **kwargs) -> ProcedureResult:
    return RLOND(delta=delta, **kwargs).run(pvalues)


def donation_r_lond(pvalues: Sequence[float], delta: float = 0.1, **kwargs) -> ProcedureResult:
    return DonationRLOND(delta=delta, **kwargs).run(pvalues)


def closed_r_lond(pvalues: Sequence[float], delta: float = 0.1, **kwargs) -> ProcedureResult:
    return ClosedRLOND(delta=delta, **kwargs).run(pvalues)
