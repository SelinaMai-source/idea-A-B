"""Online e-LOND family for UAI 2026 (NOT CCF-A Main).

Formulas from Xu, Fischer, Ramdas (PMLR v337):

    e-LOND:         α_t = δ γ_t (|R_{t-1}| + 1)
    Closed e-LOND:  DP of Theorem 2, O(t^2) per step
    Donation e-LOND: wealth W̄_t and inflated α_t of (8)

Default γ_t = 1/(t(t+1)). All three are independent_reimplementation
relative to UAI 2026 (no dedicated online repo). AISTATS 2024 LOND is
wrapped separately in official_wrappers.py.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .result import ONLINE_CODE_STATUS, ProcedureResult

NEG_INF = -1e300


def gamma_inv_t_tplus1(t: int) -> float:
    if t < 1:
        raise ValueError("t is 1-based")
    return 1.0 / (t * (t + 1.0))


def _rejects_e(alpha: float, evalue: float) -> bool:
    if not np.isfinite(alpha):
        return True
    if alpha <= 0.0:
        return False
    return bool(evalue * alpha >= 1.0)


class ELOND:
    """UAI / Xu-Ramdas e-LOND. independent_reimplementation (UAI formula)."""

    method_name = "e-LOND"
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
        self.evalues: list[float] = []

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        return self.delta * self.gamma(t) * (self.n_rej + 1)

    def process_next(self, evalue: float) -> bool:
        evalue = float(evalue)
        alpha = float(self.next_alpha())
        rejected = _rejects_e(alpha, evalue)
        self.alphas.append(alpha)
        self.rejected.append(rejected)
        self.evalues.append(evalue)
        if rejected:
            self.n_rej += 1
        return rejected

    def run(self, evalues: Sequence[float]) -> ProcedureResult:
        for e in evalues:
            self.process_next(e)
        return ProcedureResult.ok(
            self.method_name,
            np.asarray(self.rejected, dtype=bool),
            np.asarray(self.alphas, dtype=float),
            self.code_status,
        )


class ClosedELOND:
    """Closed e-LOND via the O(t^2) DP of UAI Theorem 2.

    independent_reimplementation. This is NOT offline closedeBH.
    """

    method_name = "closed-e-LOND"
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
        if n_past == 0:
            return self.delta * self.gamma(1) * r_plus

        # v[k] = max_{S ⊆ [n_past], |S|=k} |S ∩ R| - δ E_S (|R|+1)
        v_prev = np.full(n_past + 1, NEG_INF, dtype=float)
        v_prev[0] = 0.0
        v_cur = v_prev.copy()
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
                cand = self.delta * self.gamma(k + 1) * r_plus / denom
                if cand < best:
                    best = cand
        if not np.isfinite(best):
            return self.delta * self.gamma(t) * r_plus
        return float(best)

    def process_next(self, evalue: float) -> bool:
        evalue = float(evalue)
        alpha = float(self.next_alpha())
        if alpha == np.inf:
            rejected = True
        else:
            rejected = _rejects_e(alpha, evalue)
        self.alphas.append(alpha)
        self.rejected.append(rejected)
        self.evalues.append(evalue)
        self.rej_mask.append(rejected)
        return rejected

    def run(self, evalues: Sequence[float]) -> ProcedureResult:
        for e in evalues:
            self.process_next(e)
        return ProcedureResult.ok(
            self.method_name,
            np.asarray(self.rejected, dtype=bool),
            np.asarray(self.alphas, dtype=float),
            self.code_status,
            extra={"complexity": "O(t^2) per step, recomputed from scratch"},
        )


class DonationELOND:
    """Donation e-LOND (UAI Eq. 8). independent_reimplementation.

    W̄_t is evaluated in O(t) per step (same formula as the paper; the
    official O(log t) augmented tree is not available). See DEVIATION.md.
    """

    method_name = "donation-e-LOND"
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
        self.wealth_trace: list[float] = []

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
                excess = e_i - required
                wealth += g * min(excess, 1.0)
            else:
                wealth += g * min(e_i, 1.0)
        return float(max(wealth, 0.0))

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        n_rej = int(sum(self.rej_mask))
        r_plus = n_rej + 1
        wbar = self.donation_wealth()
        self.wealth_trace.append(wbar)
        cap = min(self.delta * r_plus * wbar, 1.0)
        base = self.delta * self.gamma(t) * r_plus
        if cap >= 1.0 - 1e-15:
            return float(np.inf)
        return float(base / (1.0 - cap))

    def process_next(self, evalue: float) -> bool:
        evalue = float(evalue)
        alpha = float(self.next_alpha())
        if not np.isfinite(alpha):
            rejected = True
        else:
            rejected = _rejects_e(alpha, evalue)
        self.alphas.append(alpha)
        self.rejected.append(rejected)
        self.evalues.append(evalue)
        self.rej_mask.append(rejected)
        return rejected

    def run(self, evalues: Sequence[float]) -> ProcedureResult:
        for e in evalues:
            self.process_next(e)
        return ProcedureResult.ok(
            self.method_name,
            np.asarray(self.rejected, dtype=bool),
            np.asarray(self.alphas, dtype=float),
            self.code_status,
            extra={
                "complexity": "O(t) wealth per step (formula-identical; not O(log t) tree)",
                "wealth_trace": np.asarray(self.wealth_trace, dtype=float),
            },
        )


def e_lond(evalues: Sequence[float], delta: float = 0.1, **kwargs) -> ProcedureResult:
    return ELOND(delta=delta, **kwargs).run(evalues)


def closed_e_lond(evalues: Sequence[float], delta: float = 0.1, **kwargs) -> ProcedureResult:
    return ClosedELOND(delta=delta, **kwargs).run(evalues)


def donation_e_lond(evalues: Sequence[float], delta: float = 0.1, **kwargs) -> ProcedureResult:
    return DonationELOND(delta=delta, **kwargs).run(evalues)
