"""SCORE / e-LOND / e-LORD / e-SAFFRON online procedures.

independent_reimplementation of Kuang, Gang, Xia (ICML 2026 Main).

Power (module contract, used by P0):
    average power at time t is the expected fraction of non-nulls that
    have arrived by t and have been rejected (paper §2 AP(t)).
    Power ratio is that curve divided by the family baseline
    (e-LORD or e-SAFFRON), pointwise in t (paper §6).

All thresholds are F_{t-1}-measurable: ``next_alpha()`` does not see e_t.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .gamma import gamma_inv_t_tplus1
from .result import SCOREPLUS_NA_REASON, ProcedureResult
from .wealth import (
    lond_refund,
    overshoot,
    rejects,
    saffron_score_cost,
    score_cost,
    score_plus_lord_alpha,
    score_plus_saffron_alpha,
    score_plus_wealth,
)

OmegaSpec = float | Sequence[float] | Callable[[int], float]


def _as_callable(spec: OmegaSpec, default: float) -> Callable[[int], float]:
    if callable(spec):
        return spec  # type: ignore[return-value]
    if isinstance(spec, (int, float)):
        value = float(spec)
        return lambda _t: value
    seq = list(spec)
    return lambda t: float(seq[t - 1]) if t - 1 < len(seq) else float(seq[-1])


class _BaseEProcedure:
    method_name = "base"

    def __init__(self) -> None:
        self.evalues: list[float] = []
        self.alphas: list[float] = []
        self.rejected: list[bool] = []
        self.overshoots: list[float] = []
        self.costs: list[float] = []
        self.n_rej = 0

    def next_alpha(self) -> float:
        raise NotImplementedError

    def _commit_cost(self, alpha: float, evalue: float, rejected: bool, o: float) -> float:
        return float(alpha)

    def process_next(self, evalue: float) -> bool:
        evalue = float(evalue)
        alpha = float(self.next_alpha())
        if not np.isfinite(alpha) or alpha < 0.0:
            alpha = 0.0
        rejected = rejects(alpha, evalue)
        o = overshoot(alpha, evalue)
        cost = self._commit_cost(alpha, evalue, rejected, o)
        self.evalues.append(evalue)
        self.alphas.append(alpha)
        self.rejected.append(rejected)
        self.overshoots.append(o)
        self.costs.append(cost)
        if rejected:
            self.n_rej += 1
        return rejected

    def run(self, evalues: Sequence[float]) -> ProcedureResult:
        for e in evalues:
            self.process_next(e)
        return ProcedureResult.from_traces(
            self.method_name,
            self.evalues,
            self.alphas,
            self.rejected,
            self.overshoots,
            self.costs,
            extra={"n_steps": len(self.evalues)},
        )


class ELOND(_BaseEProcedure):
    """e-LOND: α_t = α γ_t (R_{t-1} + 1)  (Xu & Ramdas 2024; SCORE §4.1)."""

    method_name = "e-LOND"

    def __init__(
        self,
        alpha: float,
        gamma: Callable[[int], float] | None = None,
    ) -> None:
        super().__init__()
        self.alpha_level = float(alpha)
        self.gamma = gamma or gamma_inv_t_tplus1

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        return self.alpha_level * self.gamma(t) * (self.n_rej + 1)


class ScoreLOND(_BaseEProcedure):
    """SCORE-LOND, Proposition 4.1.

    α_1 = γ_1 α
    α_t = γ_t (R_{t-1}+1) (α + sum_{j<t} min(O_j, α_j) / (R_{j-1}+1))
    """

    method_name = "SCORE-LOND"

    def __init__(
        self,
        alpha: float,
        gamma: Callable[[int], float] | None = None,
    ) -> None:
        super().__init__()
        self.alpha_level = float(alpha)
        self.gamma = gamma or gamma_inv_t_tplus1
        self.refund_sum = 0.0  # sum min(O,α)/(R_{j-1}+1)

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        wealth = self.alpha_level + self.refund_sum
        return self.gamma(t) * (self.n_rej + 1) * wealth

    def _commit_cost(self, alpha: float, evalue: float, rejected: bool, o: float) -> float:
        denom = self.n_rej + 1  # R_{t-1}+1 before possibly incrementing
        self.refund_sum += lond_refund(alpha, o) / denom
        return score_cost(alpha, o)


class ELORD(_BaseEProcedure):
    """e-LORD: α_t = ω_t (R_{t-1}+1) (α - sum_{j<t} α_j / (R_{j-1}+1))."""

    method_name = "e-LORD"

    def __init__(self, alpha: float, omega: OmegaSpec = 0.05) -> None:
        super().__init__()
        self.alpha_level = float(alpha)
        self.omega = _as_callable(omega, 0.05)
        self.spent = 0.0  # sum α_j / (R_{j-1}+1)

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        remaining = self.alpha_level - self.spent
        if remaining < 0.0:
            remaining = 0.0
        return self.omega(t) * (self.n_rej + 1) * remaining

    def _commit_cost(self, alpha: float, evalue: float, rejected: bool, o: float) -> float:
        denom = self.n_rej + 1
        self.spent += alpha / denom
        return float(alpha)


class ScoreLORD(_BaseEProcedure):
    """SCORE-LORD, Proposition 4.3.

    α_t = ω_t (R_{t-1}+1) (α - sum_{j<t} (α_j - O_j)_+ / (R_{j-1}+1))
    """

    method_name = "SCORE-LORD"

    def __init__(self, alpha: float, omega: OmegaSpec = 0.05) -> None:
        super().__init__()
        self.alpha_level = float(alpha)
        self.omega = _as_callable(omega, 0.05)
        self.spent = 0.0

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        remaining = max(self.alpha_level - self.spent, 0.0)
        return self.omega(t) * (self.n_rej + 1) * remaining

    def _commit_cost(self, alpha: float, evalue: float, rejected: bool, o: float) -> float:
        denom = self.n_rej + 1
        cost = score_cost(alpha, o)
        self.spent += cost / denom
        return cost


class ESAFFRON(_BaseEProcedure):
    """e-SAFFRON (Zhang et al. 2025; SCORE §4.3 baseline)."""

    method_name = "e-SAFFRON"

    def __init__(
        self,
        alpha: float,
        omega: OmegaSpec = 0.05,
        lam: OmegaSpec = 0.5,
    ) -> None:
        super().__init__()
        self.alpha_level = float(alpha)
        self.omega = _as_callable(omega, 0.05)
        self.lam = _as_callable(lam, 0.5)
        self.spent = 0.0

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        remaining = max(self.alpha_level - self.spent, 0.0)
        return self.omega(t) * (1.0 - self.lam(t)) * (self.n_rej + 1) * remaining

    def _commit_cost(self, alpha: float, evalue: float, rejected: bool, o: float) -> float:
        t = len(self.alphas) + 1
        lam = self.lam(t)
        is_candidate_fail = evalue < (1.0 / lam if lam > 0.0 else np.inf)
        if is_candidate_fail and 0.0 < lam < 1.0:
            cost = alpha / (1.0 - lam)
        else:
            cost = 0.0
        self.spent += cost / (self.n_rej + 1)
        return float(cost)


class ScoreSAFFRON(_BaseEProcedure):
    """SCORE-SAFFRON, Proposition 4.4."""

    method_name = "SCORE-SAFFRON"

    def __init__(
        self,
        alpha: float,
        omega: OmegaSpec = 0.05,
        lam: OmegaSpec = 0.5,
    ) -> None:
        super().__init__()
        self.alpha_level = float(alpha)
        self.omega = _as_callable(omega, 0.05)
        self.lam = _as_callable(lam, 0.5)
        self.spent = 0.0

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        remaining = max(self.alpha_level - self.spent, 0.0)
        return self.omega(t) * (1.0 - self.lam(t)) * (self.n_rej + 1) * remaining

    def _commit_cost(self, alpha: float, evalue: float, rejected: bool, o: float) -> float:
        t = len(self.alphas) + 1
        lam = self.lam(t)
        cost = saffron_score_cost(alpha, evalue, lam)
        denom = self.n_rej + 1
        self.spent += cost / denom
        return cost


class ScorePlusLORD(_BaseEProcedure):
    """SCORE+-LORD, Proposition 5.3. Requires CPQD.

    α_t = ω_t (R_{t-1} ∨ 1) W_t
    W_t = α - sum_{j<t} (α_j - O_j)_+ / (R_{t-1} ∨ 1)
    """

    method_name = "SCORE+-LORD"

    def __init__(self, alpha: float, omega: OmegaSpec = 0.05) -> None:
        super().__init__()
        self.alpha_level = float(alpha)
        self.omega = _as_callable(omega, 0.05)
        self.cost_history: list[float] = []
        self.wealth_history: list[float] = []
        self._pending_wealth = 0.0

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        self._pending_wealth = score_plus_wealth(self.alpha_level, self.cost_history, self.n_rej)
        return score_plus_lord_alpha(self.omega(t), self.n_rej, self._pending_wealth)

    def _commit_cost(self, alpha: float, evalue: float, rejected: bool, o: float) -> float:
        self.wealth_history.append(self._pending_wealth)
        cost = score_cost(alpha, o)
        self.cost_history.append(cost)
        return cost


class ScorePlusSAFFRON(_BaseEProcedure):
    """SCORE+-SAFFRON, Theorem 5.4. Requires CPQD."""

    method_name = "SCORE+-SAFFRON"

    def __init__(
        self,
        alpha: float,
        omega: OmegaSpec = 0.05,
        lam: OmegaSpec = 0.5,
    ) -> None:
        super().__init__()
        self.alpha_level = float(alpha)
        self.omega = _as_callable(omega, 0.05)
        self.lam = _as_callable(lam, 0.5)
        self.cost_history: list[float] = []
        self.wealth_history: list[float] = []
        self._pending_wealth = 0.0

    def next_alpha(self) -> float:
        t = len(self.alphas) + 1
        self._pending_wealth = score_plus_wealth(self.alpha_level, self.cost_history, self.n_rej)
        return score_plus_saffron_alpha(self.omega(t), self.lam(t), self.n_rej, self._pending_wealth)

    def _commit_cost(self, alpha: float, evalue: float, rejected: bool, o: float) -> float:
        t = len(self.alphas) + 1
        self.wealth_history.append(self._pending_wealth)
        cost = saffron_score_cost(alpha, evalue, self.lam(t))
        self.cost_history.append(cost)
        return cost


def _run(cls: type[_BaseEProcedure], evalues: Sequence[float], **kwargs) -> ProcedureResult:
    return cls(**kwargs).run(evalues)


def e_lond(evalues: Sequence[float], alpha: float = 0.05, **kwargs) -> ProcedureResult:
    return _run(ELOND, evalues, alpha=alpha, **kwargs)


def score_lond(evalues: Sequence[float], alpha: float = 0.05, **kwargs) -> ProcedureResult:
    return _run(ScoreLOND, evalues, alpha=alpha, **kwargs)


def e_lord(evalues: Sequence[float], alpha: float = 0.05, omega: OmegaSpec = 0.05) -> ProcedureResult:
    return _run(ELORD, evalues, alpha=alpha, omega=omega)


def score_lord(evalues: Sequence[float], alpha: float = 0.05, omega: OmegaSpec = 0.05) -> ProcedureResult:
    return _run(ScoreLORD, evalues, alpha=alpha, omega=omega)


def e_saffron(
    evalues: Sequence[float],
    alpha: float = 0.05,
    omega: OmegaSpec = 0.05,
    lam: OmegaSpec = 0.5,
) -> ProcedureResult:
    return _run(ESAFFRON, evalues, alpha=alpha, omega=omega, lam=lam)


def score_saffron(
    evalues: Sequence[float],
    alpha: float = 0.05,
    omega: OmegaSpec = 0.05,
    lam: OmegaSpec = 0.5,
) -> ProcedureResult:
    return _run(ScoreSAFFRON, evalues, alpha=alpha, omega=omega, lam=lam)


def score_plus_lord(
    evalues: Sequence[float],
    alpha: float = 0.05,
    omega: OmegaSpec = 0.05,
    *,
    assume_cpqd: bool = False,
) -> ProcedureResult:
    if not assume_cpqd:
        return ProcedureResult.na("SCORE+-LORD", SCOREPLUS_NA_REASON, assume_cpqd=False)
    proc = ScorePlusLORD(alpha=alpha, omega=omega)
    result = proc.run(evalues)
    result.assume_cpqd = True
    result.protocol_status = "ok"
    result.extra["wealth"] = list(proc.wealth_history)
    result.extra["fdr_guarantee"] = True
    return result


def score_plus_saffron(
    evalues: Sequence[float],
    alpha: float = 0.05,
    omega: OmegaSpec = 0.05,
    lam: OmegaSpec = 0.5,
    *,
    assume_cpqd: bool = False,
) -> ProcedureResult:
    if not assume_cpqd:
        return ProcedureResult.na("SCORE+-SAFFRON", SCOREPLUS_NA_REASON, assume_cpqd=False)
    proc = ScorePlusSAFFRON(alpha=alpha, omega=omega, lam=lam)
    result = proc.run(evalues)
    result.assume_cpqd = True
    result.protocol_status = "ok"
    result.extra["wealth"] = list(proc.wealth_history)
    result.extra["fdr_guarantee"] = True
    return result


DISPATCH = {
    "e-LOND": e_lond,
    "SCORE-LOND": score_lond,
    "e-LORD": e_lord,
    "SCORE-LORD": score_lord,
    "e-SAFFRON": e_saffron,
    "SCORE-SAFFRON": score_saffron,
    "SCORE+-LORD": score_plus_lord,
    "SCORE+-SAFFRON": score_plus_saffron,
}


def run_named(name: str, evalues: Sequence[float], **kwargs) -> ProcedureResult:
    if name not in DISPATCH:
        raise KeyError(f"unknown SCORE method {name!r}; known={sorted(DISPATCH)}")
    return DISPATCH[name](evalues, **kwargs)
