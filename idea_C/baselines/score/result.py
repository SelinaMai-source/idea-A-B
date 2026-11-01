"""Procedure result objects for SCORE algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CODE_STATUS = "independent_reimplementation"
PAPER = "SCORE: A Unified Framework for Overshoot Refund in Online FDR Control"
VENUE = "ICML 2026 Main"
ARXIV = "2601.20386"


@dataclass
class ProcedureResult:
    """Output of one online pass over an e-value stream.

    ``status='N/A'`` is used when SCORE+ is requested without CPQD.
    Arrays are empty in that case so later agents can skip the method
    without inventing numbers.
    """

    method: str
    status: str  # "ok" | "N/A"
    code_status: str = CODE_STATUS
    rejected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    thresholds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    evalues: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    overshoots: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    costs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    n_rejections: int = 0
    assume_cpqd: bool | None = None
    na_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def na(cls, method: str, reason: str, *, assume_cpqd: bool = False) -> ProcedureResult:
        return cls(
            method=method,
            status="N/A",
            assume_cpqd=assume_cpqd,
            na_reason=reason,
        )

    @classmethod
    def from_traces(
        cls,
        method: str,
        evalues: list[float] | np.ndarray,
        thresholds: list[float] | np.ndarray,
        rejected: list[bool] | np.ndarray,
        overshoots: list[float] | np.ndarray,
        costs: list[float] | np.ndarray,
        *,
        assume_cpqd: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ProcedureResult:
        rej = np.asarray(rejected, dtype=bool)
        return cls(
            method=method,
            status="ok",
            rejected=rej,
            thresholds=np.asarray(thresholds, dtype=float),
            evalues=np.asarray(evalues, dtype=float),
            overshoots=np.asarray(overshoots, dtype=float),
            costs=np.asarray(costs, dtype=float),
            n_rejections=int(np.sum(rej)),
            assume_cpqd=assume_cpqd,
            extra=extra or {},
        )


SCOREPLUS_NA_REASON = (
    "SCORE+ uses the retroactive denominator R_t ∨ 1 and is valid only under "
    "Conditional Positive Quadrant Dependence (CPQD, paper Assumption 5.1 / §5). "
    "Independent e-values plus monotone threshold updates are a sufficient "
    "condition (Appendix A). This procedure returns N/A rather than silently "
    "running SCORE+ when assume_cpqd is False."
)
