"""Result objects and honest code-status labels for UAI 2026 methods.

UAI 2026 is NOT CCF-A Main.
Offline neilzxu/eclosure is arXiv:2509.02517, not this paper.
evalue-omt is AISTATS 2024 e-LOND, not UAI closed/donation e-LOND.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

VENUE = "UAI 2026 (NOT CCF-A Main)"
PAPER = (
    "Improving online FDR procedures via online analogs of e-closure "
    "and compound e-values (Xu, Fischer, Ramdas)"
)

# Online UAI methods implemented here.
ONLINE_CODE_STATUS = "independent_reimplementation"
# AISTATS 2024 LOND wrapper (related, different gamma / finite-horizon).
AISTATS_ELOND_STATUS = "aistats2024_official_wrapper"
# Offline e-Closure package (related, not UAI online).
OFFLINE_ECLOSURE_STATUS = "offline_eclosure_official_clone_arxiv2509.02517"


@dataclass
class ProcedureResult:
    method: str
    status: str  # "ok" | "unavailable"
    code_status: str
    rejected: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    thresholds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    n_rejections: int = 0
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        method: str,
        rejected: np.ndarray,
        thresholds: np.ndarray,
        code_status: str,
        extra: dict[str, Any] | None = None,
    ) -> ProcedureResult:
        rej = np.asarray(rejected, dtype=bool)
        return cls(
            method=method,
            status="ok",
            code_status=code_status,
            rejected=rej,
            thresholds=np.asarray(thresholds, dtype=float),
            n_rejections=int(np.sum(rej)),
            extra=extra or {},
        )

    @classmethod
    def unavailable(cls, method: str, reason: str, code_status: str) -> ProcedureResult:
        return cls(method=method, status="unavailable", code_status=code_status, reason=reason)
