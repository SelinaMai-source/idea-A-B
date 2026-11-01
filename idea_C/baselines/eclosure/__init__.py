"""UAI 2026 online e-closure / donation (Xu, Fischer, Ramdas).

Venue is UAI 2026 (**NOT CCF-A Main**).

Official clones (gitignored trees, see MANIFEST):
    official/     neilzxu/eclosure @ 8bbd5a37 — OFFLINE e-Closure
                  (arXiv:2509.02517). Public API closedeBH / closedBY.
                  NOT UAI 2026 online closed/donation e-LOND.
    evalue-omt/   neilzxu/evalue-omt @ cda6e996 — AISTATS 2024 e-LOND.
                  Experiment name lag_comp is related, not this paper's DGP.

Online methods in this package are independent_reimplementation.
"""

from .dgp import (
    PAPER_DELTA,
    PAPER_LAG,
    PAPER_M,
    PAPER_MU0,
    PAPER_MU1,
    PAPER_RHO,
    PAPER_TRIALS,
    RELATED_OFFICIAL_EXP,
    UAI_VENUE_NOTE,
    LocalDependenceStream,
    local_dependent_gaussian,
)
from .metrics import (
    average_power_terminal,
    empirical_fdr,
    empirical_supfdr,
    fdp_path,
    fdr_terminal,
    sup_fdp,
)
from .official_wrappers import aistats_elond_on_evalues, try_offline_closedeBH
from .online_elond import (
    ClosedELOND,
    DonationELOND,
    ELOND,
    closed_e_lond,
    donation_e_lond,
    e_lond,
)
from .result import (
    AISTATS_ELOND_STATUS,
    OFFLINE_ECLOSURE_STATUS,
    ONLINE_CODE_STATUS,
    PAPER,
    VENUE,
    ProcedureResult,
)
from .rlond import (
    ClosedRLOND,
    DonationRLOND,
    RLOND,
    closed_r_lond,
    donation_r_lond,
    r_lond,
)

# e-LORD is the SCORE/Zhang procedure, reused for P1-B on the same stream.
from idea_C.baselines.score.procedures import ELORD, e_lord  # noqa: E402


def run_named(name: str, values, *, delta: float = 0.1, kind: str = "e", **kwargs):
    """Dispatch online methods. ``kind='e'`` uses e-values; ``kind='p'`` uses p-values."""
    key = name.strip()
    e_map = {
        "e-LOND": e_lond,
        "closed-e-LOND": closed_e_lond,
        "Closed e-LOND": closed_e_lond,
        "donation-e-LOND": donation_e_lond,
        "Donation e-LOND": donation_e_lond,
        "e-LORD": lambda ev, **kw: e_lord(ev, alpha=kw.pop("alpha", delta), **kw),
    }
    p_map = {
        "r-LOND": r_lond,
        "closed-r-LOND": closed_r_lond,
        "Closed r-LOND": closed_r_lond,
        "donation-r-LOND": donation_r_lond,
        "Donation r-LOND": donation_r_lond,
    }
    if key in e_map:
        return e_map[key](values, delta=delta, **kwargs) if key != "e-LORD" else e_map[key](values, **kwargs)
    if key in p_map:
        return p_map[key](values, delta=delta, **kwargs)
    raise KeyError(f"unknown UAI/online method {name!r}")


__all__ = [
    "AISTATS_ELOND_STATUS",
    "ClosedELOND",
    "ClosedRLOND",
    "DonationELOND",
    "DonationRLOND",
    "ELOND",
    "ELORD",
    "LocalDependenceStream",
    "OFFLINE_ECLOSURE_STATUS",
    "ONLINE_CODE_STATUS",
    "PAPER",
    "PAPER_DELTA",
    "PAPER_LAG",
    "PAPER_M",
    "PAPER_MU0",
    "PAPER_MU1",
    "PAPER_RHO",
    "PAPER_TRIALS",
    "ProcedureResult",
    "RELATED_OFFICIAL_EXP",
    "RLOND",
    "UAI_VENUE_NOTE",
    "VENUE",
    "aistats_elond_on_evalues",
    "average_power_terminal",
    "closed_e_lond",
    "closed_r_lond",
    "donation_e_lond",
    "donation_r_lond",
    "e_lond",
    "e_lord",
    "empirical_fdr",
    "empirical_supfdr",
    "fdp_path",
    "fdr_terminal",
    "local_dependent_gaussian",
    "r_lond",
    "run_named",
    "sup_fdp",
    "try_offline_closedeBH",
]
