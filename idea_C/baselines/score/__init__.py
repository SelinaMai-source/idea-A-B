"""ICML 2026 SCORE (Kuang, Gang, Xia). independent_reimplementation.

No official repository was found. Algorithms follow paper §4.1–4.3 (SCORE)
and §5.2–5.3 (SCORE+). SCORE+ is returned as N/A /
``INVALID_PROTOCOL`` unless ``assume_cpqd=True``; it must not claim an
FDR guarantee without that flag.

Power: AP(t) = expected fraction of arrived non-nulls rejected by time t;
power ratio is pointwise vs the same-family baseline (e-LORD / e-SAFFRON).
"""

from .dgp import (
    PAPER_ALPHA,
    PAPER_LAMBDA,
    PAPER_OMEGA,
    PAPER_PI1,
    PAPER_REPS,
    PAPER_T,
    GaussianMixtureStream,
    independent_gaussian_mixture,
    likelihood_ratio_evalue,
)
from .metrics import (
    average_power_path,
    average_power_terminal,
    fdp_path,
    fdr_terminal,
    power_ratio,
)
from .procedures import (
    DISPATCH,
    ELOND,
    ELORD,
    ESAFFRON,
    ScoreLOND,
    ScoreLORD,
    ScorePlusLORD,
    ScorePlusSAFFRON,
    ScoreSAFFRON,
    e_lond,
    e_lord,
    e_saffron,
    run_named,
    score_lond,
    score_lord,
    score_plus_lord,
    score_plus_saffron,
    score_saffron,
)
from .result import CODE_STATUS, PAPER, VENUE, ProcedureResult
from .wealth import (
    lond_refund,
    overshoot,
    saffron_score_cost,
    score_cost,
    score_plus_lord_alpha,
    score_plus_saffron_alpha,
    score_plus_wealth,
)

__all__ = [
    "CODE_STATUS",
    "DISPATCH",
    "ELOND",
    "ELORD",
    "ESAFFRON",
    "GaussianMixtureStream",
    "PAPER",
    "PAPER_ALPHA",
    "PAPER_LAMBDA",
    "PAPER_OMEGA",
    "PAPER_PI1",
    "PAPER_REPS",
    "PAPER_T",
    "ProcedureResult",
    "ScoreLOND",
    "ScoreLORD",
    "ScorePlusLORD",
    "ScorePlusSAFFRON",
    "ScoreSAFFRON",
    "VENUE",
    "average_power_path",
    "average_power_terminal",
    "e_lond",
    "e_lord",
    "e_saffron",
    "fdp_path",
    "fdr_terminal",
    "independent_gaussian_mixture",
    "likelihood_ratio_evalue",
    "lond_refund",
    "overshoot",
    "power_ratio",
    "run_named",
    "score_cost",
    "score_lond",
    "score_lord",
    "score_plus_lord",
    "score_plus_lord_alpha",
    "score_plus_saffron",
    "score_plus_saffron_alpha",
    "score_plus_wealth",
    "score_saffron",
    "saffron_score_cost",
]
