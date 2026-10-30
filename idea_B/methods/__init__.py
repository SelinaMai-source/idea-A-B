"""Idea B methods. Importable wrappers around ``methods/v1.0``."""

from __future__ import annotations

import sys
from pathlib import Path

_V10 = Path(__file__).resolve().parent / "v1.0"
_V10_STR = str(_V10)
if _V10_STR not in sys.path:
    sys.path.insert(0, _V10_STR)

from algorithm import (  # noqa: E402
    METHOD_FACTORIES,
    classify_collapse,
    run_closed_loop,
)
from feedback_controller import (  # noqa: E402
    allocate_audit,
    make_controller_state,
)
from observability import (  # noqa: E402
    accumulate_conditional_gramian,
    accumulate_observed_gramian,
    condition_number,
    eigenspectrum,
    lambda_min,
    observed_design_gramian,
    skill_information,
)
from policy_update import ridge_update, sgd_update  # noqa: E402

__all__ = [
    "METHOD_FACTORIES",
    "allocate_audit",
    "accumulate_conditional_gramian",
    "accumulate_observed_gramian",
    "classify_collapse",
    "condition_number",
    "eigenspectrum",
    "lambda_min",
    "make_controller_state",
    "observed_design_gramian",
    "ridge_update",
    "run_closed_loop",
    "sgd_update",
    "skill_information",
]
