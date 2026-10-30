"""Full Feedback Oracle.

Generic control / upper bound, not an official reproduction. Every query is
observed. The oracle must not be given *less* information than a budgeted
baseline on the same feature sequence.
"""

from __future__ import annotations

from typing import Any

METHOD_ID = "FullFeedbackOracle"
GENERIC_CONTROL = True
OFFICIAL_REPRODUCTION = False
LABEL = "Full Feedback Oracle (generic control)"


def make_method(**overrides: Any) -> dict[str, Any]:
    spec = {
        "method_name": METHOD_ID,
        "generic_control": True,
        "official_reproduction": False,
        "label": LABEL,
        "budget_override": 1.0,
    }
    spec.update(overrides)
    return spec
