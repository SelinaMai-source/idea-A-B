"""Fixed Exploration Schedule.

Generic control, not an official reproduction. Audit probability is constant
in time and across queries, equal to the budget.
"""

from __future__ import annotations

from typing import Any

METHOD_ID = "FixedExplorationSchedule"
GENERIC_CONTROL = True
OFFICIAL_REPRODUCTION = False
LABEL = "Fixed Exploration Schedule (generic control)"


def make_method(**overrides: Any) -> dict[str, Any]:
    spec = {
        "method_name": METHOD_ID,
        "generic_control": True,
        "official_reproduction": False,
        "label": LABEL,
    }
    spec.update(overrides)
    return spec
