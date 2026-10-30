"""No Feedback Controller.

Generic control, not an official reproduction. The method spends no extra
audit budget and only receives the environment's natural feedback.
"""

from __future__ import annotations

from typing import Any

METHOD_ID = "NoFeedbackController"
GENERIC_CONTROL = True
OFFICIAL_REPRODUCTION = False
LABEL = "No Feedback Controller (generic control)"


def make_method(**overrides: Any) -> dict[str, Any]:
    spec = {
        "method_name": METHOD_ID,
        "generic_control": True,
        "official_reproduction": False,
        "label": LABEL,
        "budget_override": 0.0,
    }
    spec.update(overrides)
    return spec
