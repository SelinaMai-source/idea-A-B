"""Standard Online SGD.

Generic control, not an official reproduction. Uniform audit probabilities
with a one-pass squared-loss SGD update instead of cumulative ridge.
"""

from __future__ import annotations

from typing import Any

METHOD_ID = "StandardOnlineSGD"
GENERIC_CONTROL = True
OFFICIAL_REPRODUCTION = False
LABEL = "Standard Online SGD (generic control)"


def make_method(**overrides: Any) -> dict[str, Any]:
    spec = {
        "method_name": METHOD_ID,
        "generic_control": True,
        "official_reproduction": False,
        "label": LABEL,
        "update_rule": "sgd",
    }
    spec.update(overrides)
    return spec
