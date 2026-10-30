"""Uniform Exploration.

Generic control for the Idea B synthetic protocol. This is not an official
reproduction of Performative Bandit (ICML 2024) or Performative SGD
(NeurIPS 2024). Every query receives the same audit probability equal to the
feedback budget.
"""

from __future__ import annotations

from typing import Any

METHOD_ID = "UniformExploration"
GENERIC_CONTROL = True
OFFICIAL_REPRODUCTION = False
LABEL = "Uniform Exploration (generic control)"


def make_method(**overrides: Any) -> dict[str, Any]:
    spec = {
        "method_name": METHOD_ID,
        "generic_control": True,
        "official_reproduction": False,
        "label": LABEL,
    }
    spec.update(overrides)
    return spec
