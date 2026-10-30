"""Uncertainty-based Exploration.

Generic control, not an official reproduction. Audit probabilities are
proportional to predictive leverage ``phi^T (W + lambda I)^{-1} phi``.
"""

from __future__ import annotations

from typing import Any

METHOD_ID = "UncertaintyExploration"
GENERIC_CONTROL = True
OFFICIAL_REPRODUCTION = False
LABEL = "Uncertainty-based Exploration (generic control)"


def make_method(**overrides: Any) -> dict[str, Any]:
    spec = {
        "method_name": METHOD_ID,
        "generic_control": True,
        "official_reproduction": False,
        "label": LABEL,
    }
    spec.update(overrides)
    return spec
