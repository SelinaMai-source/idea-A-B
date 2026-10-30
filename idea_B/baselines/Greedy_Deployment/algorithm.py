"""Greedy Deployment.

Generic control, not an official reproduction of Performative SGD
(NeurIPS 2024) or Performative Bandit (ICML 2024). The limited budget is
spent on the queries with the highest current predicted quality, which
reinforces already-served directions.
"""

from __future__ import annotations

from typing import Any

METHOD_ID = "GreedyDeployment"
GENERIC_CONTROL = True
OFFICIAL_REPRODUCTION = False
LABEL = "Greedy Deployment (generic control)"


def make_method(**overrides: Any) -> dict[str, Any]:
    spec = {
        "method_name": METHOD_ID,
        "generic_control": True,
        "official_reproduction": False,
        "label": LABEL,
    }
    spec.update(overrides)
    return spec
