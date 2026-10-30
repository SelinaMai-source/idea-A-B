"""Lazy Deployment.

Generic control, not an official reproduction of Performative SGD
(NeurIPS 2024). Sampling follows the greedy rule, but parameters update only
every ``lazy_period`` rounds, so extra labels need not change the deployed
policy immediately.
"""

from __future__ import annotations

from typing import Any

METHOD_ID = "LazyDeployment"
GENERIC_CONTROL = True
OFFICIAL_REPRODUCTION = False
LABEL = "Lazy Deployment (generic control)"
LAZY_PERIOD = 5


def make_method(**overrides: Any) -> dict[str, Any]:
    spec = {
        "method_name": METHOD_ID,
        "generic_control": True,
        "official_reproduction": False,
        "label": LABEL,
        "lazy_period": LAZY_PERIOD,
    }
    spec.update(overrides)
    return spec
