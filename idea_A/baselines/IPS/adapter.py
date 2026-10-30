"""Audit IPW / Horvitz–Thompson with the known audit propensity ``q``.

Statistical baseline adapter. Calls ``shared.metrics.estimators.ipw_mean``.
This is not a clone of an official repository.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.metrics.estimators import ipw_mean


def estimate(y_visible: np.ndarray, a: np.ndarray, q: np.ndarray, clip: float | None = None) -> float:
    return float(ipw_mean(y_visible, a, q, clip=clip))


def estimate_estimands(estimands_fn: Any, *args: Any, **kwargs: Any):
    kwargs = dict(kwargs)
    kwargs["estimator"] = "ipw"
    return estimands_fn(*args, **kwargs)
