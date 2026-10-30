"""Hájek / SNIPS with the known audit propensity ``q``.

Statistical baseline adapter. Calls ``shared.metrics.estimators.snips_mean``.
This is not a clone of an official repository.
"""

from __future__ import annotations

import numpy as np

from shared.metrics.estimators import snips_mean


def estimate(y_visible: np.ndarray, a: np.ndarray, q: np.ndarray, clip: float | None = None) -> float:
    return float(snips_mean(y_visible, a, q, clip=clip))
