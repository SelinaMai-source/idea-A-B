"""Uniform random audit: ``q_t = b`` for every unit.

Statistical baseline, not an official cloned repository.
"""

from __future__ import annotations

import numpy as np


def allocate(n: int, budget: float) -> np.ndarray:
    return np.full(int(n), float(budget), dtype=float)
