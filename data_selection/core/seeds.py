"""Deterministic seeding for the CPU selectors. Does not seed torch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be >= 0")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
