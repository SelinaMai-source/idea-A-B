"""Deterministic seeding for Idea C v0.

Same integer seed must reproduce Python ``random``, the NumPy legacy
global RNG, and a fresh ``numpy.random.Generator``. PyTorch is *not*
seeded here: the Idea C venv is CPU NumPy/SciPy only.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed process RNGs and return ``np.random.default_rng(seed)``.

    Parameters
    ----------
    seed:
        Non-negative integer. Booleans are rejected.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an int")
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("seed must be >= 0")

    random.seed(seed_int)
    np.random.seed(seed_int)
    return np.random.default_rng(seed_int)


def spawn_stream(rng: np.random.Generator) -> np.random.Generator:
    """Independent child Generator from ``rng.spawn(1)`` (NumPy ≥ 1.25)."""
    children = rng.spawn(1)
    return children[0]
