"""Process-wide seeds and an independent NumPy Generator.

``PYTHONHASHSEED`` is intentionally left unchanged: CPython only reads it at
startup, so setting it here would not make the running process deterministic.
"""

from __future__ import annotations

import random

import numpy as np

_TORCH = None
_TORCH_TRIED = False


def _maybe_torch():
    global _TORCH, _TORCH_TRIED
    if not _TORCH_TRIED:
        _TORCH_TRIED = True
        try:
            import torch
        except Exception:
            _TORCH = None
        else:
            _TORCH = torch
    return _TORCH


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch (if installed) and return a Generator.

    The returned ``numpy.random.Generator`` is created with
    ``np.random.default_rng(seed)``. It does not share stream state with the
    legacy global ``np.random`` RNG, but both streams are reset from ``seed``
    on every call. The same ``seed`` therefore reproduces both
    ``generator.random`` and subsequent ``np.random`` / ``random`` draws.

    Parameters
    ----------
    seed:
        Non-negative integer. Booleans are rejected.

    Returns
    -------
    np.random.Generator
        Deterministic generator bound to ``seed``.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an int")
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("seed must be >= 0")

    random.seed(seed_int)
    np.random.seed(seed_int)

    torch_mod = _maybe_torch()
    if torch_mod is not None:
        try:
            torch_mod.manual_seed(seed_int)
        except Exception:
            pass
        mps = getattr(torch_mod, "mps", None)
        manual_seed = getattr(mps, "manual_seed", None)
        if callable(manual_seed):
            try:
                manual_seed(seed_int)
            except Exception:
                pass

    return np.random.default_rng(seed_int)
