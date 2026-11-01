"""Monte Carlo mean / std / 95% CI helpers.

These are for *later* P0/P1 aggregation. They must not be used to
fabricate table numbers: if a cell was not run, leave ``NOT_RUN``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats


def mean_std_ci(
    values: Any,
    level: float = 0.95,
) -> dict[str, float]:
    """Return mean, sample std (ddof=1), and a Student-t CI.

    With ``n < 2`` the CI degenerates to ``(nan, nan)`` and std is nan.
    Non-finite inputs are dropped; if nothing remains, all fields are nan.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "level": float(level),
        }
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "n": 1.0,
            "mean": mean,
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "level": float(level),
        }
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)
    tcrit = float(sp_stats.t.ppf(0.5 + 0.5 * float(level), df=n - 1))
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "ci_low": mean - tcrit * se,
        "ci_high": mean + tcrit * se,
        "level": float(level),
    }
