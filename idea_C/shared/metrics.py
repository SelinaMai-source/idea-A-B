"""FDR / TPR / TPP / Power / SupFDR for Idea C v0.

These are *evaluation* functionals. They may look at ground-truth null /
non-null labels. Sampling rules and e-processes must never import this
module for decisions.

Definitions (single rejection set ``S``, truth partition ``H0``, ``H1``):

- **FDP** = ``|S ∩ H0| / max(|S|, 1)``
- **FDR** = ``E[FDP]``. In code, ``fdr`` on one set is the realized FDP;
  Monte Carlo FDR is the mean of per-trial FDP.
- **TPP** = ``|S ∩ H1| / max(|H1|, 1)`` (true positive *proportion*,
  AAAI Def. 2 without the outer expectation; e-PS §6 reports averaged TPP).
- **TPR** = ``E[TPP]``. On one set, ``tpr`` equals TPP. When ``|H1| = 0``,
  both are 0.0 (no true positives exist).
- **Power** = same as TPR / TPP on a fixed set: the fraction of non-nulls
  that are rejected. Streaming papers (SCORE / UAI) average this over time;
  we expose ``mean_power_path`` for that.
- **SupFDR** = ``E[ sup_t FDP(S_t) ]``. Requires a *path* of rejection sets.
  Do not mix SupFDR with FDR in tables.

``samples_at_target_tpr`` is the first time index ``t`` (1-based sample
count) at which TPP ≥ ``1 - delta``. Returns NaN if the target is never
met. An unreached trial is **right-censored** (``CENSORED``), not dropped,
not 0, and not ``NOT_RUN``. Aggregate with ``summarize_samples_at_target``;
do not average only the successful seeds. If success rates differ across
methods, do not rank by the conditional mean among successes alone.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

CENSORED = "CENSORED"
NOT_RUN = "NOT_RUN"
UNDEFINED = "undefined"
NOT_IDENTIFIABLE = "not_identifiable"


def _as_bool_mask(values: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == bool and arr.shape == (n,):
        return arr
    if arr.shape == (n,):
        return arr.astype(bool)
    raise ValueError(f"{name} must have shape ({n},)")


def fdp(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """Realized false discovery proportion of one rejection set."""
    rej = np.asarray(rejected, dtype=bool).reshape(-1)
    null = _as_bool_mask(np.asarray(is_null), rej.size, "is_null")
    n_rej = int(rej.sum())
    if n_rej == 0:
        return 0.0
    return float(np.count_nonzero(rej & null) / n_rej)


def fdr(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """FDR on a single set equals FDP; average across trials separately."""
    return fdp(rejected, is_null)


def tpp(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """True positive proportion ``|S ∩ H1| / max(|H1|, 1)``."""
    rej = np.asarray(rejected, dtype=bool).reshape(-1)
    null = _as_bool_mask(np.asarray(is_null), rej.size, "is_null")
    n_alt = int((~null).sum())
    if n_alt == 0:
        return 0.0
    return float(np.count_nonzero(rej & ~null) / n_alt)


def tpr(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """Single-set TPR (equals TPP). Monte Carlo TPR is the mean of TPP."""
    return tpp(rejected, is_null)


def power(rejected: np.ndarray, is_null: np.ndarray) -> float:
    """Fraction of non-nulls rejected; identical to TPP on one set."""
    return tpp(rejected, is_null)


def mean_power_path(rejected_path: np.ndarray, is_null: np.ndarray) -> float:
    """Time-average of Power along a ``(T, K)`` rejection path."""
    path = np.asarray(rejected_path, dtype=bool)
    if path.ndim != 2:
        raise ValueError("rejected_path must be (T, K)")
    if path.size == 0:
        return 0.0
    return float(np.mean([power(row, is_null) for row in path]))


def sup_fdp(rejected_path: np.ndarray, is_null: np.ndarray) -> float:
    """``sup_t FDP(S_t)`` along a ``(T, K)`` path (one trial)."""
    path = np.asarray(rejected_path, dtype=bool)
    if path.ndim != 2:
        raise ValueError("rejected_path must be (T, K)")
    if path.size == 0:
        return 0.0
    return float(max(fdp(row, is_null) for row in path))


def sup_fdr(rejected_paths: Iterable[np.ndarray], is_null: np.ndarray) -> float:
    """Monte Carlo SupFDR: mean of per-trial ``sup_t FDP``."""
    values = [sup_fdp(path, is_null) for path in rejected_paths]
    if not values:
        return float("nan")
    return float(np.mean(values))


def samples_at_target_tpr(
    tpp_path: np.ndarray,
    delta: float = 0.05,
    sample_index: np.ndarray | None = None,
) -> float:
    """First sample count at which TPP ≥ ``1 - delta``.

    Returns ``nan`` if the target is never reached. Call
    ``summarize_samples_at_target`` so the unreached trial is serialized as
    ``CENSORED`` rather than dropped, 0, or ``NOT_RUN``.
    """
    path = np.asarray(tpp_path, dtype=float).reshape(-1)
    target = 1.0 - float(delta)
    hits = np.flatnonzero(path >= target)
    if hits.size == 0:
        return float("nan")
    idx = int(hits[0])
    if sample_index is None:
        return float(idx + 1)
    index = np.asarray(sample_index, dtype=float).reshape(-1)
    return float(index[idx])


def _is_reached_time(value: Any) -> bool:
    """Finite positive reaching time. 0 / NaN / None are not successes."""
    if value is None:
        return False
    if isinstance(value, str):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0.0)


def summarize_samples_at_target(
    reaching_times: Sequence[Any],
    *,
    target_tpr: float,
    max_horizon: float,
    method: str | None = None,
) -> dict[str, Any]:
    """Right-censored Samples@Target summary for one method.

    Unreached trials stay in the denominator as ``CENSORED``. The
    conditional mean is reported only among successes and is **not** a
    ranking statistic when success rates differ across methods.
    """
    raw = list(reaching_times)
    n_trials = len(raw)
    reached = [float(value) for value in raw if _is_reached_time(value)]
    n_reached = len(reached)
    n_censored = n_trials - n_reached
    success_rate = float(n_reached / n_trials) if n_trials else float("nan")
    censoring_rate = float(n_censored / n_trials) if n_trials else float("nan")
    if n_reached:
        conditional_mean: float | str = float(np.mean(reached))
    else:
        conditional_mean = UNDEFINED

    # Identifiable if the failure-time median is observed on or before the
    # horizon. All-reached uses the ordinary sample median. With censoring,
    # treat unreached as +∞ and take the ((n+1)//2)-th order statistic.
    ordered: list[float] = sorted(reached + [float("inf")] * n_censored)
    median: float | str = NOT_IDENTIFIABLE
    if n_trials and n_censored == 0:
        median = float(np.median(np.asarray(reached, dtype=float)))
    elif n_trials:
        rank = (n_trials + 1) // 2
        candidate = ordered[rank - 1]
        if np.isfinite(candidate):
            median = float(candidate)

    per_trial: list[float | str] = []
    for value in raw:
        if _is_reached_time(value):
            per_trial.append(float(value))
        else:
            per_trial.append(CENSORED)

    return {
        "method": method,
        "target_tpr": float(target_tpr),
        "n_trials": int(n_trials),
        "n_reached": int(n_reached),
        "success_rate": success_rate,
        "conditional_mean_among_successes": conditional_mean,
        "median_reaching_time": median,
        "right_censoring_rate": censoring_rate,
        "max_horizon": float(max_horizon),
        "unreached_status": CENSORED,
        "per_trial_status": per_trial,
        "rank_by_conditional_mean_alone": False,
    }


def can_rank_by_conditional_mean(
    summaries: Sequence[dict[str, Any]],
    *,
    atol: float = 1e-12,
) -> bool:
    """False when success rates differ: do not rank by conditional mean alone."""
    rates: list[float] = []
    for summary in summaries:
        rate = summary.get("success_rate")
        if rate is None or not np.isfinite(rate):
            return False
        rates.append(float(rate))
    if len(rates) < 2:
        return True
    return bool((max(rates) - min(rates)) <= float(atol))


def samples_at_target_metric_cells(summary: dict[str, Any]) -> dict[str, Any]:
    """Metric cells for ``write_run``. Partial success → ``CENSORED``, never 0."""
    n_trials = int(summary.get("n_trials") or 0)
    n_reached = int(summary.get("n_reached") or 0)
    if n_trials <= 0:
        scalar: float | str = NOT_RUN
    elif n_reached < n_trials:
        scalar = CENSORED
    else:
        scalar = summary["conditional_mean_among_successes"]
    return {
        "Samples@Target TPR↓": scalar,
        "target_tpr": summary["target_tpr"],
        "n_trials": summary["n_trials"],
        "n_reached": summary["n_reached"],
        "success_rate": summary["success_rate"],
        "conditional_mean_among_successes": summary["conditional_mean_among_successes"],
        "median_reaching_time": summary["median_reaching_time"],
        "right_censoring_rate": summary["right_censoring_rate"],
        "max_horizon": summary["max_horizon"],
        "unreached_status": CENSORED,
        "rank_by_conditional_mean_alone": False,
    }
