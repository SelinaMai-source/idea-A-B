"""Mean estimators for audited outcomes and for natural observations.

Audit questions must call ``ipw_mean``, ``snips_mean``, or ``dr_mean`` with the
known audit propensity ``q``. ``naive_ips_mean`` is a different function: it
applies Horvitz–Thompson weights to a natural observation mask and a
caller-supplied natural propensity ``p_hat``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _sample_length(y: Any) -> int:
    arr = np.asarray(y)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = np.reshape(arr, -1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"y must be a non-empty 1-D sample, got shape {np.shape(y)}")
    return int(arr.shape[0])


def _as_float_vector(name: str, value: Any, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = np.reshape(arr, -1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1-D vector, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} does not match sample length {n}")
    return np.array(arr, dtype=float, copy=True)


def _as_indicator(name: str, value: Any, n: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be a length-{n} vector of 0/1 indicators")
    if arr.ndim == 2 and 1 in arr.shape:
        arr = np.reshape(arr, -1)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(
            f"{name} must be a length-{n} vector of 0/1 indicators, got shape {arr.shape}"
        )
    if arr.dtype == bool:
        return np.array(arr, dtype=bool, copy=True)
    numeric = np.asarray(arr, dtype=float)
    valid = np.isfinite(numeric) & ((numeric == 0.0) | (numeric == 1.0))
    if not np.all(valid):
        raise ValueError(f"{name} must contain only 0/1 values")
    return numeric == 1.0


def _apply_lower_clip(name: str, values: np.ndarray, clip: float | None) -> np.ndarray:
    """Lower-bound ``values`` only when ``clip`` is explicitly provided.

    ``clip=None`` returns ``values`` unchanged. There is no default floor and
    no upper clip.
    """
    if clip is None:
        return values
    if isinstance(clip, (bool, np.bool_)):
        raise TypeError(f"{name} clip must be a positive float or None")
    if not isinstance(clip, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} clip must be a positive float or None")
    clip_value = float(clip)
    if not math.isfinite(clip_value) or clip_value <= 0.0:
        raise ValueError(f"{name} clip must be a finite lower bound > 0")
    return np.maximum(values, clip_value)


def _prepare_weights(
    y: Any,
    indicator: Any,
    propensity: Any,
    clip: float | None,
    indicator_name: str,
    propensity_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _sample_length(y)
    outcomes = _as_float_vector("y", y, n)
    active = _as_indicator(indicator_name, indicator, n)
    prop = _apply_lower_clip(
        propensity_name,
        _as_float_vector(propensity_name, propensity, n),
        clip,
    )
    invalid = active & (~np.isfinite(prop) | ~(prop > 0.0))
    if np.any(invalid):
        raise ValueError(
            f"{propensity_name} must be finite and > 0 where {indicator_name}==1 "
            "(pass an explicit clip to floor the propensity; there is no silent clip)"
        )
    return outcomes, active, prop


def observed_only_mean(y: Any, observed_mask: Any) -> float:
    """Mean of ``y`` on units with ``observed_mask == 1``.

    Unobserved entries of ``y`` are not read. If no unit is observed, returns
    NaN. This is the sample mean of the observed outcomes, not an inverse-
    propensity estimate of the population mean.
    """
    n = _sample_length(y)
    outcomes = _as_float_vector("y", y, n)
    observed = _as_indicator("observed_mask", observed_mask, n)
    if not np.any(observed):
        return float("nan")
    return float(np.mean(outcomes[observed]))


def ipw_mean(y: Any, a: Any, q: Any, clip: float | None = None) -> float:
    """Horvitz–Thompson / IPW estimate of the population mean.

    .. math::

        \\hat\\mu_{IPW} = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{A_i}{q_i} Y_i

    The summand is 0 when ``A_i = 0``, and ``Y_i`` is not read on those units.
    The divisor is the population size ``n``, including unaudited units.

    Unbiased for the population mean when ``A`` is independent of ``Y`` given
    history and ``E[A] = q``, with ``q > 0`` on the units where that identity
    is required. ``q`` is the known audit propensity.

    Parameters
    ----------
    y, a, q:
        Outcome, audit indicator in ``{0, 1}``, and known audit propensity.
        A scalar ``q`` is broadcast to every unit.
    clip:
        Optional finite lower bound applied to ``q`` via ``maximum(q, clip)``.
        The default ``None`` does not clip. Values at 0 stay 0 unless the
        caller passes ``clip``.
    """
    outcomes, audited, propensity = _prepare_weights(y, a, q, clip, "a", "q")
    terms = np.zeros(outcomes.shape[0], dtype=float)
    terms[audited] = outcomes[audited] / propensity[audited]
    return float(np.mean(terms))


def snips_mean(y: Any, a: Any, q: Any, clip: float | None = None) -> float:
    """Hájek / SNIPS (self-normalized IPW) estimate of the population mean.

    .. math::

        \\hat\\mu_{SNIPS}
        = \\frac{\\sum_i (A_i / q_i) Y_i}{\\sum_i (A_i / q_i)}

    ``Y_i`` is not read where ``A_i = 0``. If every weight is 0, returns NaN.
    ``clip`` has the same meaning as in ``ipw_mean`` and defaults to no clip.
    """
    outcomes, audited, propensity = _prepare_weights(y, a, q, clip, "a", "q")
    weights = np.zeros(outcomes.shape[0], dtype=float)
    weights[audited] = 1.0 / propensity[audited]
    denominator = float(np.sum(weights))
    if denominator == 0.0:
        return float("nan")
    numerator = float(np.sum(weights[audited] * outcomes[audited]))
    return numerator / denominator


def dr_mean(
    y: Any,
    a: Any,
    q: Any,
    mu_hat: Any,
    clip: float | None = None,
) -> float:
    """AIPW / doubly robust estimate of the population mean.

    Standard AIPW:

    .. math::

        \\hat\\mu_{DR}
        = \\frac{1}{n} \\sum_i
          \\left(\\hat\\mu_i + \\frac{A_i}{q_i}(Y_i - \\hat\\mu_i)\\right)

    The residual ``(Y_i - mu_hat_i) / q_i`` is added only where ``A_i == 1``.
    Where ``A_i == 0`` the contribution is ``mu_hat_i`` alone, and the ``y``
    term is not read. ``mu_hat`` is a per-unit outcome prediction. A scalar
    ``mu_hat`` or ``q`` is broadcast across units.

    ``clip`` lower-bounds ``q`` only when the caller sets it. The default is
    ``None`` (no silent clip).
    """
    outcomes, audited, propensity = _prepare_weights(y, a, q, clip, "a", "q")
    prediction = _as_float_vector("mu_hat", mu_hat, outcomes.shape[0])
    contribution = prediction.copy()
    contribution[audited] = prediction[audited] + (
        (outcomes[audited] - prediction[audited]) / propensity[audited]
    )
    return float(np.mean(contribution))


def naive_ips_mean(
    y: Any,
    observed_mask: Any,
    p_hat: Any,
    clip: float | None = None,
) -> float:
    """Horvitz–Thompson mean that treats ``p_hat`` as a natural propensity.

    .. math::

        \\frac{1}{n} \\sum_i \\frac{O_i}{\\hat p_i} Y_i

    ``O`` is the natural observation indicator, not the audit indicator.
    ``p_hat`` is whatever propensity the caller supplies for that natural
    channel. This function does not accept or use an audit propensity ``q``.
    Call ``ipw_mean`` for known-audit IPW.

    ``Y_i`` is not read where ``O_i = 0``. ``clip`` defaults to ``None`` and
    does not silently floor ``p_hat``.
    """
    outcomes, observed, propensity = _prepare_weights(
        y,
        observed_mask,
        p_hat,
        clip,
        "observed_mask",
        "p_hat",
    )
    terms = np.zeros(outcomes.shape[0], dtype=float)
    terms[observed] = outcomes[observed] / propensity[observed]
    return float(np.mean(terms))
