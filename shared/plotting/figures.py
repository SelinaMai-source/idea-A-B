"""Matplotlib helpers that plot caller-supplied data only.

Every helper requires a title, axis labels, a legend, and ``n_seeds``. The
seed count is drawn as a subtitle under the title. Units belong in ``xlabel``
and ``ylabel``; these helpers do not invent units, series, or error bars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_n_seeds(n_seeds: Any) -> int:
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, (int, np.integer)):
        raise TypeError("n_seeds must be a positive integer")
    count = int(n_seeds)
    if count < 1:
        raise ValueError("n_seeds must be >= 1")
    return count


def _series_matrix(
    series: Mapping[str, Sequence[Any]],
    length: int,
    yerr: Mapping[str, Sequence[Any]] | None,
) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    if not isinstance(series, Mapping) or len(series) == 0:
        raise ValueError("series must be a non-empty mapping of legend label to values")
    if yerr is not None and set(yerr) != set(series):
        raise ValueError("yerr must provide one sequence for every series label")
    prepared: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for label, values in series.items():
        text = str(label)
        if not text.strip():
            raise ValueError("legend labels must be non-empty")
        array = np.asarray(values, dtype=float)
        if array.shape != (length,):
            raise ValueError(f"series {text!r} must have shape ({length},), got {array.shape}")
        error = None
        if yerr is not None:
            error = np.asarray(yerr[label], dtype=float)
            if error.shape != (length,):
                raise ValueError(f"yerr[{text!r}] must have shape ({length},), got {error.shape}")
        prepared.append((text, array, error))
    return prepared


def annotate_axes(ax: Any, title: str, xlabel: str, ylabel: str, n_seeds: int) -> None:
    """Set title, axis labels, legend, and an ``n_seeds`` subtitle.

    Labeled artists must already exist on ``ax``. The legend is built from
    those labels. ``n_seeds`` is written as a subtitle between the axes and
    the title; it is not inferred from the data.
    """
    title_text = _require_text("title", title)
    xlabel_text = _require_text("xlabel", xlabel)
    ylabel_text = _require_text("ylabel", ylabel)
    count = _require_n_seeds(n_seeds)

    ax.set_title(title_text, pad=20)
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)
    handles, labels = ax.get_legend_handles_labels()
    if not handles or not any(str(label).strip() for label in labels):
        raise ValueError("legend is required: plot labeled artists before annotation")
    legend = ax.legend(handles, labels)
    if legend is None:
        raise ValueError("legend is required")

    for text in list(ax.texts):
        if text.get_gid() == "n_seeds_subtitle":
            text.remove()
    subtitle = ax.text(
        0.5,
        1.01,
        f"n_seeds={count}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        clip_on=False,
    )
    subtitle.set_gid("n_seeds_subtitle")
    ax.figure.subplots_adjust(top=0.82, bottom=0.18)


def plot_series(
    x: Sequence[Any],
    series: Mapping[str, Sequence[Any]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    n_seeds: int,
    yerr: Mapping[str, Sequence[Any]] | None = None,
    path: str | Path | None = None,
):
    """Plot one labeled line per series. Missing error bars are omitted."""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or x_arr.size == 0:
        raise ValueError("x must be a non-empty 1-D numeric vector")
    prepared = _series_matrix(series, int(x_arr.shape[0]), yerr)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, values, error in prepared:
        ax.errorbar(x_arr, values, yerr=error, label=label, marker="o", capsize=3)
    annotate_axes(ax, title, xlabel, ylabel, n_seeds)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150)
    return fig


def plot_bars(
    labels: Sequence[Any],
    series: Mapping[str, Sequence[Any]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    n_seeds: int,
    yerr: Mapping[str, Sequence[Any]] | None = None,
    path: str | Path | None = None,
):
    """Grouped bars. ``labels`` are category tick labels supplied by the caller."""
    tick_labels = [str(label) for label in labels]
    if len(tick_labels) == 0:
        raise ValueError("labels must be non-empty")
    prepared = _series_matrix(series, len(tick_labels), yerr)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    positions = np.arange(len(tick_labels), dtype=float)
    width = 0.8 / len(prepared)
    for index, (label, values, error) in enumerate(prepared):
        offset = (index - (len(prepared) - 1) / 2.0) * width
        ax.bar(
            positions + offset,
            values,
            width=width,
            label=label,
            yerr=error,
            capsize=3,
        )
    ax.set_xticks(positions, tick_labels)
    annotate_axes(ax, title, xlabel, ylabel, n_seeds)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150)
    return fig
