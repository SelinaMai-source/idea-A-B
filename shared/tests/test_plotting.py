"""Figure helpers require labels and plot only the supplied series."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shared.plotting.figures import annotate_axes, plot_bars, plot_series


def _subtitle(ax) -> str:
    matches = [text.get_text() for text in ax.texts if text.get_gid() == "n_seeds_subtitle"]
    assert matches
    return matches[0]


def test_plot_series_keeps_caller_data_and_required_labels(tmp_path):
    destination = tmp_path / "series.png"
    fig = plot_series(
        [0.01, 0.05],
        {"IPW": [0.2, 0.1]},
        title="Audit IPW",
        xlabel="budget",
        ylabel="RMSE",
        n_seeds=3,
        yerr={"IPW": [0.01, 0.02]},
        path=destination,
    )
    ax = fig.axes[0]
    assert ax.get_title() == "Audit IPW"
    assert ax.get_xlabel() == "budget"
    assert ax.get_ylabel() == "RMSE"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["IPW"]
    assert _subtitle(ax) == "n_seeds=3"
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [0.2, 0.1])
    assert destination.stat().st_size > 0
    plt.close(fig)


def test_plot_bars_and_missing_annotations():
    fig = plot_bars(
        ["MCAR", "MNAR"],
        {"observed-only": [0.4, 0.8], "audit IPW": [0.2, 0.25]},
        title="Bias",
        xlabel="mechanism",
        ylabel="absolute bias",
        n_seeds=3,
    )
    ax = fig.axes[0]
    assert _subtitle(ax) == "n_seeds=3"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "observed-only",
        "audit IPW",
    ]
    plt.close(fig)

    empty_fig, empty_ax = plt.subplots()
    with pytest.raises(ValueError):
        annotate_axes(empty_ax, "Title", "x", "y", 3)
    plt.close(empty_fig)
    with pytest.raises(TypeError):
        plot_series([0.0, 1.0], {"A": [1.0, 2.0]}, xlabel="x", ylabel="y", n_seeds=3)
    with pytest.raises(ValueError):
        plot_series([0.0, 1.0], {"A": [1.0, 2.0]}, title=" ", xlabel="x", ylabel="y", n_seeds=3)
