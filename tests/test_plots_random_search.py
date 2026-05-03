import matplotlib
matplotlib.use("Agg")

import pandas as pd

from src.utils.plots import (
    plot_random_search_boxplots,
    plot_random_search_heatmap,
    plot_random_search_ordered_bin_lineplot,
    plot_random_search_summary_bars,
)


def test_plot_random_search_boxplots_returns_figure():
    df = pd.DataFrame(
        {
            "selector": ["f_classif", "mutual_info", "f_classif", "mutual_info"],
            "activation": ["relu", "relu", "tanh", "tanh"],
            "pr_auc_mean": [0.80, 0.82, 0.84, 0.83],
        }
    )

    fig = plot_random_search_boxplots(
        df,
        params=["selector", "activation"],
        metric="pr_auc_mean",
        n_cols=2,
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) >= 2


def test_plot_random_search_summary_bars_returns_figure():
    summary_df = pd.DataFrame(
        {
            "param_value": ["0.0", "0.1", "0.2"],
            "pr_auc_mean_median": [0.81, 0.84, 0.83],
            "top_20pct_rate": [0.10, 0.50, 0.40],
        }
    )

    fig = plot_random_search_summary_bars(
        summary_df,
        x_col="param_value",
        y_col="pr_auc_mean_median",
        title="Dropout",
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_random_search_ordered_bin_lineplot_returns_figure():
    summary_df = pd.DataFrame(
        {
            "lr_bin": ["1e-4", "3e-4", "1e-3"],
            "pr_auc_mean_median": [0.81, 0.83, 0.85],
        }
    )

    fig = plot_random_search_ordered_bin_lineplot(
        summary_df,
        x_col="lr_bin",
        y_col="pr_auc_mean_median",
        title="Learning rate",
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_random_search_heatmap_masks_low_support_cells():
    interaction_df = pd.DataFrame(
        {
            "activation": ["relu", "relu", "tanh"],
            "dropout_bin": ["0.0", "0.1", "0.1"],
            "pr_auc_mean_median": [0.81, 0.84, 0.85],
            "count": [4, 2, 5],
        }
    )

    fig = plot_random_search_heatmap(
        interaction_df,
        x_col="dropout_bin",
        y_col="activation",
        value_col="pr_auc_mean_median",
        count_col="count",
        min_count=3,
        title="Activation x Dropout",
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) >= 1
