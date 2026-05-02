import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.plots import (
    plot_calibration_curves_grid,
    plot_confusion_matrix_threshold_comparison,
    plot_confusion_matrices_grid,
    plot_fairness_feature_subplot,
    plot_holdout_pr_roc_subplot,
    plot_learning_curves_grid,
    plot_pr_curves_grid,
    plot_probability_histograms_grid,
    plot_retention_vs_roi,
    plot_roi_by_threshold,
    plot_roi_heatmap,
    plot_roc_curves_grid,
    plot_single_confusion_matrix,
    plot_single_probability_histogram,
)


def _build_oof_dict():
    return {
        name: pd.DataFrame(
            {
                "y_true": [0, 0, 1, 1, 0, 1],
                "proba": probs,
                "pred_0_5": [int(p >= 0.5) for p in probs],
            }
        )
        for name, probs in {
            "LogisticRegression": [0.1, 0.2, 0.8, 0.9, 0.4, 0.7],
            "MLP": [0.15, 0.35, 0.75, 0.85, 0.45, 0.8],
            "MLP Optuna": [0.05, 0.25, 0.78, 0.92, 0.32, 0.82],
            "XGBoost Optuna": [0.12, 0.3, 0.7, 0.88, 0.38, 0.79],
        }.items()
    }


def test_plot_roc_and_pr_curves_grid_return_figures():
    oof_predictions = _build_oof_dict()

    roc_fig = plot_roc_curves_grid(oof_predictions, show=False)
    pr_fig = plot_pr_curves_grid(oof_predictions, show=False)

    assert roc_fig is not None
    assert pr_fig is not None
    assert len(roc_fig.axes) == 4
    assert len(pr_fig.axes) == 4


def test_plot_confusion_calibration_and_histogram_grids_return_figures():
    oof_predictions = _build_oof_dict()

    cm_fig = plot_confusion_matrices_grid(oof_predictions, show=False)
    calib_fig = plot_calibration_curves_grid(oof_predictions, show=False)
    hist_fig = plot_probability_histograms_grid(oof_predictions, show=False)

    assert cm_fig is not None
    assert calib_fig is not None
    assert hist_fig is not None
    assert len(cm_fig.axes) == 4
    assert len(calib_fig.axes) == 4
    assert len(hist_fig.axes) == 4


def test_plot_learning_curves_grid_returns_figure():
    X_arr, y_arr = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.6, 0.4],
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))]
        ),
        "MLP": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200, C=0.8))]
        ),
        "MLP Optuna": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200, C=1.2))]
        ),
        "XGBoost Optuna": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200, C=1.5))]
        ),
    }

    fig = plot_learning_curves_grid(
        models=models,
        X=X,
        y=y,
        cv=cv,
        show=False,
        train_sizes=np.array([0.4, 0.7, 1.0]),
    )

    assert fig is not None
    assert len(fig.axes) == 4


def test_plot_threshold_and_stress_charts_return_figures():
    threshold_df = pd.DataFrame(
        {
            "threshold": [0.1, 0.3, 0.5, 0.7],
            "roi": [-0.3, 0.1, 0.05, -0.2],
        }
    )
    retention_df = pd.DataFrame(
        {
            "retention_rate": [0.0, 0.1, 0.2, 0.3],
            "roi": [-1.0, -0.5, 0.1, 0.3],
        }
    )
    heatmap_df = pd.DataFrame(
        {
            "campaign_cost": [100000, 100000, 200000, 200000],
            "retention_rate": [0.1, 0.2, 0.1, 0.2],
            "roi": [0.2, 0.4, -0.1, 0.05],
        }
    )

    threshold_fig = plot_roi_by_threshold(threshold_df, show=False)
    retention_fig = plot_retention_vs_roi(retention_df, show=False)
    heatmap_fig = plot_roi_heatmap(heatmap_df, show=False)

    assert threshold_fig is not None
    assert retention_fig is not None
    assert heatmap_fig is not None
    assert len(threshold_fig.axes) == 1
    assert len(retention_fig.axes) == 1
    assert len(heatmap_fig.axes) >= 1


def test_plot_holdout_and_threshold_comparison_figures_return_expected_axes():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.4, 0.8, 0.9, 0.55, 0.72])
    y_pred = (y_prob >= 0.5).astype(int)

    cm_single = plot_single_confusion_matrix(y_true, y_pred, title="Holdout", show=False)
    pr_roc = plot_holdout_pr_roc_subplot(y_true, y_prob, title="MLP Optuna", show=False)
    hist = plot_single_probability_histogram(y_prob, title="Holdout", show=False)
    cm_compare = plot_confusion_matrix_threshold_comparison(
        y_true,
        y_prob,
        threshold_a=0.5,
        threshold_b=0.7,
        show=False,
    )

    assert cm_single is not None
    assert pr_roc is not None
    assert hist is not None
    assert cm_compare is not None
    assert len(pr_roc.axes) == 2
    assert len(cm_compare.axes) == 2


def test_plot_fairness_feature_subplot_returns_three_axes():
    by_group = pd.DataFrame(
        {
            "group": ["A", "B"],
            "selection_rate": [0.3, 0.5],
            "recall": [0.7, 0.8],
            "precision": [0.6, 0.75],
            "f1_score": [0.64, 0.77],
            "fpr": [0.2, 0.1],
            "fnr": [0.3, 0.2],
        }
    )

    fig = plot_fairness_feature_subplot(
        by_group,
        feature_name="Contract",
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) == 3
