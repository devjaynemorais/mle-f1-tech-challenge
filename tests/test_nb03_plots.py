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
    plot_confusion_matrices_grid,
    plot_learning_curves_grid,
    plot_pr_curves_grid,
    plot_probability_histograms_grid,
    plot_roc_curves_grid,
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
