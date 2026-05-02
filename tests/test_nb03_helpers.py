import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.exp import (
    artifact_uri_to_local_path,
    build_cross_validate_comparison_table,
    calculate_expected_calibration_error,
    compute_campaign_economics,
    compute_generalization_gap_pct,
    generate_oof_predictions,
    rebuild_train_val_test_splits,
)


def test_artifact_uri_to_local_path_maps_mlflow_artifacts_uri(tmp_path):
    path = artifact_uri_to_local_path(
        "mlflow-artifacts:/6/run-123/artifacts",
        workspace_root=tmp_path,
        artifact_name="best_params.json",
    )

    assert path == tmp_path / "mlartifacts" / "6" / "run-123" / "artifacts" / "best_params.json"


def test_rebuild_train_val_test_splits_separates_features_target_and_metadata():
    df = pd.DataFrame(
        {
            "CustomerID": ["A", "B", "C"],
            "feature_num": [1, 2, 3],
            "feature_cat": ["x", "y", "z"],
            "CLTV": [1000, 2000, 3000],
            "Churn Value": [0, 1, 0],
        }
    )

    split_bundle = rebuild_train_val_test_splits(
        df=df,
        target_column="Churn Value",
        train_val_idx=[0, 2],
        test_idx=[1],
        metadata_columns=("CLTV", "CustomerID"),
    )

    assert list(split_bundle["X_train_val"].columns) == ["feature_num", "feature_cat"]
    assert split_bundle["y_train_val"].tolist() == [0, 0]
    assert split_bundle["X_test"].shape == (1, 2)
    assert split_bundle["metadata_train_val"]["CLTV"].tolist() == [1000, 3000]
    assert split_bundle["metadata_test"]["CustomerID"].tolist() == ["B"]


def test_compute_generalization_gap_pct_uses_train_over_validation_formula():
    gap = compute_generalization_gap_pct(train_score=0.80, valid_score=0.72)

    assert gap == pytest.approx((1 - (0.80 / 0.72)) * 100)


def test_calculate_expected_calibration_error_is_zero_for_perfect_bin_alignment():
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=2)

    assert ece == pytest.approx(0.15)


def test_compute_campaign_economics_returns_expected_components():
    y_true = np.array([1, 1, 0, 0], dtype=int)
    y_prob = np.array([0.9, 0.2, 0.8, 0.1], dtype=float)
    cltv = np.array([100.0, 200.0, 150.0, 50.0], dtype=float)

    metrics = compute_campaign_economics(
        y_true=y_true,
        y_prob=y_prob,
        cltv=cltv,
        threshold=0.5,
        campaign_cost=1000.0,
        retention_rate=0.10,
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["vr"] == pytest.approx(90.0)
    assert metrics["vrec"] == pytest.approx(9.0)
    assert metrics["vp"] == pytest.approx(40.0)
    assert metrics["cmca"] == pytest.approx(500.0)
    assert metrics["vd"] == pytest.approx(500.0)
    assert metrics["iel"] == pytest.approx(-531.0)
    assert metrics["roi"] == pytest.approx(-1.031)


def test_generate_oof_predictions_returns_one_probability_per_row():
    X_arr, y_arr = make_classification(
        n_samples=40,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.6, 0.4],
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )

    oof_df = generate_oof_predictions(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        model_name="LogisticRegression",
    )

    assert len(oof_df) == len(X)
    assert set(["row_index", "fold", "y_true", "proba", "pred_0_5", "model"]).issubset(oof_df.columns)
    assert oof_df["row_index"].is_unique
    assert oof_df["proba"].between(0, 1).all()
    assert sorted(oof_df["fold"].unique().tolist()) == [1, 2, 3, 4]


def test_build_cross_validate_comparison_table_returns_expected_columns():
    X_arr, y_arr = make_classification(
        n_samples=50,
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
        "LogisticRegressionAlt": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200, C=0.8))]
        ),
    }

    comparison_df = build_cross_validate_comparison_table(
        models,
        X=X,
        y=y,
        cv=cv,
    )

    assert list(comparison_df.columns) == [
        "modelo",
        "PR-AUC",
        "PR-AUC gap (%)",
        "ROC-AUC",
        "ROC-AUC gap (%)",
        "Recall",
        "Recall gap (%)",
        "Precision",
        "Precision gap (%)",
        "F1-Score",
        "F1-Score gap (%)",
        "fit_time(mean)",
        "score_time(mean)",
    ]
    assert len(comparison_df) == 2
