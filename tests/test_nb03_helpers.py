import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.exp import (
    artifact_uri_to_local_path,
    build_cross_validate_comparison_table,
    build_default_preprocessor,
    build_campaign_retention_roi_grid,
    build_fairness_report_for_feature,
    build_retention_roi_table,
    calculate_expected_calibration_error,
    compute_campaign_economics,
    compute_holdout_technical_metrics,
    compute_generalization_gap_pct,
    extract_selected_feature_names,
    fit_final_estimator_and_generate_predictions,
    generate_oof_predictions,
    optimize_threshold_for_roi,
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
        activation_cost=50.0,
        retention_rate=0.10,
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["vr"] == pytest.approx(90.0)
    assert metrics["vrec"] == pytest.approx(9.0)
    assert metrics["vp"] == pytest.approx(40.0)
    assert metrics["cmca"] == pytest.approx(50.0)
    assert metrics["total_campaign_cost"] == pytest.approx(100.0)
    assert metrics["vd"] == pytest.approx(50.0)
    assert metrics["iel"] == pytest.approx(-131.0)
    assert metrics["roi"] == pytest.approx(-1.31)


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


def test_optimize_threshold_for_roi_returns_best_threshold_and_metrics():
    y_true = np.array([1, 1, 0, 0], dtype=int)
    y_prob = np.array([0.9, 0.4, 0.6, 0.1], dtype=float)
    cltv = np.array([100.0, 200.0, 150.0, 50.0], dtype=float)

    threshold_df, best_row = optimize_threshold_for_roi(
        y_true=y_true,
        y_prob=y_prob,
        cltv=cltv,
        threshold_grid=np.array([0.3, 0.5, 0.7]),
        activation_cost=50.0,
        retention_rate=0.1,
    )

    assert threshold_df.shape[0] == 3
    assert "roi" in threshold_df.columns
    assert best_row["threshold"] == pytest.approx(0.3)
    assert best_row["roi"] == pytest.approx(threshold_df["roi"].max())


def test_build_retention_roi_table_varies_retention_with_fixed_campaign_cost():
    y_true = np.array([1, 1, 0, 0], dtype=int)
    y_prob = np.array([0.9, 0.4, 0.6, 0.1], dtype=float)
    cltv = np.array([100.0, 200.0, 150.0, 50.0], dtype=float)

    retention_df = build_retention_roi_table(
        y_true=y_true,
        y_prob=y_prob,
        cltv=cltv,
        threshold=0.3,
        activation_cost=50.0,
        retention_values=[0.0, 0.5, 1.0],
    )

    assert retention_df["retention_rate"].tolist() == [0.0, 0.5, 1.0]
    assert retention_df["activation_cost"].nunique() == 1
    assert retention_df["total_campaign_cost"].nunique() == 1
    assert retention_df["vrec"].iloc[0] == pytest.approx(0.0)
    assert retention_df["roi"].iloc[-1] > retention_df["roi"].iloc[0]


def test_build_campaign_retention_roi_grid_builds_cartesian_product():
    y_true = np.array([1, 1, 0, 0], dtype=int)
    y_prob = np.array([0.9, 0.4, 0.6, 0.1], dtype=float)
    cltv = np.array([100.0, 200.0, 150.0, 50.0], dtype=float)

    grid_df = build_campaign_retention_roi_grid(
        y_true=y_true,
        y_prob=y_prob,
        cltv=cltv,
        threshold=0.3,
        activation_cost_values=[25.0, 50.0],
        retention_values=[0.1, 0.2, 0.3],
    )

    assert len(grid_df) == 6
    assert sorted(grid_df["activation_cost"].unique().tolist()) == [25.0, 50.0]
    assert sorted(grid_df["retention_rate"].unique().tolist()) == [0.1, 0.2, 0.3]


def test_fit_final_estimator_and_generate_predictions_returns_holdout_frame():
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

    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )

    fitted_estimator, holdout_df = fit_final_estimator_and_generate_predictions(
        estimator=estimator,
        X_train=X.iloc[:30],
        y_train=y.iloc[:30],
        X_test=X.iloc[30:],
        y_test=y.iloc[30:],
        threshold=0.4,
        model_name="LogisticRegression",
    )

    assert fitted_estimator is not None
    assert len(holdout_df) == 10
    assert set(["row_index", "y_true", "proba", "y_pred", "threshold", "model"]).issubset(
        holdout_df.columns
    )
    assert holdout_df["proba"].between(0, 1).all()


def test_compute_holdout_technical_metrics_returns_expected_keys():
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.3, 0.8, 0.9], dtype=float)

    metrics = compute_holdout_technical_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
    )

    assert set(["pr_auc", "roc_auc", "recall", "precision", "f1_score"]).issubset(metrics.keys())
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)


def test_extract_selected_feature_names_returns_selected_columns():
    X = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "b", "a", "b"],
            "num1": [1, 2, 3, 4, 5, 6],
            "num2": [6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    estimator = Pipeline(
        [
            ("prep", build_default_preprocessor()),
            ("selector", SelectKBest(score_func=f_classif, k=2)),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    estimator.fit(X, y)

    selected = extract_selected_feature_names(estimator, X)

    assert len(selected) == 2
    assert all(isinstance(name, str) for name in selected)


def test_build_fairness_report_for_feature_returns_group_and_summary_tables():
    X_feature = pd.Series(["A", "A", "B", "B", "A", "B"], name="Contract")
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    y_pred = pd.Series([1, 1, 1, 0, 0, 0])

    report = build_fairness_report_for_feature(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_feature=X_feature,
        feature_name="Contract",
    )

    assert report["feature_name"] == "Contract"
    assert {"selection_rate", "recall", "precision", "f1_score", "fpr", "fnr"}.issubset(
        report["by_group"].columns
    )
    assert {"metric", "difference", "ratio"}.issubset(report["summary"].columns)
