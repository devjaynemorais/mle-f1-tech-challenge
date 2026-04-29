from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.models.mlp import CityEmbeddingMLP
from src.utils.exp import (
    MLPEmbeddingClassifierWrapper,
    build_k_grid,
    build_city_vocabulary,
    encode_city_ids,
    extract_selected_feature_names,
    format_selected_features_log,
    get_processed_feature_names,
    split_tabular_and_city,
    summarize_grid_search_results,
)


def test_get_processed_feature_names_returns_names_after_preprocessing():
    X = pd.DataFrame(
        {
            "City": ["A", "B", "A"],
            "Tenure Months": [1, 2, 3],
        }
    )
    y = pd.Series([0, 1, 0])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=["object", "category"]),
            ),
            (
                "num",
                "passthrough",
                make_column_selector(dtype_exclude=["object", "category"]),
            ),
        ],
        remainder="drop",
    )

    feature_names = get_processed_feature_names(preprocessor, X, y)

    assert len(feature_names) == 3
    assert any(name.startswith("cat__City_") for name in feature_names)
    assert "num__Tenure Months" in feature_names


def test_build_k_grid_includes_all_when_requested():
    k_grid = build_k_grid(12, min_k=10, include_all=True)

    assert k_grid == [10, 11, 12, "all"]


def test_build_k_grid_raises_for_small_feature_space():
    with pytest.raises(ValueError):
        build_k_grid(8, min_k=10, include_all=True)


def test_extract_selected_feature_names_uses_selector_mask():
    X = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1],
            "f2": [1, 1, 0, 0],
            "f3": [0, 0, 1, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1])

    pipeline = Pipeline(
        [
            ("selector", SelectKBest(score_func=f_classif, k=2)),
        ]
    )
    pipeline.fit(X, y)

    selected_features = extract_selected_feature_names(
        pipeline,
        ["f1", "f2", "f3"],
        selector_step="selector",
    )

    assert len(selected_features) == 2
    assert set(selected_features).issubset({"f1", "f2", "f3"})


def test_summarize_grid_search_results_reads_best_row():
    search = SimpleNamespace(
        best_index_=1,
        best_params_={"selector__score_func": f_classif, "selector__k": 12},
        cv_results_={
            "mean_test_pr_auc": [0.60, 0.70],
            "mean_test_roc_auc": [0.75, 0.80],
            "mean_test_recall": [0.65, 0.85],
            "mean_test_precision": [0.50, 0.55],
            "mean_test_f1": [0.56, 0.66],
            "mean_fit_time": [0.10, 0.20],
            "mean_score_time": [0.01, 0.02],
        },
    )

    summary = summarize_grid_search_results(
        search,
        "MLP",
        selector_label_map={f_classif: "f_classif"},
    )

    assert summary == {
        "model": "MLP",
        "selector": "f_classif",
        "k": 12,
        "pr_auc_mean": 0.70,
        "roc_auc_mean": 0.80,
        "recall_mean": 0.85,
        "precision_mean": 0.55,
        "f1_mean": 0.66,
        "fit_time_mean_s": 0.20,
        "score_time_mean_s": 0.02,
    }


def test_format_selected_features_log_mentions_all_features_when_applicable():
    log_text = format_selected_features_log(
        "XGBoost",
        {"selector__score_func": f_classif, "selector__k": "all"},
        ["f1", "f2"],
    )

    assert "XGBoost" in log_text
    assert "f_classif" in log_text
    assert "all (todas as features processadas foram mantidas)" in log_text
    assert "- f1" in log_text
    assert "- f2" in log_text


def test_build_city_vocabulary_reserves_zero_for_unknown():
    city_to_idx = build_city_vocabulary(pd.Series(["B", "A", "B", None]))

    assert city_to_idx == {"A": 1, "B": 2}


def test_encode_city_ids_maps_unknown_to_zero():
    city_ids = encode_city_ids(
        pd.Series(["A", "Z", None]),
        {"A": 1, "B": 2},
    )

    assert city_ids.tolist() == [1, 0, 0]


def test_split_tabular_and_city_removes_geo_columns_from_tabular_branch():
    X = pd.DataFrame(
        {
            "City": ["A", "B"],
            "Zip Code": [90001, 94105],
            "Latitude": [33.9, 37.7],
            "Longitude": [-118.2, -122.4],
            "Lat Long": ["33.9,-118.2", "37.7,-122.4"],
            "Tenure Months": [1, 2],
        }
    )

    city, X_tab = split_tabular_and_city(X)

    assert city.tolist() == ["A", "B"]
    assert list(X_tab.columns) == ["Tenure Months"]


def test_city_embedding_mlp_forward_returns_single_logit_per_row():
    model = CityEmbeddingMLP(
        input_dim=3,
        n_cities=5,
        embedding_dim=4,
        hidden_dim=8,
        output_dim=1,
    )
    x_tab = torch.randn(2, 3)
    x_city = torch.tensor([1, 3], dtype=torch.long)

    output = model(x_tab, x_city)

    assert output.shape == (2, 1)


def test_mlp_embedding_wrapper_fit_and_predict_proba():
    X = pd.DataFrame(
        {
            "City": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "Zip Code": [90001, 90001, 94105, 94105, 93721, 93721, 90002, 90002, 94107, 94107],
            "Latitude": [33.9, 33.9, 37.7, 37.7, 36.7, 36.7, 34.0, 34.0, 37.8, 37.8],
            "Longitude": [-118.2, -118.2, -122.4, -122.4, -119.7, -119.7, -118.3, -118.3, -122.3, -122.3],
            "Lat Long": ["a"] * 10,
            "Contract": ["Month-to-month", "One year"] * 5,
            "Monthly Charges": [50, 60, 70, 80, 90, 55, 65, 75, 85, 95],
            "Tenure Months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=["object", "category"]),
            ),
            (
                "num",
                "passthrough",
                make_column_selector(dtype_exclude=["object", "category"]),
            ),
        ],
        remainder="drop",
    )

    wrapper = MLPEmbeddingClassifierWrapper(
        preprocessor=preprocessor,
        feature_engineer=None,
        embedding_dim=4,
        hidden_dim=8,
        batch_size=4,
        max_epochs=2,
        patience=1,
        val_size=0.2,
        verbose=False,
    )

    wrapper.fit(X, y)
    proba = wrapper.predict_proba(X)

    assert proba.shape == (10, 2)
    assert set(wrapper.city_to_idx_.keys()).issubset({"A", "B", "C", "D", "E"})
