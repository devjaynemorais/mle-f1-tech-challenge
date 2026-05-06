import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.production import (
    ProductionModelSettings,
    build_inference_dataframe,
    load_production_model,
    load_production_settings,
    materialize_production_model,
    predict_with_threshold,
)


def test_load_production_settings_reads_active_model(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
production:
  active_model: prod_a
  models:
    prod_a:
      display_name: MLP Optuna
      framework: sklearn
      threshold: 0.35
      model_path: models/production/mlp_optuna/model.pkl
      local:
        metadata_path: models/production/mlp_optuna/serving_metadata.json
      pipeline:
        required_columns: [Gender, CLTV]
        optional_columns: [City]
      inference:
        input_path: data/interim/input.csv
        output_path: models/predictions/output.csv
        target_column: Churn Value
""",
        encoding="utf-8",
    )

    settings = load_production_settings(config_path, workspace_root=tmp_path)

    assert settings.model_key == "prod_a"
    assert settings.model_name == "MLP Optuna"
    assert settings.threshold == pytest.approx(0.35)
    assert settings.model_path == tmp_path / "models" / "production" / "mlp_optuna" / "model.pkl"
    assert settings.required_columns == ("Gender", "CLTV")
    assert settings.optional_columns == ("City",)


def test_load_production_settings_normalizes_accidentally_concatenated_columns(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
production:
  active_model: prod_a
  models:
    prod_a:
      display_name: MLP Optuna
      framework: sklearn
      threshold: 0.35
      model_path: models/production/mlp_optuna/model.pkl
      local:
        metadata_path: models/production/mlp_optuna/serving_metadata.json
      pipeline:
        required_columns: [Gender, "Device Protection- Tech Support", CLTV]
        optional_columns: [City]
      inference:
        input_path: data/interim/input.csv
        output_path: models/predictions/output.csv
        target_column: Churn Value
""",
        encoding="utf-8",
    )

    settings = load_production_settings(config_path, workspace_root=tmp_path)

    assert settings.required_columns == (
        "Gender",
        "Device Protection",
        "Tech Support",
        "CLTV",
    )


def test_build_inference_dataframe_normalizes_aliases_and_keeps_optional_columns():
    records = [
        {
            "gender": "Male",
            "CLTV": 1500,
            "city": "Los Angeles",
        }
    ]

    df = build_inference_dataframe(
        records,
        required_columns=("Gender", "CLTV"),
        optional_columns=("City", "Latitude"),
    )

    assert list(df.columns) == ["Gender", "CLTV", "City"]
    assert df.loc[0, "Gender"] == "Male"
    assert df.loc[0, "CLTV"] == 1500
    assert df.loc[0, "City"] == "Los Angeles"


def test_build_inference_dataframe_raises_for_missing_required_columns():
    with pytest.raises(KeyError, match="Missing required inference columns"):
        build_inference_dataframe(
            [{"CLTV": 1000}],
            required_columns=("Gender", "CLTV"),
        )


def test_predict_with_threshold_uses_positive_class_probability():
    class DummyModel:
        def predict_proba(self, X):
            return [[0.8, 0.2], [0.1, 0.9]]

    X = pd.DataFrame({"Gender": ["Male", "Female"], "CLTV": [1000, 2000]})

    predictions = predict_with_threshold(DummyModel(), X, threshold=0.5)

    assert predictions["churn_probability"].tolist() == pytest.approx([0.2, 0.9])
    assert predictions["churn_label"].tolist() == [0, 1]


def test_materialize_production_model_validates_local_artifact_and_writes_metadata(
    tmp_path,
):
    model_path = tmp_path / "models" / "production" / "mlp_optuna" / "model.pkl"
    model_path.parent.mkdir(parents=True)
    estimator = DummyClassifier(strategy="prior")
    estimator.fit([[0], [1]], [0, 1])
    joblib.dump(estimator, model_path)

    settings = ProductionModelSettings(
        model_key="prod_a",
        model_name="MLP Optuna",
        framework="sklearn",
        threshold=0.35,
        model_path=model_path,
        metadata_path=tmp_path / "models" / "production" / "mlp_optuna" / "serving_metadata.json",
        required_columns=("Gender", "CLTV"),
        optional_columns=("City",),
        input_path=tmp_path / "input.csv",
        output_path=tmp_path / "output.csv",
        target_column="Churn Value",
    )

    local_path = materialize_production_model(settings)

    assert local_path == settings.model_path
    assert settings.metadata_path.exists()


def test_load_production_model_loads_local_pickle(tmp_path):
    model_path = tmp_path / "models" / "production" / "mlp_optuna" / "model.pkl"
    model_path.parent.mkdir(parents=True)
    estimator = DummyClassifier(strategy="prior")
    estimator.fit([[0], [1]], [0, 1])
    joblib.dump(estimator, model_path)

    settings = ProductionModelSettings(
        model_key="prod_a",
        model_name="MLP Optuna",
        framework="sklearn",
        threshold=0.35,
        model_path=model_path,
        metadata_path=tmp_path / "models" / "production" / "mlp_optuna" / "serving_metadata.json",
        required_columns=("Gender", "CLTV"),
        optional_columns=("City",),
        input_path=tmp_path / "input.csv",
        output_path=tmp_path / "output.csv",
        target_column="Churn Value",
    )

    loaded_model = load_production_model(settings)

    assert isinstance(loaded_model, DummyClassifier)
