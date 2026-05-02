from pathlib import Path

import pandas as pd
import pytest

from src.models.production import (
    ProductionModelSettings,
    build_inference_dataframe,
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
      source:
        tracking_uri: sqlite:///mlflow.db
        experiment_name: exp-prod
        run_id: abc123
        model_uri: runs:/abc123/model
      local:
        artifact_dir: models/production/mlp_optuna
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
    assert settings.model_uri == "runs:/abc123/model"
    assert settings.required_columns == ("Gender", "CLTV")
    assert settings.optional_columns == ("City",)
    assert settings.local_artifact_dir == tmp_path / "models" / "production" / "mlp_optuna"


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


def test_materialize_production_model_downloads_and_writes_metadata(tmp_path, monkeypatch):
    downloaded_dir = tmp_path / "downloaded" / "model"
    downloaded_dir.mkdir(parents=True)
    (downloaded_dir / "MLmodel").write_text("flavors: {}", encoding="utf-8")

    monkeypatch.setattr(
        "src.models.production.mlflow.artifacts.download_artifacts",
        lambda artifact_uri, dst_path, tracking_uri: str(downloaded_dir),
    )

    settings = ProductionModelSettings(
        model_key="prod_a",
        model_name="MLP Optuna",
        framework="sklearn",
        threshold=0.35,
        tracking_uri="sqlite:///tmp/mlflow.db",
        model_uri="runs:/abc123/model",
        run_id="abc123",
        experiment_name="exp-prod",
        local_artifact_dir=tmp_path / "models" / "production" / "mlp_optuna",
        metadata_path=tmp_path / "models" / "production" / "mlp_optuna" / "serving_metadata.json",
        required_columns=("Gender", "CLTV"),
        optional_columns=("City",),
        input_path=tmp_path / "input.csv",
        output_path=tmp_path / "output.csv",
        target_column="Churn Value",
    )

    local_dir = materialize_production_model(settings)

    assert local_dir == settings.local_artifact_dir
    assert (local_dir / "MLmodel").exists()
    assert settings.metadata_path.exists()
