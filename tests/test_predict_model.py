import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.production import ProductionModelSettings


def test_predict_generates_default_interim_input_when_missing(tmp_path, monkeypatch):
    from src.models import predict_model

    input_path = tmp_path / "data" / "interim" / "telecom_clean.csv"
    output_path = tmp_path / "models" / "predictions" / "output.csv"

    settings = ProductionModelSettings(
        model_key="prod_a",
        model_name="MLP Optuna",
        framework="sklearn",
        threshold=0.35,
        model_path=tmp_path / "models" / "production" / "mlp_optuna" / "model.pkl",
        metadata_path=tmp_path / "models" / "production" / "mlp_optuna" / "serving_metadata.json",
        required_columns=("Gender", "CLTV"),
        optional_columns=("City",),
        input_path=input_path,
        output_path=output_path,
        target_column=None,
    )

    monkeypatch.setattr(
        predict_model,
        "load_production_settings",
        lambda *args, **kwargs: settings,
    )

    class DummyModel:
        def predict_proba(self, X):
            return [[0.8, 0.2] for _ in range(len(X))]

    monkeypatch.setattr(
        predict_model,
        "load_production_model",
        lambda settings, prefer_local=True: DummyModel(),
    )

    state = {"calls": 0}
    fake_make_dataset = types.ModuleType("src.data.make_dataset")

    def fake_process_data():
        state["calls"] += 1
        input_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"Gender": "Male", "CLTV": 1200, "City": "Los Angeles"}]
        ).to_csv(input_path, index=False)

    fake_make_dataset.process_data = fake_process_data
    monkeypatch.setitem(sys.modules, "src.data.make_dataset", fake_make_dataset)

    result_df = predict_model.predict()

    assert state["calls"] == 1
    assert input_path.exists()
    assert output_path.exists()
    assert result_df["churn_probability"].tolist() == [0.2]
    assert result_df["churn_label"].tolist() == [0]


def test_predict_does_not_generate_custom_missing_input(tmp_path, monkeypatch):
    from src.models import predict_model

    settings = ProductionModelSettings(
        model_key="prod_a",
        model_name="MLP Optuna",
        framework="sklearn",
        threshold=0.35,
        model_path=tmp_path / "models" / "production" / "mlp_optuna" / "model.pkl",
        metadata_path=tmp_path / "models" / "production" / "mlp_optuna" / "serving_metadata.json",
        required_columns=("Gender", "CLTV"),
        optional_columns=("City",),
        input_path=tmp_path / "data" / "interim" / "telecom_clean.csv",
        output_path=tmp_path / "models" / "predictions" / "output.csv",
        target_column=None,
    )

    monkeypatch.setattr(
        predict_model,
        "load_production_settings",
        lambda *args, **kwargs: settings,
    )
    monkeypatch.setattr(
        predict_model,
        "load_production_model",
        lambda settings, prefer_local=True: object(),
    )

    state = {"calls": 0}
    fake_make_dataset = types.ModuleType("src.data.make_dataset")

    def fake_process_data():
        state["calls"] += 1

    fake_make_dataset.process_data = fake_process_data
    monkeypatch.setitem(sys.modules, "src.data.make_dataset", fake_make_dataset)

    custom_input_path = tmp_path / "missing" / "custom.csv"

    with pytest.raises(FileNotFoundError):
        predict_model.predict(input_path=custom_input_path)

    assert state["calls"] == 0
