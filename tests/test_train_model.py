import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakePipeline:
    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((X.copy(), y.copy()))
        return self


def test_train_production_model_saves_pickle_metadata_and_updates_config(
    monkeypatch, tmp_path
):
    from src.models import train as train_module

    experiment_config_path = tmp_path / "best_mlp_threshold_params.yaml"
    experiment_config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "target": "Churn Value",
                    "meta_cols": ["CLTV", "CustomerID"],
                },
                "model": {
                    "name": "mlp",
                    "params": {"threshold": 0.33},
                },
                "evaluation": {"threshold": 0.33},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    production_config_path = tmp_path / "config.yaml"
    production_config_path.write_text(
        yaml.safe_dump(
            {
                "production": {
                    "active_model": "mlp_optuna_prod",
                    "models": {
                        "mlp_optuna_prod": {
                            "display_name": "MLP Optuna",
                            "framework": "sklearn",
                            "threshold": 0.35,
                            "model_path": "models/production/mlp_optuna/model.pkl",
                            "local": {
                                "metadata_path": "models/production/mlp_optuna/serving_metadata.json"
                            },
                            "pipeline": {
                                "required_columns": ["Gender", "Monthly Charges"],
                                "optional_columns": ["Churn Score"],
                            },
                            "inference": {
                                "input_path": "data/interim/telecom_clean.csv",
                                "output_path": "models/predictions/mlp_optuna_predictions.csv",
                                "target_column": "Churn Value",
                            },
                        }
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cleaned_df = pd.DataFrame(
        {
            "Gender": ["Male", "Female"],
            "Monthly Charges": [80.0, 65.0],
            "Churn Value": [1, 0],
            "CLTV": [2000.0, 1500.0],
            "CustomerID": ["A", "B"],
        }
    )
    fake_pipeline = _FakePipeline()
    dumped = {}

    monkeypatch.setattr(train_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr(
        train_module,
        "load_clean_dataset",
        lambda config: cleaned_df.copy(),
    )
    monkeypatch.setattr(
        train_module,
        "build_pipeline",
        lambda config: fake_pipeline,
    )
    monkeypatch.setattr(
        train_module.joblib,
        "dump",
        lambda model, path: dumped.update({"model": model, "path": Path(path)}),
    )

    result = train_module.train_production_model(
        experiment_config_path=experiment_config_path,
        production_config_path=production_config_path,
    )

    updated_config = yaml.safe_load(production_config_path.read_text(encoding="utf-8"))
    metadata_path = tmp_path / "models" / "production" / "mlp_optuna" / "serving_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert dumped["model"] is fake_pipeline
    assert dumped["path"] == tmp_path / "models" / "production" / "mlp_optuna" / "model.pkl"
    assert fake_pipeline.fit_calls[0][0].columns.tolist() == ["Gender", "Monthly Charges"]
    assert fake_pipeline.fit_calls[0][1].tolist() == [1, 0]
    assert updated_config["production"]["active_model"] == "mlp_optuna_prod"
    assert (
        updated_config["production"]["models"]["mlp_optuna_prod"]["threshold"] == 0.33
    )
    assert (
        updated_config["production"]["models"]["mlp_optuna_prod"]["model_path"]
        == "models/production/mlp_optuna/model.pkl"
    )
    assert metadata["threshold"] == 0.33
    assert metadata["source_experiment_config"] == "best_mlp_threshold_params.yaml"
    assert metadata["required_columns"] == ["Gender", "Monthly Charges"]
    assert metadata["optional_columns"] == ["Churn Score"]
    assert result["threshold"] == 0.33
