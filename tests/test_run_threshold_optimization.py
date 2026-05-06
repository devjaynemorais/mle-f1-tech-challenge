import os
from contextlib import contextmanager
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_run_threshold_optimization_logs_and_writes_best_threshold_yaml(
    monkeypatch, tmp_path
):
    from src.experimentation import run_threshold_optimization as threshold_module

    monkeypatch.setattr(threshold_module, "CONFIG_DIR", tmp_path)

    config_path = tmp_path / "experiments" / "best_mlp_params.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("model:\n  name: mlp\n", encoding="utf-8")

    config = {
        "model": {
            "name": "mlp",
            "params": {
                "hidden_dim": 64,
                "dropout": 0.2,
                "selector_k": 24,
                "threshold": 0.5,
                "random_state": 42,
            },
        },
        "tracking": {
            "experiment_name": "evaluation-mlp",
            "tracking_uri": "sqlite:///mlflow.db",
        },
        "cv": {
            "random_state": 42,
        },
        "evaluation": {
            "threshold": 0.5,
            "activation_cost": 50.0,
            "retention_rate": 0.1,
        },
    }

    events = []
    logged_artifacts = []
    captured_optimize = {}
    fitted_pipeline = {}
    oof_df = pd.DataFrame(
        {
            "row_index": [0, 1, 2],
            "y_true": [1, 0, 1],
            "proba": [0.8, 0.2, 0.6],
            "y_pred": [1, 0, 1],
            "threshold": [0.5, 0.5, 0.5],
            "model": ["mlp", "mlp", "mlp"],
        }
    )
    threshold_df = pd.DataFrame(
        {
            "threshold": [0.3, 0.5],
            "roi": [0.25, 0.10],
            "iel": [1200.0, 600.0],
            "pr_auc": [0.91, 0.91],
            "roc_auc": [0.94, 0.94],
            "recall": [0.88, 0.75],
            "precision": [0.71, 0.80],
            "f1_score": [0.79, 0.77],
        }
    )
    best_row = threshold_df.iloc[0].copy()

    class FakePipeline:
        def fit(self, X, y):
            fitted_pipeline["fit"] = (X, y)
            return self

    @contextmanager
    def fake_start_run(run_name, nested=False):
        events.append(("start_run", run_name, nested))
        yield object()

    monkeypatch.setattr(
        threshold_module,
        "prep_data",
        lambda cfg: {
            "X_train": "X_train",
            "y_train": pd.Series([1, 0, 1]),
            "meta_train": pd.DataFrame({"CLTV": [100.0, 80.0, 120.0]}),
        },
    )
    monkeypatch.setattr(threshold_module, "build_pipeline", lambda cfg: FakePipeline())
    monkeypatch.setattr(threshold_module, "build_cv", lambda cfg: "cv")
    monkeypatch.setattr(
        threshold_module,
        "generate_oof_predictions_df",
        lambda **kwargs: oof_df.copy(),
    )

    def fake_optimize_threshold_for_roi(**kwargs):
        captured_optimize.update(kwargs)
        return threshold_df.copy(), best_row.copy()

    monkeypatch.setattr(
        threshold_module,
        "optimize_threshold_for_roi",
        fake_optimize_threshold_for_roi,
    )
    monkeypatch.setattr(
        threshold_module,
        "log_dataframe_artifact",
        lambda df, artifact_name: logged_artifacts.append(
            ("dataframe", artifact_name, df.copy())
        ),
    )
    monkeypatch.setattr(
        threshold_module.mlflow,
        "set_tracking_uri",
        lambda uri: events.append(("tracking_uri", uri)),
    )
    monkeypatch.setattr(
        threshold_module.mlflow,
        "set_experiment",
        lambda name: events.append(("set_experiment", name)),
    )
    monkeypatch.setattr(threshold_module.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(
        threshold_module.mlflow,
        "log_params",
        lambda params: events.append(("log_params", params)),
    )
    monkeypatch.setattr(
        threshold_module.mlflow,
        "log_metrics",
        lambda metrics: events.append(("log_metrics", metrics)),
    )
    monkeypatch.setattr(
        threshold_module.mlflow,
        "log_artifact",
        lambda path: logged_artifacts.append(("artifact", Path(path).name)),
    )

    result = threshold_module.run_threshold_optimization(
        config,
        config_path=config_path,
    )

    best_yaml_path = tmp_path / "best_mlp_threshold_params.yaml"
    assert best_yaml_path.exists()
    saved = yaml.safe_load(best_yaml_path.read_text(encoding="utf-8"))

    assert result["best_threshold"] == 0.3
    assert result["best_yaml_path"] == best_yaml_path
    assert fitted_pipeline["fit"][0] == "X_train"
    assert fitted_pipeline["fit"][1].tolist() == [1, 0, 1]
    assert captured_optimize["y_true"].tolist() == [1, 0, 1]
    assert captured_optimize["y_prob"].tolist() == [0.8, 0.2, 0.6]
    assert captured_optimize["cltv"].tolist() == [100.0, 80.0, 120.0]
    assert saved["evaluation"]["threshold"] == 0.3
    assert saved["model"]["params"]["threshold"] == 0.3
    assert saved["tracking"]["experiment_name"] == "evaluation-mlp"
    assert saved["threshold_optimization"]["source_config"] == os.path.relpath(
        config_path.resolve(),
        threshold_module.REPO_ROOT,
    )
    assert saved["threshold_optimization"]["primary_metric"] == "roi"
    assert saved["threshold_optimization"]["best_threshold"] == 0.3
    assert saved["threshold_optimization"]["best_metrics"]["roi"] == 0.25
    assert ("tracking_uri", "sqlite:///mlflow.db") in events
    assert ("set_experiment", "evaluation-mlp") in events
    assert ("start_run", "mlp_threshold_optimization", False) in events
    assert ("artifact", "best_mlp_threshold_params.yaml") in logged_artifacts
