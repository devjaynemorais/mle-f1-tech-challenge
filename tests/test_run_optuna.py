import os
from contextlib import contextmanager
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_run_optuna_logs_parent_run_and_writes_best_yaml(monkeypatch, tmp_path):
    from src.experimentation import run_optuna as run_optuna_module

    config_path = tmp_path / "experiments" / "mlp.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("model:\n  name: mlp\n", encoding="utf-8")
    config = {
        "model": {"name": "mlp"},
        "tracking": {
            "experiment_name": "exp-optuna",
            "tracking_uri": "sqlite:///mlflow.db",
        },
        "cv": {
            "scoring": {
                "pr_auc": "average_precision",
                "roc_auc": "roc_auc",
                "recall": "recall",
                "precision": "precision",
                "f1_score": "f1",
            }
        },
        "tuning": {
            "primary_metric": "pr_auc",
        },
    }

    events = []
    logged_artifacts = []
    logged_model = {}

    class FakePipeline:
        def __init__(self):
            self.fit_calls = []

        def fit(self, X, y):
            self.fit_calls.append((X, y))
            return self

    best_pipeline = FakePipeline()

    @contextmanager
    def fake_start_run(run_name, nested=False):
        events.append(("start_run", run_name, nested))
        yield object()

    monkeypatch.setattr(
        run_optuna_module,
        "prep_data",
        lambda config: {"X_train": "X_train", "y_train": "y_train"},
    )
    monkeypatch.setattr(run_optuna_module, "build_cv", lambda config: "cv")
    monkeypatch.setattr(
        run_optuna_module,
        "run_optuna_study",
        lambda **kwargs: (
            object(),
            pd.DataFrame([{"trial_number": 0, "pr_auc_mean": 0.80}]),
            pd.DataFrame([{"trial_number": 0, "best_pr_auc_so_far": 0.80}]),
            {"hidden_dim": 64, "dropout": 0.2, "selector_k": 24},
            {
                "trial_number": 0,
                "pr_auc_mean": 0.80,
                "pr_auc_std": 0.02,
                "roc_auc_mean": 0.85,
                "roc_auc_std": 0.01,
                "recall_mean": 0.70,
                "recall_std": 0.03,
                "precision_mean": 0.60,
                "precision_std": 0.02,
                "f1_score_mean": 0.64,
                "f1_score_std": 0.02,
                "fit_time_mean": 1.5,
                "score_time_mean": 0.2,
            },
        ),
    )
    monkeypatch.setattr(
        run_optuna_module,
        "build_pipeline",
        lambda *, model_name, model_params, config, y_reference=None: best_pipeline,
    )
    monkeypatch.setattr(
        run_optuna_module,
        "prepare_model_params",
        lambda *, model_name, params, config, y_reference: {
            "hidden_dim": 64,
            "dropout": 0.2,
            "selector_k": 24,
            "random_state": 42,
        },
    )
    monkeypatch.setattr(
        run_optuna_module,
        "log_dataframe_artifact",
        lambda df, artifact_name: logged_artifacts.append(("dataframe", artifact_name, df.copy())),
    )
    monkeypatch.setattr(
        run_optuna_module.mlflow,
        "set_tracking_uri",
        lambda uri: events.append(("tracking_uri", uri)),
    )
    monkeypatch.setattr(
        run_optuna_module.mlflow,
        "set_experiment",
        lambda name: events.append(("set_experiment", name)),
    )
    monkeypatch.setattr(run_optuna_module.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(
        run_optuna_module.mlflow,
        "log_params",
        lambda params: events.append(("log_params", params)),
    )
    monkeypatch.setattr(
        run_optuna_module.mlflow,
        "log_metrics",
        lambda metrics: events.append(("log_metrics", metrics)),
    )
    monkeypatch.setattr(
        run_optuna_module.mlflow,
        "log_artifact",
        lambda path: logged_artifacts.append(("artifact", Path(path).name)),
    )
    monkeypatch.setattr(
        run_optuna_module.mlflow_sklearn,
        "log_model",
        lambda model, artifact_path: logged_model.update(
            {"model": model, "artifact_path": artifact_path}
        ),
    )

    result = run_optuna_module.run_optuna(config, config_path=config_path)

    best_yaml_path = run_optuna_module.CONFIG_DIR / "best_mlp_params.yaml"
    assert best_yaml_path.exists()
    saved = yaml.safe_load(best_yaml_path.read_text(encoding="utf-8"))

    assert result["best_params"] == {"hidden_dim": 64, "dropout": 0.2, "selector_k": 24}
    assert result["best_yaml_path"] == best_yaml_path
    assert saved["model"]["name"] == "mlp"
    assert saved["model"]["params"] == {
        "hidden_dim": 64,
        "dropout": 0.2,
        "selector_k": 24,
        "random_state": 42,
    }
    assert saved["tracking"]["experiment_name"] == "evaluation-exp-optuna"
    assert saved["optuna_result"]["source_config"] == os.path.relpath(
        config_path.resolve(),
        run_optuna_module.REPO_ROOT,
    )
    assert saved["optuna_result"]["best_metrics"]["pr_auc_mean"] == 0.80
    assert ("tracking_uri", "sqlite:///mlflow.db") in events
    assert ("set_experiment", "exp-optuna") in events
    assert ("start_run", "mlp", False) in events
    assert logged_model["model"] is best_pipeline
    assert logged_model["artifact_path"] == "best_model"
    assert best_pipeline.fit_calls == [("X_train", "y_train")]
    assert ("artifact", "best_mlp_params.yaml") in logged_artifacts
