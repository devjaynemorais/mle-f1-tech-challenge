from contextlib import contextmanager
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_run_experiment_logs_cross_validation_metrics_to_mlflow(monkeypatch):
    from src.experimentation import run_experiment as run_experiment_module

    logged = {
        "tracking_uri": None,
        "experiment_name": None,
        "run_name": None,
        "metrics": None,
    }

    @contextmanager
    def fake_start_run(run_name):
        logged["run_name"] = run_name
        yield object()

    monkeypatch.setattr(
        run_experiment_module,
        "prep_data",
        lambda config: {
            "X_train": pd.DataFrame({"feature": [1, 2, 3]}),
            "y_train": pd.Series([0, 1, 0]),
        },
    )
    monkeypatch.setattr(run_experiment_module, "build_pipeline", lambda config: "pipeline")
    monkeypatch.setattr(run_experiment_module, "build_cv", lambda config: "cv")
    monkeypatch.setattr(
        run_experiment_module,
        "cross_validate",
        lambda **kwargs: {
            "test_pr_auc": np.array([0.40, 0.60]),
            "test_roc_auc": np.array([0.70, 0.90]),
            "test_recall": np.array([0.80, 0.60]),
            "test_precision": np.array([0.50, 0.70]),
            "test_f1_score": np.array([0.55, 0.75]),
            "fit_time": np.array([1.0, 3.0]),
            "score_time": np.array([0.1, 0.3]),
        },
    )
    monkeypatch.setattr(
        run_experiment_module.mlflow,
        "set_tracking_uri",
        lambda uri: logged.__setitem__("tracking_uri", uri),
    )
    monkeypatch.setattr(
        run_experiment_module.mlflow,
        "set_experiment",
        lambda name: logged.__setitem__("experiment_name", name),
    )
    monkeypatch.setattr(run_experiment_module.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(
        run_experiment_module.mlflow,
        "log_metrics",
        lambda metrics: logged.__setitem__("metrics", metrics),
    )

    config = {
        "model": {"name": "logistic_regression"},
        "cv": {
            "scoring": {
                "pr_auc": "average_precision",
                "roc_auc": "roc_auc",
                "recall": "recall",
                "precision": "precision",
                "f1_score": "f1",
            }
        },
        "tracking": {
            "tracking_uri": "sqlite:///mlflow.db",
            "experiment_name": "exp-churn",
        },
    }

    summary_df = run_experiment_module.run_experiment(config)

    assert logged["tracking_uri"] == "sqlite:///mlflow.db"
    assert logged["experiment_name"] == "exp-churn"
    assert logged["run_name"] == "logistic_regression"
    assert logged["metrics"] == pytest.approx(
        {
            "pr_auc": 0.5,
            "pr_auc_std": 0.1,
            "roc_auc": 0.8,
            "recall": 0.7,
            "precision": 0.6,
            "f1_score": 0.65,
            "fit_time": 2.0,
            "score_time": 0.2,
        }
    )
    assert summary_df.loc[0, "pr_auc_mean"] == pytest.approx(0.5)
    assert summary_df.loc[0, "pr_auc_std"] == pytest.approx(0.1)
    assert summary_df.loc[0, "roc_auc_mean"] == pytest.approx(0.8)
    assert summary_df.loc[0, "recall_mean"] == pytest.approx(0.7)
    assert summary_df.loc[0, "precision_mean"] == pytest.approx(0.6)
    assert summary_df.loc[0, "f1_score_mean"] == pytest.approx(0.65)
    assert summary_df.loc[0, "fit_time_mean"] == pytest.approx(2.0)
    assert summary_df.loc[0, "score_time_mean"] == pytest.approx(0.2)


def test_run_experiment_skips_set_experiment_when_disabled(monkeypatch):
    from src.experimentation import run_experiment as run_experiment_module

    called = {"set_experiment": 0}

    @contextmanager
    def fake_start_run(run_name):
        yield object()

    monkeypatch.setattr(
        run_experiment_module,
        "prep_data",
        lambda config: {
            "X_train": pd.DataFrame({"feature": [1, 2, 3]}),
            "y_train": pd.Series([0, 1, 0]),
        },
    )
    monkeypatch.setattr(run_experiment_module, "build_pipeline", lambda config: "pipeline")
    monkeypatch.setattr(run_experiment_module, "build_cv", lambda config: "cv")
    monkeypatch.setattr(
        run_experiment_module,
        "cross_validate",
        lambda **kwargs: {
            "test_pr_auc": np.array([0.4, 0.6]),
            "test_roc_auc": np.array([0.7, 0.9]),
            "test_recall": np.array([0.8, 0.6]),
            "test_precision": np.array([0.5, 0.7]),
            "test_f1_score": np.array([0.55, 0.75]),
            "fit_time": np.array([1.0, 3.0]),
            "score_time": np.array([0.1, 0.3]),
        },
    )
    monkeypatch.setattr(run_experiment_module.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        run_experiment_module.mlflow,
        "set_experiment",
        lambda name: called.__setitem__("set_experiment", called["set_experiment"] + 1),
    )
    monkeypatch.setattr(run_experiment_module.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(run_experiment_module.mlflow, "log_metrics", lambda metrics: None)

    config = {
        "model": {"name": "logistic_regression"},
        "cv": {
            "scoring": {
                "pr_auc": "average_precision",
                "roc_auc": "roc_auc",
                "recall": "recall",
                "precision": "precision",
                "f1_score": "f1",
            }
        },
        "tracking": {
            "tracking_uri": "sqlite:///mlflow.db",
            "experiment_name": "exp-churn",
        },
    }

    run_experiment_module.run_experiment(config, set_experiment=False)

    assert called["set_experiment"] == 0


def test_load_config_reads_full_yaml_without_overrides(tmp_path):
    from src.experimentation import run_experiment as run_experiment_module

    config_path = tmp_path / "full_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  name: logistic_regression",
                "tracking:",
                "  experiment_name: exp-full",
                "  tracking_uri: sqlite:///mlflow.db",
            ]
        ),
        encoding="utf-8",
    )

    config = run_experiment_module.load_config(config_path)

    assert config == {
        "model": {"name": "logistic_regression"},
        "tracking": {
            "experiment_name": "exp-full",
            "tracking_uri": "sqlite:///mlflow.db",
        },
    }
