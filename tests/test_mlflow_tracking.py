import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_best_row_to_metrics_extracts_prefixed_metrics():
    from src.utils.mlflow_tracking import best_row_to_metrics

    row = {
        "pr_auc_mean": 0.91,
        "pr_auc_std": 0.02,
        "roc_auc_mean": 0.95,
        "recall_mean": 0.88,
        "precision_mean": 0.74,
        "f1_mean": 0.80,
        "fit_time_mean_s": 1.2,
        "score_time_mean_s": 0.1,
        "other": "ignored",
    }

    metrics = best_row_to_metrics(row, prefix="best")

    assert metrics == {
        "best_pr_auc_mean": 0.91,
        "best_pr_auc_std": 0.02,
        "best_roc_auc_mean": 0.95,
        "best_recall_mean": 0.88,
        "best_precision_mean": 0.74,
        "best_f1_mean": 0.80,
        "best_fit_time_mean_s": 1.2,
        "best_score_time_mean_s": 0.1,
    }


def test_log_dataframe_artifact_writes_utf8_csv_and_logs_artifact(monkeypatch, tmp_path):
    from src.utils import mlflow_tracking

    logged = {}

    def fake_log_artifact(path):
        logged["path"] = path

    monkeypatch.setattr(mlflow_tracking.mlflow, "log_artifact", fake_log_artifact)
    monkeypatch.setattr(mlflow_tracking.tempfile, "gettempdir", lambda: str(tmp_path))

    df = pd.DataFrame({"cidade": ["São Paulo"], "valor": [1]})
    mlflow_tracking.log_dataframe_artifact(df, "artifact.csv")

    artifact_path = Path(logged["path"])
    assert artifact_path.name == "artifact.csv"
    content = artifact_path.read_text(encoding="utf-8")
    assert "São Paulo" in content


def test_log_json_artifact_writes_utf8_json_and_logs_artifact(monkeypatch, tmp_path):
    from src.utils import mlflow_tracking

    logged = {}

    def fake_log_artifact(path):
        logged["path"] = path

    monkeypatch.setattr(mlflow_tracking.mlflow, "log_artifact", fake_log_artifact)
    monkeypatch.setattr(mlflow_tracking.tempfile, "gettempdir", lambda: str(tmp_path))

    payload = {"cidade": "São Paulo", "valor": 1}
    mlflow_tracking.log_json_artifact(payload, "artifact.json")

    artifact_path = Path(logged["path"])
    assert artifact_path.name == "artifact.json"
    content = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert content == payload


def test_log_split_artifacts_delegates_all_expected_artifacts(monkeypatch):
    from src.utils import mlflow_tracking

    captured = []

    def fake_log_dataframe_artifact(df, artifact_name):
        captured.append((artifact_name, df.copy()))

    monkeypatch.setattr(
        mlflow_tracking,
        "log_dataframe_artifact",
        fake_log_dataframe_artifact,
    )

    train_val_idx = pd.Index([10, 20], name="idx")
    test_idx = pd.Index([30], name="idx")
    metadata_train_val = pd.DataFrame({"CLTV": [100.0, 200.0]}, index=train_val_idx)
    metadata_test = pd.DataFrame({"CLTV": [300.0]}, index=test_idx)

    mlflow_tracking.log_split_artifacts(
        train_val_idx=train_val_idx,
        test_idx=test_idx,
        metadata_train_val=metadata_train_val,
        metadata_test=metadata_test,
    )

    artifact_names = [name for name, _ in captured]
    assert artifact_names == [
        "train_val_indices.csv",
        "test_indices.csv",
        "metadata_train_val.csv",
        "metadata_test.csv",
    ]
    assert captured[0][1]["idx"].tolist() == [10, 20]
    assert np.isclose(captured[2][1]["CLTV"].sum(), 300.0)
