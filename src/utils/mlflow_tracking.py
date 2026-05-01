"""Helpers de tracking no MLflow para os notebooks de experimentacao."""

from __future__ import annotations

from contextlib import contextmanager
import json
from numbers import Integral, Real
from pathlib import Path
import tempfile
from typing import Any, Iterator

import mlflow
import numpy as np
import pandas as pd


def set_tracking(tracking_uri: str) -> None:
    """Define a URI de tracking do MLflow."""
    mlflow.set_tracking_uri(tracking_uri)


def ensure_experiment(experiment_name: str) -> str:
    """Garante que o experimento exista e retorna seu ID."""
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experimento do MLflow nao encontrado: {experiment_name}")
    return experiment.experiment_id


@contextmanager
def start_experiment_run(
    experiment_name: str,
    run_name: str,
    nested: bool = False,
    tags: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Abre uma run no experimento informado."""
    ensure_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags({k: _coerce_param_value(v) for k, v in tags.items()})
        yield run


def log_params_safe(params: dict[str, Any]) -> None:
    """Loga parametros convertendo estruturas para tipos aceitos pelo MLflow."""
    safe_params = {key: _coerce_param_value(value) for key, value in params.items() if value is not None}
    if safe_params:
        mlflow.log_params(safe_params)


def log_metrics_safe(metrics: dict[str, Any]) -> None:
    """Loga metricas numericas aceitas pelo MLflow."""
    safe_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, Real) and not isinstance(value, bool):
            safe_metrics[key] = float(value)
    if safe_metrics:
        mlflow.log_metrics(safe_metrics)


def log_dataframe_artifact(df: pd.DataFrame, artifact_name: str) -> None:
    """Persiste um DataFrame em CSV UTF-8 e sobe como artefato."""
    artifact_path = _artifact_temp_path(artifact_name)
    df.to_csv(artifact_path, index=False, encoding="utf-8")
    mlflow.log_artifact(str(artifact_path))


def log_json_artifact(payload: dict | list, artifact_name: str) -> None:
    """Persiste um payload JSON UTF-8 e sobe como artefato."""
    artifact_path = _artifact_temp_path(artifact_name)
    artifact_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(artifact_path))


def log_split_artifacts(
    train_val_idx: Any,
    test_idx: Any,
    metadata_train_val: pd.DataFrame,
    metadata_test: pd.DataFrame,
) -> None:
    """Loga indices dos splits e metadados sincronizados."""
    train_val_idx_df = pd.DataFrame({"idx": list(train_val_idx)})
    test_idx_df = pd.DataFrame({"idx": list(test_idx)})

    log_dataframe_artifact(train_val_idx_df, "train_val_indices.csv")
    log_dataframe_artifact(test_idx_df, "test_indices.csv")
    log_dataframe_artifact(metadata_train_val.reset_index(drop=False), "metadata_train_val.csv")
    log_dataframe_artifact(metadata_test.reset_index(drop=False), "metadata_test.csv")


def best_row_to_metrics(best_row: pd.Series | dict[str, Any], prefix: str = "best") -> dict[str, float]:
    """Extrai metricas da melhor linha com prefixo padronizado."""
    if isinstance(best_row, pd.Series):
        row_dict = best_row.to_dict()
    else:
        row_dict = dict(best_row)

    metric_keys = [
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "recall_mean",
        "precision_mean",
        "f1_mean",
        "fit_time_mean_s",
        "score_time_mean_s",
    ]

    metrics: dict[str, float] = {}
    for key in metric_keys:
        if key not in row_dict or row_dict[key] is None:
            continue
        value = row_dict[key]
        if isinstance(value, np.generic):
            value = value.item()
        metrics[f"{prefix}_{key}"] = float(value)
    return metrics


def _artifact_temp_path(artifact_name: str) -> Path:
    artifact_path = Path(tempfile.gettempdir()) / artifact_name
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    return artifact_path


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (Integral, Real)) and not isinstance(value, bool):
        return value
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
