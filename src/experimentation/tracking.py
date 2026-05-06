from __future__ import annotations

import os
from typing import Any

import mlflow


def get_tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve tracking com precedencia para a env var usada no Docker."""
    tracking_cfg = config.get("tracking") or config.get("mlflow") or {}
    env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return {
        "tracking_uri": env_tracking_uri or tracking_cfg.get("tracking_uri"),
        "experiment_name": tracking_cfg.get("experiment_name"),
    }


def apply_tracking_config(
    tracking_cfg: dict[str, Any],
    *,
    set_experiment: bool = True,
) -> None:
    tracking_uri = tracking_cfg.get("tracking_uri")
    experiment_name = tracking_cfg.get("experiment_name")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if set_experiment and experiment_name:
        mlflow.set_experiment(experiment_name)
