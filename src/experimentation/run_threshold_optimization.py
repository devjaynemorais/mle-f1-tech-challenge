from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml

from src.experimentation.build_pipeline import build_pipeline
from src.experimentation.prep_data import prep_data
from src.experimentation.run_experiment import DEFAULT_CONFIG_PATH, build_cv, load_config
from src.experimentation.tracking import apply_tracking_config, get_tracking_config
from src.utils.exp import generate_oof_predictions, optimize_threshold_for_roi
from src.utils.mlflow_tracking import log_dataframe_artifact


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _get_evaluation_config(config: dict[str, Any]) -> dict[str, float]:
    evaluation_cfg = config.get("evaluation", {})
    return {
        "threshold": float(evaluation_cfg.get("threshold", 0.5)),
        "activation_cost": float(evaluation_cfg.get("activation_cost", 50.0)),
        "retention_rate": float(evaluation_cfg.get("retention_rate", 0.1)),
    }


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return sanitized or "model"


def _best_yaml_path(model_name: str) -> Path:
    return CONFIG_DIR / f"best_{_sanitize_model_name(model_name)}_threshold_params.yaml"


def _relative_config_path(config_path: str | Path | None) -> str | None:
    if config_path is None:
        return None

    resolved = Path(config_path).resolve()
    return os.path.relpath(resolved, REPO_ROOT)


def generate_oof_predictions_df(
    *,
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
    model_name: str,
) -> pd.DataFrame:
    oof_df = generate_oof_predictions(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        model_name=model_name,
    )
    normalized = oof_df.rename(columns={"pred_0_5": "y_pred"}).copy()
    normalized["threshold"] = 0.5
    return normalized


def _build_threshold_payload(
    *,
    config: dict[str, Any],
    config_path: str | Path | None,
    best_row: pd.Series | dict[str, Any],
) -> dict[str, Any]:
    best_series = best_row if isinstance(best_row, pd.Series) else pd.Series(best_row)
    best_threshold = float(best_series["threshold"])

    payload = copy.deepcopy(config)
    payload.setdefault("evaluation", {})
    payload["evaluation"]["threshold"] = best_threshold

    model_params = payload.setdefault("model", {}).setdefault("params", {})
    if "threshold" in model_params:
        model_params["threshold"] = best_threshold

    payload["threshold_optimization"] = {
        "source_config": _relative_config_path(config_path),
        "primary_metric": "roi",
        "best_threshold": best_threshold,
        "best_metrics": {
            key: float(value)
            for key, value in best_series.items()
            if key != "threshold" and value is not None
        },
    }
    return payload


def _write_best_yaml(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _build_mlflow_metrics(best_row: pd.Series | dict[str, Any]) -> dict[str, float]:
    best_series = best_row if isinstance(best_row, pd.Series) else pd.Series(best_row)
    metrics = {"best_threshold": float(best_series["threshold"])}
    for key, value in best_series.items():
        if key == "threshold" or value is None:
            continue
        if isinstance(value, (int, float)):
            metrics[f"best_{key}"] = float(value)
    return metrics


def run_threshold_optimization(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    set_experiment: bool = True,
) -> dict[str, Any]:
    tracking_cfg = get_tracking_config(config)
    evaluation_cfg = _get_evaluation_config(config)
    model_name = config["model"]["name"]
    apply_tracking_config(tracking_cfg, set_experiment=set_experiment)

    split_bundle = prep_data(config)
    X_train = split_bundle["X_train"]
    y_train = split_bundle["y_train"]
    meta_train = split_bundle["meta_train"]

    if "CLTV" not in meta_train.columns:
        raise KeyError("meta_train precisa conter a coluna 'CLTV'.")

    pipeline = build_pipeline(config)
    cv = build_cv(config)
    oof_df = generate_oof_predictions_df(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        model_name=model_name,
    )

    threshold_df, best_row = optimize_threshold_for_roi(
        y_true=oof_df["y_true"],
        y_prob=oof_df["proba"],
        cltv=meta_train.loc[oof_df["row_index"], "CLTV"],
        activation_cost=evaluation_cfg["activation_cost"],
        retention_rate=evaluation_cfg["retention_rate"],
    )

    pipeline.fit(X_train, y_train)
    best_yaml_path = _best_yaml_path(model_name)
    payload = _build_threshold_payload(
        config=config,
        config_path=config_path,
        best_row=best_row,
    )

    with mlflow.start_run(run_name=f"{model_name}_threshold_optimization"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "source_threshold": evaluation_cfg["threshold"],
                "primary_metric": "roi",
                "activation_cost": evaluation_cfg["activation_cost"],
                "retention_rate": evaluation_cfg["retention_rate"],
            }
        )
        mlflow.log_metrics(_build_mlflow_metrics(best_row))
        log_dataframe_artifact(oof_df, f"{model_name}_oof_predictions.csv")
        log_dataframe_artifact(threshold_df, f"{model_name}_threshold_search.csv")
        _write_best_yaml(payload, best_yaml_path)
        mlflow.log_artifact(str(best_yaml_path))

    return {
        "pipeline": pipeline,
        "oof_df": oof_df,
        "threshold_df": threshold_df,
        "best_row": best_row,
        "best_threshold": float(best_row["threshold"]),
        "best_yaml_path": best_yaml_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Otimiza o threshold economico de um modelo.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help="Caminho do YAML completo do experimento.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_config(args.config_path)
    run_threshold_optimization(config, config_path=args.config_path)


if __name__ == "__main__":
    main()
