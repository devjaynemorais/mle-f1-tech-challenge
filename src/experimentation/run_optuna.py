from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Any

import mlflow
from mlflow import sklearn as mlflow_sklearn
import yaml

from src.experimentation.build_pipeline import build_pipeline
from src.experimentation.prep_data import prep_data
from src.experimentation.run_experiment import DEFAULT_CONFIG_PATH, build_cv, load_config
from src.utils.mlflow_tracking import log_dataframe_artifact
from src.utils.optuna_search import prepare_model_params, run_optuna_study


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _get_tracking_config(config: dict[str, Any]) -> dict[str, Any]:
    tracking_cfg = config.get("tracking") or config.get("mlflow") or {}
    return {
        "tracking_uri": tracking_cfg.get("tracking_uri"),
        "experiment_name": tracking_cfg.get("experiment_name"),
    }


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return sanitized or "model"


def _best_yaml_path(config_path: str | Path | None, model_name: str) -> Path:
    del config_path
    return CONFIG_DIR / f"best_{_sanitize_model_name(model_name)}_params.yaml"


def _relative_config_path(config_path: str | Path | None) -> str | None:
    if config_path is None:
        return None

    resolved = Path(config_path).resolve()
    return os.path.relpath(resolved, REPO_ROOT)


def _build_evaluation_experiment_name(experiment_name: str | None, model_name: str) -> str:
    if experiment_name:
        if experiment_name.startswith("optuna-"):
            return f"evaluation-{experiment_name.removeprefix('optuna-')}"
        return f"evaluation-{experiment_name}"
    return f"evaluation-{_sanitize_model_name(model_name)}"


def _build_best_payload(
    *,
    config: dict[str, Any],
    model_name: str,
    config_path: str | Path | None,
    primary_metric: str,
    best_model_params: dict[str, Any],
    best_row: dict[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(config)
    payload.setdefault("model", {})
    payload["model"]["name"] = model_name
    payload["model"]["params"] = best_model_params

    tracking_cfg = payload.setdefault("tracking", {})
    tracking_cfg["experiment_name"] = _build_evaluation_experiment_name(
        tracking_cfg.get("experiment_name"),
        model_name,
    )

    payload["optuna_result"] = {
        "source_config": _relative_config_path(config_path),
        "primary_metric": primary_metric,
        "best_metrics": {
            key: value
            for key, value in best_row.items()
            if key.endswith("_mean") or key.endswith("_std")
        },
        "best_trial_number": best_row.get("trial_number"),
    }
    return payload


def _write_best_yaml(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _build_parent_metrics(best_row: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    for key, value in best_row.items():
        if key.endswith("_mean") or key.endswith("_std"):
            metrics[f"best_{key}"] = float(value)
    if "trial_number" in best_row:
        metrics["best_trial_number"] = float(best_row["trial_number"])
    return metrics


def run_optuna(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    set_experiment: bool = True,
) -> dict[str, Any]:
    tracking_cfg = _get_tracking_config(config)
    tracking_uri = tracking_cfg["tracking_uri"]
    experiment_name = tracking_cfg["experiment_name"]

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if set_experiment and experiment_name:
        mlflow.set_experiment(experiment_name)

    data = prep_data(config)
    X_train = data["X_train"]
    y_train = data["y_train"]

    cv = build_cv(config)
    scoring = config["cv"]["scoring"]
    model_name = config["model"]["name"]
    tuning_cfg = config["tuning"]
    primary_metric = tuning_cfg["primary_metric"]

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(
            {
                key: value
                for key, value in {
                    "model_name": model_name,
                    "primary_metric": primary_metric,
                    "n_trials": tuning_cfg.get("n_trials"),
                    "timeout_seconds": tuning_cfg.get("timeout_seconds"),
                }.items()
                if value is not None
            }
        )

        study, trials_df, convergence_df, best_params, best_row = run_optuna_study(
            model_name=model_name,
            config=config,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_trials=tuning_cfg.get("n_trials"),
            timeout_seconds=tuning_cfg.get("timeout_seconds", 3600),
            convergence_patience_trials=tuning_cfg.get("convergence_patience_trials", 50),
            convergence_min_improvement=tuning_cfg.get("convergence_min_improvement", 5e-3),
            random_state=config["cv"].get("random_state", 42),
        )

        mlflow.log_metrics(_build_parent_metrics(best_row))
        log_dataframe_artifact(trials_df, f"{model_name}_optuna_trials.csv")
        log_dataframe_artifact(convergence_df, f"{model_name}_optuna_convergence.csv")

        best_model_params = prepare_model_params(
            model_name=model_name,
            params=best_params,
            config=config,
            y_reference=y_train,
        )
        best_pipeline = build_pipeline(
            model_name=model_name,
            model_params=best_model_params,
            config=config,
            y_reference=y_train,
        )
        best_pipeline.fit(X_train, y_train)
        mlflow_sklearn.log_model(best_pipeline, artifact_path="best_model")

        best_yaml_path = _best_yaml_path(config_path, model_name)
        payload = _build_best_payload(
            config=config,
            model_name=model_name,
            config_path=config_path,
            primary_metric=primary_metric,
            best_model_params=best_model_params,
            best_row=best_row,
        )
        _write_best_yaml(payload, best_yaml_path)
        mlflow.log_artifact(str(best_yaml_path))

    return {
        "study": study,
        "trials_df": trials_df,
        "convergence_df": convergence_df,
        "best_params": best_params,
        "best_row": best_row,
        "best_pipeline": best_pipeline,
        "best_yaml_path": best_yaml_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa tuning de hiperparametros com Optuna.")
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
    run_optuna(config, config_path=args.config_path)


if __name__ == "__main__":
    main()
