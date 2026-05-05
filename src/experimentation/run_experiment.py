from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from src.experimentation.build_pipeline import build_pipeline
from src.experimentation.prep_data import prep_data


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "experiments" / "base_exp.yaml"
)


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_cv(config):
    cv_cfg = config["cv"]

    cv_type = cv_cfg.get("type", "stratified_kfold")
    n_splits = cv_cfg.get("n_splits", 5)
    shuffle = cv_cfg.get("shuffle", True)
    random_state = cv_cfg.get("random_state", 42)

    if cv_type == "stratified_kfold":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    if cv_type == "kfold":
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    raise ValueError(f"CV nao suportado: {cv_type}")


def _get_tracking_config(config):
    tracking_cfg = config.get("tracking") or config.get("mlflow") or {}
    return {
        "tracking_uri": tracking_cfg.get("tracking_uri"),
        "experiment_name": tracking_cfg.get("experiment_name"),
    }


def _summarize_cv_results(model_name, cv_res, metrics):
    summary = {"model": model_name}

    for metric in metrics:
        scores = cv_res[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(scores.mean())
        summary[f"{metric}_std"] = float(scores.std())

    summary["fit_time_mean"] = float(cv_res["fit_time"].mean())
    summary["score_time_mean"] = float(cv_res["score_time"].mean())

    return summary


def _build_mlflow_metrics(summary):
    return {
        "pr_auc": summary["pr_auc_mean"],
        "pr_auc_std": summary["pr_auc_std"],
        "roc_auc": summary["roc_auc_mean"],
        "recall": summary["recall_mean"],
        "precision": summary["precision_mean"],
        "f1_score": summary["f1_score_mean"],
        "fit_time": summary["fit_time_mean"],
        "score_time": summary["score_time_mean"],
    }


def run_experiment(config, set_experiment=True):
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

    pipeline = build_pipeline(config)
    cv = build_cv(config)

    scoring = config["cv"]["scoring"]
    metrics = list(scoring.keys())
    model_name = config["model"]["name"]

    print(f"\n=== AVALIANDO MODELO: {model_name} ===")

    cv_res = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=config["cv"].get("n_jobs", 1),
        return_train_score=False,
    )

    summary = _summarize_cv_results(model_name, cv_res, metrics)
    summary_df = pd.DataFrame([summary])

    with mlflow.start_run(run_name=model_name):
        mlflow.log_metrics(_build_mlflow_metrics(summary))

    print("\n=== RESULTADO DA VALIDACAO CRUZADA ===")
    print(summary_df.to_string(index=False))

    return summary_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa um experimento a partir de um YAML.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help="Caminho do YAML completo do experimento.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_experiment(load_config(args.config_path))
