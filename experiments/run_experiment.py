"""
Script de experimentação: testa combinações de features com LogisticRegression,
avalia com validação cruzada estratificada e loga métricas técnicas e KPIs de
negócio (CLTV) no MLflow.

Uso:
    python experiments/run_experiment.py --config config/base_exp.yaml
"""
# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.encoders import FrequencyEncoder


def load_data(config: dict) -> tuple:
    """Carrega dados do interim, separa features, target e metadados de negócio."""
    df = pd.read_csv(config["data"]["interim_path"])
    target = config["data"]["target"]

    drop_cols = []
    if config["features"].get("drop_churn_score", {}).get("enabled", False):
        drop_cols.append("Churn Score")
    if config["features"].get("drop_city", {}).get("enabled", False):
        drop_cols.append("City")

    meta = df[["CLTV", "CustomerID"]].copy()
    X = df.drop(
        columns=[target, "CLTV", "CustomerID"]
        + [c for c in drop_cols if c in df.columns]
    )
    y = df[target]

    return X, y, meta


def build_pipeline(config: dict, X: pd.DataFrame) -> Pipeline:
    """Constrói o pipeline de pré-processamento + LogisticRegression."""
    numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    city_enc_cfg = config["features"].get("city_freq_encoding", {})
    use_city_freq = city_enc_cfg.get("enabled", False) and "City" in categorical_cols
    if use_city_freq:
        categorical_cols.remove("City")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    steps = [("preprocessor", preprocessor)]
    if use_city_freq:
        steps.append(("city_freq_encoding", FrequencyEncoder(column="City")))
    steps.append(("scaler", StandardScaler()))
    steps.append(("model", LogisticRegression(**config["model"]["params"])))

    return Pipeline(steps)


def run_cv(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, config: dict
) -> dict:
    """
    Roda validação cruzada estratificada com CLTV como sample_weight.

    Retorna métricas técnicas e KPIs de negócio por fold.
    """
    cv_config = config["validation"]
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=config["experiment"]["random_state"],
    )

    scores = {
        k: []
        for k in [
            "roc_auc",
            "f1",
            "recall",
            "captured_value",
            "expected_loss",
            "cltv_mean",
            "capture_value_ratio",
        ]
    }

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        meta_val = meta.iloc[val_idx]

        cltv_weights = meta.iloc[train_idx]["CLTV"].values
        pipeline.fit(X_train, y_train, model__sample_weight=cltv_weights)

        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        scores["roc_auc"].append(roc_auc_score(y_val, y_proba))
        scores["f1"].append(f1_score(y_val, y_pred))
        scores["recall"].append(recall_score(y_val, y_pred))

        captured = meta_val.loc[(y_val == 1) & (y_pred == 1), "CLTV"].sum()
        lost = meta_val.loc[(y_val == 1) & (y_pred == 0), "CLTV"].sum()
        cltv_tp_mean = meta_val.loc[(y_val == 1) & (y_pred == 1), "CLTV"].mean()

        scores["captured_value"].append(captured)
        scores["expected_loss"].append(lost)
        scores["cltv_mean"].append(0.0 if np.isnan(cltv_tp_mean) else cltv_tp_mean)
        scores["capture_value_ratio"].append(
            captured / (captured + lost) if (captured + lost) > 0 else 0.0
        )

    return {k: np.array(v) for k, v in scores.items()}


def run_experiment(config_path: str) -> dict:
    """Orquestra o experimento: carrega dados, roda CV, loga no MLflow."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X, y, meta = load_data(config)
    pipeline = build_pipeline(config, X)
    scores = run_cv(pipeline, X, y, meta, config)

    metrics = {
        "recall_mean": np.mean(scores["recall"]),
        "recall_std": np.std(scores["recall"]),
        "f1_mean": np.mean(scores["f1"]),
        "roc_auc_mean": np.mean(scores["roc_auc"]),
        "captured_value_mean": np.mean(scores["captured_value"]),
        "expected_loss_mean": np.mean(scores["expected_loss"]),
        "capture_value_ratio_mean": np.mean(scores["capture_value_ratio"]),
        "cltv_captured_mean": np.mean(scores["cltv_mean"]),
    }

    run_name = config["experiment"]["name"]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config["model"]["params"])
        mlflow.log_param("cv_folds", config["validation"]["n_splits"])
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])

        for feat_name, feat_cfg in config["features"].items():
            if isinstance(feat_cfg, dict) and "enabled" in feat_cfg:
                mlflow.log_param(f"feature__{feat_name}", feat_cfg["enabled"])

        mlflow.log_metrics(metrics)
        mlflow.log_dict(config, "config.yaml")

        final_pipeline = build_pipeline(config, X)
        final_pipeline.fit(X, y, model__sample_weight=meta["CLTV"].values)
        mlflow.sklearn.log_model(final_pipeline, "model")

    _print_results(run_name, metrics)
    return metrics


def _print_results(run_name: str, metrics: dict) -> None:
    print(f"\n{'='*50}")
    print(f"Experimento: {run_name}")
    print(f"{'='*50}")

    print("\n--- Métricas Técnicas (CV) ---")
    print(f"  Recall  {metrics['recall_mean']:.4f}  ± {metrics['recall_std']:.4f}")
    print(f"  F1      {metrics['f1_mean']:.4f}")
    print(f"  AUC     {metrics['roc_auc_mean']:.4f}")

    print("\n--- KPIs de Negócio (CV médio) ---")
    print(f"  CLTV Capturado       R$ {metrics['captured_value_mean']:>10,.0f}")
    print(f"  Perda Esperada       R$ {metrics['expected_loss_mean']:>10,.0f}")
    print(f"  Capture Ratio        {metrics['capture_value_ratio_mean']:>10.2%}")
    print(f"  CLTV Médio Capturado R$ {metrics['cltv_captured_mean']:>10,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experimenta combinações de features com LogisticRegression"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho para o config YAML do experimento (ex: config/base_exp.yaml)",
    )
    args = parser.parse_args()
    run_experiment(args.config)
