"""Treina o modelo de produção definido em config.model.name."""
# ruff: noqa: E402
import io
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import socket

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.models.mlp import MLP, evaluate, train_with_early_stopping
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _check_mlflow_server(tracking_uri: str) -> None:
    if not tracking_uri.startswith("http"):
        return
    from urllib.parse import urlparse

    parsed = urlparse(tracking_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5000
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError:
        logger.error(
            "MLflow nao esta rodando em %s. Suba o servidor antes de treinar:\n"
            "  mlflow server --host 127.0.0.1 --port %d",
            tracking_uri,
            port,
        )
        sys.exit(1)


MLP_MAX_EPOCHS = 100
MLP_PATIENCE = 10
MLP_BATCH_SIZE = 32
MLP_LR = 1e-3
MLP_WD = 1e-5
MLP_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _overfitting_recall(recall_train, recall_test):
    return (recall_train - recall_test) / (recall_train + 1e-9) * 100


def _log_train_test_metrics(metrics_train, metrics_test, overfitting_recall):
    mlflow.log_metric("train_recall", metrics_train["recall"])
    mlflow.log_metric("train_precision", metrics_train["precision"])
    mlflow.log_metric("train_f1", metrics_train["f1"])
    mlflow.log_metric("train_auc", metrics_train["auc"])
    mlflow.log_metric("test_recall", metrics_test["recall"])
    mlflow.log_metric("test_precision", metrics_test["precision"])
    mlflow.log_metric("test_f1", metrics_test["f1"])
    mlflow.log_metric("test_auc", metrics_test["auc"])
    mlflow.log_metric("overfitting_recall_pct", overfitting_recall)


def train_dummy(X_train, y_train, X_test, y_test, config):
    model = DummyClassifier(
        strategy="most_frequent", random_state=config["model"]["random_state"]
    )
    model.fit(X_train, y_train)
    metrics_train = compute_metrics(y_train, model.predict_proba(X_train)[:, 1])
    metrics_test = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
    overfitting = _overfitting_recall(metrics_train["recall"], metrics_test["recall"])

    with mlflow.start_run(run_name="DummyClassifier"):
        mlflow.log_param("model", "DummyClassifier")
        mlflow.log_param("strategy", "most_frequent")
        _log_train_test_metrics(metrics_train, metrics_test, overfitting)
        mlflow.sklearn.log_model(model, "dummy_classifier")

    joblib.dump(model, config["model"]["model_path"])
    logger.info(
        "Train recall=%.4f  auc=%.4f | Test recall=%.4f  auc=%.4f",
        metrics_train["recall"],
        metrics_train["auc"],
        metrics_test["recall"],
        metrics_test["auc"],
    )


def train_logistic_regression(X_train, y_train, X_test, y_test, config):
    cv_config = config["validation"]
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=config["model"]["random_state"],
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=config["model"]["params"]["C"],
                    max_iter=config["model"]["params"]["max_iter"],
                    class_weight=config["model"]["params"]["class_weight"],
                    random_state=config["model"]["random_state"],
                ),
            ),
        ]
    )
    scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "auc": "roc_auc",
    }
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    pipeline.fit(X_train, y_train)
    metrics_train = compute_metrics(y_train, pipeline.predict_proba(X_train)[:, 1])
    metrics_test = compute_metrics(y_test, pipeline.predict_proba(X_test)[:, 1])
    overfitting = _overfitting_recall(metrics_train["recall"], metrics_test["recall"])

    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", config["model"]["params"]["C"])
        mlflow.log_param("cv_folds", cv_config["n_splits"])
        mlflow.log_metric("cv_recall", np.mean(cv_results["test_recall"]))
        mlflow.log_metric("cv_auc", np.mean(cv_results["test_auc"]))
        _log_train_test_metrics(metrics_train, metrics_test, overfitting)
        mlflow.sklearn.log_model(pipeline, "logistic_regression_pipeline")

    joblib.dump(pipeline, config["model"]["model_path"])
    cv_recall = np.mean(cv_results["test_recall"])
    cv_auc = np.mean(cv_results["test_auc"])
    logger.info("CV recall=%.4f  auc=%.4f", cv_recall, cv_auc)
    logger.info(
        "Train recall=%.4f  auc=%.4f | Test recall=%.4f  auc=%.4f | Overfit=%.1f%%",
        metrics_train["recall"],
        metrics_train["auc"],
        metrics_test["recall"],
        metrics_test["auc"],
        overfitting,
    )


def train_random_forest(X_train, y_train, X_test, y_test, config):
    model = RandomForestClassifier(random_state=config["model"]["random_state"])
    model.fit(X_train, y_train)
    metrics_train = compute_metrics(y_train, model.predict_proba(X_train)[:, 1])
    metrics_test = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
    overfitting = _overfitting_recall(metrics_train["recall"], metrics_test["recall"])

    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_param("model", "RandomForest")
        _log_train_test_metrics(metrics_train, metrics_test, overfitting)
        mlflow.sklearn.log_model(model, "rf_baseline")

    joblib.dump(model, config["model"]["model_path"])
    logger.info(
        "Train recall=%.4f  auc=%.4f | Test recall=%.4f  auc=%.4f | Overfit=%.1f%%",
        metrics_train["recall"],
        metrics_train["auc"],
        metrics_test["recall"],
        metrics_test["auc"],
        overfitting,
    )


def train_mlp(X_train, y_train, X_test, y_test, config):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)
    y_tr = y_train.astype(np.float32)
    y_te = y_test.astype(np.float32)

    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_sc), torch.tensor(y_tr)),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_sc), torch.tensor(y_te)),
        batch_size=MLP_BATCH_SIZE,
    )

    model = MLP(input_dim=X_train_sc.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)

    logger.info(
        "Dispositivo: %s | max_epochs=%d | patience=%d",
        DEVICE,
        MLP_MAX_EPOCHS,
        MLP_PATIENCE,
    )
    epochs_ran = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        DEVICE,
        MLP_MAX_EPOCHS,
        MLP_PATIENCE,
        MLP_THRESHOLD,
        logger,
    )

    _, metrics_train = evaluate(model, train_loader, criterion, DEVICE, MLP_THRESHOLD)
    _, metrics_test = evaluate(model, val_loader, criterion, DEVICE, MLP_THRESHOLD)
    overfitting = _overfitting_recall(metrics_train["recall"], metrics_test["recall"])

    with mlflow.start_run(run_name="MLP"):
        mlflow.log_param("model", "MLP")
        mlflow.log_param("hidden_dim", 64)
        mlflow.log_param("max_epochs", MLP_MAX_EPOCHS)
        mlflow.log_param("epochs_ran", epochs_ran)
        mlflow.log_param("patience", MLP_PATIENCE)
        mlflow.log_param("batch_size", MLP_BATCH_SIZE)
        mlflow.log_param("lr", MLP_LR)
        mlflow.log_param("weight_decay", MLP_WD)
        _log_train_test_metrics(metrics_train, metrics_test, overfitting)
        mlflow.pytorch.log_model(model, "mlp_baseline")

    torch.save(model.state_dict(), config["model"]["model_path"])
    joblib.dump(scaler, config["model"]["mlp_scaler_path"])
    logger.info("Modelo salvo em %s", config["model"]["model_path"])
    logger.info(
        "Train recall=%.4f  auc=%.4f | Test recall=%.4f  auc=%.4f | Overfit=%.1f%%",
        metrics_train["recall"],
        metrics_train["auc"],
        metrics_test["recall"],
        metrics_test["auc"],
        overfitting,
    )


MODELS_TO_TRAIN = {
    "dummy": train_dummy,
    "logistic_regression": train_logistic_regression,
    "random_forest": train_random_forest,
    "mlp": train_mlp,
}


def train():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    _check_mlflow_server(config["mlflow"]["tracking_uri"])

    model_name = config["model"]["name"]
    if model_name not in MODELS_TO_TRAIN:
        opts = list(MODELS_TO_TRAIN.keys())
        raise ValueError(f"Modelo '{model_name}' não reconhecido. Opções: {opts}")

    logger.info("Carregando base de treino...")
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])

    target = config["features"]["target_column"]
    X_train = train_df.drop(columns=[target]).values
    y_train = train_df[target].values
    X_test = test_df.drop(columns=[target]).values
    y_test = test_df[target].values

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Treinando modelo de produção: %s", model_name)
    MODELS_TO_TRAIN[model_name](X_train, y_train, X_test, y_test, config)
    logger.info("Modelo salvo em %s", config["model"]["model_path"])


if __name__ == "__main__":
    train()
