"""Carrega o modelo de produção e avalia no conjunto de teste."""
# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.metrics import compute_metrics
from src.models.mlp import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _predict_mlp(X: np.ndarray, model_path: str, scaler_path: str) -> np.ndarray:
    scaler = joblib.load(scaler_path)
    X_sc = scaler.transform(X).astype(np.float32)
    X_tensor = torch.tensor(X_sc).to(DEVICE)

    model = MLP(input_dim=X_sc.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_tensor)).cpu().numpy().ravel()
    return probs


def predict():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Carregando dados de teste...")
    test_df = pd.read_csv(config["data"]["test_path"])
    target = config["features"]["target_column"]
    X_test = test_df.drop(columns=[target]).values
    y_test = test_df[target].values

    model_name = config["model"]["name"]
    model_path = config["model"]["model_path"]

    print(f"Carregando modelo de produção: {model_name} ({model_path})...")

    if model_name == "mlp":
        proba = _predict_mlp(X_test, model_path, config["model"]["mlp_scaler_path"])
    else:
        model = joblib.load(model_path)
        proba = _predict_sklearn(model, X_test)

    metrics = compute_metrics(y_test, proba)

    print(f"\n=== Resultado — Modelo de Produção: {model_name} ===")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")


if __name__ == "__main__":
    predict()
