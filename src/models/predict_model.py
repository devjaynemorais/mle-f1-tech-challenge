"""Avalia todos os modelos treinados no conjunto de teste e exibe tabela comparativa."""
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

SKLEARN_MODELS = {
    "DummyClassifier": "models/dummy_classifier.pkl",
    "LogisticRegression": "models/logistic_regression.pkl",
    "RandomForest": "models/rf_baseline.pkl",
}
MLP_MODEL_PATH = "models/mlp_baseline.pt"
MLP_SCALER_PATH = "models/mlp_scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_sklearn(model, X):
    return model.predict_proba(X)[:, 1]


def predict_mlp(X, scaler):
    X_sc = scaler.transform(X).astype(np.float32)
    X_tensor = torch.tensor(X_sc).to(DEVICE)

    model = MLP(input_dim=X_sc.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=DEVICE))
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

    results = []

    for name, path in SKLEARN_MODELS.items():
        print(f"Avaliando {name}...")
        model = joblib.load(path)
        proba = predict_sklearn(model, X_test)
        metrics = compute_metrics(y_test, proba)
        results.append({"modelo": name, **metrics})

    print("Avaliando MLP...")
    mlp_scaler = joblib.load(MLP_SCALER_PATH)
    proba = predict_mlp(X_test, mlp_scaler)
    metrics = compute_metrics(y_test, proba)
    results.append({"modelo": "MLP", **metrics})

    table = pd.DataFrame(results).set_index("modelo")
    table.columns = ["Recall", "Precision", "F1", "AUC"]
    table = table.sort_values("Recall", ascending=False)

    print("\n=== Comparativo de Modelos (Test Set) ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print()


if __name__ == "__main__":
    predict()
