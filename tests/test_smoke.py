"""Smoke tests — verifica que os principais módulos carregam e executam sem erros."""
# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import torch
import yaml


def test_config_carrega():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    assert "model" in config
    assert "data" in config
    assert config["model"]["name"] in (
        "mlp",
        "logistic_regression",
        "random_forest",
        "dummy",
    )


def test_modelo_producao_existe():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    model_path = Path(config["model"]["model_path"])
    assert model_path.exists(), f"Modelo não encontrado: {model_path}"


def test_mlp_carrega_e_prediz():
    from src.models.mlp import MLP

    model = MLP(input_dim=31)
    model.eval()
    x = torch.randn(4, 31)
    with torch.no_grad():
        out = torch.sigmoid(model(x))
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_scaler_carrega():
    scaler = joblib.load("models/mlp_scaler.pkl")
    import numpy as np

    X = np.random.randn(5, 31).astype("float32")
    result = scaler.transform(X)
    assert result.shape == (5, 31)


def test_feature_columns_json_existe():
    import json

    path = Path("models/feature_columns.json")
    assert path.exists()
    cols = json.load(open(path))
    assert len(cols) > 0
    assert isinstance(cols[0], str)


def test_mlp_wrapper_fit_predict():
    from sklearn.datasets import make_classification

    from src.utils.exp import MLPClassifierWrapper

    X, y = make_classification(
        n_samples=80,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    clf = MLPClassifierWrapper(
        hidden_dim=16,
        batch_size=16,
        max_epochs=3,
        patience=2,
        val_size=0.2,
        random_state=42,
    )
    clf.fit(X, y)

    proba = clf.predict_proba(X[:10])
    pred = clf.predict(X[:10])

    assert proba.shape == (10, 2)
    assert pred.shape == (10,)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
