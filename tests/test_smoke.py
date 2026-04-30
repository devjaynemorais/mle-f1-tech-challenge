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


def test_train_with_early_stopping_respects_min_delta(monkeypatch):
    import src.models.mlp as mlp_module
    from src.models.mlp import MLP, train_with_early_stopping

    val_losses = iter([1.0, 0.9995, 0.9992, 0.9991])

    monkeypatch.setattr(mlp_module, "train_epoch", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        mlp_module,
        "evaluate",
        lambda *args, **kwargs: (next(val_losses), {}),
    )

    model = MLP(input_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    epochs_ran = train_with_early_stopping(
        model=model,
        train_loader=[],
        val_loader=[],
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        max_epochs=10,
        patience=2,
        min_delta=1e-3,
    )

    assert epochs_ran == 3


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


def test_experimentation_notebook_contains_random_search_exploration_section():
    import json

    notebook = json.loads(Path("notebooks/02_experimentation.ipynb").read_text(encoding="utf-8"))
    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "def prepare_random_search_analysis" in notebook_source
    assert "def build_metric_summary_by_param" in notebook_source
    assert "def build_top20_stability_summary" in notebook_source
    assert "def build_pairwise_interaction_summary" in notebook_source
    assert "Analise exploratoria dos resultados do RandomizedSearchCV" in notebook_source
    assert "mlp_optuna_analysis_summary" in notebook_source
    assert "xgb_optuna_analysis_summary" in notebook_source
