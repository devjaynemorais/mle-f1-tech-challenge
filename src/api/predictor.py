"""Carrega o modelo de produção e executa predições para a API."""

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.api.schemas import CustomerFeatures, PredictionResult
from src.models.mlp import MLP
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

# Mapeamento: campo snake_case do schema → nome original da coluna
_FIELD_TO_COL = {
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure_months": "Tenure Months",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "cltv": "CLTV",
}

_CAT_COLS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
_NUM_COLS = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChurnPredictor:
    """Carrega e mantém o modelo de produção em memória."""

    def __init__(self):
        with open(CONFIG_PATH) as f:
            self._config = yaml.safe_load(f)

        self.model_name: str = self._config["model"]["name"]
        self.model_path: str = self._config["model"]["model_path"]

        self._scaler = joblib.load(BASE_DIR / self._config["model"]["scaler_path"])
        with open(BASE_DIR / "models" / "feature_columns.json") as f:
            self._feature_cols: List[str] = json.load(f)

        if self.model_name == "mlp":
            self._mlp_scaler = joblib.load(
                BASE_DIR / self._config["model"]["mlp_scaler_path"]
            )
            n_features = len(self._feature_cols)
            self._model = MLP(input_dim=n_features).to(DEVICE)
            self._model.load_state_dict(
                torch.load(BASE_DIR / self.model_path, map_location=DEVICE)
            )
            self._model.eval()
        else:
            self._model = joblib.load(BASE_DIR / self.model_path)

        logger.info("Modelo carregado: %s (%s)", self.model_name, self.model_path)

    def _records_to_df(self, records: List[CustomerFeatures]) -> pd.DataFrame:
        rows = []
        for r in records:
            row = {_FIELD_TO_COL[f]: getattr(r, f) for f in _FIELD_TO_COL}
            rows.append(row)
        df = pd.DataFrame(rows)

        df = pd.get_dummies(df, columns=_CAT_COLS, drop_first=True)
        df = df.reindex(columns=self._feature_cols, fill_value=0)

        num_present = [c for c in _NUM_COLS if c in df.columns]
        df[num_present] = self._scaler.transform(df[num_present])
        return df

    def predict_batch(
        self, records: List[CustomerFeatures], threshold: float = 0.5
    ) -> List[PredictionResult]:
        df = self._records_to_df(records)
        X = df.values.astype(np.float32)

        if self.model_name == "mlp":
            X_sc = self._mlp_scaler.transform(X)
            tensor = torch.tensor(X_sc).to(DEVICE)
            with torch.no_grad():
                probs = torch.sigmoid(self._model(tensor)).cpu().numpy().ravel()
        else:
            probs = self._model.predict_proba(X)[:, 1]

        return [
            PredictionResult(
                churn_probability=float(p),
                churn_label=int(p >= threshold),
            )
            for p in probs
        ]


_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    """Singleton — carrega o modelo uma vez e reutiliza."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
