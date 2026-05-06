"""Carrega o modelo de producao serializado no MLflow e executa predicoes."""

from __future__ import annotations

from typing import List

from src.api.schemas import CustomerFeatures, PredictionResult
from src.models.production import (
    BASE_DIR,
    CONFIG_PATH,
    build_inference_dataframe,
    load_production_model,
    load_production_settings,
    materialize_production_model,
    predict_with_threshold,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ChurnPredictor:
    """Carrega e mantem o modelo de producao em memoria."""

    def __init__(self):
        self.settings = load_production_settings(
            config_path=CONFIG_PATH,
            workspace_root=BASE_DIR,
        )
        self.model_name = self.settings.model_name
        self.model_uri = str(self.settings.model_path)
        self.threshold = self.settings.threshold
        self.model_path = str(materialize_production_model(self.settings))
        self._model = load_production_model(self.settings, prefer_local=True)

        logger.info(
            "Modelo de producao carregado: %s (%s)",
            self.model_name,
            self.model_uri,
        )

    def _records_to_df(self, records: List[CustomerFeatures]):
        return build_inference_dataframe(
            records,
            required_columns=self.settings.required_columns,
            optional_columns=self.settings.optional_columns,
        )

    def predict_batch(
        self,
        records: List[CustomerFeatures],
        threshold: float | None = None,
    ) -> List[PredictionResult]:
        effective_threshold = self.threshold if threshold is None else threshold
        df = self._records_to_df(records)
        prediction_frame = predict_with_threshold(
            self._model,
            df,
            threshold=effective_threshold,
        )

        return [
            PredictionResult(
                churn_probability=float(row.churn_probability),
                churn_label=int(row.churn_label),
            )
            for row in prediction_frame.itertuples(index=False)
        ]


_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    """Singleton - carrega o modelo uma vez e reutiliza."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
