"""Executa inferencia offline com o modelo de producao materializado do MLflow."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.models.production import (
    BASE_DIR,
    CONFIG_PATH,
    build_inference_dataframe,
    evaluate_prediction_frame,
    load_production_model,
    load_production_settings,
    predict_with_threshold,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa inferencia offline usando o pipeline final do MLflow.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="CSV bruto para inferencia. Se omitido, usa o caminho configurado no YAML.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="CSV de saida com probabilidades e labels. Se omitido, usa o YAML.",
    )
    return parser.parse_args()


def predict(input_path: str | Path | None = None, output_path: str | Path | None = None):
    settings = load_production_settings(CONFIG_PATH, workspace_root=BASE_DIR)
    model = load_production_model(settings, prefer_local=True)

    csv_input_path = Path(input_path) if input_path is not None else settings.input_path
    csv_output_path = (
        Path(output_path) if output_path is not None else settings.output_path
    )

    logger.info("Carregando base de inferencia: %s", csv_input_path)
    raw_df = pd.read_csv(csv_input_path)

    y_true = None
    if settings.target_column and settings.target_column in raw_df.columns:
        y_true = raw_df[settings.target_column].to_numpy()

    X_inference = build_inference_dataframe(
        raw_df,
        required_columns=settings.required_columns,
        optional_columns=settings.optional_columns,
    )
    prediction_frame = predict_with_threshold(
        model,
        X_inference,
        threshold=settings.threshold,
    )

    result_df = raw_df.copy()
    result_df["churn_probability"] = prediction_frame["churn_probability"].to_numpy()
    result_df["churn_label"] = prediction_frame["churn_label"].to_numpy()

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_output_path, index=False)
    logger.info("Predicoes salvas em %s", csv_output_path)

    if y_true is not None:
        metrics = evaluate_prediction_frame(
            y_true,
            prediction_frame,
            threshold=settings.threshold,
        )
        logger.info("=== Resultado - Modelo de Producao: %s ===", settings.model_name)
        logger.info("  Threshold : %.2f", settings.threshold)
        logger.info("  Recall    : %.4f", metrics["recall"])
        logger.info("  Precision : %.4f", metrics["precision"])
        logger.info("  F1        : %.4f", metrics["f1"])
        logger.info("  AUC       : %.4f", metrics["auc"])

    return result_df


if __name__ == "__main__":
    cli_args = parse_args()
    predict(input_path=cli_args.input_path, output_path=cli_args.output_path)
