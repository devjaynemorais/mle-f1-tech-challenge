"""Shared serving helpers for the production model serialized as a local pickle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import compute_metrics
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

FIELD_TO_RAW_COLUMN = {
    "gender": "Gender",
    "Gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "Senior Citizen": "Senior Citizen",
    "partner": "Partner",
    "Partner": "Partner",
    "dependents": "Dependents",
    "Dependents": "Dependents",
    "tenure_months": "Tenure Months",
    "Tenure Months": "Tenure Months",
    "phone_service": "Phone Service",
    "Phone Service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "Multiple Lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "Internet Service": "Internet Service",
    "online_security": "Online Security",
    "Online Security": "Online Security",
    "online_backup": "Online Backup",
    "Online Backup": "Online Backup",
    "device_protection": "Device Protection",
    "Device Protection": "Device Protection",
    "tech_support": "Tech Support",
    "Tech Support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "Streaming TV": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "Streaming Movies": "Streaming Movies",
    "contract": "Contract",
    "Contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "Paperless Billing": "Paperless Billing",
    "payment_method": "Payment Method",
    "Payment Method": "Payment Method",
    "monthly_charges": "Monthly Charges",
    "Monthly Charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "Total Charges": "Total Charges",
    "cltv": "CLTV",
    "CLTV": "CLTV",
    "city": "City",
    "City": "City",
    "zip_code": "Zip Code",
    "Zip Code": "Zip Code",
    "latitude": "Latitude",
    "Latitude": "Latitude",
    "longitude": "Longitude",
    "Longitude": "Longitude",
    "lat_long": "Lat Long",
    "Lat Long": "Lat Long",
    "churn_score": "Churn Score",
    "Churn Score": "Churn Score",
}

KNOWN_RAW_COLUMNS = set(FIELD_TO_RAW_COLUMN.values())


@dataclass(frozen=True)
class ProductionModelSettings:
    """Configuration required to serve the production model."""

    model_key: str
    model_name: str
    framework: str
    threshold: float
    model_path: Path
    metadata_path: Path
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    input_path: Path
    output_path: Path
    target_column: str | None


def _normalize_column_list(columns: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []

    for column in columns:
        if column in KNOWN_RAW_COLUMNS:
            normalized.append(column)
            continue

        if "-" in column:
            parts = [part.strip() for part in column.split("-")]
            if parts and all(part in KNOWN_RAW_COLUMNS for part in parts):
                normalized.extend(parts)
                continue

        normalized.append(column)

    return tuple(normalized)


def load_yaml_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the project YAML configuration."""
    with open(config_path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_production_settings(
    config_path: str | Path = CONFIG_PATH,
    *,
    workspace_root: str | Path = BASE_DIR,
) -> ProductionModelSettings:
    """Read the active production-model settings from the project YAML."""
    config = load_yaml_config(config_path)
    production_cfg = config["production"]
    model_key = production_cfg["active_model"]
    model_cfg = production_cfg["models"][model_key]
    local_cfg = model_cfg["local"]
    inference_cfg = model_cfg["inference"]
    pipeline_cfg = model_cfg["pipeline"]

    return ProductionModelSettings(
        model_key=model_key,
        model_name=model_cfg["display_name"],
        framework=model_cfg["framework"],
        threshold=float(model_cfg["threshold"]),
        model_path=Path(workspace_root) / model_cfg["model_path"],
        metadata_path=Path(workspace_root) / local_cfg["metadata_path"],
        required_columns=_normalize_column_list(pipeline_cfg["required_columns"]),
        optional_columns=_normalize_column_list(pipeline_cfg.get("optional_columns", [])),
        input_path=Path(workspace_root) / inference_cfg["input_path"],
        output_path=Path(workspace_root) / inference_cfg["output_path"],
        target_column=inference_cfg.get("target_column"),
    )


def _normalize_record(record: Any) -> dict[str, Any]:
    """Convert API records or dictionaries to the raw feature schema."""
    if isinstance(record, Mapping):
        raw_record = dict(record)
    elif hasattr(record, "model_dump"):
        raw_record = record.model_dump(by_alias=True)
    elif hasattr(record, "dict"):
        raw_record = record.dict(by_alias=True)
    else:
        raise TypeError(
            "Unsupported record type. Expected Mapping or Pydantic-compatible model."
        )

    normalized: dict[str, Any] = {}
    for key, value in raw_record.items():
        normalized[FIELD_TO_RAW_COLUMN.get(key, key)] = value
    return normalized


def build_inference_dataframe(
    data: pd.DataFrame | Sequence[Any],
    *,
    required_columns: Sequence[str],
    optional_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Normalize batch input and align it to the production raw schema."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df = df.rename(columns=lambda column: FIELD_TO_RAW_COLUMN.get(column, column))
    else:
        df = pd.DataFrame([_normalize_record(record) for record in data])

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing required inference columns: {missing_str}")

    ordered_columns = list(required_columns) + [
        column for column in optional_columns if column in df.columns
    ]
    return df.loc[:, ordered_columns].copy()


def _build_serving_metadata(settings: ProductionModelSettings) -> dict[str, Any]:
    """Serialize the production-serving settings for auditability."""
    return {
        "model_key": settings.model_key,
        "model_name": settings.model_name,
        "framework": settings.framework,
        "threshold": settings.threshold,
        "model_path": str(settings.model_path),
        "required_columns": list(settings.required_columns),
        "optional_columns": list(settings.optional_columns),
    }


def materialize_production_model(
    settings: ProductionModelSettings,
    *,
    force_download: bool = False,
) -> Path:
    """Validate the local model artifact and ensure serving metadata exists."""
    del force_download
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Production model artifact not found: {settings.model_path}"
        )

    if not settings.metadata_path.exists():
        settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        settings.metadata_path.write_text(
            json.dumps(_build_serving_metadata(settings), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return settings.model_path


def load_production_model(
    settings: ProductionModelSettings,
    *,
    prefer_local: bool = True,
) -> Any:
    """Load the active production model from the configured local pickle."""
    del prefer_local
    model_path = materialize_production_model(settings)

    if settings.framework != "sklearn":
        raise ValueError(f"Unsupported production framework: {settings.framework}")

    return joblib.load(model_path)


def predict_with_threshold(
    model: Any,
    X: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Run probability inference and apply the configured decision threshold."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The production model must expose predict_proba().")

    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("predict_proba() must return a 2D array with two classes.")

    positive_proba = probabilities[:, 1]
    predictions = (positive_proba >= threshold).astype(int)
    return pd.DataFrame(
        {
            "churn_probability": positive_proba,
            "churn_label": predictions,
        },
        index=X.index,
    )


def evaluate_prediction_frame(
    y_true: Sequence[int] | np.ndarray,
    prediction_frame: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, float]:
    """Compute classification metrics from a prediction frame."""
    return compute_metrics(
        y_true,
        prediction_frame["churn_probability"].to_numpy(dtype=float),
        threshold=threshold,
    )
