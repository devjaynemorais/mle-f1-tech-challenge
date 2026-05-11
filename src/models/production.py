"""Shared serving helpers for the production model serialized as a local pickle."""

from __future__ import annotations

import json
import os
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


def _get_production_run_id(config: dict[str, Any] | None = None) -> str | None:
    """Extrai o run_id da configuracao de producao, se existir."""
    if config is None:
        config = load_yaml_config()
    production_cfg = config.get("production", {})
    source_cfg = production_cfg.get("source", {})
    return source_cfg.get("run_id")


def materialize_production_model(
    settings: ProductionModelSettings,
    *,
    force_download: bool = False,
) -> Path:
    """Validate the local model artifact and ensure serving metadata exists.

    If the model does not exist locally and force_download is True, attempt to
    download it from MLflow using the run_id from config or the best available run.
    """
    if settings.model_path.exists() and not force_download:
        logger.info("Modelo local encontrado: %s", settings.model_path)
    elif force_download:
        logger.info("Baixando modelo do MLflow para: %s", settings.model_path)
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            # Tenta obter o run_id do config.yaml
            config = load_yaml_config()
            run_id = _get_production_run_id(config)

            if run_id:
                # Verifica se o run existe
                try:
                    run = client.get_run(run_id)
                    logger.info("Usando run_id do config: %s", run_id)
                except Exception:
                    logger.warning(
                        "run_id do config nao existe (%s), buscando melhor run...", run_id
                    )
                    run_id = None

            if not run_id:
                # Busca o melhor run disponivel
                possible_names = [
                    "churn-baseline",
                    "Churn_Prediction_Pipeline",
                    "churn_prediction_pipeline",
                    "mlp_optuna",
                    "optuna-mlp",
                ]

                experiment = None
                for name in possible_names:
                    experiment = client.get_experiment_by_name(name)
                    if experiment is not None:
                        logger.info("Experimento MLflow encontrado: %s", name)
                        break

                if experiment is None:
                    all_experiments = mlflow.search_experiments()
                    exp_names = [e.name for e in all_experiments]
                    raise RuntimeError(
                        f"Experimento nao encontrado no MLflow. "
                        f"Experimentos disponiveis: {exp_names}"
                    )

                # Busca o run com melhor f1_score
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.f1_score DESC", "start_time DESC"],
                    max_results=1,
                )

                if not runs:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1,
                    )

                if not runs:
                    raise RuntimeError(
                        f"Nenhum run encontrado no experimento '{experiment.name}'"
                    )

                run_id = runs[0].info.run_id
                logger.info("Melhor run encontrado: %s", run_id)

            # Baixa o modelo do MLflow - tenta multiplas abordagens
            model_uri = f"runs:/{run_id}/model"
            logger.info("Tentando baixar modelo de: %s", model_uri)

            import shutil

            model_downloaded = False

            # Abordagem 1: Tenta copiar diretamente do filesystem do MLflow
            # Os modelos estao em mlruns/X/models/m-xxx/artifacts/model.pkl
            mlruns_dir = BASE_DIR / "mlruns"
            if mlruns_dir.exists():
                for exp_dir in mlruns_dir.iterdir():
                    if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                        continue

                    models_dir = exp_dir / "models"
                    if not models_dir.exists():
                        continue

                    for model_dir in models_dir.iterdir():
                        if not model_dir.is_dir():
                            continue

                        artifacts_dir = model_dir / "artifacts"
                        if not artifacts_dir.exists():
                            continue

                        model_file = artifacts_dir / "model.pkl"
                        if model_file.exists():
                            settings.model_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(model_file, settings.model_path)
                            logger.info(
                                "Modelo copiado do filesystem MLflow: %s -> %s",
                                model_file,
                                settings.model_path,
                            )
                            model_downloaded = True
                            break

                        # Tenta outros formatos
                        other_files = list(artifacts_dir.glob("*.pkl"))
                        if other_files:
                            settings.model_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(other_files[0], settings.model_path)
                            logger.info(
                                "Modelo copiado (arquivo: %s): %s",
                                other_files[0].name,
                                settings.model_path,
                            )
                            model_downloaded = True
                            break

                    if model_downloaded:
                        break

            # Abordagem 2: Tenta mlflow.artifacts.download_artifacts
            if not model_downloaded:
                try:
                    local_model_path = mlflow.artifacts.download_artifacts(
                        model_uri,
                        dst_path=str(settings.model_path.parent),
                    )

                    model_file = Path(local_model_path) / "model.pkl"
                    if model_file.exists():
                        settings.model_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(model_file, settings.model_path)
                        logger.info("Modelo baixado com sucesso: %s", settings.model_path)
                        model_downloaded = True
                except Exception as e1:
                    logger.warning("download_artifacts falhou: %s", e1)

            # Abordagem 3: Tenta mlflow.sklearn.load_model
            if not model_downloaded:
                try:
                    model = mlflow.sklearn.load_model(model_uri)
                    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, settings.model_path)
                    logger.info(
                        "Modelo carregado e salvo via mlflow.sklearn.load_model: %s",
                        settings.model_path,
                    )
                    model_downloaded = True
                except Exception as e2:
                    logger.error("mlflow.sklearn.load_model falhou: %s", e2)

            if not model_downloaded:
                raise RuntimeError(
                    "Nao foi possivel baixar o modelo do MLflow. "
                    "Verifique se os experimentos foram executados e se ha modelos em mlruns/"
                )

            # Atualiza o config com o run_id usado
            if config and "source" not in config.get("production", {}):
                _update_config_run_id(run_id)

        except ImportError as e:
            raise RuntimeError(
                "MLflow nao instalado. Instale com: pip install mlflow"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Erro ao baixar modelo do MLflow: {e}") from e
    else:
        raise FileNotFoundError(
            f"Production model artifact not found: {settings.model_path}\n"
            "Execute com --force-download ou rode o setup completo para baixar do MLflow."
        )

    if not settings.metadata_path.exists():
        settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        settings.metadata_path.write_text(
            json.dumps(_build_serving_metadata(settings), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return settings.model_path


def _update_config_run_id(run_id: str) -> None:
    """Adiciona o run_id ao config.yaml."""
    import re

    content = CONFIG_PATH.read_text(encoding="utf-8")

    # Adiciona secao source apos active_model
    if "source:" not in content:
        content = re.sub(
            r"(active_model:\s*\S+\n)",
            f"\\1  source:\n    run_id: {run_id}\n",
            content,
        )
        CONFIG_PATH.write_text(content, encoding="utf-8")
        logger.info("Config atualizado com run_id: %s", run_id)


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
