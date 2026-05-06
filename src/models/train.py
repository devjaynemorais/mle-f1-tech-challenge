"""Train the final production model from an experimentation YAML."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import yaml

from src.experimentation.build_pipeline import build_pipeline
from src.experimentation.prep_data import build_feature_matrices, load_clean_dataset
from src.experimentation.run_experiment import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_CONFIG_PATH = BASE_DIR / "config" / "best_mlp_threshold_params.yaml"
PRODUCTION_CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml_file(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _relative_to_base(path: Path) -> str:
    return os.path.relpath(path.resolve(), BASE_DIR.resolve()).replace("\\", "/")


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return sanitized or "model"


def _default_model_key(model_name: str) -> str:
    if model_name == "mlp":
        return "mlp_optuna_prod"
    return f"{_sanitize_model_name(model_name)}_prod"


def _default_display_name(model_name: str) -> str:
    if model_name == "mlp":
        return "MLP Optuna"
    return model_name.replace("_", " ").title()


def _default_model_dir(model_key: str) -> Path:
    return Path("models") / "production" / model_key.removesuffix("_prod")


def _resolve_threshold(config: dict[str, Any]) -> float:
    evaluation_cfg = config.get("evaluation", {})
    model_threshold = config.get("model", {}).get("params", {}).get("threshold", 0.5)
    return float(evaluation_cfg.get("threshold", model_threshold))


def _ensure_production_model_entry(
    production_config: dict[str, Any],
    *,
    model_name: str,
    feature_columns: list[str],
) -> tuple[str, dict[str, Any]]:
    production_cfg = production_config.setdefault("production", {})
    models_cfg = production_cfg.setdefault("models", {})

    model_key = production_cfg.get("active_model") or _default_model_key(model_name)
    production_cfg["active_model"] = model_key

    model_cfg = models_cfg.setdefault(model_key, {})
    model_cfg.setdefault("display_name", _default_display_name(model_name))
    model_cfg["framework"] = "sklearn"

    model_dir = _default_model_dir(model_key)
    model_cfg["model_path"] = (model_dir / "model.pkl").as_posix()

    local_cfg = model_cfg.setdefault("local", {})
    local_cfg["metadata_path"] = (model_dir / "serving_metadata.json").as_posix()

    pipeline_cfg = model_cfg.setdefault("pipeline", {})
    pipeline_cfg.setdefault("required_columns", list(feature_columns))
    pipeline_cfg.setdefault("optional_columns", [])

    inference_cfg = model_cfg.setdefault("inference", {})
    inference_cfg.setdefault("input_path", "data/interim/telecom_clean.csv")
    inference_cfg.setdefault("output_path", f"models/predictions/{model_key}_predictions.csv")
    inference_cfg.setdefault("target_column", "Churn Value")

    return model_key, model_cfg


def _build_serving_metadata(
    *,
    model_key: str,
    model_cfg: dict[str, Any],
    source_config_path: Path,
) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "model_name": model_cfg["display_name"],
        "framework": model_cfg["framework"],
        "model_path": model_cfg["model_path"],
        "threshold": float(model_cfg["threshold"]),
        "source_experiment_config": _relative_to_base(source_config_path),
        "required_columns": list(model_cfg["pipeline"]["required_columns"]),
        "optional_columns": list(model_cfg["pipeline"].get("optional_columns", [])),
    }


def _write_serving_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train_production_model(
    experiment_config_path: str | Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    production_config_path: str | Path = PRODUCTION_CONFIG_PATH,
) -> dict[str, Any]:
    experiment_config = load_config(experiment_config_path)
    clean_df = load_clean_dataset(experiment_config)
    X, y, _ = build_feature_matrices(clean_df, experiment_config)

    pipeline = build_pipeline(experiment_config)
    pipeline.fit(X, y)

    production_config = load_yaml_file(production_config_path)
    model_name = experiment_config["model"]["name"]
    threshold = _resolve_threshold(experiment_config)

    model_key, model_cfg = _ensure_production_model_entry(
        production_config,
        model_name=model_name,
        feature_columns=list(X.columns),
    )
    model_cfg["threshold"] = threshold

    model_path = BASE_DIR / model_cfg["model_path"]
    metadata_path = BASE_DIR / model_cfg["local"]["metadata_path"]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metadata_payload = _build_serving_metadata(
        model_key=model_key,
        model_cfg=model_cfg,
        source_config_path=Path(experiment_config_path),
    )
    _write_serving_metadata(metadata_path, metadata_payload)
    _write_yaml_file(production_config_path, production_config)

    logger.info("Modelo de producao treinado: %s", model_cfg["display_name"])
    logger.info("Artefato salvo em: %s", model_path)
    logger.info("Threshold de producao: %.2f", threshold)

    return {
        "model_key": model_key,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "threshold": threshold,
        "n_samples": len(X),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina o modelo final de producao a partir de um YAML de experimento.",
    )
    parser.add_argument(
        "experiment_config_path",
        nargs="?",
        default=str(DEFAULT_EXPERIMENT_CONFIG_PATH),
        help="Caminho do YAML completo do experimento aprovado para producao.",
    )
    parser.add_argument(
        "--production-config-path",
        default=str(PRODUCTION_CONFIG_PATH),
        help="Caminho do config.yaml de producao a ser atualizado.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    train_production_model(
        experiment_config_path=args.experiment_config_path,
        production_config_path=args.production_config_path,
    )


if __name__ == "__main__":
    main()
