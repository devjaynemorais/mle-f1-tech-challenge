"""Valida e prepara localmente o artefato de producao serializado em pickle."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.production import (
    CONFIG_PATH,
    BASE_DIR,
    load_production_model,
    load_production_settings,
    materialize_production_model,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida e prepara o modelo de producao local.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Parametro legado. Mantido por compatibilidade, sem efeito no fluxo local.",
    )
    return parser.parse_args()


def prepare_production_model(force_download: bool = True) -> Path:
    """Prepara o modelo de producao, baixando do MLflow se necessario.

    Args:
        force_download: Se True, baixa o modelo do MLflow mesmo que exista localmente.
                       Padrao é True para garantir que o modelo esteja sempre atualizado.
    """
    settings = load_production_settings(CONFIG_PATH, workspace_root=BASE_DIR)
    local_artifact_dir = materialize_production_model(
        settings,
        force_download=force_download,
    )
    loaded_model = load_production_model(settings, prefer_local=True)

    logger.info("Modelo ativo: %s", settings.model_name)
    logger.info("Artefato local: %s", local_artifact_dir)
    logger.info("Threshold de producao: %.2f", settings.threshold)
    logger.info("Classe carregada: %s", type(loaded_model).__name__)

    return local_artifact_dir


if __name__ == "__main__":
    cli_args = parse_args()
    prepare_production_model(force_download=cli_args.force_download)
