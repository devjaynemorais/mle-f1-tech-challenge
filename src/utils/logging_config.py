"""Configuração centralizada de logging estruturado para o projeto."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Retorna logger configurado com handler para stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # evita duplicação com root logger (MLflow, etc.)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
