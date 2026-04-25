"""Configuração centralizada de logging estruturado para o projeto."""

import io
import logging
import sys


def _utf8_stdout() -> io.TextIOWrapper:
    """Retorna stdout com encoding UTF-8 (necessário no Windows com cp1252)."""
    if hasattr(sys.stdout, "buffer"):
        return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    return sys.stdout


def get_logger(name: str) -> logging.Logger:
    """Retorna logger configurado com handler UTF-8 para stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # evita duplicação com root logger (MLflow, etc.)
    handler = logging.StreamHandler(_utf8_stdout())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
