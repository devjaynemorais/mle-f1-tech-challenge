"""Script to read raw data, clean it, and save as interim."""
# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def process_data():
    BASE_DIR = Path(__file__).resolve().parents[2]
    config_path = BASE_DIR / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = BASE_DIR / config["data"]["raw_path"]
    interim_path = BASE_DIR / config["data"]["interim_path"]

    logger.info("Lendo base raw de: %s", raw_path)
    df = pd.read_excel(raw_path)

    logger.info("Verificando dados duplicados...")
    if df.duplicated().sum() > 0:
        logger.info("Removendo %d duplicatas...", df.duplicated().sum())
        df = df.drop_duplicates()
    else:
        logger.info("Nenhum dado duplicado encontrado.")

    logger.info("Verificando dados faltantes...")
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        logger.info("%d dados faltantes encontrados.", n_missing)

    logger.info("Convertendo Total Charges para numérico...")
    df["Total Charges"] = pd.to_numeric(df["Total Charges"].replace(" ", np.nan))
    df["Total Charges"] = df["Total Charges"].fillna(0)

    drop_cols = [
        "Country",
        "State",
        "Lat Long",
        "Churn Label",
        "Churn Reason",
        "Latitude",
        "Longitude",
        "Zip Code",
        "Count",
    ]
    logger.info("Removendo colunas desnecessárias: %s", drop_cols)
    df.drop(columns=drop_cols, inplace=True)

    logger.info("Dataset final: %d linhas x %d colunas", df.shape[0], df.shape[1])

    logger.info("Salvando dados limpos em %s...", interim_path)
    df.to_csv(interim_path, index=False)
    logger.info("make_dataset.py concluído!")
    return df


if __name__ == "__main__":
    process_data()
