import numpy as np
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_data(config) -> pd.DataFrame:
    raw_path = config["data"]["raw_path"]
    file_suffix = Path(raw_path).suffix.lower()

    if file_suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(raw_path)

    return pd.read_csv(raw_path)


def sanity_check(config) -> pd.DataFrame:
    # data load
    logger.info("Lendo base raw de: %s", config["data"]["raw_path"])
    logger.info("Carregando os dados...")
    df = load_data(config)

    logger.info("Iniciando sanity check...")
    logger.info("Preparando os dados...")

    # drop cols

    drop_cols = config["data"].get("drop_cols", [])
    if drop_cols:
        logger.info("Removendo colunas: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # drop_duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.info("Removendo %d duplicatas.", n_duplicates)
        df = df.drop_duplicates()


    # convert Total Charges to numeric
    if "Total Charges" in df.columns:
        logger.info("Convertendo Total Charges para numérico.")
        total_charges = df["Total Charges"].astype("string").str.strip()
        total_charges = total_charges.mask(total_charges.eq(""), pd.NA)
        df["Total Charges"] = pd.to_numeric(total_charges, errors="coerce")
        df["Total Charges"] = df["Total Charges"].fillna(0)


    # missing values
    logger.info("Verificando dados faltantes...")
    n_missing = df.isnull().sum().sum()
    logger.info("Total de valores faltantes: %d", n_missing)

    logger.info("Sanity Check finalizado. Dados preparados com sucesso.")

    return df


def split_data(df, config):
    target = config["data"]["target"]
    meta_cols = config["data"].get("meta_cols", [])

    feature_cols = [
        col for col in df.columns
        if col not in [target] + meta_cols
    ]

    X = df[feature_cols]
    y = df[target]
    metadata = df[meta_cols]  

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def prep_data(config):
    
    df = sanity_check(config)

    X_train, X_test, y_train, y_test, meta_train, meta_test = split_data(df, config)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
    }
