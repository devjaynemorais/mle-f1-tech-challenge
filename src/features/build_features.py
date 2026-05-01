"""Script to turn interim data into processed features for modeling."""
# ruff: noqa: E402
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def execute_features():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("Lendo a base interim...")
    df = pd.read_csv(config["data"]["interim_path"])

    target = config["features"]["target_column"]
    drop_cols = config["features"]["drop_columns"]
    cat_cols = config["features"]["categorical_columns"]
    num_cols = config["features"]["numerical_columns"]

    logger.info("Removendo colunas desnecessárias: %s", drop_cols)
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    logger.info("Aplicando One-Hot Encoding nas categóricas...")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    logger.info("Separando features (X) e target (y)...")
    X = df.drop(columns=[target])
    y = df[target]

    logger.info("Executando split de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    logger.info("Escalonando variáveis numéricas...")
    num_cols_encoded = [c for c in num_cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train[num_cols_encoded] = scaler.fit_transform(X_train[num_cols_encoded])
    X_test[num_cols_encoded] = scaler.transform(X_test[num_cols_encoded])

    logger.info("Salvando feature columns para uso na API...")
    with open("models/feature_columns.json", "w") as f:
        json.dump(list(X_train.columns), f)

    logger.info("Salvando Scaler (artefato)...")
    joblib.dump(scaler, config["model"]["scaler_path"])

    logger.info("Salvando bases processadas prontas para modelagem...")
    pd.concat([X_train, y_train], axis=1).to_csv(
        config["data"]["train_path"], index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(config["data"]["test_path"], index=False)
    logger.info("build_features.py concluído!")


if __name__ == "__main__":
    execute_features()
