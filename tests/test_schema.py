"""Schema tests — valida estrutura e tipos dos dados com pandera."""
# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.pandas import Column, DataFrameSchema

_INTERIM_SCHEMA = DataFrameSchema(
    {
        "CustomerID": Column(str),
        "Tenure Months": Column(int, pa.Check.ge(0)),
        "Monthly Charges": Column(float, pa.Check.ge(0)),
        "Total Charges": Column(float, pa.Check.ge(0)),
        "Churn Value": Column(int, pa.Check.isin([0, 1])),
        "CLTV": Column(int, pa.Check.ge(0)),
    },
    coerce=True,
)

_PROCESSED_SCHEMA = DataFrameSchema(
    {
        "Tenure Months": Column(float),
        "Monthly Charges": Column(float),
        "Total Charges": Column(float),
        "CLTV": Column(float),
        "Churn Value": Column(int, pa.Check.isin([0, 1])),
    },
    coerce=True,
)


def test_interim_schema():
    df = pd.read_csv("data/interim/telecom_clean.csv")
    _INTERIM_SCHEMA.validate(df)


def test_processed_train_schema():
    df = pd.read_csv("data/processed/train.csv")
    _PROCESSED_SCHEMA.validate(df)


def test_processed_test_schema():
    df = pd.read_csv("data/processed/test.csv")
    _PROCESSED_SCHEMA.validate(df)


def test_sem_nulos_interim():
    df = pd.read_csv("data/interim/telecom_clean.csv")
    assert df.isnull().sum().sum() == 0, "Dados intermediários contêm nulos"


def test_split_proporcao():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    total = len(train) + len(test)
    test_ratio = len(test) / total
    assert (
        0.25 <= test_ratio <= 0.35
    ), f"Proporção de teste inesperada: {test_ratio:.2f}"


@pytest.mark.parametrize("col", ["Churn Value"])
def test_target_binario(col):
    for path in ("data/processed/train.csv", "data/processed/test.csv"):
        df = pd.read_csv(path)
        assert set(df[col].unique()).issubset({0, 1}), f"Target não binário em {path}"
