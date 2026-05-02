"""Schema tests — valida estrutura e tipos dos dados com pandera."""
# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pandera.pandas as pa
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


def test_interim_schema():
    df = pd.read_csv("data/interim/telecom_clean.csv")
    _INTERIM_SCHEMA.validate(df)


def test_sem_nulos_interim():
    df = pd.read_csv("data/interim/telecom_clean.csv")
    assert df.isnull().sum().sum() == 0, "Dados intermediários contêm nulos"
