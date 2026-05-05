import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_load_data_uses_read_csv_for_csv(monkeypatch):
    from src.experimentation import prep_data as prep_data_module

    captured = {}
    expected = pd.DataFrame({"a": [1]})

    monkeypatch.setattr(
        prep_data_module.pd,
        "read_csv",
        lambda path: captured.update({"path": path}) or expected,
    )

    config = {"data": {"raw_path": "data/raw/sample.csv"}}

    result = prep_data_module.load_data(config)

    assert captured["path"] == "data/raw/sample.csv"
    assert result.equals(expected)


def test_load_data_uses_read_excel_for_xlsx(monkeypatch):
    from src.experimentation import prep_data as prep_data_module

    captured = {}
    expected = pd.DataFrame({"a": [1]})

    monkeypatch.setattr(
        prep_data_module.pd,
        "read_excel",
        lambda path: captured.update({"path": path}) or expected,
    )

    config = {"data": {"raw_path": "data/raw/sample.xlsx"}}

    result = prep_data_module.load_data(config)

    assert captured["path"] == "data/raw/sample.xlsx"
    assert result.equals(expected)
