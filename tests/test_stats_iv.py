import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.stats import AnaliseIV


def make_iv_df():
    return pd.DataFrame(
        {
            "feature": ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C"],
            "target": [1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
        }
    )


def test_iv_is_non_negative_with_good_over_bad_convention():
    df = make_iv_df()

    analysis = AnaliseIV(df=df, target="target", convention="good_over_bad")

    result = analysis.get_lista_iv()
    iv_value = result.loc[result["Variavel"] == "feature", "IV"].iloc[0]

    assert iv_value > 0
    assert result.loc[result["Variavel"] == "feature", "Forca_Preditiva"].iloc[0] != "Irrelevante"


def test_iv_total_is_invariant_to_woe_convention():
    df = make_iv_df()

    event_over_nonevent = AnaliseIV(
        df=df,
        target="target",
        convention="event_over_nonevent",
    )
    good_over_bad = AnaliseIV(
        df=df,
        target="target",
        convention="good_over_bad",
    )

    iv_event = event_over_nonevent.get_lista_iv().loc[
        lambda x: x["Variavel"] == "feature", "IV"
    ].iloc[0]
    iv_good = good_over_bad.get_lista_iv().loc[
        lambda x: x["Variavel"] == "feature", "IV"
    ].iloc[0]

    assert iv_event == pytest.approx(iv_good)


def test_woe_sign_flips_between_conventions_but_iv_component_stays_positive():
    df = make_iv_df()

    event_over_nonevent = AnaliseIV(
        df=df,
        target="target",
        convention="event_over_nonevent",
    ).get_bivariada("feature")
    good_over_bad = AnaliseIV(
        df=df,
        target="target",
        convention="good_over_bad",
    ).get_bivariada("feature")

    merged = event_over_nonevent.merge(
        good_over_bad,
        on=["Variavel", "Var_Range"],
        suffixes=("_event", "_good"),
    )

    assert np.allclose(merged["WOE_event"], -merged["WOE_good"])
    assert (merged["IV_event"] >= 0).all()
    assert (merged["IV_good"] >= 0).all()
