import sys
from pathlib import Path

from sklearn.feature_selection import SelectKBest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_pipeline_adds_selector_for_mlp_when_selector_k_is_present():
    from src.experimentation.build_pipeline import build_pipeline

    config = {
        "features": {"engineering": {}, "geo": {"strategy": "drop"}},
        "preprocessing": {"scaler": {"type": "none"}},
        "model": {
            "name": "mlp",
            "params": {
                "hidden_dim": 64,
                "selector_k": 12,
            },
        },
    }

    pipeline = build_pipeline(config)

    selector = pipeline.named_steps["selector"]
    model = pipeline.named_steps["model"]

    assert isinstance(selector, SelectKBest)
    assert selector.k == 12
    assert not hasattr(model, "selector_k")


def test_build_pipeline_uses_selector_passthrough_for_mlp_without_selector_k():
    from src.experimentation.build_pipeline import build_pipeline

    config = {
        "features": {"engineering": {}, "geo": {"strategy": "drop"}},
        "preprocessing": {"scaler": {"type": "none"}},
        "model": {
            "name": "mlp",
            "params": {
                "hidden_dim": 64,
            },
        },
    }

    pipeline = build_pipeline(config)

    assert pipeline.named_steps["selector"] == "passthrough"


def test_build_pipeline_does_not_enable_selector_for_non_mlp_models():
    from src.experimentation.build_pipeline import build_pipeline

    base_config = {
        "features": {"engineering": {}, "geo": {"strategy": "drop"}},
        "preprocessing": {"scaler": {"type": "none"}},
    }

    xgb_pipeline = build_pipeline(
        {
            **base_config,
            "model": {
                "name": "xgboost",
                "params": {"selector_k": 12, "n_estimators": 10},
            },
        }
    )
    reglog_pipeline = build_pipeline(
        {
            **base_config,
            "model": {
                "name": "logistic_regression",
                "params": {"selector_k": 12, "max_iter": 10},
            },
        }
    )

    assert xgb_pipeline.named_steps["selector"] == "passthrough"
    assert reglog_pipeline.named_steps["selector"] == "passthrough"
