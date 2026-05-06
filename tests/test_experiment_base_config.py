import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_base_experiment_config_contains_required_sections_for_all_runners():
    config_path = Path("config/experiments/base_exp.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert set(config) >= {
        "data",
        "features",
        "preprocessing",
        "model",
        "cv",
        "tracking",
        "tuning",
        "evaluation",
    }

    assert set(config["data"]) >= {
        "raw_path",
        "target",
        "test_size",
        "random_state",
        "drop_cols",
        "meta_cols",
    }
    assert "CLTV" in config["data"]["meta_cols"]

    assert set(config["features"]) >= {"engineering", "geo"}
    assert "strategy" in config["features"]["geo"]

    assert set(config["preprocessing"]) >= {"scaler"}
    assert set(config["model"]) >= {"name", "params"}

    assert set(config["cv"]) >= {
        "type",
        "n_splits",
        "shuffle",
        "random_state",
        "n_jobs",
        "scoring",
    }
    assert set(config["cv"]["scoring"]) >= {
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1_score",
    }

    assert set(config["tracking"]) >= {"experiment_name", "tracking_uri"}

    assert set(config["tuning"]) >= {
        "primary_metric",
        "direction",
        "n_trials",
        "timeout_seconds",
        "convergence_patience_trials",
        "convergence_min_improvement",
        "search_space",
    }
    assert set(config["tuning"]["search_space"]) >= {
        "logistic_regression",
        "mlp",
        "xgboost",
    }

    assert set(config["evaluation"]) >= {
        "threshold",
        "activation_cost",
        "retention_rate",
    }
