import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_resolve_config_paths_supports_explicit_files_and_directory(tmp_path):
    from src.experimentation.run_experiment_suite import resolve_config_paths

    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yml"
    ignored = tmp_path / "notes.txt"

    config_a.write_text("tracking:\n  experiment_name: exp-a\n", encoding="utf-8")
    config_b.write_text("tracking:\n  experiment_name: exp-b\n", encoding="utf-8")
    ignored.write_text("ignore", encoding="utf-8")

    resolved = resolve_config_paths(
        config_paths=[str(config_b)],
        config_dir=str(tmp_path),
    )

    assert resolved == [config_a, config_b]


def test_run_experiment_suite_groups_runs_by_experiment_name(monkeypatch, tmp_path):
    from src.experimentation import run_experiment_suite as suite_module

    config_a = tmp_path / "baseline.yaml"
    config_b = tmp_path / "xgb.yaml"
    config_c = tmp_path / "other.yaml"

    config_a.write_text(
        "tracking:\n  experiment_name: exp-shared\nmodel:\n  name: baseline\n",
        encoding="utf-8",
    )
    config_b.write_text(
        "tracking:\n  experiment_name: exp-shared\nmodel:\n  name: xgboost\n",
        encoding="utf-8",
    )
    config_c.write_text(
        "tracking:\n  experiment_name: exp-other\nmodel:\n  name: mlp\n",
        encoding="utf-8",
    )

    events = []

    monkeypatch.setattr(
        suite_module,
        "resolve_config_paths",
        lambda config_paths=None, config_dir=None: [config_a, config_b, config_c],
    )
    monkeypatch.setattr(
        suite_module,
        "load_config",
        lambda config_path: {
            "tracking": {
                "experiment_name": "exp-shared" if "other" not in str(config_path) else "exp-other",
                "tracking_uri": "sqlite:///mlflow.db",
            },
            "model": {
                "name": Path(config_path).stem,
            },
        },
    )
    monkeypatch.setattr(
        suite_module.mlflow,
        "set_tracking_uri",
        lambda uri: events.append(("tracking_uri", uri)),
    )
    monkeypatch.setattr(
        suite_module.mlflow,
        "set_experiment",
        lambda name: events.append(("set_experiment", name)),
    )
    monkeypatch.setattr(
        suite_module,
        "run_experiment",
        lambda config, set_experiment=False: pd.DataFrame(
            [
                {
                    "model": config["model"]["name"],
                    "pr_auc_mean": 0.8,
                }
            ]
        ),
    )

    result = suite_module.run_experiment_suite(
        config_paths=[str(config_a)],
        config_dir=str(tmp_path),
    )

    assert events == [
        ("tracking_uri", "sqlite:///mlflow.db"),
        ("set_experiment", "exp-other"),
        ("tracking_uri", "sqlite:///mlflow.db"),
        ("set_experiment", "exp-shared"),
    ]
    assert result["config_path"].tolist() == [
        str(config_c),
        str(config_a),
        str(config_b),
    ]
    assert result["experiment_name"].tolist() == [
        "exp-other",
        "exp-shared",
        "exp-shared",
    ]
    assert result["model"].tolist() == ["other", "baseline", "xgb"]

