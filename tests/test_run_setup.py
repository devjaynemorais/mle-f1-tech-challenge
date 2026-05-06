import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_execute_setup_runs_experiment_suites_and_optuna_in_expected_order(monkeypatch):
    import run_setup

    call_log = []

    monkeypatch.setattr(
        run_setup,
        "run_experiment_suite",
        lambda *, config_paths=None, config_dir=None: call_log.append(
            ("suite", config_paths, config_dir)
        ),
    )
    monkeypatch.setattr(
        run_setup,
        "load_config",
        lambda path: {"loaded_from": str(path)},
    )
    monkeypatch.setattr(
        run_setup,
        "run_optuna",
        lambda config, config_path=None, set_experiment=True: call_log.append(
            ("optuna", config["loaded_from"], str(config_path), set_experiment)
        ),
    )

    run_setup.execute_setup()

    expected_base = str(run_setup.EXPERIMENTS_DIR / "base_exp_mlp.yaml")
    expected_base_2 = str(run_setup.EXPERIMENTS_DIR / "base_exp_reglog.yaml")
    expected_base_3 = str(run_setup.EXPERIMENTS_DIR / "base_exp_xgb.yaml")
    expected_allfeat = str(run_setup.EXPERIMENTS_DIR / "allfeat_exp_mlp.yaml")
    expected_allfeat_2 = str(run_setup.EXPERIMENTS_DIR / "allfeat_exp_reglog.yaml")
    expected_allfeat_3 = str(run_setup.EXPERIMENTS_DIR / "allfeat_exp_xgb.yaml")
    expected_optuna_mlp = str(run_setup.EXPERIMENTS_DIR / "optuna_mlp.yaml")
    expected_optuna_xgb = str(run_setup.EXPERIMENTS_DIR / "optuna_xgb.yaml")
    expected_optuna_reglog = str(run_setup.EXPERIMENTS_DIR / "optuna_reglog.yaml")

    assert call_log == [
        (
            "suite",
            [expected_base, expected_base_2, expected_base_3],
            None,
        ),
        (
            "suite",
            [expected_allfeat, expected_allfeat_2, expected_allfeat_3],
            None,
        ),
        ("optuna", expected_optuna_mlp, expected_optuna_mlp, True),
        ("optuna", expected_optuna_xgb, expected_optuna_xgb, True),
        ("optuna", expected_optuna_reglog, expected_optuna_reglog, True),
    ]
