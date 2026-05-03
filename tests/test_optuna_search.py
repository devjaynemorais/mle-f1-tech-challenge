import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeTrial:
    def __init__(self):
        self.calls = []

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, list(choices)))
        return choices[0]

    def suggest_float(self, name, low, high, log=False, step=None):
        self.calls.append(("float", name, low, high, log, step))
        return low

    def suggest_int(self, name, low, high, step=1):
        self.calls.append(("int", name, low, high, step))
        return low


def test_suggest_params_from_space_supports_categorical_int_and_float():
    from src.utils.optuna_search import suggest_params_from_space

    trial = FakeTrial()
    search_space = {
        "model__activation": ["tanh"],
        "model__hidden_dim": {"low": 32, "high": 64, "step": 2},
        "model__lr": {"low": 3e-4, "high": 1e-3, "log": True},
        "model__dropout": {"low": 0.0, "high": 0.3, "step": 0.1},
    }

    params = suggest_params_from_space(trial, search_space)

    assert params == {
        "model__activation": "tanh",
        "model__hidden_dim": 32,
        "model__lr": 3e-4,
        "model__dropout": 0.0,
    }
    assert ("categorical", "model__activation", ["tanh"]) in trial.calls
    assert ("int", "model__hidden_dim", 32, 64, 2) in trial.calls
    assert ("float", "model__lr", 3e-4, 1e-3, True, None) in trial.calls
    assert ("float", "model__dropout", 0.0, 0.3, False, 0.1) in trial.calls


def test_build_trials_df_returns_sorted_dataframe():
    from src.utils.optuna_search import build_trials_df

    records = [
        {"trial_number": 2, "pr_auc_mean": 0.91},
        {"trial_number": 1, "pr_auc_mean": 0.89},
    ]

    df = build_trials_df(records)

    assert df["trial_number"].tolist() == [1, 2]
    assert list(df.columns) == ["trial_number", "pr_auc_mean"]


def test_build_convergence_history_tracks_best_so_far():
    from src.utils.optuna_search import build_convergence_history

    trials_df = pd.DataFrame(
        [
            {"trial_number": 0, "pr_auc_mean": 0.80},
            {"trial_number": 1, "pr_auc_mean": 0.81},
            {"trial_number": 2, "pr_auc_mean": 0.805},
        ]
    )

    history = build_convergence_history(trials_df)

    assert history["best_pr_auc_so_far"].tolist() == [0.80, 0.81, 0.81]
    assert history["improved_best"].tolist() == [True, True, False]


def test_should_stop_for_convergence_respects_min_improvement_threshold():
    from src.utils.optuna_search import should_stop_for_convergence

    trials_df = pd.DataFrame(
        [
            {"trial_number": 0, "pr_auc_mean": 0.9000},
            {"trial_number": 1, "pr_auc_mean": 0.9030},
            {"trial_number": 2, "pr_auc_mean": 0.9040},
            {"trial_number": 3, "pr_auc_mean": 0.9045},
        ]
    )

    should_stop = should_stop_for_convergence(
        trials_df,
        patience_trials=3,
        min_improvement=5e-3,
    )

    assert should_stop is True


def test_should_not_stop_when_recent_trial_improves_enough():
    from src.utils.optuna_search import should_stop_for_convergence

    trials_df = pd.DataFrame(
        [
            {"trial_number": 0, "pr_auc_mean": 0.9000},
            {"trial_number": 1, "pr_auc_mean": 0.9010},
            {"trial_number": 2, "pr_auc_mean": 0.9065},
        ]
    )

    should_stop = should_stop_for_convergence(
        trials_df,
        patience_trials=2,
        min_improvement=5e-3,
    )

    assert should_stop is False
