import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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


def test_prepare_model_params_preserves_selector_k_for_mlp():
    from src.utils.optuna_search import prepare_model_params

    prepared = prepare_model_params(
        model_name="mlp",
        params={"hidden_dim": 96, "selector_k": 24},
        config={"model": {"params": {}}, "cv": {"random_state": 42}},
        y_reference="y_train",
    )

    assert prepared["hidden_dim"] == 96
    assert prepared["selector_k"] == 24
    assert prepared["random_state"] == 42


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


def test_generic_objective_logs_nested_trial_runs(monkeypatch):
    from src.utils import optuna_search

    trial = SimpleNamespace(number=7)
    trial_records = []
    logged = {"params": None, "metrics": []}
    build_call = {}

    @contextmanager
    def fake_start_run(run_name, nested=False):
        logged["run_name"] = run_name
        logged["nested"] = nested
        yield object()

    monkeypatch.setattr(
        optuna_search,
        "suggest_params",
        lambda trial, model_name, config: {
            "max_depth": 4,
            "learning_rate": 0.1,
            "selector_k": 18,
        },
    )
    monkeypatch.setattr(
        optuna_search,
        "build_pipeline",
        lambda *, model_name, model_params, config, y_reference=None: build_call.update(
            {
                "model_name": model_name,
                "model_params": model_params,
                "config": config,
                "y_reference": y_reference,
            }
        )
        or "pipeline",
    )
    monkeypatch.setattr(
        optuna_search,
        "cross_validate",
        lambda estimator, X, y, cv, scoring, n_jobs, return_train_score: {
            "test_pr_auc": pd.Series([0.80, 0.90]),
            "test_roc_auc": pd.Series([0.70, 0.75]),
            "test_recall": pd.Series([0.60, 0.65]),
            "test_precision": pd.Series([0.50, 0.55]),
            "test_f1_score": pd.Series([0.54, 0.59]),
            "fit_time": pd.Series([1.0, 2.0]),
            "score_time": pd.Series([0.1, 0.2]),
        },
    )
    monkeypatch.setattr(optuna_search.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(
        optuna_search.mlflow,
        "log_params",
        lambda params: logged.__setitem__("params", params),
    )
    monkeypatch.setattr(
        optuna_search.mlflow,
        "log_metric",
        lambda name, value: logged["metrics"].append((name, value)),
    )

    config = {
        "tuning": {
            "primary_metric": "pr_auc",
        },
        "cv": {
            "scoring": {
                "pr_auc": "average_precision",
                "roc_auc": "roc_auc",
                "recall": "recall",
                "precision": "precision",
                "f1_score": "f1",
            }
        },
    }

    objective = optuna_search.make_optuna_objective(
        model_name="xgboost",
        config=config,
        X="X_train",
        y="y_train",
        cv="cv",
        scoring=config["cv"]["scoring"],
        trial_records=trial_records,
    )

    score = objective(trial)

    assert score == pytest.approx(0.85)
    assert build_call["model_name"] == "xgboost"
    assert build_call["config"] == config
    assert build_call["y_reference"] == "y_train"
    assert build_call["model_params"]["max_depth"] == 4
    assert build_call["model_params"]["learning_rate"] == pytest.approx(0.1)
    assert build_call["model_params"]["selector_k"] == 18
    assert build_call["model_params"]["objective"] == "binary:logistic"
    assert logged["run_name"] == "trial_7"
    assert logged["nested"] is True
    assert logged["params"] == {
        "max_depth": 4,
        "learning_rate": 0.1,
        "selector_k": 18,
    }
    metrics_dict = dict(logged["metrics"])
    assert metrics_dict["pr_auc_mean"] == pytest.approx(0.85)
    assert metrics_dict["pr_auc_std"] == pytest.approx(0.05)
    assert metrics_dict["f1_score_mean"] == pytest.approx(0.565)
    assert len(trial_records) == 1
    assert trial_records[0]["trial_number"] == 7
    assert trial_records[0]["pr_auc_mean"] == pytest.approx(0.85)


def test_run_optuna_study_dispatches_to_legacy_contract(monkeypatch):
    from src.utils import optuna_search

    captured = {}

    monkeypatch.setattr(
        optuna_search,
        "_run_legacy_optuna_study",
        lambda **kwargs: captured.update(kwargs) or ("study", "trials", "history", {}, {}),
    )

    result = optuna_search.run_optuna_study(
        model_name="MLP",
        search_space={"selector__k": {"low": 10, "high": 20}},
        pipeline_factory=lambda params: params,
        X="X_train",
        y="y_train",
        cv="cv",
        scoring={"pr_auc": "average_precision"},
    )

    assert result[0] == "study"
    assert captured["model_name"] == "MLP"
    assert captured["search_space"] == {"selector__k": {"low": 10, "high": 20}}
    assert callable(captured["pipeline_factory"])


def test_legacy_notebook_pipeline_builders_remain_available():
    from sklearn.preprocessing import StandardScaler

    from src.utils.exp import DEFAULT_ROUND4_FE_PARAMS
    from src.utils.optuna_search import build_mlp_optuna_pipeline

    preprocessor = StandardScaler(with_mean=False)
    pipeline = build_mlp_optuna_pipeline(
        preprocessor=preprocessor,
        fe_params=DEFAULT_ROUND4_FE_PARAMS,
        params={
            "selector__k": 12,
            "model__activation": "relu",
            "model__hidden_dim": 64,
            "model__dropout": 0.1,
            "model__lr": 1e-3,
            "model__weight_decay": 1e-5,
            "model__batch_size": 32,
        },
    )

    assert [name for name, _ in pipeline.steps] == [
        "fe",
        "geo",
        "prep",
        "selector",
        "scaler",
        "model",
    ]
    assert pipeline.named_steps["selector"].k == 12
    assert pipeline.named_steps["prep"] is not preprocessor
