from contextlib import contextmanager
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_economic_eval_metric_uses_campaign_economics(monkeypatch):
    from src.experimentation import evaluate as evaluate_module

    captured = {}
    monkeypatch.setattr(
        evaluate_module,
        "compute_campaign_economics",
        lambda **kwargs: captured.update(kwargs) or {"iel": 123.0, "roi": 1.5},
    )

    metric = evaluate_module.build_economic_eval_metric(
        metric_name="iel",
        threshold=0.4,
        activation_cost=80.0,
        retention_rate=0.2,
    )

    eval_df = pd.DataFrame(
        {
            "score": [0.8, 0.3],
            "prediction": [1, 0],
            "target": [1, 0],
            "CLTV": [1000.0, 500.0],
        }
    )

    value = metric.eval_fn(eval_df, metrics={})

    assert value == 123.0
    assert captured["threshold"] == pytest.approx(0.4)
    assert captured["activation_cost"] == pytest.approx(80.0)
    assert captured["retention_rate"] == pytest.approx(0.2)
    assert captured["y_true"].tolist() == [1, 0]
    assert captured["y_prob"].tolist() == [0.8, 0.3]
    assert captured["cltv"].tolist() == [1000.0, 500.0]


def test_evaluate_candidate_trains_logs_model_and_calls_mlflow_evaluate(monkeypatch):
    from src.experimentation import evaluate as evaluate_module

    events = []
    fit_calls = []

    class FakePipeline:
        def fit(self, X, y):
            fit_calls.append((X.copy(), y.copy()))
            return self

        def predict_proba(self, X):
            return pd.DataFrame([[0.2, 0.8], [0.7, 0.3]]).to_numpy()

    @contextmanager
    def fake_start_run(run_name, nested=False):
        events.append(("start_run", run_name, nested))
        yield object()

    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 1, 0], name="target")
    X_test = pd.DataFrame({"feature": [4.0, 5.0]})
    y_test = pd.Series([1, 0], name="target")
    meta_test = pd.DataFrame({"CLTV": [1000.0, 500.0]})

    monkeypatch.setattr(
        evaluate_module,
        "prep_data",
        lambda config: {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "meta_test": meta_test,
        },
    )
    monkeypatch.setattr(evaluate_module, "build_pipeline", lambda config: FakePipeline())
    monkeypatch.setattr(
        evaluate_module.mlflow,
        "set_tracking_uri",
        lambda uri: events.append(("tracking_uri", uri)),
    )
    monkeypatch.setattr(
        evaluate_module.mlflow,
        "set_experiment",
        lambda name: events.append(("set_experiment", name)),
    )
    monkeypatch.setattr(evaluate_module.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(
        evaluate_module.mlflow_sklearn,
        "log_model",
        lambda model, artifact_path: events.append(("log_model", artifact_path, model)),
    )
    monkeypatch.setattr(
        evaluate_module.mlflow_pyfunc,
        "log_model",
        lambda artifact_path, python_model: type(
            "FakeModelInfo",
            (),
            {"model_uri": f"runs:/fake/{artifact_path}", "python_model": python_model},
        )(),
    )

    eval_calls = {}

    def fake_make_metric(*, eval_fn, greater_is_better, name=None, **kwargs):
        return type(
            "FakeMetric",
            (),
            {
                "name": name,
                "greater_is_better": greater_is_better,
                "eval_fn": eval_fn,
            },
        )()

    def fake_evaluate(
        model,
        data,
        *,
        model_type,
        targets,
        predictions,
        evaluators,
        evaluator_config,
        extra_metrics,
        **kwargs,
    ):
        eval_calls["model"] = model
        eval_calls["data"] = data.copy()
        eval_calls["model_type"] = model_type
        eval_calls["targets"] = targets
        eval_calls["predictions"] = predictions
        eval_calls["evaluators"] = evaluators
        eval_calls["evaluator_config"] = evaluator_config
        eval_calls["extra_metrics"] = extra_metrics
        return {"metrics": {"accuracy_score": 1.0}}

    monkeypatch.setattr(evaluate_module, "make_metric", fake_make_metric)
    monkeypatch.setattr(evaluate_module.mlflow, "evaluate", fake_evaluate)

    config = {
        "model": {"name": "logistic_regression"},
        "tracking": {
            "experiment_name": "exp-eval",
            "tracking_uri": "sqlite:///mlflow.db",
        },
        "evaluation": {
            "threshold": 0.35,
            "activation_cost": 50.0,
            "retention_rate": 0.1,
            "explainability_algorithm": "permutation",
        },
    }

    result = evaluate_module.evaluate_candidate(config)

    assert ("tracking_uri", "sqlite:///mlflow.db") in events
    assert ("set_experiment", "exp-eval") in events
    assert ("start_run", "logistic_regression_holdout", False) in events
    assert fit_calls[0][0].equals(X_train)
    assert fit_calls[0][1].equals(y_train)
    assert eval_calls["model"] == "runs:/fake/candidate_eval_model"
    assert eval_calls["model_type"] == "classifier"
    assert eval_calls["targets"] == "target"
    assert eval_calls["predictions"] == "prediction"
    assert eval_calls["evaluators"] == ["default"]
    assert eval_calls["evaluator_config"]["default"]["log_explainer"] is True
    assert eval_calls["evaluator_config"]["default"]["explainability_algorithm"] == "permutation"
    assert eval_calls["data"]["CLTV"].tolist() == [1000.0, 500.0]
    assert eval_calls["data"]["score"].tolist() == [0.8, 0.3]
    assert eval_calls["data"]["prediction"].tolist() == [1, 0]
    assert len(eval_calls["extra_metrics"]) == 2
    assert [metric.name for metric in eval_calls["extra_metrics"]] == ["iel", "roi"]
    assert result["pipeline"].__class__.__name__ == "FakePipeline"
    assert result["evaluation_result"] == {"metrics": {"accuracy_score": 1.0}}
