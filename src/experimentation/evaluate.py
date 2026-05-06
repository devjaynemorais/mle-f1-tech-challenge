from __future__ import annotations

from typing import Any

import mlflow
from mlflow import pyfunc as mlflow_pyfunc
from mlflow import sklearn as mlflow_sklearn
from mlflow.models import make_metric
import pandas as pd

from src.experimentation.build_pipeline import build_pipeline
from src.experimentation.prep_data import prep_data
from src.experimentation.tracking import apply_tracking_config, get_tracking_config
from src.utils.exp import compute_campaign_economics


class FeatureSubsetPyfuncModel(mlflow_pyfunc.PythonModel):
    """Wrapper pyfunc para ignorar colunas auxiliares durante o evaluate."""

    def __init__(self, pipeline: Any, feature_columns: list[str], threshold: float):
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.threshold = threshold

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.feature_columns]

    def predict(self, context, model_input, params=None):
        del context, params
        X = self._select_features(model_input)
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X)
            if getattr(proba, "ndim", 1) == 2:
                return (proba[:, 1] >= self.threshold).astype(int)
            return proba
        return self.pipeline.predict(X)


def _get_evaluation_config(config: dict[str, Any]) -> dict[str, float]:
    evaluation_cfg = config.get("evaluation", {})
    return {
        "threshold": float(evaluation_cfg.get("threshold", 0.5)),
        "activation_cost": float(evaluation_cfg.get("activation_cost", 50.0)),
        "retention_rate": float(evaluation_cfg.get("retention_rate", 0.1)),
        "explainability_algorithm": evaluation_cfg.get("explainability_algorithm", "permutation"),
    }


def build_economic_eval_metric(
    *,
    metric_name: str,
    threshold: float,
    activation_cost: float,
    retention_rate: float,
):
    """Cria uma metrica economica compatível com `mlflow.evaluate`."""

    def _eval_fn(
        predictions,
        targets=None,
        metrics=None,
        CLTV=None,
        score=None,
    ) -> float:
        del metrics
        if isinstance(predictions, pd.DataFrame):
            eval_df = predictions
            y_true = eval_df["target"]
            y_prob = eval_df["score"] if "score" in eval_df.columns else eval_df["prediction"]
            cltv = eval_df["CLTV"]
        else:
            y_true = targets
            y_prob = score if score is not None else predictions
            if CLTV is None:
                raise KeyError("A metrica economica requer a coluna 'CLTV'.")
            cltv = CLTV

        economics = compute_campaign_economics(
            y_true=y_true,
            y_prob=y_prob,
            cltv=cltv,
            threshold=threshold,
            activation_cost=activation_cost,
            retention_rate=retention_rate,
        )
        return float(economics[metric_name])

    greater_is_better = metric_name in {"iel", "roi", "vrec", "vr"}
    return make_metric(
        eval_fn=_eval_fn,
        greater_is_better=greater_is_better,
        name=metric_name,
    )


def _build_holdout_eval_dataframe(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    y_prob: Any,
    threshold: float,
) -> pd.DataFrame:
    if "CLTV" not in meta_test.columns:
        raise KeyError("meta_test precisa conter a coluna 'CLTV'.")

    eval_df = X_test.reset_index(drop=True).copy()
    eval_df["target"] = y_test.reset_index(drop=True)
    eval_df["score"] = pd.Series(y_prob, index=eval_df.index, dtype=float)
    eval_df["prediction"] = (eval_df["score"] >= threshold).astype(int)
    eval_df["CLTV"] = meta_test.reset_index(drop=True)["CLTV"].astype(float)
    return eval_df


def _predict_positive_class_proba(pipeline: Any, X_test: pd.DataFrame):
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if getattr(proba, "ndim", 1) == 2:
            return proba[:, 1]
        return proba
    return pipeline.predict(X_test)


def evaluate_candidate(config: dict[str, Any], set_experiment: bool = True) -> dict[str, Any]:
    tracking_cfg = get_tracking_config(config)
    evaluation_cfg = _get_evaluation_config(config)
    economic_metric_cfg = {
        "threshold": evaluation_cfg["threshold"],
        "activation_cost": evaluation_cfg["activation_cost"],
        "retention_rate": evaluation_cfg["retention_rate"],
    }

    model_name = config["model"]["name"]
    apply_tracking_config(tracking_cfg, set_experiment=set_experiment)

    split_bundle = prep_data(config)
    X_train = split_bundle["X_train"]
    y_train = split_bundle["y_train"]
    X_test = split_bundle["X_test"]
    y_test = split_bundle["y_test"]
    meta_test = split_bundle["meta_test"]

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_prob = _predict_positive_class_proba(pipeline, X_test)
    eval_df = _build_holdout_eval_dataframe(
        X_test,
        y_test,
        meta_test,
        y_prob,
        threshold=evaluation_cfg["threshold"],
    )

    model_for_evaluation = FeatureSubsetPyfuncModel(
        pipeline=pipeline,
        feature_columns=list(X_train.columns),
        threshold=evaluation_cfg["threshold"],
    )

    extra_metrics = [
        build_economic_eval_metric(metric_name="iel", **economic_metric_cfg),
        build_economic_eval_metric(metric_name="roi", **economic_metric_cfg),
    ]

    with mlflow.start_run(run_name=f"{model_name}_holdout"):
        mlflow_sklearn.log_model(pipeline, artifact_path="candidate_pipeline")
        eval_model_info = mlflow_pyfunc.log_model(
            artifact_path="candidate_eval_model",
            python_model=model_for_evaluation,
        )
        evaluation_result = mlflow.evaluate(
            model=eval_model_info.model_uri,
            data=eval_df,
            model_type="classifier",
            targets="target",
            predictions="prediction",
            evaluators=["default"],
            evaluator_config={
                "default": {
                    "log_explainer": True,
                    "explainability_algorithm": evaluation_cfg["explainability_algorithm"],
                }
            },
            extra_metrics=extra_metrics,
        )

    return {
        "pipeline": pipeline,
        "evaluation_result": evaluation_result,
        "evaluation_dataframe": eval_df,
    }
