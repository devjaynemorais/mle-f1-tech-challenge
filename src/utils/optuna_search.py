"""Helpers para executar a etapa de tuning com Optuna."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from src.experimentation.build_pipeline import build_pipeline

if TYPE_CHECKING:
    import optuna


def suggest_params_from_space(
    trial: Any, search_space: dict[str, Any]
) -> dict[str, Any]:
    """Converte um dicionario de busca reduzido em chamadas suggest_* do Optuna."""
    suggested: dict[str, Any] = {}

    for param_name, spec in search_space.items():
        if isinstance(spec, list):
            suggested[param_name] = trial.suggest_categorical(param_name, spec)
            continue

        if not isinstance(spec, dict):
            raise TypeError(f"Search space invalido para {param_name}: {spec!r}")

        low = spec["low"]
        high = spec["high"]
        step = spec.get("step")
        log = bool(spec.get("log", False))

        is_int_range = (
            isinstance(low, Integral) and isinstance(high, Integral) and not log
        )
        is_int_step = step is not None and isinstance(step, Integral)

        if is_int_range and (step is None or is_int_step):
            suggested[param_name] = trial.suggest_int(
                param_name,
                int(low),
                int(high),
                step=int(step) if step is not None else 1,
            )
            continue

        if step is not None:
            suggested[param_name] = trial.suggest_float(
                param_name,
                float(low),
                float(high),
                step=float(step),
                log=log,
            )
            continue

        suggested[param_name] = trial.suggest_float(
            param_name,
            float(low),
            float(high),
            log=log,
        )

    return suggested


def suggest_params(trial: Any, model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Despacha a sugestao de hiperparametros pelo nome do modelo."""
    search_space = _get_search_space(config, model_name)
    return suggest_params_from_space(trial, search_space)


def prepare_model_params(
    model_name: str,
    params: dict[str, Any],
    config: dict[str, Any],
    y_reference: Any,
) -> dict[str, Any]:
    """Aplica defaults e parametros especiais por modelo sem poluir a objective."""
    base_params = dict(config.get("model", {}).get("params", {}))
    prepared = {**base_params, **params}

    if model_name == "mlp":
        prepared.setdefault("output_dim", 1)
        prepared.setdefault("max_epochs", 80)
        prepared.setdefault("patience", 16)
        prepared.setdefault("min_delta", 1e-3)
        prepared.setdefault("threshold", 0.5)
        prepared.setdefault("val_size", 0.15)
        prepared.setdefault("random_state", config["cv"].get("random_state", 42))
        prepared.setdefault("verbose", False)

    if model_name == "xgboost":
        y_arr = np.asarray(y_reference)
        positive_count = float((y_arr == 1).sum())
        negative_count = float((y_arr == 0).sum())
        prepared.setdefault("objective", "binary:logistic")
        prepared.setdefault("eval_metric", "logloss")
        prepared.setdefault("random_state", config["cv"].get("random_state", 42))
        prepared.setdefault("n_jobs", 1)
        prepared.setdefault(
            "scale_pos_weight",
            negative_count / max(positive_count, 1.0),
        )

    return prepared


def summarize_cv_results(
    cv_res: dict[str, Any],
    scoring: dict[str, Any],
) -> dict[str, float]:
    """Resume as metricas medias e desvios a partir do cross_validate."""
    metrics: dict[str, float] = {}

    for metric_name in scoring:
        scores = cv_res[f"test_{metric_name}"]
        metrics[f"{metric_name}_mean"] = float(np.mean(scores))
        metrics[f"{metric_name}_std"] = float(np.std(scores))

    metrics["fit_time_mean"] = float(np.mean(cv_res["fit_time"]))
    metrics["score_time_mean"] = float(np.mean(cv_res["score_time"]))
    return metrics


def objective(
    trial: Any,
    model_name: str,
    config: dict[str, Any],
    X: Any,
    y: Any,
    cv: Any,
    scoring: dict[str, Any],
    trial_records: list[dict[str, Any]],
) -> float:
    """Objective generica do Optuna, com logging de cada trial em nested run."""
    params = suggest_params(trial, model_name, config)
    prepared_params = prepare_model_params(model_name, params, config, y)

    estimator = build_pipeline(
        model_name=model_name,
        model_params=prepared_params,
        config=config,
        y_reference=y,
    )

    cv_res = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    metrics = summarize_cv_results(cv_res, scoring)
    metric_name = config["tuning"]["primary_metric"]
    score = metrics[f"{metric_name}_mean"]

    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        for metric_key, metric_value in metrics.items():
            mlflow.log_metric(metric_key, metric_value)

    trial_records.append(
        {
            "trial_number": trial.number,
            "model_name": model_name,
            **params,
            **metrics,
        }
    )
    return score


def make_optuna_objective(
    model_name: str,
    config: dict[str, Any],
    X: Any,
    y: Any,
    cv: Any,
    scoring: dict[str, Any],
    trial_records: list[dict[str, Any]],
) -> Callable[[optuna.trial.Trial], float]:
    """Cria a funcao objetivo do Optuna a partir do contrato generico."""

    def _objective(trial: Any) -> float:
        return objective(
            trial=trial,
            model_name=model_name,
            config=config,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            trial_records=trial_records,
        )

    return _objective


def build_trials_df(trial_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Consolida os trials em um DataFrame ordenado pelo numero do trial."""
    if not trial_records:
        return pd.DataFrame()

    trials_df = pd.DataFrame(trial_records)
    if "trial_number" in trials_df.columns:
        trials_df = trials_df.sort_values("trial_number").reset_index(drop=True)
    return trials_df


def build_convergence_history(
    trials_df: pd.DataFrame,
    objective_col: str = "pr_auc_mean",
) -> pd.DataFrame:
    """Gera historico de convergencia com melhor valor acumulado."""
    if trials_df.empty:
        return pd.DataFrame(
            columns=[
                "trial_number",
                "current_pr_auc",
                "best_pr_auc_so_far",
                "improved_best",
            ]
        )

    ordered = trials_df.sort_values("trial_number").reset_index(drop=True).copy()
    ordered["current_pr_auc"] = ordered[objective_col].astype(float)
    ordered["best_pr_auc_so_far"] = ordered["current_pr_auc"].cummax()
    previous_best = ordered["best_pr_auc_so_far"].shift(1, fill_value=-np.inf)
    ordered["improved_best"] = ordered["current_pr_auc"] > previous_best
    return ordered[
        ["trial_number", "current_pr_auc", "best_pr_auc_so_far", "improved_best"]
    ]


def should_stop_for_convergence(
    trials_df: pd.DataFrame,
    patience_trials: int,
    min_improvement: float,
    objective_col: str = "pr_auc_mean",
) -> bool:
    """Decide se a busca deve parar pela ausencia de melhora minima relevante."""
    if trials_df.empty or len(trials_df) < patience_trials:
        return False

    ordered = trials_df.sort_values("trial_number").reset_index(drop=True)
    significant_best = float(ordered.loc[0, objective_col])
    stale_trials = 0

    for value in ordered[objective_col].iloc[1:]:
        current = float(value)
        if current >= significant_best + min_improvement:
            significant_best = current
            stale_trials = 0
        else:
            stale_trials += 1
            if stale_trials >= patience_trials:
                return True

    return False


def run_optuna_study(
    *,
    model_name: str,
    config: dict[str, Any],
    X: Any,
    y: Any,
    cv: Any,
    scoring: dict[str, Any],
    n_trials: int | None = None,
    timeout_seconds: int = 3600,
    convergence_patience_trials: int = 50,
    convergence_min_improvement: float = 5e-3,
    random_state: int = 42,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Executa um estudo do Optuna com objective generica e nested runs por trial."""
    optuna = _require_optuna()
    sampler = optuna.samplers.TPESampler(seed=random_state)
    direction = config.get("tuning", {}).get("direction", "maximize")
    study = optuna.create_study(direction=direction, sampler=sampler)
    trial_records: list[dict[str, Any]] = []

    optuna_objective = make_optuna_objective(
        model_name=model_name,
        config=config,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        trial_records=trial_records,
    )

    primary_metric = config["tuning"]["primary_metric"]

    def _convergence_callback(study_: Any, trial: Any) -> None:
        del study_, trial
        current_df = build_trials_df(trial_records)
        if should_stop_for_convergence(
            current_df,
            patience_trials=convergence_patience_trials,
            min_improvement=convergence_min_improvement,
            objective_col=f"{primary_metric}_mean",
        ):
            study.stop()

    study.optimize(
        optuna_objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        callbacks=[_convergence_callback],
    )

    trials_df = build_trials_df(trial_records)
    convergence_df = build_convergence_history(
        trials_df,
        objective_col=f"{primary_metric}_mean",
    )

    if trials_df.empty:
        raise RuntimeError(f"O estudo do Optuna nao gerou trials para {model_name}.")

    best_trial_number = study.best_trial.number
    best_row = (
        trials_df.loc[trials_df["trial_number"] == best_trial_number]
        .iloc[0]
        .to_dict()
    )
    best_params = dict(study.best_trial.params)
    return study, trials_df, convergence_df, best_params, best_row


def _get_search_space(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    tuning_cfg = config.get("tuning", {})
    search_space = tuning_cfg.get("search_space", {})
    if model_name in search_space and isinstance(search_space[model_name], dict):
        return search_space[model_name]
    return search_space


def _require_optuna():
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "optuna nao esta instalado no ambiente atual."
        ) from exc
    return optuna
