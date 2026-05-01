"""Helpers para executar a etapa de tuning com Optuna."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_engineer_transformer import FeatureEngineerTransformer
from src.features.geo_transformer import GeoTransformer
from src.utils.exp import MLPClassifierWrapper, require_xgboost

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


def build_mlp_optuna_pipeline(
    preprocessor: Any, fe_params: dict[str, Any], params: dict[str, Any]
) -> Pipeline:
    """Reconstrói o pipeline da MLP para a etapa de Optuna."""
    return Pipeline(
        [
            ("fe", FeatureEngineerTransformer(**fe_params)),
            ("geo", GeoTransformer(strategy="drop")),
            ("prep", preprocessor),
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=int(params["selector__k"]),
                ),
            ),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                MLPClassifierWrapper(
                    output_dim=1,
                    activation=params["model__activation"],
                    hidden_dim=int(params["model__hidden_dim"]),
                    dropout=float(params["model__dropout"]),
                    lr=float(params["model__lr"]),
                    weight_decay=float(params["model__weight_decay"]),
                    batch_size=int(params["model__batch_size"]),
                    max_epochs=80,
                    patience=16,
                    min_delta=1e-3,
                    threshold=0.5,
                    val_size=0.15,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )


def build_xgb_optuna_pipeline(
    preprocessor: Any,
    fe_params: dict[str, Any],
    params: dict[str, Any],
    y_reference: Any,
) -> Pipeline:
    """Reconstrói o pipeline do XGBoost para a etapa de Optuna."""
    xgb_classifier = require_xgboost()
    y_arr = np.asarray(y_reference)
    scale_pos_weight = float((y_arr == 0).sum()) / max(float((y_arr == 1).sum()), 1.0)

    return Pipeline(
        [
            ("fe", FeatureEngineerTransformer(**fe_params)),
            ("geo", GeoTransformer(strategy="drop")),
            ("prep", preprocessor),
            (
                "model",
                xgb_classifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=int(params["model__n_estimators"]),
                    max_depth=int(params["model__max_depth"]),
                    learning_rate=float(params["model__learning_rate"]),
                    min_child_weight=int(params["model__min_child_weight"]),
                    subsample=float(params["model__subsample"]),
                    colsample_bytree=float(params["model__colsample_bytree"]),
                    gamma=float(params["model__gamma"]),
                    reg_alpha=float(params["model__reg_alpha"]),
                    reg_lambda=float(params["model__reg_lambda"]),
                ),
            ),
        ]
    )


def evaluate_candidate_cv(
    estimator: Any, X: Any, y: Any, cv: Any, scoring: Any
) -> dict[str, float]:
    """Executa CV e devolve o pacote padronizado de métricas."""
    cv_res = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    return {
        "pr_auc_mean": float(cv_res["test_pr_auc"].mean()),
        "pr_auc_std": float(cv_res["test_pr_auc"].std()),
        "roc_auc_mean": float(cv_res["test_roc_auc"].mean()),
        "recall_mean": float(cv_res["test_recall"].mean()),
        "precision_mean": float(cv_res["test_precision"].mean()),
        "f1_mean": float(cv_res["test_f1"].mean()),
        "fit_time_mean_s": float(cv_res["fit_time"].mean()),
        "score_time_mean_s": float(cv_res["score_time"].mean()),
    }


def make_optuna_objective(
    model_name: str,
    pipeline_factory: Callable[[dict[str, Any]], Any],
    X: Any,
    y: Any,
    cv: Any,
    scoring: Any,
    search_space: dict[str, Any],
    trial_records: list[dict[str, Any]],
) -> Callable[[optuna.trial.Trial], float]:
    """Cria a função objetivo do Optuna a partir de um builder de pipeline."""

    def objective(trial: Any) -> float:
        params = suggest_params_from_space(trial, search_space)
        estimator = pipeline_factory(params)
        metrics = evaluate_candidate_cv(estimator, X, y, cv, scoring)

        trial_record = {
            "trial_number": trial.number,
            "model_name": model_name,
            **params,
            **metrics,
        }
        trial_records.append(trial_record)
        return metrics["pr_auc_mean"]

    return objective


def build_trials_df(trial_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Consolida os trials em um DataFrame ordenado pelo número do trial."""
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
    """Gera histórico de convergência com melhor valor acumulado."""
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
    """Decide se a busca deve parar pela ausência de melhora mínima relevante."""
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
    search_space: dict[str, Any],
    pipeline_factory: Callable[[dict[str, Any]], Any],
    X: Any,
    y: Any,
    cv: Any,
    scoring: Any,
    timeout_seconds: int = 3600,
    convergence_patience_trials: int = 50,
    convergence_min_improvement: float = 5e-3,
    random_state: int = 42,
) -> tuple[Any, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Executa um estudo do Optuna com parada por convergência."""
    optuna = _require_optuna()
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_records: list[dict[str, Any]] = []

    objective = make_optuna_objective(
        model_name=model_name,
        pipeline_factory=pipeline_factory,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        search_space=search_space,
        trial_records=trial_records,
    )

    def _convergence_callback(study_: Any, trial: Any) -> None:
        del study_, trial
        current_df = build_trials_df(trial_records)
        if should_stop_for_convergence(
            current_df,
            patience_trials=convergence_patience_trials,
            min_improvement=convergence_min_improvement,
        ):
            study.stop()

    study.optimize(
        objective,
        timeout=timeout_seconds,
        callbacks=[_convergence_callback],
    )

    trials_df = build_trials_df(trial_records)
    convergence_df = build_convergence_history(trials_df)

    if trials_df.empty:
        raise RuntimeError(f"O estudo do Optuna nao gerou trials para {model_name}.")

    best_row = (
        trials_df.sort_values(
            ["pr_auc_mean", "roc_auc_mean", "recall_mean"],
            ascending=[False, False, False],
        )
        .iloc[0]
        .to_dict()
    )

    best_params = {key: value for key, value in best_row.items() if key in search_space}

    return study, trials_df, convergence_df, best_params, best_row


def _require_optuna():
    try:
        import optuna  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "optuna nao esta instalado no ambiente atual."
        ) from exc
    return optuna
