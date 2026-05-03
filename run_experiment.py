"""
Fase completa de experimentação — extração dos notebooks 02 e 03 (análise).

Etapas executadas em sequência:
  1. Baseline        — 6 modelos, CV k-fold, log no MLflow
  2. FE Round 1      — features de hipótese (sem Churn Score)
  3. FE Round 2      — inclui Churn Score
  4. FE Round 3      — estratégias de City encoding
  5. Feature Selection — SelectKBest por k-grid + L1-based
  6. RandomSearch    — MLP e XGBoost (n_iter=100)
  7. Optuna          — MLP e XGBoost (busca fina)
  8. Persiste config/best_params.yaml com os melhores hiperparâmetros

Saída principal: config/best_params.yaml (versionado no git).
run_train.py consome este arquivo e é completamente independente deste script.
"""
# ruff: noqa: E402
from __future__ import annotations

import io
import os
import sys
import warnings
from functools import partial
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_in_venv = sys.prefix != sys.base_prefix or bool(os.environ.get("RUNNING_IN_DOCKER"))
if not _in_venv:
    print(
        "ERRO: ambiente virtual nao esta ativo.\n"
        "Ative antes de executar:\n"
        "  PowerShell : .venv\\Scripts\\Activate.ps1\n"
        "  Git Bash   : source .venv/Scripts/activate\n"
        "  Bash/Mac   : source .venv/bin/activate"
    )
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
BEST_PARAMS_PATH = BASE_DIR / "config" / "best_params.yaml"

sys.path.insert(0, str(BASE_DIR))

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings(
    "ignore", message=".*max_iter.*was reached.*", category=UserWarning
)
try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data.make_dataset import process_data
from src.features.feature_engineer_transformer import FeatureEngineerTransformer
from src.features.geo_transformer import GeoTransformer
from src.utils.exp import (
    DEFAULT_BASELINE_FE_PARAMS,
    DEFAULT_ROUND4_FE_PARAMS,
    MLPClassifierWrapper,
    build_default_cv,
    build_default_preprocessor,
    build_default_scoring,
    evaluate_round3_model_strategies,
    require_xgboost,
    resolve_tracking_uri,
    summarize_grid_search_results,
)
from src.utils.mlflow_tracking import (
    best_row_to_metrics,
    log_dataframe_artifact,
    log_json_artifact,
    log_metrics_safe,
    log_split_artifacts,
    start_experiment_run,
)
from src.utils.optuna_search import (
    build_mlp_optuna_pipeline,
    build_xgb_optuna_pipeline,
    run_optuna_study,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

META_COLS = ["CLTV", "CustomerID"]

MLFLOW_EXPERIMENTS = {
    "baselines": "tc-f1-nb02-baselines",
    "feature_engineering": "tc-f1-nb02-feature-engineering",
    "round3_city_encoding": "tc-f1-nb02-round3-city-encoding",
    "feature_selection": "tc-f1-nb02-feature-selection",
    "tuning_stage1_randomsearch": "tc-f1-nb02-tuning-stage1-randomsearch",
    "tuning_stage2_optuna": "tc-f1-nb02-tuning-stage2-optuna",
}

ROUND1_FE_PARAMS = {
    "drop_churn_score": True,
    "add_engagement_score": True,
    "add_tenure_group": True,
    "add_tenure_log": True,
    "add_contract_ordinal": True,
    "add_family_stability": True,
    "add_fiber_no_support": True,
    "add_support_gap_count": True,
    "add_payment_automatic_flag": True,
    "add_electronic_check_flag": True,
    "add_paperless_echeck_flag": True,
    "add_price_pressure_ratio": True,
}

ROUND2_FE_PARAMS = {**ROUND1_FE_PARAMS, "drop_churn_score": False}

MLP_PARAM_DISTRIBUTIONS = {
    "selector__score_func": [f_classif],
    "selector__k": [20, 30, 40],
    "model__activation": ["relu", "gelu", "tanh"],
    "model__hidden_dim": [32, 64, 128],
    "model__dropout": [0.0, 0.1, 0.2, 0.3],
    "model__lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "model__weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "model__batch_size": [32, 64, 128],
}

XGB_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [200, 300, 500, 700],
    "model__max_depth": [3, 5, 7, 9, 12, 20],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
    "model__gamma": [0.0, 0.1, 0.3, 0.5],
    "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "model__reg_lambda": [1.0, 2.0, 5.0, 10.0],
}

# Espaços reduzidos após análise do RandomSearch (derivados do notebook 02)
MLP_OPTUNA_SEARCH_SPACE = {
    "selector__k": {"low": 35, "high": 45, "step": 1},
    "model__activation": ["tanh"],
    "model__hidden_dim": {"low": 32, "high": 64, "step": 2},
    "model__dropout": {"low": 0.0, "high": 0.3, "step": 0.1},
    "model__lr": {"low": 3e-4, "high": 1e-3, "log": True},
    "model__weight_decay": [0.0, 1e-6, 1e-5],
    "model__batch_size": [64, 80, 96, 112, 128],
}

XGB_OPTUNA_SEARCH_SPACE = {
    "model__n_estimators": [200, 250, 300],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": {"low": 0.01, "high": 0.03, "log": True},
    "model__min_child_weight": [1, 2, 3],
    "model__subsample": [0.7],
    "model__colsample_bytree": [0.7],
    "model__gamma": [0.3, 0.4, 0.5],
    "model__reg_alpha": {"low": 0.0, "high": 0.5, "step": 0.1},
    "model__reg_lambda": {"low": 2.0, "high": 10.0, "step": 2.0},
}

_SMOKE_TEST = os.environ.get("EXPERIMENT_SMOKE_TEST") == "1"
OPTUNA_TIMEOUT_SECONDS = 30 if _SMOKE_TEST else 600
OPTUNA_CONVERGENCE_PATIENCE = 3 if _SMOKE_TEST else 30
OPTUNA_CONVERGENCE_MIN_IMPROVEMENT = 5e-3
_RANDOMSEARCH_N_ITER = 3 if _SMOKE_TEST else 50
_CV_N_SPLITS = 2 if _SMOKE_TEST else None  # None = usa config.yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cv_metrics(cv_result: dict) -> dict:
    return {
        "pr_auc_mean": float(cv_result["test_pr_auc"].mean()),
        "pr_auc_std": float(cv_result["test_pr_auc"].std()),
        "roc_auc_mean": float(cv_result["test_roc_auc"].mean()),
        "recall_mean": float(cv_result["test_recall"].mean()),
        "precision_mean": float(cv_result["test_precision"].mean()),
        "f1_mean": float(cv_result["test_f1"].mean()),
        "fit_time_mean_s": float(cv_result["fit_time"].mean()),
    }


def _build_baseline_models(fe_params: dict, preprocessor, y_ref) -> dict:
    """Constrói os 6 pipelines de baseline."""
    XGBClassifier = require_xgboost()
    y_arr = np.asarray(y_ref)
    scale_pos_weight = float((y_arr == 0).sum()) / max(float((y_arr == 1).sum()), 1.0)

    def _pipe(model):
        return Pipeline(
            [
                ("fe", FeatureEngineerTransformer(**fe_params)),
                ("geo", GeoTransformer(strategy="drop")),
                ("prep", preprocessor),
                ("model", model),
            ]
        )

    return {
        "Dummy": _pipe(DummyClassifier(strategy="stratified", random_state=42)),
        "LogisticRegression": _pipe(
            LogisticRegression(
                C=0.1,
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
            )
        ),
        "DecisionTree": _pipe(DecisionTreeClassifier(max_depth=5, random_state=42)),
        "RandomForest": _pipe(
            RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                random_state=42,
            )
        ),
        "XGBoost": _pipe(
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=42,
            )
        ),
        "MLP": Pipeline(
            [
                ("fe", FeatureEngineerTransformer(**fe_params)),
                ("geo", GeoTransformer(strategy="drop")),
                ("prep", preprocessor),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    MLPClassifierWrapper(
                        output_dim=1,
                        activation="relu",
                        hidden_dim=64,
                        dropout=0.1,
                        lr=1e-3,
                        weight_decay=0.0,
                        batch_size=64,
                        max_epochs=50,
                        patience=10,
                        min_delta=1e-3,
                        threshold=0.5,
                        val_size=0.15,
                        random_state=42,
                        verbose=False,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Fases de experimentação
# ---------------------------------------------------------------------------


def run_baselines(X_tv, y_tv, meta_tv, X_test, meta_test, cv, scoring) -> str:
    """Fase 1: baseline com 6 modelos. Retorna o run_id do pai para uso futuro."""
    print("\n[Baseline] Executando CV para 6 modelos...")
    preprocessor = build_default_preprocessor()
    models = _build_baseline_models(DEFAULT_BASELINE_FE_PARAMS, preprocessor, y_tv)

    with start_experiment_run(
        MLFLOW_EXPERIMENTS["baselines"],
        "baseline_inicial",
        tags={"phase": "baseline_inicial"},
    ) as parent_run:
        log_split_artifacts(X_tv.index, X_test.index, meta_tv, meta_test)

        for model_name, pipeline in models.items():
            cv_result = cross_validate(
                pipeline,
                X_tv,
                y_tv,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
            )
            metrics = _cv_metrics(cv_result)
            print(f"  {model_name:20s}  PR-AUC={metrics['pr_auc_mean']:.4f}")

            with start_experiment_run(
                MLFLOW_EXPERIMENTS["baselines"],
                model_name,
                nested=True,
                tags={
                    "phase": "baseline_inicial",
                    "model_name": model_name,
                    "mlflow.parentRunId": parent_run.info.run_id,
                },
            ):
                log_metrics_safe(metrics)

    return parent_run.info.run_id


def run_fe_round(
    round_name: str,
    fe_params: dict,
    X_tv,
    y_tv,
    cv,
    scoring,
    preprocessor=None,
) -> None:
    """Fases 2 e 3: feature engineering rounds 1 e 2."""
    print(f"\n[FE {round_name}] Executando CV para 6 modelos...")
    preprocessor = preprocessor or build_default_preprocessor()
    models = _build_baseline_models(fe_params, preprocessor, y_tv)

    with start_experiment_run(
        MLFLOW_EXPERIMENTS["feature_engineering"],
        round_name,
        tags={"phase": round_name},
    ):
        for model_name, pipeline in models.items():
            cv_result = cross_validate(
                pipeline,
                X_tv,
                y_tv,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
            )
            metrics = _cv_metrics(cv_result)
            print(f"  {model_name:20s}  PR-AUC={metrics['pr_auc_mean']:.4f}")

            with start_experiment_run(
                MLFLOW_EXPERIMENTS["feature_engineering"],
                model_name,
                nested=True,
                tags={"phase": round_name, "model_name": model_name},
            ):
                log_metrics_safe(metrics)


# zip_region e geo_cluster omitidos: requerem Latitude/Longitude/Zip Code,
# colunas ausentes no interim CSV gerado por process_data().
_ROUND3_STRATEGY_SPECS = {
    "LogisticRegression": [
        ("drop", "lr_drop"),
        ("frequency", "lr_frequency"),
        ("target", "lr_target"),
        ("risk_band", "lr_risk_band"),
    ],
    "XGBoost": [
        ("drop", "xgb_drop"),
        ("frequency", "xgb_frequency"),
        ("target", "xgb_target"),
        ("risk_band", "xgb_risk_band"),
    ],
    "MLP": [
        ("drop", "mlp_drop"),
        ("frequency", "mlp_frequency"),
        ("target", "mlp_target"),
        ("risk_band", "mlp_risk_band"),
        ("city_embedding", "mlp_city_embedding"),
    ],
}


def run_round3_city_encoding(X_tv, y_tv, cv, scoring) -> None:
    """Fase 4: testa estratégias de City encoding para cada modelo."""
    preprocessor = build_default_preprocessor()
    all_summaries = []

    for model_name, strategy_specs in _ROUND3_STRATEGY_SPECS.items():
        print(f"\n[Round 3 City Encoding] {model_name}...")
        summary_df, _ = evaluate_round3_model_strategies(
            model_name,
            strategy_specs,
            X=X_tv,
            y=y_tv,
            cv=cv,
            scoring=scoring,
            preprocessor=preprocessor,
            fe_params=ROUND2_FE_PARAMS,
            y_reference=y_tv,
        )
        all_summaries.append(summary_df)
        best = summary_df.sort_values("pr_auc_mean", ascending=False).iloc[0]
        print(f"  melhor: {best.get('strategy', '')}  PR-AUC={best['pr_auc_mean']:.4f}")

    results = pd.concat(all_summaries, ignore_index=True)
    with start_experiment_run(
        MLFLOW_EXPERIMENTS["round3_city_encoding"],
        "round3_city_encoding",
        tags={"phase": "round3_city_encoding"},
    ):
        log_dataframe_artifact(results, "round3_city_encoding_results.csv")
    print(f"  {len(results)} combinações avaliadas no total")


def run_feature_selection(X_tv, y_tv, cv, scoring) -> None:
    """Fase 5: SelectKBest e seleção baseada em L1."""
    print("\n[Feature Selection] Avaliando SelectKBest e L1...")
    preprocessor = build_default_preprocessor()

    # SelectKBest — testa k = [10, 20, 30, 40] para LR e MLP
    rows = []
    for k in [10, 20, 30, 40]:
        for model_name, model in [
            (
                "LogisticRegression",
                LogisticRegression(
                    C=0.1, max_iter=2000, solver="lbfgs", random_state=42
                ),
            ),
            (
                "MLP",
                MLPClassifierWrapper(
                    output_dim=1,
                    activation="relu",
                    hidden_dim=64,
                    dropout=0.1,
                    lr=1e-3,
                    weight_decay=0.0,
                    batch_size=64,
                    max_epochs=50,
                    patience=10,
                    min_delta=1e-3,
                    val_size=0.15,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]:
            pipeline = Pipeline(
                [
                    ("fe", FeatureEngineerTransformer(**DEFAULT_ROUND4_FE_PARAMS)),
                    ("geo", GeoTransformer(strategy="drop")),
                    ("prep", preprocessor),
                    ("selector", SelectKBest(score_func=f_classif, k=k)),
                    ("scaler", StandardScaler(with_mean=False)),
                    ("model", model),
                ]
            )
            cv_result = cross_validate(
                pipeline,
                X_tv,
                y_tv,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
            )
            metrics = _cv_metrics(cv_result)
            rows.append({"model": model_name, "k": k, **metrics})
            print(f"  k={k:2d}  {model_name:20s}  PR-AUC={metrics['pr_auc_mean']:.4f}")

    results_df = pd.DataFrame(rows)
    with start_experiment_run(
        MLFLOW_EXPERIMENTS["feature_selection"],
        "feature_selection",
        tags={"phase": "feature_selection"},
    ):
        log_dataframe_artifact(results_df, "feature_selection_results.csv")


def run_random_search(X_tv, y_tv, cv, scoring) -> None:
    """Fase 6: RandomizedSearchCV para MLP e XGBoost."""
    XGBClassifier = require_xgboost()
    y_arr = np.asarray(y_tv)
    scale_pos_weight = float((y_arr == 0).sum()) / max(float((y_arr == 1).sum()), 1.0)
    preprocessor = build_default_preprocessor()

    # MLP
    print(f"\n[RandomSearch] MLP (n_iter={_RANDOMSEARCH_N_ITER})...")
    mlp_pipeline = Pipeline(
        [
            ("fe", FeatureEngineerTransformer(**DEFAULT_ROUND4_FE_PARAMS)),
            ("geo", GeoTransformer(strategy="drop")),
            ("prep", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=20)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                MLPClassifierWrapper(
                    output_dim=1,
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
    mlp_search = RandomizedSearchCV(
        mlp_pipeline,
        MLP_PARAM_DISTRIBUTIONS,
        n_iter=_RANDOMSEARCH_N_ITER,
        cv=cv,
        scoring=scoring,
        refit="pr_auc",
        n_jobs=1,
        random_state=42,
        return_train_score=False,
    )
    mlp_search.fit(X_tv, y_tv)
    best_mlp_rs = summarize_grid_search_results(mlp_search, "MLP")
    mlp_results_df = pd.DataFrame([best_mlp_rs])
    print(f"  Melhor PR-AUC: {best_mlp_rs.get('pr_auc_mean', 0):.4f}")

    with start_experiment_run(
        MLFLOW_EXPERIMENTS["tuning_stage1_randomsearch"],
        "randomsearch_mlp",
        tags={"phase": "tuning_stage1", "model_name": "MLP"},
    ):
        log_dataframe_artifact(mlp_results_df, "mlp_randomsearch_results.csv")
        log_metrics_safe(
            {k: v for k, v in best_mlp_rs.items() if isinstance(v, (int, float))}
        )

    # XGBoost
    print(f"\n[RandomSearch] XGBoost (n_iter={_RANDOMSEARCH_N_ITER})...")
    xgb_pipeline = Pipeline(
        [
            ("fe", FeatureEngineerTransformer(**DEFAULT_ROUND4_FE_PARAMS)),
            ("geo", GeoTransformer(strategy="drop")),
            ("prep", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    xgb_search = RandomizedSearchCV(
        xgb_pipeline,
        XGB_PARAM_DISTRIBUTIONS,
        n_iter=_RANDOMSEARCH_N_ITER,
        cv=cv,
        scoring=scoring,
        refit="pr_auc",
        n_jobs=1,
        random_state=42,
        return_train_score=False,
    )
    xgb_search.fit(X_tv, y_tv)
    best_idx = xgb_search.best_index_
    cv_res = xgb_search.cv_results_
    best_xgb_rs = {
        "model": "XGBoost",
        "pr_auc_mean": float(cv_res["mean_test_pr_auc"][best_idx]),
        "roc_auc_mean": float(cv_res["mean_test_roc_auc"][best_idx]),
        "recall_mean": float(cv_res["mean_test_recall"][best_idx]),
        "precision_mean": float(cv_res["mean_test_precision"][best_idx]),
        "f1_mean": float(cv_res["mean_test_f1"][best_idx]),
        "fit_time_mean_s": float(cv_res["mean_fit_time"][best_idx]),
        **{k: v for k, v in xgb_search.best_params_.items()},
    }
    xgb_results_df = pd.DataFrame([best_xgb_rs])
    print(f"  Melhor PR-AUC: {best_xgb_rs.get('pr_auc_mean', 0):.4f}")

    with start_experiment_run(
        MLFLOW_EXPERIMENTS["tuning_stage1_randomsearch"],
        "randomsearch_xgboost",
        tags={"phase": "tuning_stage1", "model_name": "XGBoost"},
    ):
        log_dataframe_artifact(xgb_results_df, "xgb_randomsearch_results.csv")
        log_metrics_safe(
            {k: v for k, v in best_xgb_rs.items() if isinstance(v, (int, float))}
        )


def run_optuna(X_tv, y_tv, cv, scoring) -> tuple[dict, dict]:
    """Fase 7: Optuna para MLP e XGBoost. Retorna (best_mlp_params, best_xgb_params)."""
    preprocessor = build_default_preprocessor()

    # MLP
    print("\n[Optuna] MLP...")
    mlp_factory = partial(
        build_mlp_optuna_pipeline, preprocessor, DEFAULT_ROUND4_FE_PARAMS
    )
    _, mlp_trials, mlp_convergence, best_mlp_params, best_mlp_row = run_optuna_study(
        model_name="MLP",
        search_space=MLP_OPTUNA_SEARCH_SPACE,
        pipeline_factory=mlp_factory,
        X=X_tv,
        y=y_tv,
        cv=cv,
        scoring=scoring,
        timeout_seconds=OPTUNA_TIMEOUT_SECONDS,
        convergence_patience_trials=OPTUNA_CONVERGENCE_PATIENCE,
        convergence_min_improvement=OPTUNA_CONVERGENCE_MIN_IMPROVEMENT,
    )
    pr_auc_mlp = best_mlp_row.get("pr_auc_mean", 0)
    print(f"  Trials: {len(mlp_trials)}  |  Melhor PR-AUC: {pr_auc_mlp:.4f}")
    print(f"  Melhores params: {best_mlp_params}")

    with start_experiment_run(
        MLFLOW_EXPERIMENTS["tuning_stage2_optuna"],
        "optuna_mlp",
        tags={"phase": "tuning_stage2", "model_name": "MLP"},
    ):
        log_json_artifact(best_mlp_params, "best_mlp_optuna_params.json")
        log_dataframe_artifact(mlp_trials, "mlp_optuna_trials.csv")
        log_dataframe_artifact(mlp_convergence, "mlp_optuna_convergence.csv")
        log_metrics_safe(best_row_to_metrics(best_mlp_row))

    # XGBoost
    print("\n[Optuna] XGBoost...")

    def xgb_factory(params):
        return build_xgb_optuna_pipeline(
            preprocessor, DEFAULT_ROUND4_FE_PARAMS, params, y_tv
        )  # noqa: E731

    _, xgb_trials, xgb_convergence, best_xgb_params, best_xgb_row = run_optuna_study(
        model_name="XGBoost",
        search_space=XGB_OPTUNA_SEARCH_SPACE,
        pipeline_factory=xgb_factory,
        X=X_tv,
        y=y_tv,
        cv=cv,
        scoring=scoring,
        timeout_seconds=OPTUNA_TIMEOUT_SECONDS,
        convergence_patience_trials=OPTUNA_CONVERGENCE_PATIENCE,
        convergence_min_improvement=OPTUNA_CONVERGENCE_MIN_IMPROVEMENT,
    )
    pr_auc_xgb = best_xgb_row.get("pr_auc_mean", 0)
    print(f"  Trials: {len(xgb_trials)}  |  Melhor PR-AUC: {pr_auc_xgb:.4f}")
    print(f"  Melhores params: {best_xgb_params}")

    with start_experiment_run(
        MLFLOW_EXPERIMENTS["tuning_stage2_optuna"],
        "optuna_xgboost",
        tags={"phase": "tuning_stage2", "model_name": "XGBoost"},
    ):
        log_json_artifact(best_xgb_params, "best_xgb_optuna_params.json")
        log_dataframe_artifact(xgb_trials, "xgb_optuna_trials.csv")
        log_dataframe_artifact(xgb_convergence, "xgb_optuna_convergence.csv")
        log_metrics_safe(best_row_to_metrics(best_xgb_row))

    return best_mlp_params, best_xgb_params


def write_best_params(mlp_params: dict, config: dict) -> None:
    """Persiste os melhores hiperparâmetros do MLP em config/best_params.yaml."""
    current: dict = {}
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, encoding="utf-8") as f:
            current = yaml.safe_load(f) or {}

    # Preserva split e production; atualiza apenas mlp_optuna
    updated = {
        "split": current.get(
            "split",
            {
                "random_state": config["experiment"]["random_state"],
                "test_size": config["experiment"]["test_size"],
            },
        ),
        "mlp_optuna": {k: v for k, v in mlp_params.items()},
        "production": current.get(
            "production",
            {
                "threshold": 0.35,
                "activation_cost": 50.0,
                "retention_rate": 0.10,
            },
        ),
    }

    header = (
        "# Gerado por run_experiment.py após tuning com Optuna.\n"
        "# Commitado no git — representa os melhores hiperparâmetros encontrados.\n"
        "# run_train.py lê este arquivo para treinar o modelo de produção.\n\n"
    )
    BEST_PARAMS_PATH.write_text(
        header + yaml.dump(updated, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"\n  best_params.yaml atualizado: {BEST_PARAMS_PATH}")
    print(f"  MLP Optuna params: {mlp_params}")


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------


def main() -> None:
    config = _load_config()

    tracking_uri = resolve_tracking_uri(
        config["mlflow"]["tracking_uri"], workspace_root=BASE_DIR
    )
    mlflow.set_tracking_uri(tracking_uri)

    print("\n[0/7] Preparando dados...")
    interim_path = BASE_DIR / config["data"]["interim_path"]
    if not interim_path.exists():
        process_data()
    df = pd.read_csv(interim_path)
    target = config["data"]["target"]
    feature_cols = [c for c in df.columns if c not in [target] + META_COLS]

    X_tv, X_test, y_tv, y_test, meta_tv, meta_test = train_test_split(
        df[feature_cols],
        df[target],
        df[META_COLS],
        test_size=float(config["experiment"]["test_size"]),
        stratify=df[target],
        random_state=int(config["experiment"]["random_state"]),
    )
    print(f"  train/val: {len(X_tv)} | test: {len(X_test)}")

    n_splits = _CV_N_SPLITS if _CV_N_SPLITS else config["validation"]["n_splits"]
    cv = build_default_cv(
        n_splits=n_splits,
        random_state=int(config["experiment"]["random_state"]),
    )
    scoring = build_default_scoring()

    print("\n[1/7] Baseline...")
    run_baselines(X_tv, y_tv, meta_tv, X_test, meta_test, cv, scoring)

    print("\n[2/7] Feature Engineering Round 1 (sem Churn Score)...")
    run_fe_round(
        "round1_feature_engineering", ROUND1_FE_PARAMS, X_tv, y_tv, cv, scoring
    )

    print("\n[3/7] Feature Engineering Round 2 (com Churn Score)...")
    run_fe_round(
        "round2_feature_engineering", ROUND2_FE_PARAMS, X_tv, y_tv, cv, scoring
    )

    print("\n[4/7] Feature Engineering Round 3 (City Encoding)...")
    run_round3_city_encoding(X_tv, y_tv, cv, scoring)

    print("\n[5/7] Feature Selection...")
    run_feature_selection(X_tv, y_tv, cv, scoring)

    print("\n[6/7] RandomSearch (MLP + XGBoost)...")
    run_random_search(X_tv, y_tv, cv, scoring)

    print("\n[7/7] Optuna (MLP + XGBoost)...")
    best_mlp_params, _ = run_optuna(X_tv, y_tv, cv, scoring)

    print("\n[+] Persistindo melhores hiperparâmetros...")
    write_best_params(best_mlp_params, config)

    print("\n[+] EXPERIMENTACAO CONCLUIDA COM SUCESSO!")
    print("    Proximo passo: commite config/best_params.yaml e execute make train")


if __name__ == "__main__":
    main()
