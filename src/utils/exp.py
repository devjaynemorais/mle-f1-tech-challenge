"""Utilitarios para experimentacao e wrapper sklearn da MLP."""

from __future__ import annotations

from collections.abc import Sequence
import inspect
import json
from pathlib import Path
import tempfile
from typing import Any

import copy

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency is absent locally
    XGBClassifier = None

from src.features.feature_engineer_transformer import FeatureEngineerTransformer
from src.features.geo_transformer import GeoTransformer
from src.models.mlp import CityEmbeddingMLP, DEFAULT_DEVICE, MLP

DEFAULT_METRICS = ("pr_auc", "roc_auc", "recall", "precision", "f1")
DEFAULT_METADATA_COLUMNS = ("CLTV", "CustomerID")
DEFAULT_BASELINE_FE_PARAMS = {
    "drop_churn_score": True,
    "add_engagement_score": False,
    "add_tenure_group": False,
    "add_tenure_log": False,
    "add_contract_ordinal": False,
    "add_family_stability": False,
    "add_fiber_no_support": False,
    "add_support_gap_count": False,
    "add_payment_automatic_flag": False,
    "add_electronic_check_flag": False,
    "add_paperless_echeck_flag": False,
    "add_price_pressure_ratio": False,
}
DEFAULT_ROUND4_FE_PARAMS = {
    "drop_churn_score": False,
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


def resolve_tracking_uri(tracking_uri: str, workspace_root: str | Path | None = None) -> str:
    """Resolve URIs locais do MLflow para caminhos absolutos quando necessário."""
    if tracking_uri.startswith("sqlite:///") and workspace_root is not None:
        db_name = tracking_uri.replace("sqlite:///", "", 1)
        db_path = Path(workspace_root) / db_name
        return f"sqlite:///{db_path.as_posix()}"
    return tracking_uri


def set_mlflow_tracking(
    tracking_uri: str,
    *,
    workspace_root: str | Path | None = None,
) -> MlflowClient:
    """Configura a URI de tracking e devolve um cliente do MLflow."""
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri, workspace_root))
    return MlflowClient()


def build_default_preprocessor() -> ColumnTransformer:
    """Cria o pre-processador padrao usado nos notebooks de experimentacao."""
    ohe = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("cat", ohe, make_column_selector(dtype_include=["object", "category"])),
            ("num", "passthrough", make_column_selector(dtype_exclude=["object", "category"])),
        ],
        remainder="drop",
    )


def build_default_cv(
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
):
    """Cria um StratifiedKFold padrao."""
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def build_default_scoring() -> dict[str, str]:
    """Pacote padrao de metricas usado na comparacao tecnica."""
    return {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }


def artifact_uri_to_local_path(
    artifact_uri: str,
    workspace_root: str | Path,
    artifact_name: str | None = None,
) -> Path:
    """Converte artifact_uri do MLflow para caminho local no workspace."""
    workspace_path = Path(workspace_root)

    if artifact_uri.startswith("mlflow-artifacts:/"):
        relative = artifact_uri.replace("mlflow-artifacts:/", "", 1).strip("/")
        base_path = workspace_path / "mlartifacts" / Path(relative)
    elif artifact_uri.startswith("file://"):
        base_path = Path(artifact_uri.replace("file://", "", 1))
    else:
        base_path = Path(artifact_uri)
        if not base_path.is_absolute():
            base_path = workspace_path / base_path

    return base_path / artifact_name if artifact_name else base_path


def find_run_by_name(
    client: MlflowClient,
    experiment_name: str,
    run_name: str,
    *,
    tags: dict[str, Any] | None = None,
) -> Any:
    """Localiza uma run pelo nome e opcionalmente por tags."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experimento do MLflow nao encontrado: {experiment_name}")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=50,
    )

    for run in runs:
        if tags is None:
            return run
        if all(str(run.data.tags.get(key)) == str(value) for key, value in tags.items()):
            return run

    raise RuntimeError(
        f"Run '{run_name}' nao encontrada no experimento '{experiment_name}'"
        + (f" com tags {tags}" if tags else "")
    )


def resolve_notebook03_runs(
    client: MlflowClient,
    *,
    baseline_experiment: str = "tc-f1-nb02-baselines",
    baseline_parent_run_name: str = "baseline_inicial",
    tuning_experiment: str = "tc-f1-nb02-tuning-stage2-optuna",
) -> dict[str, Any]:
    """Resolve as runs necessarias para a comparacao do notebook 03."""
    baseline_parent = find_run_by_name(
        client,
        baseline_experiment,
        baseline_parent_run_name,
        tags={"phase": "baseline_inicial"},
    )
    baseline_experiment_id = client.get_experiment_by_name(baseline_experiment).experiment_id
    child_runs = client.search_runs(
        [baseline_experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{baseline_parent.info.run_id}'",
        max_results=100,
    )
    child_by_model = {
        run.data.tags.get("model_name") or run.data.tags.get("mlflow.runName"): run
        for run in child_runs
    }

    return {
        "baseline_parent": baseline_parent,
        "logistic_baseline": child_by_model["LogisticRegression"],
        "mlp_baseline": child_by_model["MLP"],
        "mlp_optuna": find_run_by_name(
            client,
            tuning_experiment,
            "optuna_mlp",
            tags={"phase": "tuning_stage2", "model_name": "MLP"},
        ),
        "xgb_optuna": find_run_by_name(
            client,
            tuning_experiment,
            "optuna_xgboost",
            tags={"phase": "tuning_stage2", "model_name": "XGBoost"},
        ),
    }


def load_run_csv_artifact(
    run: Any,
    artifact_name: str,
    *,
    workspace_root: str | Path,
) -> pd.DataFrame:
    """Le um artefato tabular CSV diretamente do diretório local de artefatos."""
    artifact_path = artifact_uri_to_local_path(
        run.info.artifact_uri,
        workspace_root=workspace_root,
        artifact_name=artifact_name,
    )
    return pd.read_csv(artifact_path)


def load_run_json_artifact(
    run: Any,
    artifact_name: str,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Le um artefato JSON diretamente do diretório local de artefatos."""
    artifact_path = artifact_uri_to_local_path(
        run.info.artifact_uri,
        workspace_root=workspace_root,
        artifact_name=artifact_name,
    )
    with artifact_path.open(encoding="utf-8") as fp:
        return json.load(fp)


def rebuild_train_val_test_splits(
    df: pd.DataFrame,
    *,
    target_column: str,
    train_val_idx: Sequence[int],
    test_idx: Sequence[int],
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> dict[str, Any]:
    """Reconstrói train_val, holdout e metadata a partir dos índices salvos."""
    train_val_index = pd.Index(train_val_idx, name="index")
    test_index = pd.Index(test_idx, name="index")

    feature_columns = [
        col for col in df.columns if col not in {target_column, *metadata_columns}
    ]

    X = df.loc[:, feature_columns]
    y = df.loc[:, target_column]
    metadata = df.loc[:, list(metadata_columns)]

    return {
        "X_train_val": X.loc[train_val_index].copy(),
        "y_train_val": y.loc[train_val_index].copy(),
        "X_test": X.loc[test_index].copy(),
        "y_test": y.loc[test_index].copy(),
        "metadata_train_val": metadata.loc[train_val_index].copy(),
        "metadata_test": metadata.loc[test_index].copy(),
    }


def load_split_bundle_from_mlflow(
    *,
    client: MlflowClient,
    workspace_root: str | Path,
    data_path: str | Path,
    target_column: str,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
    baseline_experiment: str = "tc-f1-nb02-baselines",
    baseline_parent_run_name: str = "baseline_inicial",
) -> dict[str, Any]:
    """Recupera os splits logados no notebook 02 e reconstrói as bases."""
    baseline_parent = find_run_by_name(
        client,
        baseline_experiment,
        baseline_parent_run_name,
        tags={"phase": "baseline_inicial"},
    )
    train_val_idx_df = load_run_csv_artifact(
        baseline_parent,
        "train_val_indices.csv",
        workspace_root=workspace_root,
    )
    test_idx_df = load_run_csv_artifact(
        baseline_parent,
        "test_indices.csv",
        workspace_root=workspace_root,
    )
    df = pd.read_csv(data_path)
    return rebuild_train_val_test_splits(
        df=df,
        target_column=target_column,
        train_val_idx=train_val_idx_df["idx"].tolist(),
        test_idx=test_idx_df["idx"].tolist(),
        metadata_columns=metadata_columns,
    )


def _bool_param(params: dict[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _int_param(params: dict[str, Any], key: str, default: int) -> int:
    return int(float(params.get(key, default)))


def _float_param(params: dict[str, Any], key: str, default: float) -> float:
    return float(params.get(key, default))


def _str_param(params: dict[str, Any], key: str, default: str) -> str:
    return str(params.get(key, default))


def build_baseline_logistic_pipeline(run_params: dict[str, Any] | None = None) -> Pipeline:
    """Reconstrói o baseline inicial da regressão logística."""
    run_params = run_params or {}
    logistic_params = {
        "max_iter": _int_param(run_params, "max_iter", 1000),
        "class_weight": _str_param(run_params, "class_weight", "balanced"),
        "random_state": _int_param(run_params, "random_state", 42),
        "C": _float_param(run_params, "C", 1.0),
    }
    return Pipeline(
        [
            ("fe", FeatureEngineerTransformer(**DEFAULT_BASELINE_FE_PARAMS)),
            ("geo", GeoTransformer(strategy="drop")),
            ("prep", build_default_preprocessor()),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(**logistic_params)),
        ]
    )


def build_baseline_mlp_pipeline(run_params: dict[str, Any] | None = None) -> Pipeline:
    """Reconstrói o baseline inicial da MLP."""
    run_params = run_params or {}
    return Pipeline(
        [
            ("fe", FeatureEngineerTransformer(**DEFAULT_BASELINE_FE_PARAMS)),
            ("geo", GeoTransformer(strategy="drop")),
            ("prep", build_default_preprocessor()),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                MLPClassifierWrapper(
                    hidden_dim=_int_param(run_params, "hidden_dim", 64),
                    output_dim=_int_param(run_params, "output_dim", 1),
                    activation=_str_param(run_params, "activation", "relu"),
                    batch_size=_int_param(run_params, "batch_size", 64),
                    lr=_float_param(run_params, "lr", 1e-3),
                    weight_decay=_float_param(run_params, "weight_decay", 1e-5),
                    dropout=_float_param(run_params, "dropout", 0.0),
                    max_epochs=_int_param(run_params, "max_epochs", 80),
                    patience=_int_param(run_params, "patience", 8),
                    min_delta=_float_param(run_params, "min_delta", 1e-3),
                    val_size=_float_param(run_params, "val_size", 0.15),
                    threshold=_float_param(run_params, "threshold", 0.5),
                    random_state=_int_param(run_params, "random_state", 42),
                    normalize_sample_weight_flag=_bool_param(
                        run_params, "normalize_sample_weight_flag", True
                    ),
                    verbose=_bool_param(run_params, "verbose", False),
                ),
            ),
        ]
    )


def build_optuna_mlp_pipeline(optuna_params: dict[str, Any]) -> Pipeline:
    """Reconstrói a MLP vencedora do Optuna."""
    return Pipeline(
        [
            ("fe", FeatureEngineerTransformer(**DEFAULT_ROUND4_FE_PARAMS)),
            ("geo", GeoTransformer(strategy="drop")),
            ("prep", build_default_preprocessor()),
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=int(optuna_params["selector__k"]),
                ),
            ),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                MLPClassifierWrapper(
                    output_dim=1,
                    activation=str(optuna_params["model__activation"]),
                    hidden_dim=int(optuna_params["model__hidden_dim"]),
                    dropout=float(optuna_params["model__dropout"]),
                    lr=float(optuna_params["model__lr"]),
                    weight_decay=float(optuna_params["model__weight_decay"]),
                    batch_size=int(optuna_params["model__batch_size"]),
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


def build_optuna_xgb_pipeline(
    optuna_params: dict[str, Any],
    *,
    y_reference: Any,
) -> Pipeline:
    """Reconstrói o XGBoost vencedor do Optuna."""
    xgb_classifier = require_xgboost()
    y_arr = np.asarray(y_reference)
    scale_pos_weight = float((y_arr == 0).sum()) / max(float((y_arr == 1).sum()), 1.0)
    return Pipeline(
        [
            ("fe", FeatureEngineerTransformer(**DEFAULT_ROUND4_FE_PARAMS)),
            ("geo", GeoTransformer(strategy="drop")),
            ("prep", build_default_preprocessor()),
            (
                "model",
                xgb_classifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=int(optuna_params["model__n_estimators"]),
                    max_depth=int(optuna_params["model__max_depth"]),
                    learning_rate=float(optuna_params["model__learning_rate"]),
                    min_child_weight=int(optuna_params["model__min_child_weight"]),
                    subsample=float(optuna_params["model__subsample"]),
                    colsample_bytree=float(optuna_params["model__colsample_bytree"]),
                    gamma=float(optuna_params["model__gamma"]),
                    reg_alpha=float(optuna_params["model__reg_alpha"]),
                    reg_lambda=float(optuna_params["model__reg_lambda"]),
                ),
            ),
        ]
    )


def build_notebook03_estimators(
    run_registry: dict[str, Any],
    *,
    workspace_root: str | Path,
    y_reference: Any,
) -> dict[str, Any]:
    """Reconstrói os quatro modelos comparados no notebook 03."""
    mlp_optuna_params = load_run_json_artifact(
        run_registry["mlp_optuna"],
        "best_mlp_optuna_params.json",
        workspace_root=workspace_root,
    )
    xgb_optuna_params = load_run_json_artifact(
        run_registry["xgb_optuna"],
        "best_xgb_optuna_params.json",
        workspace_root=workspace_root,
    )
    return {
        "LogisticRegression": build_baseline_logistic_pipeline(
            run_registry["logistic_baseline"].data.params
        ),
        "MLP": build_baseline_mlp_pipeline(run_registry["mlp_baseline"].data.params),
        "MLP Optuna": build_optuna_mlp_pipeline(mlp_optuna_params),
        "XGBoost Optuna": build_optuna_xgb_pipeline(
            xgb_optuna_params,
            y_reference=y_reference,
        ),
    }


def _fit_params_for_estimator(estimator: Any, sample_weight: Any | None) -> dict[str, Any]:
    """Monta fit_params seguros para estimadores e pipelines com sample_weight."""
    if sample_weight is None:
        return {}

    if isinstance(estimator, Pipeline):
        final_step_name, final_estimator = estimator.steps[-1]
        if "sample_weight" in inspect.signature(final_estimator.fit).parameters:
            return {f"{final_step_name}__sample_weight": sample_weight}
        return {}

    if "sample_weight" in inspect.signature(estimator.fit).parameters:
        return {"sample_weight": sample_weight}
    return {}


def generate_oof_predictions(
    *,
    estimator: Any,
    X: Any,
    y: Any,
    cv: Any,
    sample_weight: Any | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Gera probabilidades OOF para um estimador."""
    y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=getattr(X, "index", None))
    if hasattr(X, "index"):
        index_values = pd.Index(X.index)
    else:
        index_values = pd.Index(np.arange(len(y_series)))

    fold_frames: list[pd.DataFrame] = []
    for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X, y_series), start=1):
        estimator_fold = clone(estimator)
        X_train = rows(X, train_idx)
        X_valid = rows(X, valid_idx)
        y_train = rows(y_series, train_idx)
        y_valid = rows(y_series, valid_idx)
        fit_params = _fit_params_for_estimator(
            estimator_fold,
            None if sample_weight is None else rows(sample_weight, train_idx),
        )
        estimator_fold.fit(X_train, y_train, **fit_params)

        if hasattr(estimator_fold, "predict_proba"):
            proba = estimator_fold.predict_proba(X_valid)[:, 1]
        else:
            decision = estimator_fold.decision_function(X_valid)
            proba = 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))

        fold_frames.append(
            pd.DataFrame(
                {
                    "row_index": index_values[valid_idx].to_numpy(),
                    "fold": fold_number,
                    "y_true": np.asarray(y_valid, dtype=int),
                    "proba": np.asarray(proba, dtype=float),
                    "pred_0_5": (np.asarray(proba, dtype=float) >= 0.5).astype(int),
                    "model": model_name,
                }
            )
        )

    return (
        pd.concat(fold_frames, ignore_index=True)
        .sort_values("row_index")
        .reset_index(drop=True)
    )


def generate_oof_predictions_by_model(
    models: dict[str, Any],
    *,
    X: Any,
    y: Any,
    cv: Any,
    sample_weight: Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Gera um dicionário model_name -> DataFrame de probabilidades OOF."""
    return {
        model_name: generate_oof_predictions(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            sample_weight=sample_weight,
            model_name=model_name,
        )
        for model_name, estimator in models.items()
    }


def compute_generalization_gap_pct(train_score: float, valid_score: float) -> float:
    """Calcula o gap percentual conforme convenção do notebook."""
    train_value = float(train_score)
    valid_value = float(valid_score)
    if np.isclose(valid_value, 0.0):
        return np.nan
    return (1.0 - (train_value / valid_value)) * 100.0


def build_cross_validate_comparison_table(
    models: dict[str, Any],
    *,
    X: Any,
    y: Any,
    cv: Any,
    scoring: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Gera a tabela consolidada de treino/validação para comparação técnica."""
    scoring = scoring or build_default_scoring()
    metric_label_map = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1-Score",
    }
    rows_list: list[dict[str, Any]] = []

    for model_name, estimator in models.items():
        cv_res = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=True,
        )

        row = {"modelo": model_name}
        for metric_key, metric_label in metric_label_map.items():
            train_mean = float(np.mean(cv_res[f"train_{metric_key}"]))
            valid_mean = float(np.mean(cv_res[f"test_{metric_key}"]))
            row[metric_label] = valid_mean
            row[f"{metric_label} gap (%)"] = compute_generalization_gap_pct(
                train_mean,
                valid_mean,
            )

        row["fit_time(mean)"] = float(np.mean(cv_res["fit_time"]))
        row["score_time(mean)"] = float(np.mean(cv_res["score_time"]))
        rows_list.append(row)

    return pd.DataFrame(rows_list).sort_values("PR-AUC", ascending=False).reset_index(drop=True)


def calculate_expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Calcula o ECE com bins uniformes em [0, 1]."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if np.isclose(upper, 1.0):
            mask = (y_prob_arr >= lower) & (y_prob_arr <= upper)
        else:
            mask = (y_prob_arr >= lower) & (y_prob_arr < upper)
        if not np.any(mask):
            continue

        acc = float(y_true_arr[mask].mean())
        conf = float(y_prob_arr[mask].mean())
        weight = float(mask.mean())
        ece += abs(acc - conf) * weight

    return float(ece)


def summarize_calibration_for_models(
    oof_predictions_by_model: dict[str, pd.DataFrame],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Consolida Brier Score e ECE por modelo."""
    rows_list: list[dict[str, Any]] = []
    for model_name, oof_df in oof_predictions_by_model.items():
        y_true = oof_df["y_true"].to_numpy(dtype=int)
        y_prob = oof_df["proba"].to_numpy(dtype=float)
        rows_list.append(
            {
                "modelo": model_name,
                "brier_score": float(brier_score_loss(y_true, y_prob)),
                "ece": calculate_expected_calibration_error(
                    y_true,
                    y_prob,
                    n_bins=n_bins,
                ),
            }
        )
    return pd.DataFrame(rows_list).sort_values("modelo").reset_index(drop=True)


def compute_campaign_economics(
    *,
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    cltv: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
    activation_cost: float = 50.0,
    retention_rate: float = 0.1,
) -> dict[str, float]:
    """Calcula VR, Vrec, VP, CMCA, custo total, VD, IEL e ROI."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    cltv_arr = np.asarray(cltv, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tp_mask = (y_pred == 1) & (y_true_arr == 1)
    fp_mask = (y_pred == 1) & (y_true_arr == 0)
    fn_mask = (y_pred == 0) & (y_true_arr == 1)
    tn_mask = (y_pred == 0) & (y_true_arr == 0)

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    tn = int(tn_mask.sum())

    vr = float(np.sum(y_prob_arr[tp_mask] * cltv_arr[tp_mask]))
    vrec = float(vr * retention_rate)
    vp = float(np.sum(y_prob_arr[fn_mask] * cltv_arr[fn_mask]))
    total_acted = tp + fp
    cmca = float(activation_cost)
    total_campaign_cost = float(total_acted * activation_cost)
    vd = float(fp * activation_cost)
    iel = float(vrec - vp - total_campaign_cost)
    roi = (
        float((vrec - vp - total_campaign_cost) / total_campaign_cost)
        if total_campaign_cost
        else np.nan
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "vr": vr,
        "vrec": vrec,
        "vp": vp,
        "cmca": cmca,
        "total_campaign_cost": total_campaign_cost,
        "vd": vd,
        "iel": iel,
        "roi": roi,
        "threshold": float(threshold),
        "activation_cost": float(activation_cost),
        "retention_rate": float(retention_rate),
    }


def build_economic_comparison_table(
    oof_predictions_by_model: dict[str, pd.DataFrame],
    *,
    metadata: pd.DataFrame,
    activation_cost: float = 50.0,
    retention_rate: float = 0.1,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Consolida a análise econômica por modelo."""
    if "CLTV" not in metadata.columns:
        raise KeyError("metadata precisa conter a coluna 'CLTV'.")

    rows_list: list[dict[str, Any]] = []
    cltv_by_index = metadata["CLTV"]

    for model_name, oof_df in oof_predictions_by_model.items():
        aligned_cltv = cltv_by_index.loc[oof_df["row_index"]].to_numpy(dtype=float)
        metrics = compute_campaign_economics(
            y_true=oof_df["y_true"].to_numpy(dtype=int),
            y_prob=oof_df["proba"].to_numpy(dtype=float),
            cltv=aligned_cltv,
            threshold=threshold,
            activation_cost=activation_cost,
            retention_rate=retention_rate,
        )
        rows_list.append({"modelo": model_name, **metrics})

    return pd.DataFrame(rows_list).sort_values("iel", ascending=False).reset_index(drop=True)


def compute_binary_classification_metrics(
    *,
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula metricas tecnicas binarizadas a partir de probabilidades."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }


def optimize_threshold_for_roi(
    *,
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    cltv: Sequence[float] | np.ndarray,
    threshold_grid: Sequence[float] | np.ndarray | None = None,
    activation_cost: float = 50.0,
    retention_rate: float = 0.1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Varre thresholds e retorna tabela completa mais a melhor linha por ROI."""
    thresholds = (
        np.asarray(list(threshold_grid), dtype=float)
        if threshold_grid is not None
        else np.round(np.arange(0.0, 1.01, 0.01), 4)
    )
    rows_list: list[dict[str, Any]] = []

    for threshold in thresholds:
        economics = compute_campaign_economics(
            y_true=y_true,
            y_prob=y_prob,
            cltv=cltv,
            threshold=float(threshold),
            activation_cost=activation_cost,
            retention_rate=retention_rate,
        )
        technical = compute_binary_classification_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
        )
        rows_list.append({**economics, **technical})

    threshold_df = pd.DataFrame(rows_list).sort_values("threshold").reset_index(drop=True)
    best_idx = threshold_df["roi"].astype(float).idxmax()
    best_row = threshold_df.loc[best_idx].copy()
    return threshold_df, best_row


def compute_percentage_gain(reference_value: float, candidate_value: float) -> float:
    """Calcula ganho percentual de uma metrica em relacao a um baseline."""
    reference = float(reference_value)
    candidate = float(candidate_value)
    if np.isclose(reference, 0.0):
        return np.nan
    return ((candidate - reference) / abs(reference)) * 100.0


def build_threshold_comparison_summary(
    *,
    threshold_metrics: pd.Series | dict[str, Any],
    reference_rows: pd.DataFrame | None = None,
    roi_column: str = "roi",
) -> pd.DataFrame:
    """Resume o threshold otimo e compara ganho percentual de ROI vs referencias."""
    threshold_series = (
        threshold_metrics if isinstance(threshold_metrics, pd.Series) else pd.Series(threshold_metrics)
    )
    summary_rows = [
        {"item": "best_threshold", "value": float(threshold_series["threshold"])},
        {"item": "roi", "value": float(threshold_series["roi"])},
        {"item": "iel", "value": float(threshold_series["iel"])},
        {"item": "vr", "value": float(threshold_series["vr"])},
        {"item": "vrec", "value": float(threshold_series["vrec"])},
        {"item": "vp", "value": float(threshold_series["vp"])},
        {"item": "vd", "value": float(threshold_series["vd"])},
        {"item": "pr_auc", "value": float(threshold_series["pr_auc"])},
        {"item": "roc_auc", "value": float(threshold_series["roc_auc"])},
        {"item": "recall", "value": float(threshold_series["recall"])},
        {"item": "precision", "value": float(threshold_series["precision"])},
        {"item": "f1_score", "value": float(threshold_series["f1_score"])},
    ]

    if reference_rows is not None and not reference_rows.empty:
        for _, row in reference_rows.iterrows():
            model_name = str(row["modelo"])
            summary_rows.append(
                {
                    "item": f"roi_gain_pct_vs_{model_name.lower().replace(' ', '_')}",
                    "value": compute_percentage_gain(row[roi_column], threshold_series["roi"]),
                }
            )

    return pd.DataFrame(summary_rows)


def build_retention_roi_table(
    *,
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    cltv: Sequence[float] | np.ndarray,
    threshold: float,
    activation_cost: float,
    retention_values: Sequence[float],
) -> pd.DataFrame:
    """Gera tabela de sensibilidade do ROI a diferentes taxas de retencao."""
    rows_list: list[dict[str, Any]] = []
    for retention_rate in retention_values:
        rows_list.append(
            compute_campaign_economics(
                y_true=y_true,
                y_prob=y_prob,
                cltv=cltv,
                threshold=threshold,
                activation_cost=activation_cost,
                retention_rate=float(retention_rate),
            )
        )

    return pd.DataFrame(rows_list).sort_values("retention_rate").reset_index(drop=True)


def build_campaign_retention_roi_grid(
    *,
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    cltv: Sequence[float] | np.ndarray,
    threshold: float,
    activation_cost_values: Sequence[float],
    retention_values: Sequence[float],
) -> pd.DataFrame:
    """Gera grid custo x retencao com ROI e componentes economicos."""
    rows_list: list[dict[str, Any]] = []
    for activation_cost in activation_cost_values:
        for retention_rate in retention_values:
            rows_list.append(
                compute_campaign_economics(
                    y_true=y_true,
                    y_prob=y_prob,
                    cltv=cltv,
                    threshold=threshold,
                    activation_cost=float(activation_cost),
                    retention_rate=float(retention_rate),
                )
            )
    return pd.DataFrame(rows_list).sort_values(
        ["activation_cost", "retention_rate"]
    ).reset_index(drop=True)


def fit_final_estimator_and_generate_predictions(
    *,
    estimator: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    threshold: float,
    model_name: str,
    sample_weight: Any | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Refita o estimador no train_val completo e gera probabilidades no holdout."""
    fitted_estimator = clone(estimator)
    fit_params = _fit_params_for_estimator(fitted_estimator, sample_weight)
    fitted_estimator.fit(X_train, y_train, **fit_params)

    if hasattr(fitted_estimator, "predict_proba"):
        y_prob = fitted_estimator.predict_proba(X_test)[:, 1]
    else:
        decision = fitted_estimator.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))

    index_values = (
        pd.Index(X_test.index)
        if hasattr(X_test, "index")
        else pd.Index(np.arange(len(np.asarray(y_prob))))
    )
    holdout_df = pd.DataFrame(
        {
            "row_index": index_values.to_numpy(),
            "y_true": np.asarray(y_test, dtype=int),
            "proba": np.asarray(y_prob, dtype=float),
            "y_pred": (np.asarray(y_prob, dtype=float) >= float(threshold)).astype(int),
            "threshold": float(threshold),
            "model": model_name,
        }
    )
    return fitted_estimator, holdout_df


def compute_holdout_technical_metrics(
    *,
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Alias semantico para metricas tecnicas do holdout."""
    return compute_binary_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )


def _safe_ratio(values: pd.Series) -> float:
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(max_value, 0.0):
        return np.nan
    return float(min_value / max_value)


def _binary_group_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    positives = y_true == 1
    negatives = y_true == 0
    tp = int(((y_pred == 1) & positives).sum())
    fp = int(((y_pred == 1) & negatives).sum())
    fn = int(((y_pred == 0) & positives).sum())
    tn = int(((y_pred == 0) & negatives).sum())

    selection_rate = float((y_pred == 1).mean()) if len(y_pred) else np.nan
    recall = float(tp / (tp + fn)) if (tp + fn) else np.nan
    precision = float(tp / (tp + fp)) if (tp + fp) else np.nan
    f1 = float((2 * precision * recall) / (precision + recall)) if precision + recall else np.nan
    fpr = float(fp / (fp + tn)) if (fp + tn) else np.nan
    fnr = float(fn / (fn + tp)) if (fn + tp) else np.nan

    return {
        "support": int(len(y_true)),
        "selection_rate": selection_rate,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "fpr": fpr,
        "fnr": fnr,
    }


def build_fairness_report_for_feature(
    *,
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | pd.Series,
    sensitive_feature: Sequence[Any] | pd.Series,
    feature_name: str,
) -> dict[str, Any]:
    """Gera relatorio tabular de fairness por grupo e resumo de disparidade."""
    frame = pd.DataFrame(
        {
            "y_true": pd.Series(y_true, dtype=int).reset_index(drop=True),
            "y_pred": pd.Series(y_pred, dtype=int).reset_index(drop=True),
            "group": pd.Series(sensitive_feature, dtype="object").reset_index(drop=True),
        }
    )

    by_group_rows: list[dict[str, Any]] = []
    for group_value, group_df in frame.groupby("group", dropna=False):
        metrics = _binary_group_metrics(group_df["y_true"], group_df["y_pred"])
        by_group_rows.append({"group": str(group_value), **metrics})

    by_group = pd.DataFrame(by_group_rows).sort_values("group").reset_index(drop=True)
    tracked_metrics = ["selection_rate", "recall", "precision", "f1_score", "fpr", "fnr"]
    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "difference": float(by_group[metric].max() - by_group[metric].min()),
                "ratio": _safe_ratio(by_group[metric]),
            }
            for metric in tracked_metrics
        ]
    )

    selection_diff = float(by_group["selection_rate"].max() - by_group["selection_rate"].min())
    tpr_diff = float(by_group["recall"].max() - by_group["recall"].min())
    fpr_diff = float(by_group["fpr"].max() - by_group["fpr"].min())
    fairness_metrics = pd.DataFrame(
        [
            {"metric": "demographic_parity_difference", "value": selection_diff},
            {"metric": "equalized_odds_difference", "value": max(tpr_diff, fpr_diff)},
            {
                "metric": "equalized_odds_ratio",
                "value": np.nanmin(
                    [
                        _safe_ratio(by_group["recall"]),
                        _safe_ratio(by_group["fpr"]),
                    ]
                ),
            },
        ]
    )

    return {
        "feature_name": feature_name,
        "by_group": by_group,
        "summary": summary,
        "fairness_metrics": fairness_metrics,
    }


def build_fairness_reports_bundle(
    *,
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | pd.Series,
    X_sensitive: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Gera relatorios de fairness para multiplas features sensiveis."""
    reports: dict[str, dict[str, Any]] = {}
    for feature_name in features:
        reports[feature_name] = build_fairness_report_for_feature(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_feature=X_sensitive[feature_name],
            feature_name=feature_name,
        )
    return reports


def consolidate_fairness_reports(
    fairness_reports: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolida tabelas de fairness por grupo e metricas agregadas."""
    by_group_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for feature_name, report in fairness_reports.items():
        by_group_frames.append(report["by_group"].assign(feature_name=feature_name))
        summary_frames.append(
            report["summary"].assign(feature_name=feature_name, table="summary")
        )
        summary_frames.append(
            report["fairness_metrics"].assign(feature_name=feature_name, table="fairness_metrics")
        )

    by_group_df = pd.concat(by_group_frames, ignore_index=True) if by_group_frames else pd.DataFrame()
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return by_group_df, summary_df


def compute_shap_summary_payload(
    *,
    fitted_estimator: Any,
    X_reference: pd.DataFrame,
    sample_size: int = 200,
    background_size: int = 50,
    random_state: int = 42,
) -> dict[str, Any]:
    """Calcula SHAP values para o modelo final usando a representacao transformada."""
    try:
        import shap
    except ModuleNotFoundError as exc:  # pragma: no cover - depende do ambiente do usuario
        raise ModuleNotFoundError(
            "shap nao esta instalado no ambiente atual. "
            "Instale a dependencia no kernel do notebook para gerar os graficos de explicabilidade."
        ) from exc

    if not isinstance(fitted_estimator, Pipeline):
        raise TypeError("compute_shap_summary_payload requer um sklearn Pipeline ajustado.")

    model_step = fitted_estimator.named_steps["model"]
    pre_model_pipeline = fitted_estimator[:-1]

    sample_df = X_reference.sample(
        n=min(sample_size, len(X_reference)),
        random_state=random_state,
    ).copy()
    transformed_sample = to_dense_float32(pre_model_pipeline.transform(sample_df))
    background = transformed_sample[: min(background_size, len(transformed_sample))]

    def predict_fn(data: np.ndarray) -> np.ndarray:
        return model_step.predict_proba(data)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(transformed_sample, nsamples="auto")
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    feature_names = extract_selected_feature_names(fitted_estimator, X_reference)
    return {
        "shap_values": np.asarray(shap_values, dtype=float),
        "transformed_values": transformed_sample,
        "feature_names": feature_names,
        "sample_index": sample_df.index.tolist(),
    }


def build_artifact_name(prefix: str, model_name: str, suffix: str) -> str:
    """Padroniza nomes de artefatos do notebook 03."""
    slug = (
        model_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    return f"{prefix}_{slug}.{suffix}"


def save_figure_temp(fig: Any, artifact_name: str) -> Path:
    """Salva uma figura em diretório temporário para logging posterior no MLflow."""
    artifact_path = Path(tempfile.gettempdir()) / artifact_name
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(artifact_path, dpi=150, bbox_inches="tight")
    return artifact_path


def require_xgboost():
    """Retorna XGBClassifier ou falha com mensagem orientativa."""
    if XGBClassifier is None:
        raise ModuleNotFoundError(
            "xgboost nao esta instalado no ambiente atual. "
            "Rode `uv sync --extra dev` apos adicionar a dependencia ao projeto."
        )
    return XGBClassifier


def rows(X: Any, idx: Sequence[int]) -> Any:
    """Seleciona linhas de estruturas pandas, numpy ou sparse."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def to_dense_float32(X: Any) -> np.ndarray:
    """Converte entrada tabular para matriz densa float32."""
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def to_float32_vector(y: Any) -> np.ndarray:
    """Converte alvo para vetor float32 unidimensional."""
    return np.asarray(y, dtype=np.float32).reshape(-1)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve o device a ser usado pelo PyTorch."""
    if device is None:
        return DEFAULT_DEVICE
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def normalize_sample_weight(
    sample_weight: Any | None,
    normalize: bool = True,
) -> np.ndarray | None:
    """Normaliza pesos para media 1, evitando escalas extremas na loss."""
    if sample_weight is None:
        return None

    weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    if np.any(weights < 0):
        raise ValueError("sample_weight deve conter apenas valores nao negativos.")

    if normalize and weights.size > 0:
        mean_weight = float(weights.mean())
        if mean_weight > 0:
            weights = weights / mean_weight

    return weights


def make_dataloader(
    X: Any,
    y: Any,
    sample_weight: Any | None = None,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Cria DataLoader PyTorch a partir de arrays densos/sparse/pandas."""
    X_arr = torch.tensor(to_dense_float32(X), dtype=torch.float32)
    y_arr = torch.tensor(to_float32_vector(y), dtype=torch.float32)

    tensors = [X_arr, y_arr]
    if sample_weight is not None:
        weights = torch.tensor(
            normalize_sample_weight(sample_weight, normalize=False),
            dtype=torch.float32,
        )
        tensors.append(weights)

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_embedding_dataloader(
    X_tabular: Any,
    city_ids: Any,
    y: Any,
    sample_weight: Any | None = None,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Cria DataLoader para MLP com duas entradas: tabular e city embedding."""
    X_tab_arr = torch.tensor(to_dense_float32(X_tabular), dtype=torch.float32)
    city_arr = torch.tensor(np.asarray(city_ids, dtype=np.int64).reshape(-1), dtype=torch.long)
    y_arr = torch.tensor(to_float32_vector(y), dtype=torch.float32)

    tensors = [X_tab_arr, city_arr, y_arr]
    if sample_weight is not None:
        weights = torch.tensor(
            normalize_sample_weight(sample_weight, normalize=False),
            dtype=torch.float32,
        )
        tensors.append(weights)

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fold_results_frame(folds: list[dict[str, Any]]) -> pd.DataFrame:
    """Converte lista de metricas por fold em DataFrame."""
    return pd.DataFrame(folds)


def summarize_fold_results(
    model_name: str,
    fold_df: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Gera linha de resumo medio a partir de resultados por fold."""
    row = {"model": model_name}
    for metric in metrics:
        row[f"{metric}_mean"] = fold_df[metric].mean()

    if "fit_time_s" in fold_df:
        row["fit_time_mean_s"] = fold_df["fit_time_s"].mean()
    if "score_time_s" in fold_df:
        row["score_time_mean_s"] = fold_df["score_time_s"].mean()

    return row


def upsert_model_summary(
    results_df: pd.DataFrame,
    summary_row: dict[str, Any],
    sort_by: str = "pr_auc_mean",
) -> pd.DataFrame:
    """Atualiza ou insere linha de resumo de um modelo no ranking final."""
    model_name = summary_row["model"]
    updated = pd.concat(
        [results_df[results_df["model"] != model_name], pd.DataFrame([summary_row])],
        ignore_index=True,
    )
    return updated.sort_values(sort_by, ascending=False).reset_index(drop=True)


def get_processed_feature_names(
    preprocessor: Any,
    X: Any,
    y: Any | None = None,
) -> np.ndarray:
    """Ajusta um preprocessor clonado e retorna os nomes das features geradas."""
    preprocessor_fitted = clone(preprocessor)
    preprocessor_fitted.fit(X, y)
    return np.asarray(preprocessor_fitted.get_feature_names_out(), dtype=object)


def build_k_grid(
    n_features_processed: int,
    min_k: int = 10,
    include_all: bool = True,
    step: int = 1,
) -> list[int | str]:
    """Monta grid de k para SelectKBest com validacao minima."""
    if n_features_processed < min_k:
        raise ValueError(
            "Numero de features processadas insuficiente para o grid de selecao: "
            f"{n_features_processed} < {min_k}."
        )
    if step <= 0:
        raise ValueError(f"step deve ser maior que zero. Recebido: {step}.")

    k_grid: list[int | str] = list(range(min_k, n_features_processed + 1, step))
    if k_grid[-1] != n_features_processed:
        k_grid.append(n_features_processed)
    if include_all:
        k_grid.append("all")
    return k_grid


def extract_selected_feature_names(
    best_estimator: Any,
    feature_names: Sequence[str] | pd.DataFrame,
    selector_step: str = "selector",
) -> list[str]:
    """Extrai os nomes das features mantidas pelo seletor do pipeline vencedor."""
    if not isinstance(best_estimator, Pipeline):
        raise TypeError("extract_selected_feature_names requer um sklearn Pipeline.")

    if isinstance(feature_names, pd.DataFrame):
        transformed = feature_names.copy()
        current_feature_names = np.asarray(feature_names.columns, dtype=object)
        for step_name, step in best_estimator.steps:
            if step_name in {"model", "scaler"}:
                continue
            if step_name == selector_step:
                support_mask = step.get_support()
                return current_feature_names[support_mask].astype(str).tolist()
            if step_name == "prep":
                transformed = step.transform(transformed)
                current_feature_names = np.asarray(step.get_feature_names_out(), dtype=object)
            else:
                transformed = step.transform(transformed)
                current_feature_names = np.asarray(
                    getattr(transformed, "columns", current_feature_names),
                    dtype=object,
                )
        raise ValueError(f"O pipeline informado nao possui a etapa '{selector_step}'.")

    selector = best_estimator.named_steps[selector_step]
    support_mask = selector.get_support()
    feature_names_arr = np.asarray(feature_names, dtype=object)
    selected = feature_names_arr[support_mask]
    return selected.astype(str).tolist()


def summarize_grid_search_results(
    search: Any,
    model_name: str,
    selector_label_map: dict[Any, str] | None = None,
) -> dict[str, Any]:
    """Resume a melhor linha de um GridSearchCV para exibicao tabular."""
    selector_label_map = selector_label_map or {}
    best_idx = search.best_index_
    cv_results = search.cv_results_
    best_params = search.best_params_
    score_func = best_params["selector__score_func"]
    score_func_name = getattr(score_func, "__name__", str(score_func))

    return {
        "model": model_name,
        "selector": selector_label_map.get(score_func, score_func_name),
        "k": best_params["selector__k"],
        "pr_auc_mean": cv_results["mean_test_pr_auc"][best_idx],
        "roc_auc_mean": cv_results["mean_test_roc_auc"][best_idx],
        "recall_mean": cv_results["mean_test_recall"][best_idx],
        "precision_mean": cv_results["mean_test_precision"][best_idx],
        "f1_mean": cv_results["mean_test_f1"][best_idx],
        "fit_time_mean_s": cv_results["mean_fit_time"][best_idx],
        "score_time_mean_s": cv_results["mean_score_time"][best_idx],
    }


def format_selected_features_log(
    model_name: str,
    best_params: dict[str, Any],
    selected_features: Sequence[str],
) -> str:
    """Formata um log legivel com a configuracao vencedora e features mantidas."""
    score_func = best_params["selector__score_func"]
    score_func_name = getattr(score_func, "__name__", str(score_func))
    k_value = best_params["selector__k"]
    selected_features_list = list(selected_features)
    selected_lines = "\n".join(f"- {feature}" for feature in selected_features_list)

    if k_value == "all":
        k_message = (
            "all (todas as features processadas foram mantidas)"
        )
    else:
        k_message = str(k_value)

    return (
        f"=== FEATURES SELECIONADAS: {model_name} ===\n"
        f"Seletor: {score_func_name}\n"
        f"k vencedor: {k_message}\n"
        f"Quantidade final: {len(selected_features_list)}\n"
        "Features selecionadas:\n"
        f"{selected_lines}"
    )


def build_city_vocabulary(
    city_series: pd.Series,
    unknown_index: int = 0,
) -> dict[str, int]:
    """Cria vocabulario city->id reservando 0 para categoria desconhecida."""
    valid_cities = city_series.dropna().astype(str)
    unique_cities = sorted(valid_cities.unique().tolist())
    start_idx = 1 if unknown_index == 0 else 0
    return {city: idx for idx, city in enumerate(unique_cities, start=start_idx)}


def encode_city_ids(
    city_series: pd.Series,
    city_to_idx: dict[str, int],
    unknown_index: int = 0,
) -> np.ndarray:
    """Converte City em ids inteiros com fallback para unknown."""
    return (
        city_series.astype("string")
        .fillna("__missing__")
        .astype(str)
        .map(city_to_idx)
        .fillna(unknown_index)
        .astype(np.int64)
        .to_numpy()
    )


def split_tabular_and_city(
    X: pd.DataFrame,
    city_column: str = "City",
    geo_drop_columns: Sequence[str] = ("Zip Code", "Latitude", "Longitude", "Lat Long"),
) -> tuple[pd.Series, pd.DataFrame]:
    """Separa a coluna City do ramo tabular e remove geografia bruta do tabular."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("split_tabular_and_city requer um pandas.DataFrame.")
    if city_column not in X.columns:
        raise KeyError(f"Column '{city_column}' was not found in the input DataFrame.")

    city_series = X[city_column].copy()
    columns_to_drop = [city_column, *geo_drop_columns]
    X_tabular = X.drop(columns=columns_to_drop, errors="ignore").copy()
    return city_series, X_tabular


def infer_city_embedding_dim(n_cities: int) -> int:
    """Inferencia simples para dimensao da embedding de City."""
    if n_cities <= 0:
        return 4
    return int(min(32, max(4, np.ceil(np.sqrt(n_cities)))))


def build_round3_estimator(
    model_name: str,
    strategy_name: str,
    *,
    preprocessor: Any,
    fe_params: dict[str, Any],
    y_reference: Any,
    target_smoothing: float = 20.0,
    logistic_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
    mlp_params: dict[str, Any] | None = None,
    embedding_params: dict[str, Any] | None = None,
) -> Any:
    """Constroi um estimador do Round 3 para um modelo e estrategia geoespacial."""
    logistic_defaults = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    }
    if logistic_params is not None:
        logistic_defaults.update(logistic_params)

    xgb_defaults = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": float((np.asarray(y_reference) == 0).sum())
        / max(float((np.asarray(y_reference) == 1).sum()), 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }
    if xgb_params is not None:
        xgb_defaults.update(xgb_params)

    mlp_defaults = {
        "hidden_dim": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.0,
        "max_epochs": 80,
        "patience": 8,
        "min_delta": 1e-3,
        "val_size": 0.15,
        "threshold": 0.5,
        "random_state": 42,
        "verbose": False,
    }
    if mlp_params is not None:
        mlp_defaults.update(mlp_params)

    embedding_defaults = {
        "city_column": "City",
        "geo_drop_columns": ("Zip Code", "Latitude", "Longitude", "Lat Long"),
        "embedding_dim": None,
    }
    if embedding_params is not None:
        embedding_defaults.update(embedding_params)

    feature_engineer = FeatureEngineerTransformer(**fe_params)

    if model_name == "LogisticRegression":
        return Pipeline(
            [
                ("fe", feature_engineer),
                (
                    "geo",
                    GeoTransformer(
                        strategy=strategy_name,
                        target_smoothing=target_smoothing,
                    ),
                ),
                ("prep", preprocessor),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(**logistic_defaults)),
            ]
        )

    if model_name == "XGBoost":
        xgb_classifier = require_xgboost()
        return Pipeline(
            [
                ("fe", feature_engineer),
                (
                    "geo",
                    GeoTransformer(
                        strategy=strategy_name,
                        target_smoothing=target_smoothing,
                    ),
                ),
                ("prep", preprocessor),
                ("model", xgb_classifier(**xgb_defaults)),
            ]
        )

    if model_name == "MLP":
        if strategy_name == "city_embedding":
            return MLPEmbeddingClassifierWrapper(
                preprocessor=preprocessor,
                feature_engineer=feature_engineer,
                **embedding_defaults,
                **mlp_defaults,
            )

        return Pipeline(
            [
                ("fe", feature_engineer),
                (
                    "geo",
                    GeoTransformer(
                        strategy=strategy_name,
                        target_smoothing=target_smoothing,
                    ),
                ),
                ("prep", preprocessor),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", MLPClassifierWrapper(**mlp_defaults)),
            ]
        )

    raise ValueError(f"Modelo nao suportado no Round 3: {model_name}")


def evaluate_round3_model_strategies(
    model_name: str,
    strategy_specs: Sequence[tuple[str, str]],
    *,
    X: Any,
    y: Any,
    cv: Any,
    scoring: Any,
    preprocessor: Any,
    fe_params: dict[str, Any],
    y_reference: Any,
    metrics: Sequence[str] = DEFAULT_METRICS,
    target_smoothing: float = 20.0,
    logistic_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
    mlp_params: dict[str, Any] | None = None,
    embedding_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Avalia todas as estrategias do Round 3 para um unico modelo."""
    rows_list: list[dict[str, Any]] = []
    fold_results: dict[str, pd.DataFrame] = {}

    for strategy_name, experiment_name in strategy_specs:
        estimator = build_round3_estimator(
            model_name,
            strategy_name,
            preprocessor=preprocessor,
            fe_params=fe_params,
            y_reference=y_reference,
            target_smoothing=target_smoothing,
            logistic_params=logistic_params,
            xgb_params=xgb_params,
            mlp_params=mlp_params,
            embedding_params=embedding_params,
        )

        cv_res = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        fold_df = pd.DataFrame(
            {
                "fold": np.arange(1, len(cv_res["fit_time"]) + 1),
                **{metric: cv_res[f"test_{metric}"] for metric in metrics},
                "pr_auc_std": cv_res["test_pr_auc"].std(),
                "fit_time_s": cv_res["fit_time"],
                "score_time_s": cv_res["score_time"],
            }
        )
        fold_results[experiment_name] = fold_df

        summary_row = summarize_fold_results(
            experiment_name,
            fold_df,
            metrics=metrics,
        )
        summary_row["base_model"] = model_name
        summary_row["strategy"] = strategy_name
        rows_list.append(summary_row)

    results_df = (
        pd.DataFrame(rows_list)
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    return results_df, fold_results


class MLPClassifierWrapper(ClassifierMixin, BaseEstimator):
    """Wrapper sklearn para a arquitetura MLP definida em src.models.mlp."""

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 1,
        activation: str = "relu",
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.0,
        max_epochs: int = 80,
        patience: int = 8,
        min_delta: float = 1e-3,
        val_size: float = 0.15,
        threshold: float = 0.5,
        random_state: int = 42,
        device: str | torch.device | None = None,
        normalize_sample_weight_flag: bool = True,
        verbose: bool = False,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_size = val_size
        self.threshold = threshold
        self.random_state = random_state
        self.device = device
        self.normalize_sample_weight_flag = normalize_sample_weight_flag
        self.verbose = verbose

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None):
        """Treina a MLP com split interno para early stopping."""
        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=("csr", "csc", "coo"),
            dtype=None,
        )

        classes = unique_labels(y_checked)
        if len(classes) != 2:
            raise ValueError("MLPClassifierWrapper suporta apenas classificacao binaria.")

        self.classes_ = np.sort(classes)
        self.n_features_in_ = X_checked.shape[1]
        self.device_ = resolve_device(self.device)

        y_encoded = (np.asarray(y_checked) == self.classes_[1]).astype(np.float32)
        y_float = to_float32_vector(y_encoded)
        weights = normalize_sample_weight(
            sample_weight,
            normalize=self.normalize_sample_weight_flag,
        )
        if weights is not None and len(weights) != len(y_float):
            raise ValueError("sample_weight deve ter o mesmo tamanho de X e y.")

        train_idx, val_idx = self._make_train_val_split(y_float)

        X_train = rows(X_checked, train_idx)
        X_val = rows(X_checked, val_idx)
        y_train = y_float[train_idx]
        y_val = y_float[val_idx]

        w_train = None if weights is None else weights[train_idx]
        w_val = None if weights is None else weights[val_idx]

        train_loader = make_dataloader(
            X_train,
            y_train,
            sample_weight=w_train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = make_dataloader(
            X_val,
            y_val,
            sample_weight=w_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model_ = MLP(
            input_dim=self.n_features_in_,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device_)

        pos_weight_value = self._compute_pos_weight(y_train)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device_)

        self.criterion_ = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="none",
        )
        self.optimizer_ = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._evaluate_loss(val_loader)

            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and epoch % 10 == 0:
                print(
                    f"[MLP] epoch={epoch} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} patience={patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.epochs_trained_ = epoch
        self.best_val_loss_ = best_val_loss
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Retorna probabilidades no formato sklearn (classe 0, classe 1)."""
        prob_pos = self._predict_positive_proba(X)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: Any) -> np.ndarray:
        """Retorna classes previstas com base no threshold configurado."""
        prob_pos = self._predict_positive_proba(X)
        return np.where(prob_pos >= self.threshold, self.classes_[1], self.classes_[0])

    def decision_function(self, X: Any) -> np.ndarray:
        """Retorna logits crus da rede, no formato 1D."""
        check_is_fitted(self, "model_")
        X_checked = check_array(X, accept_sparse=("csr", "csc", "coo"), dtype=None)
        X_tensor = torch.tensor(to_dense_float32(X_checked), dtype=torch.float32).to(
            self.device_
        )

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze(1)
        return logits.cpu().numpy()

    def _predict_positive_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        logits = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-logits))

    def _make_train_val_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(len(y))
        class_counts = np.bincount(y.astype(int))

        if (
            self.val_size <= 0
            or self.val_size >= 1
            or len(y) < 10
            or class_counts.min() < 2
        ):
            return idx, idx

        train_idx, val_idx = train_test_split(
            idx,
            test_size=self.val_size,
            stratify=y,
            random_state=self.random_state,
        )
        return train_idx, val_idx

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> float:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        return neg / max(pos, 1.0)

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model_.train()
        total_loss = 0.0

        for batch in dataloader:
            X_batch, y_batch, sample_weight = self._unpack_batch(batch)

            self.optimizer_.zero_grad()
            logits = self.model_(X_batch)
            loss = self._compute_weighted_loss(logits, y_batch, sample_weight)
            loss.backward()
            self.optimizer_.step()
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        self.model_.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                X_batch, y_batch, sample_weight = self._unpack_batch(batch)
                logits = self.model_(X_batch)
                total_loss += self._compute_weighted_loss(
                    logits,
                    y_batch,
                    sample_weight,
                ).item()

        return total_loss / max(len(dataloader), 1)

    def _unpack_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        X_batch = batch[0].to(self.device_)
        y_batch = batch[1].to(self.device_).unsqueeze(1)
        sample_weight = None
        if len(batch) > 2:
            sample_weight = batch[2].to(self.device_).unsqueeze(1)
        return X_batch, y_batch, sample_weight

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        losses = self.criterion_(logits, targets)
        if sample_weight is None:
            return losses.mean()

        weighted_losses = losses * sample_weight
        denom = sample_weight.sum().clamp_min(torch.finfo(weighted_losses.dtype).eps)
        return weighted_losses.sum() / denom


class MLPEmbeddingClassifierWrapper(ClassifierMixin, BaseEstimator):
    """Wrapper sklearn para MLP com embedding dedicado para City."""

    _estimator_type = "classifier"

    def __init__(
        self,
        preprocessor: Any,
        feature_engineer: Any | None = None,
        city_column: str = "City",
        geo_drop_columns: tuple[str, ...] = ("Zip Code", "Latitude", "Longitude", "Lat Long"),
        embedding_dim: int | None = None,
        unknown_city_index: int = 0,
        hidden_dim: int = 64,
        output_dim: int = 1,
        activation: str = "relu",
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.0,
        max_epochs: int = 80,
        patience: int = 8,
        min_delta: float = 1e-3,
        val_size: float = 0.15,
        threshold: float = 0.5,
        random_state: int = 42,
        device: str | torch.device | None = None,
        normalize_sample_weight_flag: bool = True,
        verbose: bool = False,
    ) -> None:
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.city_column = city_column
        self.geo_drop_columns = geo_drop_columns
        self.embedding_dim = embedding_dim
        self.unknown_city_index = unknown_city_index
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_size = val_size
        self.threshold = threshold
        self.random_state = random_state
        self.device = device
        self.normalize_sample_weight_flag = normalize_sample_weight_flag
        self.verbose = verbose

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MLPEmbeddingClassifierWrapper requer X como pandas.DataFrame.")

        y_array = np.asarray(y)
        if len(X) != len(y_array):
            raise ValueError("X e y devem possuir o mesmo numero de linhas.")

        classes = unique_labels(y_array)
        if len(classes) != 2:
            raise ValueError("MLPEmbeddingClassifierWrapper suporta apenas classificacao binaria.")

        self.classes_ = np.sort(classes)
        self.device_ = resolve_device(self.device)
        self.n_features_in_ = X.shape[1]

        y_encoded = (y_array == self.classes_[1]).astype(np.float32)
        y_float = to_float32_vector(y_encoded)
        weights = normalize_sample_weight(
            sample_weight,
            normalize=self.normalize_sample_weight_flag,
        )
        if weights is not None and len(weights) != len(y_float):
            raise ValueError("sample_weight deve ter o mesmo tamanho de X e y.")

        train_idx, val_idx = self._make_train_val_split(y_float)
        X_train_raw = rows(X, train_idx).copy()
        X_val_raw = rows(X, val_idx).copy()
        y_train = y_float[train_idx]
        y_val = y_float[val_idx]
        w_train = None if weights is None else weights[train_idx]
        w_val = None if weights is None else weights[val_idx]

        if self.feature_engineer is not None:
            self.feature_engineer_ = clone(self.feature_engineer)
            X_train_fe = self.feature_engineer_.fit_transform(X_train_raw, y_train)
            X_val_fe = self.feature_engineer_.transform(X_val_raw)
        else:
            self.feature_engineer_ = None
            X_train_fe = X_train_raw
            X_val_fe = X_val_raw

        city_train, X_train_tab = split_tabular_and_city(
            X_train_fe,
            city_column=self.city_column,
            geo_drop_columns=self.geo_drop_columns,
        )
        city_val, X_val_tab = split_tabular_and_city(
            X_val_fe,
            city_column=self.city_column,
            geo_drop_columns=self.geo_drop_columns,
        )

        self.preprocessor_ = clone(self.preprocessor)
        X_train_tab_enc = self.preprocessor_.fit_transform(X_train_tab, y_train)
        X_val_tab_enc = self.preprocessor_.transform(X_val_tab)

        self.city_to_idx_ = build_city_vocabulary(
            city_train,
            unknown_index=self.unknown_city_index,
        )
        self.n_cities_ = len(self.city_to_idx_)
        self.embedding_dim_ = (
            self.embedding_dim
            if self.embedding_dim is not None
            else infer_city_embedding_dim(self.n_cities_)
        )

        city_train_ids = encode_city_ids(
            city_train,
            self.city_to_idx_,
            unknown_index=self.unknown_city_index,
        )
        city_val_ids = encode_city_ids(
            city_val,
            self.city_to_idx_,
            unknown_index=self.unknown_city_index,
        )

        train_loader = make_embedding_dataloader(
            X_train_tab_enc,
            city_train_ids,
            y_train,
            sample_weight=w_train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = make_embedding_dataloader(
            X_val_tab_enc,
            city_val_ids,
            y_val,
            sample_weight=w_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.tabular_input_dim_ = to_dense_float32(X_train_tab_enc).shape[1]
        self.model_ = CityEmbeddingMLP(
            input_dim=self.tabular_input_dim_,
            n_cities=self.n_cities_,
            embedding_dim=self.embedding_dim_,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device_)

        pos_weight_value = self._compute_pos_weight(y_train)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device_)

        self.criterion_ = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="none",
        )
        self.optimizer_ = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._evaluate_loss(val_loader)

            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and epoch % 10 == 0:
                print(
                    f"[MLP+Embedding] epoch={epoch} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} patience={patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.epochs_trained_ = epoch
        self.best_val_loss_ = best_val_loss
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        prob_pos = self._predict_positive_proba(X)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: Any) -> np.ndarray:
        prob_pos = self._predict_positive_proba(X)
        return np.where(prob_pos >= self.threshold, self.classes_[1], self.classes_[0])

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        X_tab_enc, city_ids = self._prepare_eval_inputs(X)
        x_tab_tensor = torch.tensor(to_dense_float32(X_tab_enc), dtype=torch.float32).to(self.device_)
        x_city_tensor = torch.tensor(np.asarray(city_ids, dtype=np.int64), dtype=torch.long).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x_tab_tensor, x_city_tensor).squeeze(1)
        return logits.cpu().numpy()

    def _predict_positive_proba(self, X: Any) -> np.ndarray:
        logits = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-logits))

    def _prepare_eval_inputs(self, X: Any) -> tuple[Any, np.ndarray]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MLPEmbeddingClassifierWrapper requer X como pandas.DataFrame.")

        X_eval = X.copy()
        if self.feature_engineer_ is not None:
            X_eval = self.feature_engineer_.transform(X_eval)

        city_eval, X_eval_tab = split_tabular_and_city(
            X_eval,
            city_column=self.city_column,
            geo_drop_columns=self.geo_drop_columns,
        )
        X_eval_tab_enc = self.preprocessor_.transform(X_eval_tab)
        city_eval_ids = encode_city_ids(
            city_eval,
            self.city_to_idx_,
            unknown_index=self.unknown_city_index,
        )
        return X_eval_tab_enc, city_eval_ids

    def _make_train_val_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(len(y))
        class_counts = np.bincount(y.astype(int))

        if (
            self.val_size <= 0
            or self.val_size >= 1
            or len(y) < 10
            or class_counts.min() < 2
        ):
            return idx, idx

        train_idx, val_idx = train_test_split(
            idx,
            test_size=self.val_size,
            stratify=y,
            random_state=self.random_state,
        )
        return train_idx, val_idx

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> float:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        return neg / max(pos, 1.0)

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model_.train()
        total_loss = 0.0

        for batch in dataloader:
            x_tab_batch, x_city_batch, y_batch, sample_weight = self._unpack_embedding_batch(batch)

            self.optimizer_.zero_grad()
            logits = self.model_(x_tab_batch, x_city_batch)
            loss = self._compute_weighted_loss(logits, y_batch, sample_weight)
            loss.backward()
            self.optimizer_.step()
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        self.model_.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                x_tab_batch, x_city_batch, y_batch, sample_weight = self._unpack_embedding_batch(batch)
                logits = self.model_(x_tab_batch, x_city_batch)
                total_loss += self._compute_weighted_loss(
                    logits,
                    y_batch,
                    sample_weight,
                ).item()

        return total_loss / max(len(dataloader), 1)

    def _unpack_embedding_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x_tab_batch = batch[0].to(self.device_)
        x_city_batch = batch[1].to(self.device_)
        y_batch = batch[2].to(self.device_).unsqueeze(1)
        sample_weight = None
        if len(batch) > 3:
            sample_weight = batch[3].to(self.device_).unsqueeze(1)
        return x_tab_batch, x_city_batch, y_batch, sample_weight

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        losses = self.criterion_(logits, targets)
        if sample_weight is None:
            return losses.mean()

        weighted_losses = losses * sample_weight
        denom = sample_weight.sum().clamp_min(torch.finfo(weighted_losses.dtype).eps)
        return weighted_losses.sum() / denom
