"""
Treina o modelo MLP de produção de forma autossuficiente.

Lê os hiperparâmetros de config/best_params.yaml (gerado por run_experiment.py
e versionado no git), recria o split com a mesma seed, treina o MLP Optuna,
avalia no holdout, registra no MLflow e materializa os artefatos.

Não depende de nenhum run anterior no MLflow.
"""
# ruff: noqa: E402
from __future__ import annotations

import io
import os
import re
import socket
import subprocess
import sys
import time
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

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

sys.path.insert(0, str(BASE_DIR))

import mlflow
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.make_dataset import process_data
from src.utils.exp import (
    build_default_cv,
    build_optuna_mlp_pipeline,
    build_retention_roi_table,
    compute_holdout_technical_metrics,
    fit_final_estimator_and_generate_predictions,
    generate_oof_predictions,
    optimize_threshold_for_roi,
    resolve_tracking_uri,
)
from src.utils.mlflow_tracking import (
    log_dataframe_artifact,
    log_metrics_safe,
    log_params_safe,
    start_experiment_run,
)

MLFLOW_EXPERIMENT = "tc-f1-nb03-MLP-production-performance"
PRODUCTION_RUN_NAME = "nb03_mlp_production_model_threshold_035"
META_COLS = ["CLTV", "CustomerID"]
MLFLOW_PORT = 5000


def _ensure_mlflow_ui(db_uri: str) -> None:
    """Sobe o servidor MLflow em background se a porta não estiver ocupada."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        already_running = s.connect_ex(("localhost", MLFLOW_PORT)) == 0

    if already_running:
        print(f"  MLflow UI ja rodando em http://localhost:{MLFLOW_PORT}")
        return

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--host",
            "localhost",
            "--port",
            str(MLFLOW_PORT),
            "--backend-store-uri",
            db_uri,
            "--allowed-hosts",
            f"localhost,localhost:{MLFLOW_PORT}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(BASE_DIR),
    )
    time.sleep(2)
    print(f"  MLflow UI iniciado em http://localhost:{MLFLOW_PORT}")


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_best_params() -> dict:
    if not BEST_PARAMS_PATH.exists():
        print(
            f"AVISO: {BEST_PARAMS_PATH.name} nao encontrado. "
            "Execute run_experiment.py para gerar os melhores hiperparametros, "
            "ou use os valores padrao do arquivo existente."
        )
        sys.exit(1)
    with open(BEST_PARAMS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_data(
    config: dict,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame
]:
    interim_path = BASE_DIR / config["data"]["interim_path"]
    if not interim_path.exists():
        print("Criando dados intermediarios...")
        process_data()

    df = pd.read_csv(interim_path)
    target = config["data"]["target"]
    feature_cols = [c for c in df.columns if c not in [target] + META_COLS]

    return df, df[feature_cols], df[target], df[META_COLS]


def _split(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, split_params: dict):
    return train_test_split(
        X,
        y,
        meta,
        test_size=float(split_params["test_size"]),
        stratify=y,
        random_state=int(split_params["random_state"]),
    )


def _update_config(run_id: str) -> None:
    content = CONFIG_PATH.read_text(encoding="utf-8")
    content = re.sub(r"(run_id:\s*)\S+", f"run_id: {run_id}", content)
    content = re.sub(
        r"(model_uri:\s*)runs:/[^\s/]+/model",
        f"model_uri: runs:/{run_id}/model",
        content,
    )
    CONFIG_PATH.write_text(content, encoding="utf-8")
    print(f"  run_id   : {run_id}")
    print(f"  model_uri: runs:/{run_id}/model")


def main() -> None:
    config = _load_config()
    best_params = _load_best_params()

    mlp_params = best_params["mlp_optuna"]
    split_params = best_params["split"]
    prod_params = best_params["production"]

    raw_uri = config["mlflow"]["tracking_uri"]
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or resolve_tracking_uri(
        raw_uri, workspace_root=BASE_DIR
    )
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_mlflow_ui(raw_uri)

    phases = tqdm(
        [
            "dados",
            "pipeline",
            "OOF+threshold",
            "treino final",
            "MLflow",
            "materializar",
        ],
        desc="make train",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    phases.set_description("[1/6] dados")
    _, X, y, meta = _prepare_data(config)
    X_tv, X_test, y_tv, y_test, meta_tv, meta_test = _split(X, y, meta, split_params)
    print(f"\n  train/val: {len(X_tv)} | test: {len(X_test)}")
    phases.update(1)

    phases.set_description("[2/6] pipeline")
    pipeline = build_optuna_mlp_pipeline(mlp_params)
    cv = build_default_cv(
        n_splits=config["validation"]["n_splits"],
        random_state=int(split_params["random_state"]),
    )
    phases.update(1)

    phases.set_description("[3/6] OOF + threshold")
    oof_df = generate_oof_predictions(
        estimator=pipeline,
        X=X_tv,
        y=y_tv,
        cv=cv,
        model_name="MLP Optuna",
    )
    cltv_oof = meta_tv.loc[oof_df["row_index"].values, "CLTV"].values

    threshold_df, best_threshold_row = optimize_threshold_for_roi(
        y_true=oof_df["y_true"].values,
        y_prob=oof_df["proba"].values,
        cltv=cltv_oof,
        activation_cost=float(prod_params["activation_cost"]),
        retention_rate=float(prod_params["retention_rate"]),
    )
    best_threshold = float(best_threshold_row["threshold"])
    print(f"\n  threshold otimo (ROI): {best_threshold:.2f}")
    print(f"  threshold producao fixo: {prod_params['threshold']}")
    phases.update(1)

    phases.set_description("[4/6] treino final")
    production_threshold = float(prod_params["threshold"])
    final_estimator, holdout_df = fit_final_estimator_and_generate_predictions(
        estimator=pipeline,
        X_train=X_tv,
        y_train=y_tv,
        X_test=X_test,
        y_test=y_test,
        threshold=production_threshold,
        model_name="MLP Optuna",
    )

    holdout_metrics = compute_holdout_technical_metrics(
        y_true=holdout_df["y_true"].values,
        y_prob=holdout_df["proba"].values,
        threshold=production_threshold,
    )
    roi_table = build_retention_roi_table(
        y_true=holdout_df["y_true"].values,
        y_prob=holdout_df["proba"].values,
        cltv=meta_test["CLTV"].values,
        threshold=production_threshold,
        activation_cost=float(prod_params["activation_cost"]),
        retention_values=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    )
    print(f"\n  PR-AUC  : {holdout_metrics.get('pr_auc', 0):.4f}")
    print(f"  ROC-AUC : {holdout_metrics.get('roc_auc', 0):.4f}")
    print(f"  Recall  : {holdout_metrics.get('recall', 0):.4f}")
    print(f"  F1      : {holdout_metrics.get('f1_score', 0):.4f}")
    phases.update(1)

    phases.set_description("[5/6] MLflow")
    with start_experiment_run(
        MLFLOW_EXPERIMENT,
        PRODUCTION_RUN_NAME,
        tags={"phase": "production", "model_name": "MLP Optuna"},
    ) as run:
        log_params_safe({f"mlp__{k}": v for k, v in mlp_params.items()})
        log_params_safe(
            {
                "threshold": production_threshold,
                "activation_cost": prod_params["activation_cost"],
                "retention_rate": prod_params["retention_rate"],
                "split_random_state": split_params["random_state"],
                "split_test_size": split_params["test_size"],
            }
        )
        log_metrics_safe(holdout_metrics)
        log_dataframe_artifact(threshold_df, "threshold_search.csv")
        log_dataframe_artifact(holdout_df, "holdout_predictions.csv")
        log_dataframe_artifact(roi_table, "roi_by_retention_rate.csv")

        mlflow.sklearn.log_model(
            sk_model=final_estimator,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=X_test.head(5),
        )
        run_id = run.info.run_id

    print(f"\n  run_id: {run_id}")
    phases.update(1)

    phases.set_description("[6/6] materializar")
    _update_config(run_id)
    result = subprocess.run(
        [sys.executable, "src/models/prepare_production_model.py"],
        env=ENV,
    )
    if result.returncode != 0:
        print("Erro ao materializar o modelo. Verifique logs acima.")
        sys.exit(1)
    phases.update(1)
    phases.close()

    print("\n[+] TREINAMENTO CONCLUIDO COM SUCESSO!")
    print(f"    run_id  : {run_id}")
    print(f"    threshold: {production_threshold}")
    print("    modelo  : models/production/mlp_optuna/")
    print("\nProximos passos:")
    print("  make api          # API local")
    print("  make compose-full # stack Docker completo")


if __name__ == "__main__":
    main()
