"""
Pipeline de setup — executa 5 etapas em sequência:

1. Interim data: roda make_dataset.py para criar data/interim/telecom_clean.csv
   a partir do Excel bruto. Pula se já existir.

2. Split baseline: carrega o interim CSV, faz o train/val/test split (70/30,
   estratificado, seed 42) e registra os índices no MLflow como artefatos CSV.
   Pula se o run já existir. Replica o que o notebook 02 faz.

3. Stub runs de comparação: cria no MLflow os 4 runs que o notebook 03 espera
   encontrar do notebook 02:
   - LogisticRegression e MLP como child runs do baseline
   - optuna_mlp e optuna_xgboost no experimento de tuning, cada um com um JSON
     de hiperparâmetros razoáveis para o dataset Telco

4. Notebook 03: executa 03_modelo_mvp.ipynb de forma não-interativa via nbconvert.
   O notebook carrega os splits e os 4 modelos, treina todos, faz a análise
   comparativa, seleciona o MLP Optuna como modelo de produção e registra o run
   no MLflow.

5. Materialização: extrai o run_id do run recém-criado, atualiza
   config/config.yaml, e roda prepare_production_model.py para baixar os
   artefatos do MLflow para models/production/mlp_optuna/.

Após isso, `make api` e `make compose-full` funcionam sem nenhum passo manual.
"""
# ruff: noqa: E402
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
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
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "03_modelo_mvp.ipynb"
NOTEBOOK_OUT = BASE_DIR / "notebooks" / "03_modelo_mvp_executed.ipynb"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

BASELINE_EXPERIMENT = "tc-f1-nb02-baselines"
BASELINE_RUN_NAME = "baseline_inicial"
OPTUNA_EXPERIMENT = "tc-f1-nb02-tuning-stage2-optuna"

# Hiperparâmetros padrão quando notebook 02 não foi executado.
# Valores razoáveis baseados em boas práticas para o dataset Telco.
_DEFAULT_MLP_OPTUNA_PARAMS = {
    "selector__k": 20,
    "model__activation": "relu",
    "model__hidden_dim": 128,
    "model__dropout": 0.3,
    "model__lr": 0.001,
    "model__weight_decay": 1e-4,
    "model__batch_size": 64,
}

_DEFAULT_XGB_OPTUNA_PARAMS = {
    "model__n_estimators": 200,
    "model__max_depth": 4,
    "model__learning_rate": 0.05,
    "model__min_child_weight": 1,
    "model__subsample": 0.8,
    "model__colsample_bytree": 0.8,
    "model__gamma": 0.0,
    "model__reg_alpha": 0.1,
    "model__reg_lambda": 1.0,
}

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

sys.path.insert(0, str(BASE_DIR))


def _run(description: str, cmd: list[str], cwd: str | None = None) -> None:
    print(f"\n{'=' * 40}\n{description}\n{'=' * 40}")
    result = subprocess.run(cmd, env=ENV, cwd=cwd)
    if result.returncode != 0:
        print(f"Erro fatal em: {description}. Abortando.")
        sys.exit(1)


def _load_config() -> dict:
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_splits() -> None:
    """Cria interim data e registra no MLflow dependencias do notebook 03."""
    print(f"\n{'=' * 40}\nPreparando dados e dependencias MLflow\n{'=' * 40}")

    import json
    import tempfile

    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient
    from sklearn.model_selection import train_test_split

    from src.data.make_dataset import process_data
    from src.utils.exp import resolve_tracking_uri
    from src.utils.mlflow_tracking import log_split_artifacts, start_experiment_run

    config = _load_config()
    interim_path = BASE_DIR / config["data"]["interim_path"]

    # 1. Interim data
    if not interim_path.exists():
        print("Criando dados intermediarios (make_dataset)...")
        process_data()
    else:
        print(f"Dados intermediarios ja existem: {interim_path.name}")

    tracking_uri = resolve_tracking_uri(
        config["mlflow"]["tracking_uri"], workspace_root=BASE_DIR
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 2. Baseline split run (parent)
    baseline_parent = None
    exp = client.get_experiment_by_name(BASELINE_EXPERIMENT)
    if exp is not None:
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"attributes.run_name = '{BASELINE_RUN_NAME}'",
            max_results=1,
        )
        if runs:
            baseline_parent = runs[0]
            print(
                f"Split baseline ja existe"
                f" (run_id={baseline_parent.info.run_id[:8]}...). Pulando."
            )

    if baseline_parent is None:
        df = pd.read_csv(interim_path)
        target = config["data"]["target"]
        meta_cols = ["CLTV", "CustomerID"]
        feature_cols = [col for col in df.columns if col not in [target] + meta_cols]

        X_tv, X_test, _, _, meta_tv, meta_test = train_test_split(
            df[feature_cols],
            df[target],
            df[meta_cols],
            test_size=config["experiment"]["test_size"],
            stratify=df[target],
            random_state=config["experiment"]["random_state"],
        )
        with start_experiment_run(
            BASELINE_EXPERIMENT,
            BASELINE_RUN_NAME,
            tags={"phase": BASELINE_RUN_NAME},
        ) as run:
            log_split_artifacts(X_tv.index, X_test.index, meta_tv, meta_test)
            baseline_parent = run
        print(f"Splits registrados: {len(X_tv)} treino/val | {len(X_test)} teste")

    # 3. Baseline child runs (LogisticRegression, MLP)
    baseline_exp_id = client.get_experiment_by_name(BASELINE_EXPERIMENT).experiment_id
    child_runs = client.search_runs(
        [baseline_exp_id],
        filter_string=f"tags.mlflow.parentRunId = '{baseline_parent.info.run_id}'",
        max_results=100,
    )
    child_models = {r.data.tags.get("model_name") for r in child_runs}

    for model_name in ("LogisticRegression", "MLP"):
        if model_name in child_models:
            print(f"Child run '{model_name}' ja existe. Pulando.")
            continue
        with start_experiment_run(
            BASELINE_EXPERIMENT,
            model_name,
            tags={
                "phase": BASELINE_RUN_NAME,
                "model_name": model_name,
                "mlflow.parentRunId": baseline_parent.info.run_id,
            },
        ):
            pass
        print(f"Child run '{model_name}' criada.")

    # 4. Optuna runs (MLP + XGBoost) com hiperparametros padrao
    optuna_configs = [
        (
            "optuna_mlp",
            "MLP",
            "best_mlp_optuna_params.json",
            _DEFAULT_MLP_OPTUNA_PARAMS,
        ),
        (
            "optuna_xgboost",
            "XGBoost",
            "best_xgb_optuna_params.json",
            _DEFAULT_XGB_OPTUNA_PARAMS,
        ),
    ]
    optuna_exp = client.get_experiment_by_name(OPTUNA_EXPERIMENT)
    existing_optuna = set()
    if optuna_exp is not None:
        for run_name, _, _, _ in optuna_configs:
            runs = client.search_runs(
                [optuna_exp.experiment_id],
                filter_string=f"attributes.run_name = '{run_name}'",
                max_results=1,
            )
            if runs:
                existing_optuna.add(run_name)

    for run_name, model_name, artifact_name, params in optuna_configs:
        if run_name in existing_optuna:
            print(f"Optuna run '{run_name}' ja existe. Pulando.")
            continue
        with start_experiment_run(
            OPTUNA_EXPERIMENT,
            run_name,
            tags={"phase": "tuning_stage2", "model_name": model_name},
        ):
            artifact_path = Path(tempfile.gettempdir()) / artifact_name
            artifact_path.write_text(
                json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            mlflow.log_artifact(str(artifact_path))
        print(f"Optuna run '{run_name}' criada com {artifact_name}.")


def execute_notebook() -> None:
    # cwd=NOTEBOOKS_DIR garante que Path.cwd().resolve().parent no notebook 03
    # resolva para a raiz do projeto (e nao para o pai dela)
    _run(
        "Executando notebook 03 (treino + registro MLflow)",
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=900",
            "--output",
            str(NOTEBOOK_OUT),
            str(NOTEBOOK_PATH),
        ],
        cwd=str(NOTEBOOKS_DIR),
    )


def extract_run_id() -> str:
    print(f"\n{'=' * 40}\nExtraindo run_id do MLflow\n{'=' * 40}")

    from mlflow.tracking import MlflowClient

    from src.utils.exp import resolve_tracking_uri

    config = _load_config()
    prod_cfg = config["production"]
    source_cfg = prod_cfg["models"][prod_cfg["active_model"]]["source"]
    tracking_uri = resolve_tracking_uri(
        source_cfg["tracking_uri"], workspace_root=BASE_DIR
    )
    experiment_name = source_cfg["experiment_name"]

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experimento '{experiment_name}' nao encontrado.")
        print("Verifique se o notebook 03 foi executado com sucesso.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(f"Nenhum run encontrado em '{experiment_name}'.")
        sys.exit(1)

    run_id = runs[0].info.run_id
    print(f"run_id: {run_id}")
    return run_id


def update_config(run_id: str) -> None:
    print(f"\n{'=' * 40}\nAtualizando config/config.yaml\n{'=' * 40}")
    content = CONFIG_PATH.read_text(encoding="utf-8")

    content = re.sub(r"(run_id:\s*)\S+", f"run_id: {run_id}", content)
    content = re.sub(
        r"(model_uri:\s*)runs:/[^\s/]+/model",
        f"model_uri: runs:/{run_id}/model",
        content,
    )

    CONFIG_PATH.write_text(content, encoding="utf-8")
    print(f"run_id  : {run_id}")
    print(f"model_uri: runs:/{run_id}/model")


if __name__ == "__main__":
    prepare_splits()
    execute_notebook()
    run_id = extract_run_id()
    update_config(run_id)
    _run(
        "Materializando modelo de producao",
        [sys.executable, "src/models/prepare_production_model.py"],
    )
    print("\n[+] SETUP CONCLUIDO COM SUCESSO!")
    print(f"    run_id : {run_id}")
    print("    modelo : models/production/mlp_optuna/")
    print("\nProximos passos:")
    print("  make api          # API local")
    print("  make compose-full # stack Docker completo")
