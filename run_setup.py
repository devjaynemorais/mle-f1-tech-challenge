"""
Pipeline de setup — executa experimentos, tuning e materializacao em sequencia:

1. Experimentos: roda as suites de experimentos (base e allfeat) para MLP,
   Regressao Logistica e XGBoost, registrando tudo no MLflow.

2. Optuna: executa otimizacao de hiperparametros para cada modelo.

3. Materializacao: extrai o melhor run_id do MLflow e materializa o modelo
   de producao em models/production/mlp_optuna/.

Apos isso, `make api` e `make compose-full` funcionam sem nenhum passo manual.
"""
# ruff: noqa: E402
from __future__ import annotations

import io
import os
import re
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

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
EXPERIMENTS_DIR = BASE_DIR / "config" / "experiments"

sys.path.insert(0, str(BASE_DIR))

from src.experimentation.run_experiment import load_config
from src.experimentation.run_experiment_suite import run_experiment_suite
from src.experimentation.run_optuna import run_optuna
from src.models.prepare_production_model import prepare_production_model


EXPERIMENT_SUITES: list[list[str]] = [
    [
        "base_exp_mlp.yaml",
        "base_exp_reglog.yaml",
        "base_exp_xgb.yaml",
    ],
    [
        "allfeat_exp_mlp.yaml",
        "allfeat_exp_reglog.yaml",
        "allfeat_exp_xgb.yaml",
    ],
]

OPTUNA_CONFIGS: list[str] = [
    "optuna_mlp.yaml",
    "optuna_xgb.yaml",
    "optuna_reglog.yaml",
]

DEFAULT_LOCAL_MLFLOW_URI = "http://localhost:5000"


def _resolve_experiment_paths(file_names: list[str]) -> list[str]:
    return [str((EXPERIMENTS_DIR / file_name).resolve()) for file_name in file_names]


def _has_local_mlflow_server(tracking_uri: str = DEFAULT_LOCAL_MLFLOW_URI) -> bool:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
        "localhost",
        "127.0.0.1",
    }:
        return False

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=0.5):
            return True
    except OSError:
        return False


def _bootstrap_tracking_uri() -> str | None:
    env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_tracking_uri:
        return env_tracking_uri

    if os.environ.get("RUNNING_IN_DOCKER"):
        return None

    if _has_local_mlflow_server():
        os.environ["MLFLOW_TRACKING_URI"] = DEFAULT_LOCAL_MLFLOW_URI
        print(
            "MLflow local detectado em "
            f"{DEFAULT_LOCAL_MLFLOW_URI}; registrando runs nesse servidor."
        )
        return DEFAULT_LOCAL_MLFLOW_URI

    return None


def execute_setup() -> None:
    _bootstrap_tracking_uri()

    for suite_file_names in EXPERIMENT_SUITES:
        config_paths = _resolve_experiment_paths(suite_file_names)
        print(f"\nExecutando suite: {', '.join(Path(path).name for path in config_paths)}")
        run_experiment_suite(config_paths=config_paths)

    for config_name in OPTUNA_CONFIGS:
        config_path = (EXPERIMENTS_DIR / config_name).resolve()
        print(f"\nExecutando Optuna: {config_path.name}")
        config = load_config(config_path)
        run_optuna(config, config_path=config_path)


def extract_best_run_id() -> str | None:
    """Extrai o run_id do melhor modelo do experimento de producao."""
    print(f"\n{'=' * 40}\nExtraindo run_id do melhor modelo\n{'=' * 40}")
    _bootstrap_tracking_uri()

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from src.utils.exp import resolve_tracking_uri

        config = _load_config()
        env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        tracking_uri = resolve_tracking_uri(
            env_tracking_uri
            or config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db"),
            workspace_root=BASE_DIR,
        )
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)

        # Tenta encontrar o experimento de producao
        possible_names = [
            "churn-baseline",
            "Churn_Prediction_Pipeline",
            "churn_prediction_pipeline",
            "mlp_optuna",
            "optuna-mlp",
        ]

        experiment = None
        for name in possible_names:
            experiment = client.get_experiment_by_name(name)
            if experiment is not None:
                print(f"Experimento MLflow encontrado: {name}")
                break

        if experiment is None:
            all_experiments = mlflow.search_experiments()
            exp_names = [e.name for e in all_experiments]
            print(f"Experimento nao encontrado. Experimentos disponiveis: {exp_names}")
            return None

        # Busca o run mais recente com melhor metrica
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1_score DESC", "start_time DESC"],
            max_results=1,
        )

        if not runs:
            # Fallback: qualquer run
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )

        if not runs:
            print(f"Nenhum run encontrado em '{experiment.name}'.")
            return None

        run_id = runs[0].info.run_id
        print(f"Melhor run_id: {run_id}")
        return run_id

    except Exception as e:
        print(f"Erro ao extrair run_id: {e}")
        return None


def update_config_with_run_id(run_id: str) -> None:
    """Atualiza o config.yaml com o run_id do modelo de producao."""
    print(f"\n{'=' * 40}\nAtualizando config/config.yaml\n{'=' * 40}")

    content = CONFIG_PATH.read_text(encoding="utf-8")

    # Adiciona ou atualiza secao source se existir padrao de run_id
    if "run_id:" in content:
        content = re.sub(r"(run_id:\s*)\S+", f"run_id: {run_id}", content)
    else:
        # Adiciona run_id na secao production.models.mlp_optuna_prod
        content = re.sub(
            r"(active_model: mlp_optuna_prod\n)",
            f"active_model: mlp_optuna_prod\n  source:\n    run_id: {run_id}\n",
            content,
        )

    # Atualiza model_uri se existir
    if "model_uri:" in content:
        content = re.sub(
            r"(model_uri:\s*)runs:/[^\s/]+/model",
            f"model_uri: runs:/{run_id}/model",
            content,
        )

    CONFIG_PATH.write_text(content, encoding="utf-8")
    print(f"run_id atualizado: {run_id}")
    if "model_uri:" in content:
        print(f"model_uri: runs:/{run_id}/model")


def _load_config() -> dict:
    """Carrega o config.yaml principal."""
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def materialize_models() -> None:
    """Materializa os modelos de producao a partir do MLflow."""
    print("\nMaterializando modelos de producao...")
    _bootstrap_tracking_uri()

    # Tenta extrair o run_id primeiro
    run_id = extract_best_run_id()
    if run_id:
        update_config_with_run_id(run_id)

    # Materializa o modelo (baixa do MLflow se necessario)
    try:
        prepare_production_model(force_download=True)
        print("[+] Modelos materializados com sucesso!")
    except FileNotFoundError as e:
        print(f"[!] Modelo nao encontrado: {e}")
        print("    Execute os experimentos primeiro ou verifique o MLflow.")
    except Exception as e:
        print(f"[!] Erro na materializacao: {e}")


if __name__ == "__main__":
    execute_setup()
    materialize_models()
    print("\n[+] SETUP CONCLUIDO COM SUCESSO!")
    print("\nProximos passos:")
    print("  Para usar localmente (sem Docker):")
    print("    make api          # Inicia a API local")
    print("\n  Para usar com Docker:")
    print("    1. make compose-build   # Build das imagens Docker")
    print("    2. make compose-full    # Sobe a stack completa (API + banco + monitoring)")
