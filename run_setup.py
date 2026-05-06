"""Orquestra a execucao dos experimentos formais e tunings do projeto."""
# ruff: noqa: E402
from __future__ import annotations

import io
import os
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
EXPERIMENTS_DIR = BASE_DIR / "config" / "experiments"

sys.path.insert(0, str(BASE_DIR))

from src.experimentation.run_experiment import load_config
from src.experimentation.run_experiment_suite import run_experiment_suite
from src.experimentation.run_optuna import run_optuna


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


def _resolve_experiment_paths(file_names: list[str]) -> list[str]:
    return [str((EXPERIMENTS_DIR / file_name).resolve()) for file_name in file_names]


def execute_setup() -> None:
    for suite_file_names in EXPERIMENT_SUITES:
        config_paths = _resolve_experiment_paths(suite_file_names)
        print(f"\nExecutando suite: {', '.join(Path(path).name for path in config_paths)}")
        run_experiment_suite(config_paths=config_paths)

    for config_name in OPTUNA_CONFIGS:
        config_path = (EXPERIMENTS_DIR / config_name).resolve()
        print(f"\nExecutando Optuna: {config_path.name}")
        config = load_config(config_path)
        run_optuna(config, config_path=config_path)


if __name__ == "__main__":
    execute_setup()
    print("\n[+] SETUP CONCLUIDO COM SUCESSO!")
