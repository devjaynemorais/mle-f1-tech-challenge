from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import mlflow
import pandas as pd

from src.experimentation.run_experiment import load_config, run_experiment


def resolve_config_paths(
    config_paths: list[str] | None = None,
    config_dir: str | None = None,
) -> list[Path]:
    """Resolve arquivos YAML explicitamente informados e/ou encontrados em um diretorio."""
    resolved_paths: set[Path] = set()

    for config_path in config_paths or []:
        path = Path(config_path).resolve()
        if path.suffix.lower() in {".yaml", ".yml"}:
            resolved_paths.add(path)

    if config_dir:
        config_dir_path = Path(config_dir).resolve()
        for pattern in ("*.yaml", "*.yml"):
            for path in config_dir_path.glob(pattern):
                resolved_paths.add(path.resolve())

    return sorted(resolved_paths)


def run_experiment_suite(
    config_paths: list[str] | None = None,
    config_dir: str | None = None,
) -> pd.DataFrame:
    """
    Executa uma suite de experimentos.

    Cada YAML representa uma run. Runs com o mesmo `tracking.experiment_name`
    ficam debaixo do mesmo experimento no MLflow.
    """
    resolved_paths = resolve_config_paths(config_paths=config_paths, config_dir=config_dir)
    if not resolved_paths:
        raise ValueError("Nenhum arquivo YAML valido foi informado para a suite.")

    grouped_configs: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    tracking_uris: dict[str, str | None] = {}

    for config_path in resolved_paths:
        config = load_config(config_path)
        tracking_cfg = config["tracking"]
        experiment_name = tracking_cfg["experiment_name"]
        tracking_uri = tracking_cfg.get("tracking_uri")

        if experiment_name in tracking_uris and tracking_uris[experiment_name] != tracking_uri:
            raise ValueError(
                f"Tracking URI inconsistente para o experimento '{experiment_name}'."
            )

        tracking_uris[experiment_name] = tracking_uri
        grouped_configs[experiment_name].append((config_path, config))

    suite_rows = []

    for experiment_name in sorted(grouped_configs):
        tracking_uri = tracking_uris[experiment_name]
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        for config_path, config in grouped_configs[experiment_name]:
            summary_df = run_experiment(config, set_experiment=False).copy()
            summary_df.insert(0, "experiment_name", experiment_name)
            summary_df.insert(0, "config_path", str(config_path))
            suite_rows.append(summary_df)

    return pd.concat(suite_rows, ignore_index=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa uma suite de experimentos a partir de multiplos YAMLs."
    )
    parser.add_argument(
        "config_paths",
        nargs="*",
        help="Lista explicita de arquivos YAML para rodar.",
    )
    parser.add_argument(
        "--config-dir",
        dest="config_dir",
        help="Diretorio contendo YAMLs de experimento.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    results = run_experiment_suite(
        config_paths=args.config_paths,
        config_dir=args.config_dir,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
