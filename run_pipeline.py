import os
import subprocess
import sys

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run_step(script_path):
    print(f"\n{'='*40}\nExecutando: {script_path}\n{'='*40}")
    result = subprocess.run([sys.executable, script_path], env=ENV)
    if result.returncode != 0:
        print(f"Erro fatal executando {script_path}. Abortando pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    scripts = [
        "src/data/make_dataset.py",
        "src/features/build_features.py",
        "src/models/train_model.py",
        "src/models/predict_model.py",
    ]
    for script in scripts:
        run_step(script)
    print("\n[+] PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
