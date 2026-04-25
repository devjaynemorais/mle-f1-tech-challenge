import os
import subprocess
import sys
from pathlib import Path

_in_venv = sys.prefix != sys.base_prefix
_expected_venv = Path(__file__).parent / ".venv"
if not _in_venv:
    print(
        "ERRO: ambiente virtual não está ativo.\n"
        "Ative antes de executar:\n"
        "  PowerShell : .venv\\Scripts\\Activate.ps1\n"
        "  Git Bash   : source .venv/Scripts/activate\n"
        "  Bash/Mac   : source .venv/bin/activate"
    )
    sys.exit(1)

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
    ]
    for script in scripts:
        run_step(script)
    print("\n[+] PIPELINE DE TREINO CONCLUÍDO COM SUCESSO!")
