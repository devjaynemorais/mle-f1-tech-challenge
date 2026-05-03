import os
import subprocess
import sys

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

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run_step(script_path: str) -> None:
    print(f"\n{'=' * 40}\nExecutando: {script_path}\n{'=' * 40}")
    result = subprocess.run([sys.executable, script_path], env=ENV)
    if result.returncode != 0:
        print(f"Erro fatal executando {script_path}. Abortando pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    run_step("src/models/prepare_production_model.py")
    print("\n[+] PREPARO DO MODELO DE PRODUCAO CONCLUIDO COM SUCESSO!")
