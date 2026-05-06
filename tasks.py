"""
Alternativa ao Makefile para Git Bash / Windows sem make instalado.

Uso:
    python tasks.py <target>

Targets disponíveis:
    lint          ruff check .
    test          pytest tests/ -v
    train         python run_train.py
    inference     python run_inference.py
    api           uvicorn src.api.main:app --reload --host localhost --port 8080
    docker-build  docker build -t churn-api:latest .
    docker-run    docker run -p 8080:8080 churn-api:latest
"""

import subprocess
import sys

PY = sys.executable  # usa o Python do venv atual


TASKS = {
    "lint": [PY, "-m", "ruff", "check", "."],
    "test": [PY, "-m", "pytest", "tests/", "-v"],
    "train": [PY, "run_train.py"],
    "inference": [PY, "run_inference.py"],
    "api": [
        PY,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--reload",
        "--host",
        "localhost",
        "--port",
        "8080",
    ],
    "docker-build": ["docker", "build", "-t", "churn-api:latest", "."],
    "docker-run": ["docker", "run", "-p", "8080:8080", "churn-api:latest"],
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        print(__doc__)
        sys.exit(0)

    target = sys.argv[1]
    cmd = TASKS[target]
    print(f"[tasks] {' '.join(str(c) for c in cmd)}")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
