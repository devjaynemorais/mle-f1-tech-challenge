#!/bin/sh
set -e

case "$1" in
  train)
    exec python run_train.py
    ;;
  inference)
    exec python run_inference.py
    ;;
  api)
    exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    ;;
  mlflow)
    exec mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri sqlite:////mlflow/mlflow.db \
      --default-artifact-root /mlflow/mlartifacts \
      --allowed-hosts "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"
    ;;
  *)
    echo "Uso: docker run churn-api [train|inference|api|mlflow]"
    exit 1
    ;;
esac
