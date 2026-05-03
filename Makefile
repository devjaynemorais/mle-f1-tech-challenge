.PHONY: env lint test experiment setup train inference api mlflow \
        compose-build compose-up compose-train compose-down compose-monitoring compose-full

env:
	uv sync --extra dev

experiment:
	uv run python run_experiment.py

train:
	uv run python run_train.py

setup:
	uv run python src/models/prepare_production_model.py

lint:
	uv run ruff check .

test:
	uv run pytest tests/ -v

inference:
	uv run python run_inference.py

api:
	uv run uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

mlflow:
	uv run mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --allowed-hosts "127.0.0.1,127.0.0.1:5000"

# --- Docker Compose ---
compose-build:
	docker compose build

compose-train:
	docker compose up -d mlflow
	docker compose run --rm train

compose-up:
	docker compose up -d mlflow api

compose-down:
	docker compose down

compose-monitoring:
	docker compose up -d prometheus grafana

compose-full:
	docker compose up -d mlflow api prometheus grafana
