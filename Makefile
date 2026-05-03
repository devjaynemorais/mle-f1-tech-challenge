.PHONY: env lint test experiment setup train inference api mlflow \
        compose-build compose-up compose-experiment compose-train compose-down compose-monitoring compose-full

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
	uv run uvicorn src.api.main:app --reload --host localhost --port 8000

mlflow:
	uv run mlflow ui --host localhost --port 5000 --backend-store-uri sqlite:///mlflow.db --allowed-hosts "localhost,localhost:5000"

# --- Docker Compose ---
compose-build:
	docker compose build

compose-experiment:
	docker compose up -d mlflow
	docker compose run --rm experiment

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
