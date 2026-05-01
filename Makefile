.PHONY: lint test train inference api mlflow \
        compose-build compose-up compose-train compose-down

lint:
	ruff check .

test:
	pytest tests/ -v

train:
	python run_train.py

inference:
	python run_inference.py

api:
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

mlflow:
	mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --allowed-hosts "127.0.0.1,127.0.0.1:5000"

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
