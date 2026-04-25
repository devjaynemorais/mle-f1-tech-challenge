.PHONY: lint test train inference api docker-build docker-run

lint:
	ruff check .

test:
	pytest tests/ -v

train:
	python run_train.py

inference:
	python run_inference.py

api:
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8080

docker-build:
	docker build -t churn-api:latest .

docker-run:
	docker run -p 8080:8080 churn-api:latest
