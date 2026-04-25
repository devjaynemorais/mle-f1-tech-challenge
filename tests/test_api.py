"""Testes da API FastAPI usando TestClient."""
# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

_SAMPLE_RECORD = {
    "gender": "Male",
    "Senior Citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "Tenure Months": 24,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 85.0,
    "Total Charges": 2040.0,
    "CLTV": 3500,
}


def test_health_retorna_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "model_path" in data


def test_predict_batch_single_record():
    response = client.post("/predict", json={"records": [_SAMPLE_RECORD]})
    assert response.status_code == 200
    data = response.json()
    assert data["n_records"] == 1
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert pred["churn_label"] in (0, 1)


def test_predict_batch_multiple_records():
    response = client.post("/predict", json={"records": [_SAMPLE_RECORD] * 5})
    assert response.status_code == 200
    data = response.json()
    assert data["n_records"] == 5
    assert len(data["predictions"]) == 5


def test_predict_body_invalido_retorna_422():
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 422


def test_predict_campo_ausente_retorna_422():
    record_incompleto = {k: v for k, v in _SAMPLE_RECORD.items() if k != "gender"}
    response = client.post("/predict", json={"records": [record_incompleto]})
    assert response.status_code == 422


def test_health_latency_header():
    response = client.get("/health")
    assert "x-latency-ms" in response.headers


def test_predict_modelo_no_response():
    response = client.post("/predict", json={"records": [_SAMPLE_RECORD]})
    assert response.status_code == 200
    assert "model" in response.json()
