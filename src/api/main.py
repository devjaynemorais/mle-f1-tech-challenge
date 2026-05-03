"""FastAPI — API de inferência batch para predição de churn."""

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.metrics import CHURN_PROBABILITY, PREDICTIONS_TOTAL
from src.api.predictor import get_predictor
from src.api.schemas import BatchPredictRequest, BatchPredictResponse, HealthResponse
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API de inferência batch para predição de churn de clientes de telecom.",  # noqa: E501
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d  latency=%.1fms",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Erro não tratado em %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500, content={"detail": "Erro interno do servidor."}
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Verifica se a API e o modelo estão disponíveis."""
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model=predictor.model_name,
        model_path=predictor.model_path,
        model_uri=predictor.model_uri,
        threshold=predictor.threshold,
    )


@app.post("/predict", response_model=BatchPredictResponse, tags=["Predict"])
def predict_batch(body: BatchPredictRequest):
    """
    Predição de churn em batch.

    Recebe uma lista de registros de clientes e retorna a probabilidade
    e o label de churn para cada um.
    """
    predictor = get_predictor()
    logger.info("Recebida requisição batch com %d registros.", len(body.records))
    predictions = predictor.predict_batch(body.records)
    PREDICTIONS_TOTAL.labels(model=predictor.model_name).inc(len(predictions))
    for pred in predictions:
        CHURN_PROBABILITY.observe(pred.churn_probability)
    return BatchPredictResponse(
        model=predictor.model_name,
        threshold=predictor.threshold,
        n_records=len(body.records),
        predictions=predictions,
    )
