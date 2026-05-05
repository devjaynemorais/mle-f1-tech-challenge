"""Métricas Prometheus customizadas da API de churn."""

from prometheus_client import Counter, Histogram, Gauge, Enum

PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total de predições de churn realizadas",
    ["model"],
)

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribuição das probabilidades de churn retornadas",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Métricas de latência
REQUEST_LATENCY = Histogram(
    "churn_api_request_latency_seconds",
    "Latência das requisições da API de churn em segundos",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
)

PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Tempo gasto para realizar predições em segundos",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0],
)

# Métricas de erro
REQUESTS_TOTAL = Counter(
    "churn_api_requests_total",
    "Total de requisições para a API de churn",
    ["endpoint", "method", "status"],
)

REQUEST_ERRORS_TOTAL = Counter(
    "churn_api_request_errors_total",
    "Total de erros nas requisições da API",
    ["endpoint", "error_type"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "churn_prediction_errors_total",
    "Total de erros durante predições",
    ["model", "error_type"],
)

# Métrica de status do modelo
MODEL_LOADED = Gauge(
    "churn_model_loaded",
    "Indica se o modelo foi carregado com sucesso (1 = carregado, 0 = não carregado)",
    ["model_name"],
)

MODEL_LOAD_TIME = Gauge(
    "churn_model_load_time_seconds",
    "Tempo gasto para carregar o modelo em segundos",
    ["model_name"],
)

# Métrica de confiança das predições
PREDICTION_CONFIDENCE = Gauge(
    "churn_prediction_confidence",
    "Média da confiança (probabilidade) das últimas predições",
    ["model"],
)
