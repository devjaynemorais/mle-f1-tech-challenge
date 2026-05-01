"""Métricas Prometheus customizadas da API de churn."""

from prometheus_client import Counter, Histogram

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
