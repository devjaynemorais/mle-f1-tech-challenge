"""Funções utilitárias para cálculo de métricas de avaliação de modelos."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Calcula recall, precision, f1 e auc dado probabilidades preditas.

    Parâmetros
    ----------
    y_true : array-like — rótulos verdadeiros
    y_proba : array-like — probabilidades preditas para a classe positiva
    threshold : float — limiar de classificação (default 0.5)

    Retorna
    -------
    dict com recall, precision, f1 e auc
    """
    y_true = np.array(y_true).ravel()
    y_pred = (np.array(y_proba) >= threshold).astype(int)

    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
    }


def avg_metrics(metrics_list: list) -> dict:
    """
    Calcula a média de um conjunto de dicionários de métricas (ex: resultados de CV).

    Parâmetros
    ----------
    metrics_list : list[dict] — lista de dicionários retornados por compute_metrics

    Retorna
    -------
    dict com a média de cada métrica
    """
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
