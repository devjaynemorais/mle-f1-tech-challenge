"""
    Este módulo é responsável por orquestrar a execução dos experimentos, aplicando as transformações
     de engenharia de features selecionadas no arquivo de configuração .yaml e avaliando o modelo
     utilizando validação cruzada. As métricas de desempenho são logadas no MLflow para acompanhamento
     e comparação entre diferentes configurações de experimentos.
    
    As transformações de engenharia de features são aplicadas de forma condicional, permitindo que
    cada uma seja facilmente ativada ou desativada do pipeline de experimentação a partir de um arquivo
    de configuração yaml.
    
    O pipeline é construído dinamicamente com base nas transformações selecionadas, garantindo flexibilidade
    e modularidade na definição do processo de experimentação.
    
    As métricas avaliadas incluem ROC AUC, F1 Score e Recall, com foco especial na estabilidade do Recall,
    que é crucial para o contexto de churn prediction.
    
    O uso do MLflow permite um acompanhamento detalhado dos experimentos, facilitando a comparação entre
    diferentes configurações e a identificação das melhores práticas para o modelo de churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from features.encoders import FrequencyEncoder


def build_pipeline(config, X):
    """
    Summary:
        Esta função constrói o pipeline de experimentação com base nas transformações
        de engenharia de features.

    Args:
        config (yaml): Arquivo de configuração yaml contendo as transformações de engenharia de features
         e os parâmetros do modelo.
        X (DataFrame): DataFrame de entrada para o pipeline, utilizado para identificar as colunas numéricas e categóricas.

    Returns:
        Pipeline: Pipeline do sklearn contendo as etapas de pré-processamento e o modelo final.
    """

    steps = []
    model = LogisticRegression(**config["model"]["params"])

    numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if config["features"]["city_freq_encoding"]["enabled"] and "City" in categorical_cols:
        categorical_cols.remove("City")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    steps.append(("preprocessor", preprocessor))

    if config["features"]["city_freq_encoding"]["enabled"]:
        steps.append(("city_freq_encoding", FrequencyEncoder(column="City")))
    
    steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))

    pipeline = Pipeline(steps)

    return pipeline


def run_cv(model, X, y, meta, config):
    """
    Summary:
        Roda uma Validação Cruzada e retorna as métricas de desempenho.

    Args:
        model (Pipeline): Pipeline do sklearn contendo as etapas de pré-processamento e o modelo final.
        X (DataFrame): DataFrame de entrada para a validação cruzada.
        y (Series): Séries de alvo para a validação cruzada.
        meta (DataFrame): DataFrame de metadados para a validação cruzada.
        config (yaml): Arquivo de configuração yaml contendo os parâmetros da validação cruzada.

    Returns:
        dict: Dicionário contendo as métricas de desempenho da validação cruzada.
    """

    cv_config = config["validation"]

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=config["experiment"]["random_state"]
    )

    roc_auc_scores = []
    f1_scores = []
    recall_scores = []
    expected_losses = []
    cltv_means = []
    captured_values = []
    capture_value_ratios = []

    for train_idx, val_idx in cv.split(X, y):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        meta_train, meta_val = meta.iloc[train_idx], meta.iloc[val_idx]

        # CLTV como peso para o modelo, dando mais importância para os clientes com maior valor de vida útil
        sample_weight = meta_train["CLTV"].values

        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # KPI tradicionais
        from sklearn.metrics import roc_auc_score, f1_score, recall_score

        roc_auc_scores.append(roc_auc_score(y_val, y_proba))
        f1_scores.append(f1_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val, y_pred))

        # KPI de negócio -> abstrair depois em business_metrics.py
        captured_value = meta_val.loc[(y_val == 1) & (y_pred == 1), "CLTV"].sum()
        captured_values.append(captured_value)

        cltv_mean = meta_val.loc[(y_val == 1) & (y_pred == 1), "CLTV"].mean()
        cltv_means.append(cltv_mean)

        expected_loss = meta_val.loc[(y_val == 1) & (y_pred == 0), "CLTV"].sum()
        expected_losses.append(expected_loss)
        
        capture_value_ratio = captured_value / (captured_value + expected_loss) if (captured_value + expected_loss) > 0 else 0
        capture_value_ratios.append(capture_value_ratio)



    return {
        "roc_auc": roc_auc_scores,
        "f1": f1_scores,
        "recall": recall_scores,
        "cltv_mean": cltv_means,
        "expected_loss": expected_losses,
        "captured_value": captured_values,
        "capture_value_ratio": capture_value_ratios
    }


def run_experiment(df, config):
    """
        Esta função é responsável por orquestrar a execução do experimento e realiza o tracking
        no MLflow, logando as métricas de desempenho e os parâmetros do modelo para acompanhamento e comparação entre diferentes experimentos.

    Args:
        df (DataFrame): DataFrame de entrada para o experimento.
        config (yaml): Arquivo de configuração yaml contendo as transformações de engenharia de features
         e os parâmetros do modelo.

    Returns:
        dict: Dicionário contendo as métricas de desempenho do experimento.
    """

    target = config["data"]["target"]

    X = df.drop(columns=[target, "CLTV", "CustomerID"])
    y = df[target]
    meta = df[['CLTV', "CustomerID"]]

    pipeline = build_pipeline(config, X)

    scores = run_cv(pipeline, X, y, meta, config)

    metrics = {
        "roc_auc_mean": np.mean(scores["roc_auc"]),
        "f1_mean": np.mean(scores["f1"]),
        "recall_mean": np.mean(scores["recall"]),
        "recall_std": np.std(scores["recall"]),
        "cltv_mean": np.mean(scores["cltv_mean"]),
        "expected_loss_mean": np.mean(scores["expected_loss"]),
        "captured_value_mean": np.mean(scores["captured_value"]),
        "capture_value_ratio_mean": np.mean(scores["capture_value_ratio"])
    }


    mlflow.log_params(config["model"]["params"])
    mlflow.log_metrics(metrics)
    mlflow.log_dict(config, "config.yaml")

    for feat_name, feat_cfg in config["features"].items():
        if isinstance(feat_cfg, dict) and "enabled" in feat_cfg:
            mlflow.log_params({f"feature__{feat_name}": feat_cfg["enabled"]})

    return metrics