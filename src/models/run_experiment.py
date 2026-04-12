"""
Orquestra experimentos de ML com validação cruzada e tracking no MLflow.

As transformações de engenharia de features são aplicadas de forma condicional,
permitindo ativar/desativar cada uma via arquivo de configuração YAML.

O pipeline é construído dinamicamente, garantindo flexibilidade e modularidade.

As métricas avaliadas incluem ROC AUC, F1 e Recall, com foco na estabilidade
do Recall — crucial para churn prediction.

O MLflow registra métricas, parâmetros e o modelo final de cada experimento,
facilitando a comparação entre diferentes configurações.
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.encoders import FrequencyEncoder


def build_pipeline(config, X):
    """
    Summary:
        Constrói o pipeline de experimentação com base nas transformações
        de engenharia de features.

    Args:
        config (yaml): Configuração yaml com transformações e parâmetros do modelo.
        X (DataFrame): DataFrame de entrada para identificar colunas numéricas
            e categóricas.

    Returns:
        Pipeline: Pipeline do sklearn com pré-processamento e modelo final.
    """

    steps = []
    model = LogisticRegression(**config["model"]["params"])

    numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if (
        config["features"]["city_freq_encoding"]["enabled"]
        and "City" in categorical_cols
    ):
        categorical_cols.remove("City")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
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
        model (Pipeline): Pipeline com pré-processamento e modelo final.
        X (DataFrame): DataFrame de entrada para a validação cruzada.
        y (Series): Séries de alvo para a validação cruzada.
        meta (DataFrame): DataFrame de metadados para a validação cruzada.
        config (yaml): Configuração yaml com parâmetros da validação cruzada.

    Returns:
        dict: Dicionário com as métricas de desempenho da validação cruzada.
    """

    cv_config = config["validation"]

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=config["experiment"]["random_state"],
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

        # CLTV como peso: clientes de maior valor têm mais influência no treino
        sample_weight = meta_train["CLTV"].values

        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # KPI tradicionais
        from sklearn.metrics import f1_score, recall_score, roc_auc_score

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

        capture_value_ratio = (
            captured_value / (captured_value + expected_loss)
            if (captured_value + expected_loss) > 0
            else 0
        )
        capture_value_ratios.append(capture_value_ratio)

    return {
        "roc_auc": roc_auc_scores,
        "f1": f1_scores,
        "recall": recall_scores,
        "cltv_mean": cltv_means,
        "expected_loss": expected_losses,
        "captured_value": captured_values,
        "capture_value_ratio": capture_value_ratios,
    }


def run_experiment(df, config):
    """
    Orquestra o experimento e realiza o tracking no MLflow.

    Loga métricas, parâmetros e o modelo final para comparação entre
    diferentes configurações de experimentos.

    Args:
        df (DataFrame): DataFrame de entrada para o experimento.
        config (yaml): Configuração yaml com transformações de features
            e parâmetros do modelo.

    Returns:
        dict: Dicionário com as métricas de desempenho do experimento.
    """

    target = config["data"]["target"]

    X = df.drop(columns=[target, "CLTV", "CustomerID"])
    y = df[target]
    meta = df[["CLTV", "CustomerID"]]

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
        "capture_value_ratio_mean": np.mean(scores["capture_value_ratio"]),
    }

    mlflow.log_params(config["model"]["params"])
    mlflow.log_metrics(metrics)
    mlflow.log_dict(config, "config.yaml")

    for feat_name, feat_cfg in config["features"].items():
        if isinstance(feat_cfg, dict) and "enabled" in feat_cfg:
            mlflow.log_params({f"feature__{feat_name}": feat_cfg["enabled"]})

    # O CV treina em folds parciais, deixando o pipeline no estado do último fold.
    # Por isso, criamos um novo pipeline e retreinamos no dataset completo,
    # garantindo que o modelo registrado no MLflow foi ajustado em todos os dados.
    final_pipeline = build_pipeline(config, X)
    # Usa CLTV como sample_weight — model__sample_weight atravessa o Pipeline
    # do sklearn até o LogisticRegression.
    sample_weight_full = meta["CLTV"].values
    final_pipeline.fit(X, y, model__sample_weight=sample_weight_full)
    mlflow.sklearn.log_model(final_pipeline, artifact_path="model")

    return metrics
