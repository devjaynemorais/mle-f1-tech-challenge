"""Utilitarios para experimentacao e wrapper sklearn da MLP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import CityEmbeddingMLP, DEFAULT_DEVICE, MLP

DEFAULT_METRICS = ("pr_auc", "roc_auc", "recall", "precision", "f1")


def rows(X: Any, idx: Sequence[int]) -> Any:
    """Seleciona linhas de estruturas pandas, numpy ou sparse."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def to_dense_float32(X: Any) -> np.ndarray:
    """Converte entrada tabular para matriz densa float32."""
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def to_float32_vector(y: Any) -> np.ndarray:
    """Converte alvo para vetor float32 unidimensional."""
    return np.asarray(y, dtype=np.float32).reshape(-1)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve o device a ser usado pelo PyTorch."""
    if device is None:
        return DEFAULT_DEVICE
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def normalize_sample_weight(
    sample_weight: Any | None,
    normalize: bool = True,
) -> np.ndarray | None:
    """Normaliza pesos para media 1, evitando escalas extremas na loss."""
    if sample_weight is None:
        return None

    weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    if np.any(weights < 0):
        raise ValueError("sample_weight deve conter apenas valores nao negativos.")

    if normalize and weights.size > 0:
        mean_weight = float(weights.mean())
        if mean_weight > 0:
            weights = weights / mean_weight

    return weights


def make_dataloader(
    X: Any,
    y: Any,
    sample_weight: Any | None = None,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Cria DataLoader PyTorch a partir de arrays densos/sparse/pandas."""
    X_arr = torch.tensor(to_dense_float32(X), dtype=torch.float32)
    y_arr = torch.tensor(to_float32_vector(y), dtype=torch.float32)

    tensors = [X_arr, y_arr]
    if sample_weight is not None:
        weights = torch.tensor(
            normalize_sample_weight(sample_weight, normalize=False),
            dtype=torch.float32,
        )
        tensors.append(weights)

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_embedding_dataloader(
    X_tabular: Any,
    city_ids: Any,
    y: Any,
    sample_weight: Any | None = None,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Cria DataLoader para MLP com duas entradas: tabular e city embedding."""
    X_tab_arr = torch.tensor(to_dense_float32(X_tabular), dtype=torch.float32)
    city_arr = torch.tensor(np.asarray(city_ids, dtype=np.int64).reshape(-1), dtype=torch.long)
    y_arr = torch.tensor(to_float32_vector(y), dtype=torch.float32)

    tensors = [X_tab_arr, city_arr, y_arr]
    if sample_weight is not None:
        weights = torch.tensor(
            normalize_sample_weight(sample_weight, normalize=False),
            dtype=torch.float32,
        )
        tensors.append(weights)

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fold_results_frame(folds: list[dict[str, Any]]) -> pd.DataFrame:
    """Converte lista de metricas por fold em DataFrame."""
    return pd.DataFrame(folds)


def summarize_fold_results(
    model_name: str,
    fold_df: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Gera linha de resumo medio a partir de resultados por fold."""
    row = {"model": model_name}
    for metric in metrics:
        row[f"{metric}_mean"] = fold_df[metric].mean()

    if "fit_time_s" in fold_df:
        row["fit_time_mean_s"] = fold_df["fit_time_s"].mean()
    if "score_time_s" in fold_df:
        row["score_time_mean_s"] = fold_df["score_time_s"].mean()

    return row


def upsert_model_summary(
    results_df: pd.DataFrame,
    summary_row: dict[str, Any],
    sort_by: str = "pr_auc_mean",
) -> pd.DataFrame:
    """Atualiza ou insere linha de resumo de um modelo no ranking final."""
    model_name = summary_row["model"]
    updated = pd.concat(
        [results_df[results_df["model"] != model_name], pd.DataFrame([summary_row])],
        ignore_index=True,
    )
    return updated.sort_values(sort_by, ascending=False).reset_index(drop=True)


def get_processed_feature_names(
    preprocessor: Any,
    X: Any,
    y: Any | None = None,
) -> np.ndarray:
    """Ajusta um preprocessor clonado e retorna os nomes das features geradas."""
    preprocessor_fitted = clone(preprocessor)
    preprocessor_fitted.fit(X, y)
    return np.asarray(preprocessor_fitted.get_feature_names_out(), dtype=object)


def build_k_grid(
    n_features_processed: int,
    min_k: int = 10,
    include_all: bool = True,
) -> list[int | str]:
    """Monta grid de k para SelectKBest com validacao minima."""
    if n_features_processed < min_k:
        raise ValueError(
            "Numero de features processadas insuficiente para o grid de selecao: "
            f"{n_features_processed} < {min_k}."
        )

    k_grid: list[int | str] = list(range(min_k, n_features_processed + 1))
    if include_all:
        k_grid.append("all")
    return k_grid


def extract_selected_feature_names(
    best_estimator: Any,
    feature_names: Sequence[str],
    selector_step: str = "selector",
) -> list[str]:
    """Extrai os nomes das features mantidas pelo seletor do pipeline vencedor."""
    selector = best_estimator.named_steps[selector_step]
    support_mask = selector.get_support()
    feature_names_arr = np.asarray(feature_names, dtype=object)
    selected = feature_names_arr[support_mask]
    return selected.tolist()


def summarize_grid_search_results(
    search: Any,
    model_name: str,
    selector_label_map: dict[Any, str] | None = None,
) -> dict[str, Any]:
    """Resume a melhor linha de um GridSearchCV para exibicao tabular."""
    selector_label_map = selector_label_map or {}
    best_idx = search.best_index_
    cv_results = search.cv_results_
    best_params = search.best_params_
    score_func = best_params["selector__score_func"]
    score_func_name = getattr(score_func, "__name__", str(score_func))

    return {
        "model": model_name,
        "selector": selector_label_map.get(score_func, score_func_name),
        "k": best_params["selector__k"],
        "pr_auc_mean": cv_results["mean_test_pr_auc"][best_idx],
        "roc_auc_mean": cv_results["mean_test_roc_auc"][best_idx],
        "recall_mean": cv_results["mean_test_recall"][best_idx],
        "precision_mean": cv_results["mean_test_precision"][best_idx],
        "f1_mean": cv_results["mean_test_f1"][best_idx],
        "fit_time_mean_s": cv_results["mean_fit_time"][best_idx],
        "score_time_mean_s": cv_results["mean_score_time"][best_idx],
    }


def format_selected_features_log(
    model_name: str,
    best_params: dict[str, Any],
    selected_features: Sequence[str],
) -> str:
    """Formata um log legivel com a configuracao vencedora e features mantidas."""
    score_func = best_params["selector__score_func"]
    score_func_name = getattr(score_func, "__name__", str(score_func))
    k_value = best_params["selector__k"]
    selected_features_list = list(selected_features)
    selected_lines = "\n".join(f"- {feature}" for feature in selected_features_list)

    if k_value == "all":
        k_message = (
            "all (todas as features processadas foram mantidas)"
        )
    else:
        k_message = str(k_value)

    return (
        f"=== FEATURES SELECIONADAS: {model_name} ===\n"
        f"Seletor: {score_func_name}\n"
        f"k vencedor: {k_message}\n"
        f"Quantidade final: {len(selected_features_list)}\n"
        "Features selecionadas:\n"
        f"{selected_lines}"
    )


def build_city_vocabulary(
    city_series: pd.Series,
    unknown_index: int = 0,
) -> dict[str, int]:
    """Cria vocabulario city->id reservando 0 para categoria desconhecida."""
    valid_cities = city_series.dropna().astype(str)
    unique_cities = sorted(valid_cities.unique().tolist())
    start_idx = 1 if unknown_index == 0 else 0
    return {city: idx for idx, city in enumerate(unique_cities, start=start_idx)}


def encode_city_ids(
    city_series: pd.Series,
    city_to_idx: dict[str, int],
    unknown_index: int = 0,
) -> np.ndarray:
    """Converte City em ids inteiros com fallback para unknown."""
    return (
        city_series.astype("string")
        .fillna("__missing__")
        .astype(str)
        .map(city_to_idx)
        .fillna(unknown_index)
        .astype(np.int64)
        .to_numpy()
    )


def split_tabular_and_city(
    X: pd.DataFrame,
    city_column: str = "City",
    geo_drop_columns: Sequence[str] = ("Zip Code", "Latitude", "Longitude", "Lat Long"),
) -> tuple[pd.Series, pd.DataFrame]:
    """Separa a coluna City do ramo tabular e remove geografia bruta do tabular."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("split_tabular_and_city requer um pandas.DataFrame.")
    if city_column not in X.columns:
        raise KeyError(f"Column '{city_column}' was not found in the input DataFrame.")

    city_series = X[city_column].copy()
    columns_to_drop = [city_column, *geo_drop_columns]
    X_tabular = X.drop(columns=columns_to_drop, errors="ignore").copy()
    return city_series, X_tabular


def infer_city_embedding_dim(n_cities: int) -> int:
    """Inferencia simples para dimensao da embedding de City."""
    if n_cities <= 0:
        return 4
    return int(min(32, max(4, np.ceil(np.sqrt(n_cities)))))


class MLPClassifierWrapper(ClassifierMixin, BaseEstimator):
    """Wrapper sklearn para a arquitetura MLP definida em src.models.mlp."""

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 1,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 80,
        patience: int = 8,
        val_size: float = 0.15,
        threshold: float = 0.5,
        random_state: int = 42,
        device: str | torch.device | None = None,
        normalize_sample_weight_flag: bool = True,
        verbose: bool = False,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_size = val_size
        self.threshold = threshold
        self.random_state = random_state
        self.device = device
        self.normalize_sample_weight_flag = normalize_sample_weight_flag
        self.verbose = verbose

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None):
        """Treina a MLP com split interno para early stopping."""
        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=("csr", "csc", "coo"),
            dtype=None,
        )

        classes = unique_labels(y_checked)
        if len(classes) != 2:
            raise ValueError("MLPClassifierWrapper suporta apenas classificacao binaria.")

        self.classes_ = np.sort(classes)
        self.n_features_in_ = X_checked.shape[1]
        self.device_ = resolve_device(self.device)

        y_encoded = (np.asarray(y_checked) == self.classes_[1]).astype(np.float32)
        y_float = to_float32_vector(y_encoded)
        weights = normalize_sample_weight(
            sample_weight,
            normalize=self.normalize_sample_weight_flag,
        )
        if weights is not None and len(weights) != len(y_float):
            raise ValueError("sample_weight deve ter o mesmo tamanho de X e y.")

        train_idx, val_idx = self._make_train_val_split(y_float)

        X_train = rows(X_checked, train_idx)
        X_val = rows(X_checked, val_idx)
        y_train = y_float[train_idx]
        y_val = y_float[val_idx]

        w_train = None if weights is None else weights[train_idx]
        w_val = None if weights is None else weights[val_idx]

        train_loader = make_dataloader(
            X_train,
            y_train,
            sample_weight=w_train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = make_dataloader(
            X_val,
            y_val,
            sample_weight=w_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model_ = MLP(
            input_dim=self.n_features_in_,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        ).to(self.device_)

        pos_weight_value = self._compute_pos_weight(y_train)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device_)

        self.criterion_ = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="none",
        )
        self.optimizer_ = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._evaluate_loss(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and epoch % 10 == 0:
                print(
                    f"[MLP] epoch={epoch} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} patience={patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.epochs_trained_ = epoch
        self.best_val_loss_ = best_val_loss
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Retorna probabilidades no formato sklearn (classe 0, classe 1)."""
        prob_pos = self._predict_positive_proba(X)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: Any) -> np.ndarray:
        """Retorna classes previstas com base no threshold configurado."""
        prob_pos = self._predict_positive_proba(X)
        return np.where(prob_pos >= self.threshold, self.classes_[1], self.classes_[0])

    def decision_function(self, X: Any) -> np.ndarray:
        """Retorna logits crus da rede, no formato 1D."""
        check_is_fitted(self, "model_")
        X_checked = check_array(X, accept_sparse=("csr", "csc", "coo"), dtype=None)
        X_tensor = torch.tensor(to_dense_float32(X_checked), dtype=torch.float32).to(
            self.device_
        )

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze(1)
        return logits.cpu().numpy()

    def _predict_positive_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        logits = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-logits))

    def _make_train_val_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(len(y))
        class_counts = np.bincount(y.astype(int))

        if (
            self.val_size <= 0
            or self.val_size >= 1
            or len(y) < 10
            or class_counts.min() < 2
        ):
            return idx, idx

        train_idx, val_idx = train_test_split(
            idx,
            test_size=self.val_size,
            stratify=y,
            random_state=self.random_state,
        )
        return train_idx, val_idx

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> float:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        return neg / max(pos, 1.0)

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model_.train()
        total_loss = 0.0

        for batch in dataloader:
            X_batch, y_batch, sample_weight = self._unpack_batch(batch)

            self.optimizer_.zero_grad()
            logits = self.model_(X_batch)
            loss = self._compute_weighted_loss(logits, y_batch, sample_weight)
            loss.backward()
            self.optimizer_.step()
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        self.model_.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                X_batch, y_batch, sample_weight = self._unpack_batch(batch)
                logits = self.model_(X_batch)
                total_loss += self._compute_weighted_loss(
                    logits,
                    y_batch,
                    sample_weight,
                ).item()

        return total_loss / max(len(dataloader), 1)

    def _unpack_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        X_batch = batch[0].to(self.device_)
        y_batch = batch[1].to(self.device_).unsqueeze(1)
        sample_weight = None
        if len(batch) > 2:
            sample_weight = batch[2].to(self.device_).unsqueeze(1)
        return X_batch, y_batch, sample_weight

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        losses = self.criterion_(logits, targets)
        if sample_weight is None:
            return losses.mean()

        weighted_losses = losses * sample_weight
        denom = sample_weight.sum().clamp_min(torch.finfo(weighted_losses.dtype).eps)
        return weighted_losses.sum() / denom


class MLPEmbeddingClassifierWrapper(ClassifierMixin, BaseEstimator):
    """Wrapper sklearn para MLP com embedding dedicado para City."""

    _estimator_type = "classifier"

    def __init__(
        self,
        preprocessor: Any,
        feature_engineer: Any | None = None,
        city_column: str = "City",
        geo_drop_columns: tuple[str, ...] = ("Zip Code", "Latitude", "Longitude", "Lat Long"),
        embedding_dim: int | None = None,
        unknown_city_index: int = 0,
        hidden_dim: int = 64,
        output_dim: int = 1,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 80,
        patience: int = 8,
        val_size: float = 0.15,
        threshold: float = 0.5,
        random_state: int = 42,
        device: str | torch.device | None = None,
        normalize_sample_weight_flag: bool = True,
        verbose: bool = False,
    ) -> None:
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.city_column = city_column
        self.geo_drop_columns = geo_drop_columns
        self.embedding_dim = embedding_dim
        self.unknown_city_index = unknown_city_index
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_size = val_size
        self.threshold = threshold
        self.random_state = random_state
        self.device = device
        self.normalize_sample_weight_flag = normalize_sample_weight_flag
        self.verbose = verbose

    def fit(self, X: Any, y: Any, sample_weight: Any | None = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MLPEmbeddingClassifierWrapper requer X como pandas.DataFrame.")

        y_array = np.asarray(y)
        if len(X) != len(y_array):
            raise ValueError("X e y devem possuir o mesmo numero de linhas.")

        classes = unique_labels(y_array)
        if len(classes) != 2:
            raise ValueError("MLPEmbeddingClassifierWrapper suporta apenas classificacao binaria.")

        self.classes_ = np.sort(classes)
        self.device_ = resolve_device(self.device)
        self.n_features_in_ = X.shape[1]

        y_encoded = (y_array == self.classes_[1]).astype(np.float32)
        y_float = to_float32_vector(y_encoded)
        weights = normalize_sample_weight(
            sample_weight,
            normalize=self.normalize_sample_weight_flag,
        )
        if weights is not None and len(weights) != len(y_float):
            raise ValueError("sample_weight deve ter o mesmo tamanho de X e y.")

        train_idx, val_idx = self._make_train_val_split(y_float)
        X_train_raw = rows(X, train_idx).copy()
        X_val_raw = rows(X, val_idx).copy()
        y_train = y_float[train_idx]
        y_val = y_float[val_idx]
        w_train = None if weights is None else weights[train_idx]
        w_val = None if weights is None else weights[val_idx]

        if self.feature_engineer is not None:
            self.feature_engineer_ = clone(self.feature_engineer)
            X_train_fe = self.feature_engineer_.fit_transform(X_train_raw, y_train)
            X_val_fe = self.feature_engineer_.transform(X_val_raw)
        else:
            self.feature_engineer_ = None
            X_train_fe = X_train_raw
            X_val_fe = X_val_raw

        city_train, X_train_tab = split_tabular_and_city(
            X_train_fe,
            city_column=self.city_column,
            geo_drop_columns=self.geo_drop_columns,
        )
        city_val, X_val_tab = split_tabular_and_city(
            X_val_fe,
            city_column=self.city_column,
            geo_drop_columns=self.geo_drop_columns,
        )

        self.preprocessor_ = clone(self.preprocessor)
        X_train_tab_enc = self.preprocessor_.fit_transform(X_train_tab, y_train)
        X_val_tab_enc = self.preprocessor_.transform(X_val_tab)

        self.city_to_idx_ = build_city_vocabulary(
            city_train,
            unknown_index=self.unknown_city_index,
        )
        self.n_cities_ = len(self.city_to_idx_)
        self.embedding_dim_ = (
            self.embedding_dim
            if self.embedding_dim is not None
            else infer_city_embedding_dim(self.n_cities_)
        )

        city_train_ids = encode_city_ids(
            city_train,
            self.city_to_idx_,
            unknown_index=self.unknown_city_index,
        )
        city_val_ids = encode_city_ids(
            city_val,
            self.city_to_idx_,
            unknown_index=self.unknown_city_index,
        )

        train_loader = make_embedding_dataloader(
            X_train_tab_enc,
            city_train_ids,
            y_train,
            sample_weight=w_train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = make_embedding_dataloader(
            X_val_tab_enc,
            city_val_ids,
            y_val,
            sample_weight=w_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.tabular_input_dim_ = to_dense_float32(X_train_tab_enc).shape[1]
        self.model_ = CityEmbeddingMLP(
            input_dim=self.tabular_input_dim_,
            n_cities=self.n_cities_,
            embedding_dim=self.embedding_dim_,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        ).to(self.device_)

        pos_weight_value = self._compute_pos_weight(y_train)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.device_)

        self.criterion_ = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="none",
        )
        self.optimizer_ = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss = self._evaluate_loss(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and epoch % 10 == 0:
                print(
                    f"[MLP+Embedding] epoch={epoch} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} patience={patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.epochs_trained_ = epoch
        self.best_val_loss_ = best_val_loss
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        prob_pos = self._predict_positive_proba(X)
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def predict(self, X: Any) -> np.ndarray:
        prob_pos = self._predict_positive_proba(X)
        return np.where(prob_pos >= self.threshold, self.classes_[1], self.classes_[0])

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        X_tab_enc, city_ids = self._prepare_eval_inputs(X)
        x_tab_tensor = torch.tensor(to_dense_float32(X_tab_enc), dtype=torch.float32).to(self.device_)
        x_city_tensor = torch.tensor(np.asarray(city_ids, dtype=np.int64), dtype=torch.long).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x_tab_tensor, x_city_tensor).squeeze(1)
        return logits.cpu().numpy()

    def _predict_positive_proba(self, X: Any) -> np.ndarray:
        logits = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-logits))

    def _prepare_eval_inputs(self, X: Any) -> tuple[Any, np.ndarray]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MLPEmbeddingClassifierWrapper requer X como pandas.DataFrame.")

        X_eval = X.copy()
        if self.feature_engineer_ is not None:
            X_eval = self.feature_engineer_.transform(X_eval)

        city_eval, X_eval_tab = split_tabular_and_city(
            X_eval,
            city_column=self.city_column,
            geo_drop_columns=self.geo_drop_columns,
        )
        X_eval_tab_enc = self.preprocessor_.transform(X_eval_tab)
        city_eval_ids = encode_city_ids(
            city_eval,
            self.city_to_idx_,
            unknown_index=self.unknown_city_index,
        )
        return X_eval_tab_enc, city_eval_ids

    def _make_train_val_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(len(y))
        class_counts = np.bincount(y.astype(int))

        if (
            self.val_size <= 0
            or self.val_size >= 1
            or len(y) < 10
            or class_counts.min() < 2
        ):
            return idx, idx

        train_idx, val_idx = train_test_split(
            idx,
            test_size=self.val_size,
            stratify=y,
            random_state=self.random_state,
        )
        return train_idx, val_idx

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> float:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        return neg / max(pos, 1.0)

    def _train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model_.train()
        total_loss = 0.0

        for batch in dataloader:
            x_tab_batch, x_city_batch, y_batch, sample_weight = self._unpack_embedding_batch(batch)

            self.optimizer_.zero_grad()
            logits = self.model_(x_tab_batch, x_city_batch)
            loss = self._compute_weighted_loss(logits, y_batch, sample_weight)
            loss.backward()
            self.optimizer_.step()
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        self.model_.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                x_tab_batch, x_city_batch, y_batch, sample_weight = self._unpack_embedding_batch(batch)
                logits = self.model_(x_tab_batch, x_city_batch)
                total_loss += self._compute_weighted_loss(
                    logits,
                    y_batch,
                    sample_weight,
                ).item()

        return total_loss / max(len(dataloader), 1)

    def _unpack_embedding_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x_tab_batch = batch[0].to(self.device_)
        x_city_batch = batch[1].to(self.device_)
        y_batch = batch[2].to(self.device_).unsqueeze(1)
        sample_weight = None
        if len(batch) > 3:
            sample_weight = batch[3].to(self.device_).unsqueeze(1)
        return x_tab_batch, x_city_batch, y_batch, sample_weight

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        losses = self.criterion_(logits, targets)
        if sample_weight is None:
            return losses.mean()

        weighted_losses = losses * sample_weight
        denom = sample_weight.sum().clamp_min(torch.finfo(weighted_losses.dtype).eps)
        return weighted_losses.sum() / denom
