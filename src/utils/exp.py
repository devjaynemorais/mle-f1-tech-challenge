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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import DEFAULT_DEVICE, MLP

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
