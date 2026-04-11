"""Arquitetura MLP e funções de treino/avaliação com PyTorch."""

import torch
import torch.nn as nn

from src.evaluation.metrics import compute_metrics

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """
    Rede Neural Multicamadas (MLP) para classificação binária.

    Parâmetros
    ----------
    input_dim : int — número de features de entrada
    hidden_dim : int — número de neurônios na camada oculta (default 64)
    output_dim : int — dimensão da saída (default 1 para classificação binária)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(X))


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device = DEFAULT_DEVICE,
) -> float:
    """
    Executa uma época de treino.

    Retorna
    -------
    float — loss médio da época
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device = DEFAULT_DEVICE,
    threshold: float = 0.5,
) -> tuple:
    """
    Avalia o modelo em um dataloader.

    Retorna
    -------
    tuple (loss médio, dict de métricas)
    """
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            probs = torch.sigmoid(outputs)

            preds.append(probs.cpu())
            targets.append(y_batch.cpu())
            total_loss += loss.item()

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    metrics = compute_metrics(targets, preds, threshold=threshold)

    return total_loss / len(dataloader), metrics
