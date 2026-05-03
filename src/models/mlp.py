"""Arquitetura MLP e funções de treino/avaliação com PyTorch."""

import torch
import torch.nn as nn

from src.evaluation.metrics import compute_metrics

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_activation(activation: str = "relu") -> nn.Module:
    """Constroi a função de ativação a partir de um identificador estável."""
    activation_name = activation.lower()
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }

    if activation_name not in activations:
        supported = ", ".join(sorted(activations))
        raise ValueError(
            f"Unsupported activation '{activation}'. Supported values: {supported}."
        )

    return activations[activation_name]()


class MLP(nn.Module):
    """
    Rede Neural Multicamadas (MLP) para classificação binária.

    Parâmetros
    ----------
    input_dim : int — número de features de entrada
    hidden_dim : int — número de neurônios na camada oculta (default 64)
    output_dim : int — dimensão da saída (default 1 para classificação binária)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()
        self.activation_name = activation
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(p=dropout),
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(X))


class CityEmbeddingMLP(nn.Module):
    """MLP tabular com embedding dedicado para a coluna City."""

    def __init__(
        self,
        input_dim: int,
        n_cities: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.activation_name = activation
        self.dropout = dropout
        self.city_embedding = nn.Embedding(
            num_embeddings=n_cities + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.features = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(p=dropout),
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x_tabular: torch.Tensor,
        x_city: torch.Tensor,
    ) -> torch.Tensor:
        city_vec = self.city_embedding(x_city)
        x = torch.cat([x_tabular, city_vec], dim=1)
        return self.out(self.features(x))


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device = DEFAULT_DEVICE,
) -> float:
    """Executa uma época de treino. Retorna loss médio."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
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
    """Avalia o modelo. Retorna (loss médio, dict de métricas)."""
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            outputs = model(X_batch)
            total_loss += criterion(outputs, y_batch).item()
            preds.append(torch.sigmoid(outputs).cpu())
            targets.append(y_batch.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / len(dataloader), compute_metrics(targets, preds, threshold)


def train_with_early_stopping(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 7,
    min_delta: float = 1e-3,
    threshold: float = 0.5,
    logger=None,
) -> int:
    """
    Treina o MLP com early stopping baseado na val loss.

    Restaura os melhores pesos ao final.
    Retorna o número de épocas efetivamente treinadas.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _ = evaluate(model, val_loader, criterion, device, threshold)

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 and logger:
            logger.info(
                "Época %3d/%d  train_loss=%.4f  val_loss=%.4f  paciência=%d/%d",
                epoch,
                max_epochs,
                train_loss,
                val_loss,
                patience_counter,
                patience,
            )

        if patience_counter >= patience:
            if logger:
                logger.info(
                    "Early stopping na época %d (melhor val_loss=%.4f)",
                    epoch,
                    best_val_loss,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return epoch
