"""Funções utilitárias de visualização para análise exploratória."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve


def plot_univariate(df: pd.DataFrame, col: str):
    """
    Gera histograma + KDE e QQplot lado a lado para uma variável numérica.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df[col], kde=True, color="skyblue", ax=axes[0])
    axes[0].set_title(f"Histograma e KDE - {col}", fontsize=12)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequência")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    stats.probplot(df[col].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f"QQ Plot - {col}", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Análise Univariada — {col}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def grid_cat_freq_event_rate(
    df: pd.DataFrame, cat_vars: list, target: str, n_cols: int = 2
):
    """
    Grid de gráficos combinados para variáveis categóricas:
    barras (frequência relativa %) + linha (taxa da classe positiva %).
    """
    if not cat_vars:
        print("Nenhuma variável categórica informada.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.0)
    n_vars = len(cat_vars)
    n_rows = math.ceil(n_vars / n_cols)
    taxa_global = df[target].mean() * 100

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, var in enumerate(cat_vars):
        ax = axes[i]
        ax2 = ax.twinx()

        freq = df[var].value_counts(normalize=True) * 100
        taxa = df.groupby(var)[target].mean() * 100

        freq = freq.reindex(taxa.index)

        ax.bar(
            freq.index.astype(str),
            freq.values,
            color="steelblue",
            alpha=0.6,
            label="Freq. (%)",
        )
        ax2.plot(
            taxa.index.astype(str),
            taxa.values,
            color="red",
            marker="o",
            linewidth=2,
            label="Taxa Churn (%)",
        )
        ax2.axhline(
            y=taxa_global, color="darkred", linestyle="--", linewidth=1, alpha=0.7
        )

        for x, y in zip(range(len(taxa)), taxa.values):
            ax2.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=8,
                color="red",
            )

        ax.set_title(var, fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequência (%)")
        ax2.set_ylabel("Taxa Churn (%)")
        ax.tick_params(axis="x", rotation=30)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Frequência e Taxa de Churn por Categoria", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def boxplots_target_binaria(
    df: pd.DataFrame, target: str, num_vars: list, n_cols: int = 3
):
    """
    Gera boxplots de variáveis numéricas segmentados pela variável alvo binária.
    """
    n_vars = len(num_vars)
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    sns.set_style("whitegrid")
    plt.suptitle(
        "Boxplots por variável numérica (Target binária)",
        fontsize=16,
        fontweight="bold",
    )

    for i, var in enumerate(num_vars):
        sns.boxplot(data=df, x=target, y=var, ax=axes[i], palette="Set2")
        axes[i].set_title(var, fontsize=11)
        axes[i].set_xlabel(target)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def _hide_unused_axes(axes, used_count: int) -> None:
    axes_array = np.atleast_1d(axes).reshape(-1)
    for idx in range(used_count, len(axes_array)):
        axes_array[idx].set_visible(False)


def _finalize_figure(fig: plt.Figure, show: bool) -> plt.Figure:
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_random_search_boxplots(
    df: pd.DataFrame,
    params: list[str],
    metric: str = "pr_auc_mean",
    n_cols: int = 3,
    figsize_scale: tuple[float, float] = (5.5, 4.0),
    show: bool = True,
) -> plt.Figure:
    """Gera boxplots do metric por hiperparametro discreto/categorico."""
    if not params:
        raise ValueError("params nao pode ser vazio.")

    sns.set_style("whitegrid")
    n_rows = math.ceil(len(params) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows),
    )
    axes = np.atleast_1d(axes).reshape(-1)

    for idx, param in enumerate(params):
        ax = axes[idx]
        sns.boxplot(data=df, x=param, y=metric, ax=ax, color="skyblue")
        ax.set_title(param, fontsize=11)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)

    _hide_unused_axes(axes, len(params))
    fig.suptitle(f"Distribuicao de {metric} por hiperparametro", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_random_search_summary_bars(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    show: bool = True,
) -> plt.Figure:
    """Plota barras ordenadas para sumarizacoes marginais do RandomizedSearchCV."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=summary_df, x=x_col, y=y_col, ax=ax, color="steelblue")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=30)
    return _finalize_figure(fig, show)


def plot_random_search_ordered_bin_lineplot(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    show: bool = True,
) -> plt.Figure:
    """Plota linha/pontos para bins ordenados de hiperparametros numericos."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.pointplot(data=summary_df, x=x_col, y=y_col, ax=ax, color="darkorange")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=30)
    return _finalize_figure(fig, show)


def plot_random_search_heatmap(
    interaction_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str = "pr_auc_mean_median",
    count_col: str = "count",
    min_count: int = 3,
    title: str | None = None,
    cmap: str = "YlGnBu",
    show: bool = True,
) -> plt.Figure:
    """Plota heatmap de interacao entre dois hiperparametros com mascara por suporte minimo."""
    pivot_values = interaction_df.pivot(index=y_col, columns=x_col, values=value_col)
    pivot_counts = interaction_df.pivot(index=y_col, columns=x_col, values=count_col)
    mask = pivot_counts.fillna(0) < min_count

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot_values,
        mask=mask,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title or f"{value_col} por {y_col} x {x_col}", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return _finalize_figure(fig, show)


def _make_grid_figure(
    total_panels: int,
    n_cols: int = 2,
    figsize_scale: tuple[float, float] = (6.0, 4.5),
) -> tuple[plt.Figure, np.ndarray]:
    n_rows = math.ceil(total_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows),
    )
    axes_arr = np.atleast_1d(axes).reshape(-1)
    return fig, axes_arr


def plot_roc_curves_grid(
    oof_predictions: dict[str, pd.DataFrame],
    *,
    n_cols: int = 2,
    show: bool = True,
) -> plt.Figure:
    """Plota um grid 2x2 de curvas ROC a partir das probabilidades OOF."""
    fig, axes = _make_grid_figure(len(oof_predictions), n_cols=n_cols)
    sns.set_style("whitegrid")

    for ax, (model_name, oof_df) in zip(axes, oof_predictions.items()):
        y_true = oof_df["y_true"].to_numpy(dtype=int)
        y_prob = oof_df["proba"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    _hide_unused_axes(axes, len(oof_predictions))
    fig.suptitle("Curvas ROC por Modelo", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_pr_curves_grid(
    oof_predictions: dict[str, pd.DataFrame],
    *,
    n_cols: int = 2,
    show: bool = True,
) -> plt.Figure:
    """Plota um grid 2x2 de curvas Precision-Recall."""
    fig, axes = _make_grid_figure(len(oof_predictions), n_cols=n_cols)
    sns.set_style("whitegrid")

    for ax, (model_name, oof_df) in zip(axes, oof_predictions.items()):
        y_true = oof_df["y_true"].to_numpy(dtype=int)
        y_prob = oof_df["proba"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color="darkorange", linewidth=2, label=f"AP = {pr_auc:.4f}")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")

    _hide_unused_axes(axes, len(oof_predictions))
    fig.suptitle("Curvas Precision-Recall por Modelo", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_learning_curves_grid(
    models: dict[str, object],
    *,
    X,
    y,
    cv,
    scoring: str = "average_precision",
    train_sizes: np.ndarray | None = None,
    n_cols: int = 2,
    show: bool = True,
) -> plt.Figure:
    """Plota learning curves em grid 2x2 para os modelos comparados."""
    fig, axes = _make_grid_figure(len(models), n_cols=n_cols)
    sns.set_style("whitegrid")
    train_sizes = train_sizes if train_sizes is not None else np.linspace(0.1, 1.0, 5)

    for ax, (model_name, estimator) in zip(axes, models.items()):
        sizes, train_scores, valid_scores = learning_curve(
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            train_sizes=train_sizes,
        )
        train_mean = train_scores.mean(axis=1)
        valid_mean = valid_scores.mean(axis=1)
        ax.plot(sizes, train_mean, marker="o", color="steelblue", label="Treino")
        ax.plot(sizes, valid_mean, marker="s", color="darkorange", label="Validação")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Amostras de treino")
        ax.set_ylabel(scoring)
        ax.legend(loc="best")

    _hide_unused_axes(axes, len(models))
    fig.suptitle("Learning Curves por Modelo", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_confusion_matrices_grid(
    oof_predictions: dict[str, pd.DataFrame],
    *,
    threshold: float = 0.5,
    n_cols: int = 2,
    show: bool = True,
) -> plt.Figure:
    """Plota um grid 2x2 de matrizes de confusão."""
    fig, axes = _make_grid_figure(len(oof_predictions), n_cols=n_cols)

    for ax, (model_name, oof_df) in zip(axes, oof_predictions.items()):
        y_true = oof_df["y_true"].to_numpy(dtype=int)
        y_pred = (oof_df["proba"].to_numpy(dtype=float) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
        ax.set_title(model_name, fontsize=11, fontweight="bold")

    _hide_unused_axes(axes, len(oof_predictions))
    fig.suptitle("Matrizes de Confusão por Modelo", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_calibration_curves_grid(
    oof_predictions: dict[str, pd.DataFrame],
    *,
    n_bins: int = 10,
    n_cols: int = 2,
    show: bool = True,
) -> plt.Figure:
    """Plota um grid 2x2 de curvas de calibração."""
    fig, axes = _make_grid_figure(len(oof_predictions), n_cols=n_cols)
    sns.set_style("whitegrid")

    for ax, (model_name, oof_df) in zip(axes, oof_predictions.items()):
        y_true = oof_df["y_true"].to_numpy(dtype=int)
        y_prob = oof_df["proba"].to_numpy(dtype=float)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", color="steelblue", linewidth=2, label="Modelo")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Probabilidade média prevista")
        ax.set_ylabel("Fração observada de churn")
        ax.legend(loc="best")

    _hide_unused_axes(axes, len(oof_predictions))
    fig.suptitle("Curvas de Calibração por Modelo", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_probability_histograms_grid(
    oof_predictions: dict[str, pd.DataFrame],
    *,
    bins: int = 20,
    n_cols: int = 2,
    show: bool = True,
) -> plt.Figure:
    """Plota histogramas das probabilidades previstas em grid 2x2."""
    fig, axes = _make_grid_figure(len(oof_predictions), n_cols=n_cols)
    sns.set_style("whitegrid")

    for ax, (model_name, oof_df) in zip(axes, oof_predictions.items()):
        sns.histplot(oof_df["proba"], bins=bins, kde=False, color="steelblue", ax=ax)
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Probabilidade prevista")
        ax.set_ylabel("Frequência")

    _hide_unused_axes(axes, len(oof_predictions))
    fig.suptitle("Histograma de Probabilidades por Modelo", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_roi_by_threshold(
    threshold_df: pd.DataFrame,
    *,
    show: bool = True,
) -> plt.Figure:
    """Plota a curva de ROI por threshold."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=threshold_df, x="threshold", y="roi", marker="o", ax=ax, color="steelblue")
    best_idx = threshold_df["roi"].astype(float).idxmax()
    best_row = threshold_df.loc[best_idx]
    ax.axvline(best_row["threshold"], linestyle="--", color="darkorange", linewidth=1.5)
    ax.scatter([best_row["threshold"]], [best_row["roi"]], color="darkorange", s=60, zorder=3)
    ax.set_title("ROI por Threshold", fontsize=12, fontweight="bold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("ROI")
    return _finalize_figure(fig, show)


def plot_confusion_matrix_threshold_comparison(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    threshold_a: float,
    threshold_b: float,
    labels: tuple[str, str] | None = None,
    show: bool = True,
) -> plt.Figure:
    """Compara matrizes de confusao em dois thresholds."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    labels = labels or (f"thr={threshold_a:.2f}", f"thr={threshold_b:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, threshold, label in zip(axes, [threshold_a, threshold_b], labels):
        y_pred = (y_prob_arr >= threshold).astype(int)
        cm = confusion_matrix(y_true_arr, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
        ax.set_title(label, fontsize=11, fontweight="bold")

    fig.suptitle("Comparacao de Matrizes de Confusao", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_retention_vs_roi(
    retention_df: pd.DataFrame,
    *,
    show: bool = True,
) -> plt.Figure:
    """Plota a relacao entre taxa de retencao e ROI."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(
        data=retention_df,
        x="retention_rate",
        y="roi",
        marker="o",
        ax=ax,
        color="darkorange",
    )
    ax.set_title("Retencao x ROI", fontsize=12, fontweight="bold")
    ax.set_xlabel("Taxa de retencao")
    ax.set_ylabel("ROI")
    return _finalize_figure(fig, show)


def plot_roi_heatmap(
    roi_grid_df: pd.DataFrame,
    *,
    show: bool = True,
) -> plt.Figure:
    """Plota heatmap de ROI por custo de campanha e retencao."""
    cost_column = "activation_cost" if "activation_cost" in roi_grid_df.columns else "campaign_cost"
    pivot = roi_grid_df.pivot(index="retention_rate", columns=cost_column, values="roi")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdBu",
        center=0.0,
        cbar=True,
        ax=ax,
    )
    ax.set_title("Heatmap de ROI por Custo e Retencao", fontsize=12, fontweight="bold")
    ax.set_xlabel("Custo por acionamento" if cost_column == "activation_cost" else "Custo de campanha")
    ax.set_ylabel("Taxa de retencao")
    return _finalize_figure(fig, show)


def plot_single_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    title: str,
    show: bool = True,
) -> plt.Figure:
    """Plota uma unica matriz de confusao."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = confusion_matrix(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax, colorbar=False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_holdout_pr_roc_subplot(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    title: str,
    show: bool = True,
) -> plt.Figure:
    """Plota PR e ROC lado a lado para o holdout final."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    precision, recall, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    pr_auc = average_precision_score(y_true_arr, y_prob_arr)
    axes[0].plot(recall, precision, color="darkorange", linewidth=2, label=f"AP = {pr_auc:.4f}")
    axes[0].set_title(f"{title} - PR", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left")

    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    roc_auc = roc_auc_score(y_true_arr, y_prob_arr)
    axes[1].plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[1].set_title(f"{title} - ROC", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    return _finalize_figure(fig, show)


def plot_single_probability_histogram(
    y_prob: np.ndarray | pd.Series,
    *,
    title: str,
    bins: int = 20,
    show: bool = True,
) -> plt.Figure:
    """Plota histograma simples das probabilidades do holdout."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(np.asarray(y_prob, dtype=float), bins=bins, kde=False, color="steelblue", ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequencia")
    return _finalize_figure(fig, show)


def plot_fairness_feature_subplot(
    by_group_df: pd.DataFrame,
    *,
    feature_name: str,
    show: bool = True,
) -> plt.Figure:
    """Plota selection rate, metricas principais e taxas de erro por grupo."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.set_style("whitegrid")

    sns.barplot(data=by_group_df, x="group", y="selection_rate", ax=axes[0], color="steelblue")
    axes[0].set_title(f"{feature_name} - Selection Rate", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Grupo")
    axes[0].set_ylabel("Selection Rate")

    metrics_long = by_group_df.melt(
        id_vars="group",
        value_vars=["recall", "precision", "f1_score"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=metrics_long, x="group", y="value", hue="metric", ax=axes[1])
    axes[1].set_title(f"{feature_name} - Recall / Precision / F1", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Grupo")
    axes[1].set_ylabel("Valor")
    axes[1].legend(title="Metrica", loc="best")

    error_long = by_group_df.melt(
        id_vars="group",
        value_vars=["fpr", "fnr"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=error_long, x="group", y="value", hue="metric", ax=axes[2])
    axes[2].set_title(f"{feature_name} - FPR / FNR", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Grupo")
    axes[2].set_ylabel("Valor")
    axes[2].legend(title="Metrica", loc="best")

    fig.suptitle(f"Fairness por Grupo - {feature_name}", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)


def plot_shap_summary_subplot(
    shap_values: np.ndarray,
    transformed_values: np.ndarray,
    feature_names: list[str],
    *,
    show: bool = True,
) -> plt.Figure:
    """Plota SHAP summary dot e bar lado a lado."""
    try:
        import shap
    except ModuleNotFoundError as exc:  # pragma: no cover - depende do ambiente do usuario
        raise ModuleNotFoundError(
            "shap nao esta instalado no ambiente atual. "
            "Instale a dependencia no kernel do notebook para gerar os plots SHAP."
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.sca(axes[0])
    shap.summary_plot(
        shap_values,
        transformed_values,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )
    axes[0].set_title("SHAP Summary Dot", fontsize=11, fontweight="bold")

    plt.sca(axes[1])
    shap.summary_plot(
        shap_values,
        transformed_values,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    axes[1].set_title("SHAP Summary Bar", fontsize=11, fontweight="bold")
    fig.suptitle("Explicabilidade SHAP - MLP Optuna", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, show)
