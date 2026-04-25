"""Funções utilitárias de visualização para análise exploratória."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


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
