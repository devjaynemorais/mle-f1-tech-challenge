"""Funções utilitárias para Análise Exploratória de Dados (EDA)."""

import pandas as pd


def sanity_check(df: pd.DataFrame, unique_threshold: int = 20):
    """Resumo de qualidade: shape, missing values, duplicados e cardinalidade."""
    print("=" * 60)
    print("SANITY CHECK DO DATASET")
    print("=" * 60)

    print(f"\nDimensao do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")

    print("\nValores faltantes (% por coluna):")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct.sort_values(ascending=False)
    for col, val in missing_pct.items():
        print(f"  {col}: {val:.2f}%")

    duplicated_count = df.duplicated().sum()
    pct_dup = (duplicated_count / len(df)) * 100
    print(f"\nRegistros duplicados: {duplicated_count} ({pct_dup:.2f}%)")

    print(f"\nCardinalidade por coluna (threshold={unique_threshold}):")
    for col in df.columns:
        n_unique = df[col].nunique()
        flag = " [alta cardinalidade]" if n_unique > unique_threshold else ""
        print(f"  {col}: {n_unique} valores únicos{flag}")


def freq_table(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Retorna tabela com frequência absoluta, relativa (%) e acumulada (%)
    para uma variável categórica.
    """
    freq_abs = df[var].value_counts(dropna=False)
    freq_rel = df[var].value_counts(normalize=True, dropna=False) * 100
    freq_cum = freq_rel.cumsum()

    return (
        pd.DataFrame(
            {
                "Frequência Absoluta": freq_abs,
                "Frequência Relativa (%)": freq_rel.round(2),
                "Frequência Acumulada (%)": freq_cum.round(2),
            }
        )
        .reset_index()
        .rename(columns={"index": var})
    )


def taxa_churn_categoria(
    df: pd.DataFrame, variavel: str, target: str = "Churn Value"
) -> pd.DataFrame:
    """
    Retorna taxa de churn por categoria de uma variável categórica.
    """
    return (
        df.groupby(variavel)[target]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "count": "Total Clientes",
                "sum": "Qtd Churn",
                "mean": "Taxa Churn",
            }
        )
        .sort_values(by="Taxa Churn", ascending=False)
        .assign(**{"Taxa Churn (%)": lambda x: x["Taxa Churn"] * 100})
    )
