"""Funções utilitárias de análise estatística."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calcula o VIF (Variance Inflation Factor) para variáveis numéricas.

    Retorna DataFrame com VIF por variável, ordenado de forma decrescente.
    """
    X = df[features].dropna()
    vif_df = pd.DataFrame()
    vif_df["variavel"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)


def classify_vif(vif_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica o grau de multicolinearidade com base no VIF:
    - VIF < 5       : Baixa
    - 5 ≤ VIF < 10  : Moderada
    - VIF ≥ 10      : Alta
    """
    conditions = [
        vif_df["VIF"] < 5,
        (vif_df["VIF"] >= 5) & (vif_df["VIF"] < 10),
        vif_df["VIF"] >= 10,
    ]
    choices = [
        "Baixa multicolinearidade",
        "Multicolinearidade moderada",
        "Alta multicolinearidade",
    ]
    vif_df = vif_df.copy()
    vif_df["classificacao"] = np.select(conditions, choices, default="Indefinido")
    return vif_df


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Calcula o V de Cramér para medir associação entre variáveis categóricas."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))


def chi2_analysis(
    df: pd.DataFrame, cat_vars: list, target: str, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Realiza análise Qui-Quadrado + V de Cramér para variáveis categóricas.

    Retorna DataFrame com significância, força de associação e insight por variável.
    """
    results = []

    for var in cat_vars:
        contingency_table = pd.crosstab(df[var], df[target])
        chi2, p, dof, _ = chi2_contingency(contingency_table)
        v = cramers_v(contingency_table)

        significance = "Significativo" if p < alpha else "Não significativo"

        if v < 0.1:
            strength = "Muito fraco"
        elif v < 0.3:
            strength = "Fraco"
        elif v < 0.5:
            strength = "Moderado"
        else:
            strength = "Forte"

        if significance == "Significativo" and v >= 0.3:
            insight = "Boa variável preditiva"
        elif significance == "Significativo":
            insight = "Relação existe, mas fraca"
        else:
            insight = "Sem evidência de associação"

        results.append(
            {
                "Variavel": var,
                "Chi2": chi2,
                "p_value": p,
                "Cramer_V": v,
                "Significancia": significance,
                "Forca_Associacao": strength,
                "Insight": insight,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by=["Cramer_V", "p_value"], ascending=[False, True])
        .reset_index(drop=True)
    )


class AnaliseIV:
    """
    Calcula Information Value (IV) e Weight of Evidence (WOE) para variáveis preditoras.

    Parâmetros
    ----------
    df : pd.DataFrame
    target : str — nome da coluna alvo (binária: 0/1)
    nbins : int — número de bins para variáveis numéricas
    convention : str — 'event_over_nonevent' ou 'good_over_bad'
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        nbins: int = 10,
        convention: str = "event_over_nonevent",
    ):
        if convention not in ["event_over_nonevent", "good_over_bad"]:
            raise ValueError(
                "convention deve ser 'event_over_nonevent' ou 'good_over_bad'"
            )

        self.df_original = df.copy()
        self.target = target
        self.nbins = nbins
        self.convention = convention
        self.df = self._preparar_variaveis()
        self.df_tabs_iv = pd.DataFrame()

        for var in self.df.drop(columns=[self.target]).columns:
            try:
                self._criar_tabela_bivariada(var)
            except Exception as e:
                print(f"[AVISO] Variável '{var}' ignorada: {e}")

        if not self.df_tabs_iv.empty:
            self.df_tabs_iv = self.df_tabs_iv.drop_duplicates(
                subset=["Variavel", "Var_Range"], keep="last"
            )

    def _preparar_variaveis(self) -> pd.DataFrame:
        df = self.df_original.copy()
        df_num = df.select_dtypes(include=["int32", "int64", "float64"]).drop(
            columns=[self.target], errors="ignore"
        )
        for col in df_num.columns:
            try:
                df[col] = pd.qcut(df[col], q=self.nbins, duplicates="drop")
            except Exception:
                pass
        return df

    def _criar_tabela_bivariada(self, var: str):
        df_aux = self.df[[var, self.target]].copy()
        tabela = pd.crosstab(df_aux[var], df_aux[self.target])

        for classe in [0, 1]:
            if classe not in tabela.columns:
                tabela[classe] = 0

        tabela = tabela.rename(columns={0: "NonEvent", 1: "Event"}).reset_index()
        tabela["Total"] = tabela["NonEvent"] + tabela["Event"]

        total_event = tabela["Event"].sum()
        total_nonevent = tabela["NonEvent"].sum()

        eps = 1e-6
        tabela["Dist_Event"] = (
            tabela["Event"] / total_event if total_event > 0 else 0
        ).replace(0, eps)
        tabela["Dist_NonEvent"] = (
            tabela["NonEvent"] / total_nonevent if total_nonevent > 0 else 0
        ).replace(0, eps)

        if self.convention == "event_over_nonevent":
            tabela["WOE"] = np.log(tabela["Dist_Event"] / tabela["Dist_NonEvent"])
        else:
            tabela["WOE"] = np.log(tabela["Dist_NonEvent"] / tabela["Dist_Event"])

        tabela["IV"] = (tabela["Dist_Event"] - tabela["Dist_NonEvent"]) * tabela["WOE"]
        tabela["Variavel"] = var
        tabela = tabela.rename(columns={var: "Var_Range"})
        self.df_tabs_iv = pd.concat([self.df_tabs_iv, tabela], axis=0)

    def get_lista_iv(self) -> pd.DataFrame:
        """Retorna IV total por variável com classificação da força preditiva."""
        if self.df_tabs_iv.empty:
            raise ValueError("Nenhuma tabela de IV foi gerada.")

        def classificar_iv(iv):
            if iv < 0.02:
                return "Irrelevante"
            elif iv < 0.1:
                return "Fraca"
            elif iv < 0.3:
                return "Média"
            elif iv < 0.5:
                return "Forte"
            return "Muito forte (verificar possível overfitting)"

        lista = (
            self.df_tabs_iv.groupby("Variavel")["IV"]
            .sum()
            .sort_values(ascending=False)
            .round(3)
            .reset_index()
        )
        lista["Forca_Preditiva"] = lista["IV"].apply(classificar_iv)
        return lista

    def get_bivariada(self, var: str = None) -> pd.DataFrame:
        """Retorna tabela detalhada de WOE/IV. Se var=None, retorna todas."""
        if self.df_tabs_iv.empty:
            raise ValueError("Nenhuma tabela disponível.")
        if var is None:
            return self.df_tabs_iv
        return self.df_tabs_iv[self.df_tabs_iv["Variavel"] == var]
