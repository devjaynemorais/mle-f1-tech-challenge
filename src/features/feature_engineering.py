"""
    Esse Módulo contém todas a engenharia de feature encapsuladas para
    a etapa de experimentação.

    Cada função é responsável por criar uma nova feature, e pode ser facilmente
    adicionada ou removida do pipeline de experimentação a partir de um arquivo
    de configuração yaml.
"""


import pandas as pd
import numpy as np



# 1. CLTV
def drop_cltv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a coluna CLTV do dataset.
    """
    return df.drop(columns=["CLTV"], errors="ignore")


# 2. Churn Score
def drop_churn_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a coluna de churn score do dataset.
    """
    return df.drop(columns=["Churn Score"], errors="ignore")


# 3. City
def drop_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a coluna de cidade do dataset.
    """
    return df.drop(columns=["City"], errors="ignore")


# Cliente valioso alto churn risk - Flag de clientes com alto CLTV e alto risco de churn
def add_valuable_high_risk(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Cria flag de clientes valiosos com alto risco de churn.
    """

    df = df.copy()

    cltv_col = config.get("cltv_col", "CLTV")
    churn_score_col = config.get("churn_score_col", "Churn Score")

    cltv_threshold = config.get("cltv_threshold", 5000)
    churn_score_threshold = config.get("churn_score_threshold", 75)

    df["Valuable_HighRisk"] = (
        (df[cltv_col] > cltv_threshold) & (df[churn_score_col] > churn_score_threshold)
    ).astype(int)

    return df


# 4. Engajamento de Serviços
def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria score de engajamento baseado em serviços ativos.
    """

    df = df.copy()

    services = [
        'Online Security',
        'Tech Support',
        'Online Backup',
        'Device Protection'
    ]

    df['service_score'] = sum((df[s] == 'Yes').astype(int) for s in services)

    return df


# 6. Tenure Group
def add_tenure_group(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Cria grupos de tempo de contrato.
    """

    df = df.copy()

    tenure_col = config.get("tenure_col", "Tenure Months")

    bins = config.get("bins", [0, 6, 12, np.inf])
    labels = config.get("labels", ["new", "mid", "loyal"])

    df["Tenure_Group"] = pd.cut(
        df[tenure_col],
        bins=bins,
        labels=labels
    )

    return df


# tranformação logarítmica para Tenure
def add_tenure_log(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """ Cria uma nova feature de log do tempo de contrato para lidar com skewness. """

    df = df.copy()

    df['Tenure_Log'] = np.log1p(df['Tenure Months'])

    return df


# 7. Contract Ordinal Encoding
def add_contract_ordinal(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Codificação ordinal baseada em risco de churn (monotônica).
    """

    df = df.copy()

    contract_col = config.get("contract_col", "Contract")

    mapping = config.get("mapping", {
        "Month-to-month": 2,  # maior risco
        "One year": 1,
        "Two year": 0         # menor risco
    })

    df["Contract_Ordinal"] = df[contract_col].map(mapping)

    return df


# 8. Family Stability Flag
def add_family_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria flag de estabilidade familiar.
    """

    df = df.copy()

    df["Family_Stability"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    return df


# Fiber Optic No Support - Flag de clientes com fibra ótica sem suporte técnico
def add_fiber_no_support(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria flag de clientes com fibra ótica sem suporte técnico.
    """

    df = df.copy()

    df["Fiber_No_Support"] = (
        (df["Internet Service"] == "Fiber optic") & (df["Tech Support"] == "No")
    ).astype(int)

    return df


# Encoding Strategy City - Mapping de região por agrupamento de condados
def add_city_region_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica mapping de região baseado no nome da cidade.
    """

    df = df.copy()

    NORCAL_COUNTIES = {
        "Alameda","Contra Costa","Marin","Napa","San Francisco","San Mateo",
        "Santa Clara","Solano","Sonoma","Sacramento","Yolo","Placer","El Dorado",
        "Nevada","Yuba","Sutter","Butte","Shasta","Humboldt","Mendocino","Lake",
        "Del Norte","Trinity","Tehama","Siskiyou","Modoc","Lassen","Plumas","Sierra","Alpine","Amador","Calaveras","Tuolumne"
    }

    SOCAL_COUNTIES = {
        "Los Angeles","Orange","San Diego","Ventura"
    }

    CENTRAL_COUNTIES = {
        "Fresno","Kern","Kings","Tulare","Madera",
        "San Joaquin","Stanislaus","Merced",
        "Monterey","San Luis Obispo","Santa Barbara","Santa Cruz"
    }

    DESERT_COUNTIES = {
        "Riverside","San Bernardino","Imperial","Inyo"
    }


    def classify_region_from_county(county):
        if county in NORCAL_COUNTIES:
            return "norcal"
        elif county in SOCAL_COUNTIES:
            return "socal"
        elif county in CENTRAL_COUNTIES:
            return "central"
        elif county in DESERT_COUNTIES:
            return "socal"  # desert incluído como socal
        else:
            return "other"
        
    df['City_Enc'] = df['City'].apply(classify_region_from_county)

    return df

