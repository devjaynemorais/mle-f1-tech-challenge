"""
    Esse Módulo atua com um plug&play das funções do módulo de engenharia de features,
    aplicando as transformações selecionadas no arquivo de configuração .yaml, durante a etapa
    de experimentação.
"""

from .feature_engineering import *


def apply_feature_engineering(df, config):

    if config["features"]["drop_churn_score"]["enabled"]:
        df = drop_churn_score(df)
    
    if config["features"]["drop_city"]["enabled"]:
        df = drop_city(df)

    if config["features"]["engagement_score"]["enabled"]:
        df = add_engagement_score(df)
    
    if config["features"]["tenure_log"]["enabled"]:
        df = add_tenure_log(df)

    if config["features"]["tenure_group"]["enabled"]:
        df = add_tenure_group(df, config["features"]["tenure_group"])

    if config["features"]["contract_ordinal"]["enabled"]:
        df = add_contract_ordinal(df, config["features"]["contract_ordinal"])

    if config["features"]["family_stability"]["enabled"]:
        df = add_family_stability(df)

    if config["features"]["fiber_no_support"]["enabled"]:
        df = add_fiber_no_support(df)
    
    if config["features"]["city_region_mapping"]["enabled"]:
        df = add_city_region_mapping(df)

    return df