from fastapi import logger
import pandas as pd
import numpy as np


def remove_duplicates(df):
    initial_shape = df.shape[0]

    duplicated_mask = df.duplicated()
    n_duplicates = duplicated_mask.sum()

    if n_duplicates > 0:
        logger.info(f"Removendo {n_duplicates} linhas duplicadas...")
        df = df[~duplicated_mask]
    else:
        logger.info("Nenhum dado duplicado encontrado.")

    final_shape = df.shape[0]

    logger.info(f"Linhas antes: {initial_shape} | depois: {final_shape}")

    return df


def handle_total_charges(df):
    df = df.copy()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"].replace(" ", np.nan))
    df["Total Charges"] = df["Total Charges"].fillna(0)
    return df


def drop_unused_columns(df, cols):
    return df.drop(columns=cols)