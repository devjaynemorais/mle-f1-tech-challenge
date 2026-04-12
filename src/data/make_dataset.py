"""Script to read raw data, clean it, and save as interim."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def process_data():
    # load config.yaml
    BASE_DIR = Path(__file__).resolve().parents[2]
    config_path = BASE_DIR / "config" / "config.yaml"


    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = BASE_DIR / config["data"]["raw_path"]
    interim_path = BASE_DIR / config["data"]["interim_path"]

    # load raw data
    print(f"Lendo base raw de: {raw_path}")
    df = pd.read_excel(raw_path)

    # cleaning data
    print("Iniciando limpeza dos dados...")
    print("Verificando dados duplicados...")
    if df.duplicated().sum() > 0:
        print("removendo dados duplicados...")
        df = df.drop_duplicates()
    else:
        print("Nenhum dado duplicado encontrado.")

    # sessão para dados faltantes (ver depois)
    print("Verificando dados faltantes...")
    if df.isnull().sum().sum() > 0:
        print(f"{df.isnull().sum().sum()} dados faltantes encontrados.")
    else:
        print("Nenhum dado faltante encontrado.")

    print("Convertendo Total Charges para numérico (preenchendo vazios com 0)...")
    df["Total Charges"] = pd.to_numeric(df["Total Charges"].replace(" ", np.nan))
    df['Total Charges'] = df['Total Charges'].fillna(0)


    print("Removendo Colunas desnecessárias...")
    drop_cols = [
        'Country',
        'State',
        'Lat Long',
        'Churn Label',
        'Churn Reason',
        'Latitude',
        'Longitude',
        'Zip Code',
        'Count'
    ]

    df.drop(columns=drop_cols, inplace=True)

    print("Verificando se há valores nulos...")
    if df.isnull().sum().sum() > 0:
        print(f"{df.isnull().sum().sum()} valores nulos encontrados.")
    else:
        print("Nenhum valor nulo encontrado.")
    
    # checando informações gerais
    print("===== INFORMAÇÕES GERAIS =====\n")
    print(f"Total de linhas: {df.shape[0]}")
    print(f"Total de colunas: {df.shape[1]}")
    print("===== TIPOS DE DADOS =====\n")
    print(df.info())

    # salvando dados limpos em interim
    print(f"Salvando dados limpos em {interim_path}...")
    df.to_csv(interim_path, index=False)
    print("make_dataset.py concluído!")

    return df

if __name__ == "__main__":
    process_data()
