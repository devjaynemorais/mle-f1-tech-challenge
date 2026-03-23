"""Script to read raw data, clean it, and save as interim."""
import numpy as np
import pandas as pd
import yaml


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Lendo base raw de: {config['data']['raw_path']}")
    df = pd.read_csv(config["data"]["raw_path"])

    # Limpeza
    print("Convertendo TotalCharges para numérico (preenchendo vazios com NaN)...")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan))

    print("Removendo nulos (11 linhas) e a coluna customerID...")
    df = df.dropna(subset=["TotalCharges"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    interim_path = config["data"]["interim_path"]
    print(f"Salvando dados limpos em {interim_path}...")
    df.to_csv(interim_path, index=False)
    print("make_dataset.py concluído!")


if __name__ == "__main__":
    main()
