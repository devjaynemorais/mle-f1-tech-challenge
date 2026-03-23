"""Script to turn interim data into processed features for modeling."""
import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def execute_features():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Lendo a base interim...")
    df = pd.read_csv(config["data"]["interim_path"])

    # Mapeando target binário
    target = config["features"]["target_column"]
    df[target] = df[target].map({"Yes": 1, "No": 0})

    cat_cols = config["features"]["categorical_columns"]
    num_cols = config["features"]["numerical_columns"]

    print("Aplicando One-Hot Encoding nas categóricas...")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print("Separando features (X) e target (y)...")
    X = df.drop(columns=[target])
    y = df[target]

    print("Executando split de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    print("Escalonando variáveis numéricas...")
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    print("Salvando Scaler (artefato)...")
    joblib.dump(scaler, config["model"]["scaler_path"])

    print("Salvando bases processadas prontas para a Rede Neural...")
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(config["data"]["train_path"], index=False)
    test_df.to_csv(config["data"]["test_path"], index=False)
    print("build_features.py concluído!")


if __name__ == "__main__":
    execute_features()
