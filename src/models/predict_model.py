"""Script to score new data with trained models."""
import joblib
import pandas as pd
import yaml


def predict():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Carregando modelo...")
    model = joblib.load(config["model"]["model_path"])

    print("Carregando dados de teste para inferência...")
    test_df = pd.read_csv(config["data"]["test_path"])
    X_test = test_df.drop(columns=[config["features"]["target_column"]])

    preds = model.predict(X_test)
    print("Predições (primeiras 10):", preds[:10])

    print("predict_model.py concluído!")


if __name__ == "__main__":
    predict()
