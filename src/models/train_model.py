"""Script to train machine learning models."""
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Carregando base de treino...")
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])

    target = config["features"]["target_column"]
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    print("Treitando modelo Baseline (Random Forest)...")
    model = RandomForestClassifier(random_state=config["model"]["random_state"])

    # MLflow Tracking
    mlflow.set_experiment("Churn_Prediction_Baseline")
    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"Acurácia do Modelo: {acc:.4f}")

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "rf_baseline")

    print(f"Salvando o modelo final em {config['model']['model_path']}...")
    joblib.dump(model, config["model"]["model_path"])
    print("train_model.py concluído!")


if __name__ == "__main__":
    train()
