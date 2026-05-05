import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate

from src.experimentation.config_loader import load_config
from src.experimentation.prep_data import prep_data
from src.experimentation.build_pipeline import build_pipeline


def build_cv(config):
    cv_cfg = config["cv"]

    cv_type = cv_cfg.get("type", "stratified_kfold")
    n_splits = cv_cfg.get("n_splits", 5)
    shuffle = cv_cfg.get("shuffle", True)
    random_state = cv_cfg.get("random_state", 42)

    if cv_type == "stratified_kfold":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    if cv_type == "kfold":
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    raise ValueError(f"CV não suportado: {cv_type}")


def run_experiment(config):
    data = prep_data(config)

    X_train = data["X_train"]
    y_train = data["y_train"]

    pipeline = build_pipeline(config)
    cv = build_cv(config)

    scoring = config["cv"]["scoring"]
    metrics = list(scoring.keys())

    model_name = config["model"]["name"]

    print(f"\n=== AVALIANDO MODELO: {model_name} ===")

    cv_res = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=config["cv"].get("n_jobs", 1),
        return_train_score=False,
    )

    summary = {
        "model": model_name,
    }

    for metric in metrics:
        scores = cv_res[f"test_{metric}"]

        summary[f"{metric}_mean"] = scores.mean()
        summary[f"{metric}_std"] = scores.std()

    summary_df = pd.DataFrame([summary])

    print("\n=== RESULTADO DA VALIDAÇÃO CRUZADA ===")
    print(summary_df.to_string(index=False))

    return summary_df


if __name__ == "__main__":
    run_experiment()