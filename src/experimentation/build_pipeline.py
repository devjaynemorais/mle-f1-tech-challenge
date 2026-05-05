

from features.custom_transformers import FeatureEngineerTransformer, GeoTransformer
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from utils.exp import MLPClassifierWrapper
from xgboost import XGBClassifier


def build_preprocessor():
    ohe = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, make_column_selector(dtype_include=["object", "category"])),
            ("num", "passthrough", make_column_selector(dtype_exclude=["object", "category"])),
        ],
        remainder="drop",
    )

    return preprocessor


def get_scaler(config):
    scaler_cfg = config["preprocessing"].get("scaler", {})
    scaler_type = scaler_cfg.get("type", "none")
    scaler_params = scaler_cfg.get("params", {})

    if scaler_type == "standard":
        return StandardScaler(**scaler_params)

    if scaler_type == "minmax":
        return MinMaxScaler(**scaler_params)

    if scaler_type == "robust":
        return RobustScaler(**scaler_params)

    if scaler_type in [None, "none"]:
        return "passthrough"

    raise ValueError(f"Scaler não suportado: {scaler_type}")


def get_model(config):
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    model_params = model_cfg.get("params", {})

    if model_name == "dummy":
        return DummyClassifier(**model_params)

    if model_name == "logistic_regression":
        return LogisticRegression(**model_params)

    if model_name == "xgboost":
        return XGBClassifier(**model_params)

    if model_name == "mlp":
        return MLPClassifierWrapper(**model_params)

    raise ValueError(f"Modelo não suportado: {model_name}")


def build_pipeline(config):
    
    feat_params = config["features"].get("engineering", {})
    geo_strategy = config["features"].get("geo", {}).get("strategy", "none")

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineerTransformer(**feat_params)),
            ("geo", GeoTransformer(strategy=geo_strategy)),
            ("preprocessor", build_preprocessor(config)),
            ("scaler", get_scaler(config)),
            ("model", get_model(config)),
        ]
    )

    return pipeline