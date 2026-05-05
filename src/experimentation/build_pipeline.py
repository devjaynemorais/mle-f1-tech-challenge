from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from xgboost import XGBClassifier

from src.features.custom_transformers import FeatureEngineerTransformer, GeoTransformer
from src.utils.exp import MLPClassifierWrapper


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

    raise ValueError(f"Scaler nao suportado: {scaler_type}")


def get_model(config, model_name=None, model_params=None):
    model_cfg = config["model"]
    model_name = model_name or model_cfg["name"]
    merged_model_params = {
        **model_cfg.get("params", {}),
        **(model_params or {}),
    }
    merged_model_params.pop("selector_k", None)

    if model_name == "dummy":
        return DummyClassifier(**merged_model_params)

    if model_name == "logistic_regression":
        return LogisticRegression(**merged_model_params)

    if model_name == "xgboost":
        return XGBClassifier(**merged_model_params)

    if model_name == "mlp":
        return MLPClassifierWrapper(**merged_model_params)

    raise ValueError(f"Modelo nao suportado: {model_name}")


def get_selector(config, model_name=None, model_params=None):
    model_cfg = config["model"]
    resolved_model_name = model_name or model_cfg["name"]
    merged_model_params = {
        **model_cfg.get("params", {}),
        **(model_params or {}),
    }
    selector_k = merged_model_params.get("selector_k")

    if resolved_model_name == "mlp" and selector_k is not None:
        return SelectKBest(score_func=f_classif, k=int(selector_k))

    return "passthrough"


def build_pipeline(config, model_name=None, model_params=None, y_reference=None):
    del y_reference
    feat_params = config["features"].get("engineering", {})
    geo_strategy = config["features"].get("geo", {}).get("strategy", "none")

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineerTransformer(**feat_params)),
            ("geo", GeoTransformer(strategy=geo_strategy)),
            ("preprocessor", build_preprocessor()),
            ("selector", get_selector(config, model_name=model_name, model_params=model_params)),
            ("scaler", get_scaler(config)),
            ("model", get_model(config, model_name=model_name, model_params=model_params)),
        ]
    )

    return pipeline
