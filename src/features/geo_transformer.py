"""Geographic feature transformer used during experimentation."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .encoders import (
    FrequencyEncoder,
    GeoClusterEncoder,
    RiskBandEncoder,
    TargetEncoder,
    ZipRegionEncoder,
)


class GeoTransformer(BaseEstimator, TransformerMixin):
    """Apply a single geographic strategy inside an sklearn pipeline."""

    VALID_STRATEGIES = {
        "drop",
        "frequency",
        "target",
        "risk_band",
        "zip_region",
        "geo_cluster",
    }

    def __init__(
        self,
        strategy: str = "drop",
        city_column: str = "City",
        zip_code_column: str = "Zip Code",
        latitude_column: str = "Latitude",
        longitude_column: str = "Longitude",
        lat_long_column: str = "Lat Long",
        frequency_output_column: str = "City_Frequency",
        target_output_column: str = "City_Target",
        target_smoothing: float = 20.0,
        risk_band_output_column: str = "City_Risk_Band",
        risk_band_smoothing: float = 20.0,
        risk_band_labels: tuple[str, str, str] = ("low_risk", "mid_risk", "high_risk"),
        risk_band_unknown_label: str = "mid_risk",
        zip_region_output_column: str = "Geo_Region",
        geo_cluster_output_column: str = "Geo_Cluster",
        geo_cluster_min_clusters: int = 2,
        geo_cluster_max_clusters: int = 10,
        geo_cluster_fallback_clusters: int = 4,
        geo_cluster_missing_label: str = "cluster_missing",
        random_state: int = 42,
    ) -> None:
        self.strategy = strategy
        self.city_column = city_column
        self.zip_code_column = zip_code_column
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.lat_long_column = lat_long_column
        self.frequency_output_column = frequency_output_column
        self.target_output_column = target_output_column
        self.target_smoothing = target_smoothing
        self.risk_band_output_column = risk_band_output_column
        self.risk_band_smoothing = risk_band_smoothing
        self.risk_band_labels = risk_band_labels
        self.risk_band_unknown_label = risk_band_unknown_label
        self.zip_region_output_column = zip_region_output_column
        self.geo_cluster_output_column = geo_cluster_output_column
        self.geo_cluster_min_clusters = geo_cluster_min_clusters
        self.geo_cluster_max_clusters = geo_cluster_max_clusters
        self.geo_cluster_fallback_clusters = geo_cluster_fallback_clusters
        self.geo_cluster_missing_label = geo_cluster_missing_label
        self.random_state = random_state

    @property
    def geo_columns_(self) -> list[str]:
        return [
            self.city_column,
            self.zip_code_column,
            self.latitude_column,
            self.longitude_column,
            self.lat_long_column,
        ]

    def _drop_geo_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy().drop(columns=self.geo_columns_, errors="ignore")

    def fit(self, X: pd.DataFrame, y=None) -> "GeoTransformer":
        if self.strategy not in self.VALID_STRATEGIES:
            options = ", ".join(sorted(self.VALID_STRATEGIES))
            raise ValueError(f"Invalid strategy '{self.strategy}'. Expected one of: {options}.")

        if self.strategy == "drop":
            self.encoder_ = None
            return self

        if self.strategy == "frequency":
            self.encoder_ = FrequencyEncoder(
                column=self.city_column,
                output_column=self.frequency_output_column,
                drop_original=True,
            ).fit(X)
            return self

        if self.strategy == "target":
            self.encoder_ = TargetEncoder(
                column=self.city_column,
                output_column=self.target_output_column,
                drop_original=True,
                smoothing=self.target_smoothing,
            ).fit(X, y)
            return self

        if self.strategy == "risk_band":
            self.encoder_ = RiskBandEncoder(
                column=self.city_column,
                output_column=self.risk_band_output_column,
                drop_original=True,
                smoothing=self.risk_band_smoothing,
                labels=self.risk_band_labels,
                unknown_label=self.risk_band_unknown_label,
            ).fit(X, y)
            return self

        if self.strategy == "zip_region":
            self.encoder_ = ZipRegionEncoder(
                column=self.zip_code_column,
                latitude_column=self.latitude_column,
                longitude_column=self.longitude_column,
                output_column=self.zip_region_output_column,
                drop_original=True,
            ).fit(X)
            return self

        self.encoder_ = GeoClusterEncoder(
            latitude_column=self.latitude_column,
            longitude_column=self.longitude_column,
            output_column=self.geo_cluster_output_column,
            drop_original=True,
            min_clusters=self.geo_cluster_min_clusters,
            max_clusters=self.geo_cluster_max_clusters,
            fallback_clusters=self.geo_cluster_fallback_clusters,
            missing_label=self.geo_cluster_missing_label,
            random_state=self.random_state,
        ).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.strategy == "drop":
            return self._drop_geo_columns(X)

        transformed = self.encoder_.transform(X)
        return transformed.drop(columns=self.geo_columns_, errors="ignore")
