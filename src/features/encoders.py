"""Custom encoders used by the experimentation pipelines."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


def _validate_column(X: pd.DataFrame, column: str) -> None:
    if column not in X.columns:
        raise KeyError(f"Column '{column}' was not found in the input DataFrame.")


def _validate_columns(X: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in X.columns]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Columns not found in the input DataFrame: {missing_str}")


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace a categorical column with its observed frequency."""

    def __init__(
        self,
        column: str,
        output_column: str | None = None,
        drop_original: bool = True,
        normalize: bool = True,
        fill_value: float = 0.0,
    ) -> None:
        self.column = column
        self.output_column = output_column or f"{column}_Frequency"
        self.drop_original = drop_original
        self.normalize = normalize
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        _validate_column(X, self.column)
        self.frequency_map_ = X[self.column].value_counts(
            normalize=self.normalize,
            dropna=False,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_column(X, self.column)
        transformed = X.copy()
        transformed[self.output_column] = (
            transformed[self.column].map(self.frequency_map_).fillna(self.fill_value)
        )
        if self.drop_original:
            transformed = transformed.drop(columns=[self.column])
        return transformed


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Replace a categorical column with a smoothed target mean learned on fit."""

    def __init__(
        self,
        column: str,
        output_column: str | None = None,
        drop_original: bool = True,
        smoothing: float = 20.0,
    ) -> None:
        self.column = column
        self.output_column = output_column or f"{column}_Target"
        self.drop_original = drop_original
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y) -> "TargetEncoder":
        _validate_column(X, self.column)
        if y is None:
            raise ValueError("TargetEncoder requires y during fit.")

        target = pd.Series(y, index=X.index, name="target")
        self.global_mean_ = float(target.mean())

        stats = (
            pd.DataFrame({self.column: X[self.column], "target": target})
            .groupby(self.column, dropna=False)["target"]
            .agg(["mean", "count"])
        )

        if self.smoothing > 0:
            weights = stats["count"] / (stats["count"] + self.smoothing)
            encoded = self.global_mean_ * (1 - weights) + stats["mean"] * weights
        else:
            encoded = stats["mean"]

        self.target_map_ = encoded
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_column(X, self.column)
        transformed = X.copy()
        transformed[self.output_column] = (
            transformed[self.column].map(self.target_map_).fillna(self.global_mean_)
        )
        if self.drop_original:
            transformed = transformed.drop(columns=[self.column])
        return transformed


class ClusterEncoder(BaseEstimator, TransformerMixin):
    """Map a categorical column into a smaller set of named clusters."""

    def __init__(
        self,
        column: str,
        cluster_map: dict[str, str],
        output_column: str | None = None,
        drop_original: bool = True,
        default_value: str = "other",
    ) -> None:
        self.column = column
        self.cluster_map = cluster_map
        self.output_column = output_column or f"{column}_Cluster"
        self.drop_original = drop_original
        self.default_value = default_value

    def fit(self, X: pd.DataFrame, y=None) -> "ClusterEncoder":
        _validate_column(X, self.column)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_column(X, self.column)
        transformed = X.copy()
        transformed[self.output_column] = (
            transformed[self.column].map(self.cluster_map).fillna(self.default_value)
        )
        if self.drop_original:
            transformed = transformed.drop(columns=[self.column])
        return transformed


class RiskBandEncoder(BaseEstimator, TransformerMixin):
    """Learn low/mid/high risk bands from a smoothed target mean by category."""

    def __init__(
        self,
        column: str,
        output_column: str | None = None,
        drop_original: bool = True,
        smoothing: float = 20.0,
        low_threshold: float = 1 / 3,
        high_threshold: float = 2 / 3,
        labels: tuple[str, str, str] = ("low_risk", "mid_risk", "high_risk"),
        unknown_label: str = "mid_risk",
    ) -> None:
        self.column = column
        self.output_column = output_column or f"{column}_Risk_Band"
        self.drop_original = drop_original
        self.smoothing = smoothing
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.labels = labels
        self.unknown_label = unknown_label

    def fit(self, X: pd.DataFrame, y) -> "RiskBandEncoder":
        _validate_column(X, self.column)
        if y is None:
            raise ValueError("RiskBandEncoder requires y during fit.")

        target = pd.Series(y, index=X.index, name="target")
        self.global_mean_ = float(target.mean())

        stats = (
            pd.DataFrame({self.column: X[self.column], "target": target})
            .groupby(self.column, dropna=False)["target"]
            .agg(["mean", "count"])
        )

        if self.smoothing > 0:
            weights = stats["count"] / (stats["count"] + self.smoothing)
            smoothed_scores = self.global_mean_ * (1 - weights) + stats["mean"] * weights
        else:
            smoothed_scores = stats["mean"]

        risk_rank = smoothed_scores.rank(method="average", pct=True)
        low_label, mid_label, high_label = self.labels

        def assign_band(rank_value: float) -> str:
            if rank_value <= self.low_threshold:
                return low_label
            if rank_value <= self.high_threshold:
                return mid_label
            return high_label

        self.smoothed_score_map_ = smoothed_scores
        self.risk_band_map_ = risk_rank.map(assign_band)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_column(X, self.column)
        transformed = X.copy()
        transformed[self.output_column] = (
            transformed[self.column]
            .map(self.risk_band_map_)
            .fillna(self.unknown_label)
        )
        if self.drop_original:
            transformed = transformed.drop(columns=[self.column])
        return transformed


class ZipRegionEncoder(BaseEstimator, TransformerMixin):
    """Map ZIP codes into macro-regions using their learned geographic centroid."""

    def __init__(
        self,
        column: str = "Zip Code",
        latitude_column: str = "Latitude",
        longitude_column: str = "Longitude",
        output_column: str = "Geo_Region",
        drop_original: bool = True,
        default_value: str = "other",
    ) -> None:
        self.column = column
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.output_column = output_column
        self.drop_original = drop_original
        self.default_value = default_value

    @staticmethod
    def _map_coordinates_to_region(latitude: float, longitude: float) -> str:
        if pd.isna(latitude) or pd.isna(longitude):
            return "other"

        # The thresholds are anchored on the actual spatial distribution of CA ZIPs
        # in the raw experiment dataset. This is more faithful than broad ZIP ranges
        # because it uses the real centroid of each ZIP observed during fit.
        if latitude >= 37.0:
            return "norcal"
        if latitude >= 35.0:
            return "central"
        return "socal"

    @staticmethod
    def _normalize_zip(zip_code):
        try:
            return int(float(zip_code))
        except (TypeError, ValueError):
            return np.nan

    def fit(self, X: pd.DataFrame, y=None) -> "ZipRegionEncoder":
        _validate_columns(X, (self.column, self.latitude_column, self.longitude_column))

        zip_frame = pd.DataFrame(
            {
                self.column: X[self.column].map(self._normalize_zip),
                self.latitude_column: pd.to_numeric(X[self.latitude_column], errors="coerce"),
                self.longitude_column: pd.to_numeric(X[self.longitude_column], errors="coerce"),
            },
            index=X.index,
        ).dropna(subset=[self.column, self.latitude_column, self.longitude_column])

        if zip_frame.empty:
            self.zip_region_map_ = {}
            return self

        zip_centroids = (
            zip_frame.groupby(self.column)[[self.latitude_column, self.longitude_column]]
            .median()
        )
        self.zip_region_map_ = {
            zip_code: self._map_coordinates_to_region(
                row[self.latitude_column],
                row[self.longitude_column],
            )
            for zip_code, row in zip_centroids.iterrows()
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_column(X, self.column)
        transformed = X.copy()
        normalized_zip = transformed[self.column].map(self._normalize_zip)
        transformed[self.output_column] = (
            normalized_zip.map(self.zip_region_map_).fillna(self.default_value)
        )
        if self.drop_original:
            transformed = transformed.drop(columns=[self.column])
        return transformed


class GeoClusterEncoder(BaseEstimator, TransformerMixin):
    """Cluster geographic coordinates and emit a categorical cluster label."""

    def __init__(
        self,
        latitude_column: str = "Latitude",
        longitude_column: str = "Longitude",
        output_column: str = "Geo_Cluster",
        drop_original: bool = True,
        min_clusters: int = 2,
        max_clusters: int = 10,
        fallback_clusters: int = 4,
        missing_label: str = "cluster_missing",
        random_state: int = 42,
    ) -> None:
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.output_column = output_column
        self.drop_original = drop_original
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.fallback_clusters = fallback_clusters
        self.missing_label = missing_label
        self.random_state = random_state

    def _prepare_coordinates(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        _validate_columns(X, (self.latitude_column, self.longitude_column))
        coords = pd.DataFrame(
            {
                self.latitude_column: pd.to_numeric(X[self.latitude_column], errors="coerce"),
                self.longitude_column: pd.to_numeric(X[self.longitude_column], errors="coerce"),
            },
            index=X.index,
        )
        valid_mask = coords.notna().all(axis=1)
        return coords, valid_mask

    @staticmethod
    def _distance_to_line(x0: float, y0: float, x1: float, y1: float, x: float, y: float) -> float:
        numerator = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
        denominator = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def fit(self, X: pd.DataFrame, y=None) -> "GeoClusterEncoder":
        coords, valid_mask = self._prepare_coordinates(X)
        valid_coords = coords.loc[valid_mask]

        self.kmeans_ = None
        self.n_clusters_ = None

        if valid_coords.empty:
            return self

        n_samples = len(valid_coords)
        max_k = min(self.max_clusters, n_samples)

        if max_k < self.min_clusters:
            chosen_k = min(max(self.fallback_clusters, 1), n_samples)
            self.kmeans_ = KMeans(
                n_clusters=chosen_k,
                random_state=self.random_state,
                n_init="auto",
            ).fit(valid_coords)
            self.n_clusters_ = chosen_k
            return self

        inertias: list[float] = []
        candidate_ks = list(range(self.min_clusters, max_k + 1))
        models: dict[int, KMeans] = {}

        for k in candidate_ks:
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init="auto",
            ).fit(valid_coords)
            models[k] = model
            inertias.append(float(model.inertia_))

        if len(candidate_ks) == 1:
            chosen_k = candidate_ks[0]
        else:
            x0, y0 = candidate_ks[0], inertias[0]
            x1, y1 = candidate_ks[-1], inertias[-1]
            distances = [
                self._distance_to_line(x0, y0, x1, y1, k, inertia)
                for k, inertia in zip(candidate_ks, inertias)
            ]
            chosen_idx = int(np.argmax(distances))
            chosen_k = candidate_ks[chosen_idx]

        self.kmeans_ = models[chosen_k]
        self.n_clusters_ = chosen_k
        self.inertias_ = dict(zip(candidate_ks, inertias))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        coords, valid_mask = self._prepare_coordinates(X)
        transformed = X.copy()
        transformed[self.output_column] = self.missing_label

        if self.kmeans_ is not None and valid_mask.any():
            labels = self.kmeans_.predict(coords.loc[valid_mask])
            transformed.loc[valid_mask, self.output_column] = [
                f"cluster_{label}" for label in labels
            ]

        if self.drop_original:
            transformed = transformed.drop(
                columns=[self.latitude_column, self.longitude_column],
                errors="ignore",
            )
        return transformed
