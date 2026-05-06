"""
Feature engineering transformers for heuristics experimentation.

"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Iterable

from .encoders import (
    FrequencyEncoder,
    GeoClusterEncoder,
    RiskBandEncoder,
    TargetEncoder,
    ZipRegionEncoder,
)



def _require_columns(
    X: pd.DataFrame,
    columns: Iterable[str],
    transform_name: str,
) -> None:
    missing = [column for column in columns if column not in X.columns]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"{transform_name} requires columns: {missing_str}")


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """Apply deterministic feature engineering steps before preprocessing."""

    def __init__(
        self,
        drop_churn_score: bool = False,
        add_engagement_score: bool = False,
        add_tenure_group: bool = False,
        add_tenure_log: bool = False,
        add_contract_ordinal: bool = False,
        add_family_stability: bool = False,
        add_fiber_no_support: bool = False,
        add_support_gap_count: bool = False,
        add_payment_automatic_flag: bool = False,
        add_electronic_check_flag: bool = False,
        add_paperless_echeck_flag: bool = False,
        add_price_pressure_ratio: bool = False,
        churn_score_column: str = "Churn Score",
        service_columns: tuple[str, ...] = (
            "Online Security",
            "Tech Support",
            "Online Backup",
            "Device Protection",
        ),
        engagement_positive_value: str = "Yes",
        engagement_output_column: str = "service_score",
        tenure_column: str = "Tenure Months",
        tenure_bins: tuple[float, ...] = (0, 6, 12, np.inf),
        tenure_labels: tuple[str, ...] = ("new", "mid", "loyal"),
        tenure_mapping: dict[str, int] | None = None,
        tenure_output_column: str = "Tenure_Group_Ordinal",
        tenure_log_output_column: str = "Tenure_Log",
        contract_column: str = "Contract",
        contract_mapping: dict[str, int] | None = None,
        contract_output_column: str = "Contract_Ordinal",
        partner_column: str = "Partner",
        dependents_column: str = "Dependents",
        family_output_column: str = "Family_Stability",
        internet_service_column: str = "Internet Service",
        tech_support_column: str = "Tech Support",
        fiber_value: str = "Fiber optic",
        no_support_value: str = "No",
        fiber_output_column: str = "Fiber_No_Support",
        support_gap_output_column: str = "Support_Gap_Count",
        payment_method_column: str = "Payment Method",
        automatic_payment_values: tuple[str, ...] = (
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ),
        payment_automatic_output_column: str = "Payment_Automatic_Flag",
        electronic_check_value: str = "Electronic check",
        electronic_check_output_column: str = "Electronic_Check_Flag",
        paperless_billing_column: str = "Paperless Billing",
        paperless_positive_value: str = "Yes",
        paperless_echeck_output_column: str = "Paperless_ECheck_Flag",
        monthly_charges_column: str = "Monthly Charges",
        price_pressure_output_column: str = "Price_Pressure_Ratio",
    ) -> None:
        self.drop_churn_score = drop_churn_score
        self.add_engagement_score = add_engagement_score
        self.add_tenure_group = add_tenure_group
        self.add_tenure_log = add_tenure_log
        self.add_contract_ordinal = add_contract_ordinal
        self.add_family_stability = add_family_stability
        self.add_fiber_no_support = add_fiber_no_support
        self.add_support_gap_count = add_support_gap_count
        self.add_payment_automatic_flag = add_payment_automatic_flag
        self.add_electronic_check_flag = add_electronic_check_flag
        self.add_paperless_echeck_flag = add_paperless_echeck_flag
        self.add_price_pressure_ratio = add_price_pressure_ratio
        self.churn_score_column = churn_score_column
        self.service_columns = service_columns
        self.engagement_positive_value = engagement_positive_value
        self.engagement_output_column = engagement_output_column
        self.tenure_column = tenure_column
        self.tenure_bins = tenure_bins
        self.tenure_labels = tenure_labels
        self.tenure_mapping = tenure_mapping or {"new": 2, "mid": 1, "loyal": 0}
        self.tenure_output_column = tenure_output_column
        self.tenure_log_output_column = tenure_log_output_column
        self.contract_column = contract_column
        self.contract_mapping = contract_mapping or {
            "Month-to-month": 2,
            "One year": 1,
            "Two year": 0,
        }
        self.contract_output_column = contract_output_column
        self.partner_column = partner_column
        self.dependents_column = dependents_column
        self.family_output_column = family_output_column
        self.internet_service_column = internet_service_column
        self.tech_support_column = tech_support_column
        self.fiber_value = fiber_value
        self.no_support_value = no_support_value
        self.fiber_output_column = fiber_output_column
        self.support_gap_output_column = support_gap_output_column
        self.payment_method_column = payment_method_column
        self.automatic_payment_values = automatic_payment_values
        self.payment_automatic_output_column = payment_automatic_output_column
        self.electronic_check_value = electronic_check_value
        self.electronic_check_output_column = electronic_check_output_column
        self.paperless_billing_column = paperless_billing_column
        self.paperless_positive_value = paperless_positive_value
        self.paperless_echeck_output_column = paperless_echeck_output_column
        self.monthly_charges_column = monthly_charges_column
        self.price_pressure_output_column = price_pressure_output_column

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineerTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()

        if self.drop_churn_score:
            transformed = transformed.drop(
                columns=[self.churn_score_column], errors="ignore"
            )
        elif self.churn_score_column not in transformed.columns:
            transformed[self.churn_score_column] = 0

        if self.add_engagement_score:
            _require_columns(
                transformed,
                self.service_columns,
                "add_engagement_score",
            )
            transformed[self.engagement_output_column] = sum(
                (transformed[column] == self.engagement_positive_value).astype(int)
                for column in self.service_columns
            )

        if self.add_tenure_group:
            _require_columns(transformed, [self.tenure_column], "add_tenure_group")
            tenure_group = pd.cut(
                transformed[self.tenure_column],
                bins=list(self.tenure_bins),
                labels=list(self.tenure_labels),
                include_lowest=True,
            )
            transformed[self.tenure_output_column] = tenure_group.map(
                self.tenure_mapping
            ).astype("Int64")

        if self.add_tenure_log:
            _require_columns(transformed, [self.tenure_column], "add_tenure_log")
            transformed[self.tenure_log_output_column] = np.log1p(
                transformed[self.tenure_column]
            )

        if self.add_contract_ordinal:
            _require_columns(
                transformed, [self.contract_column], "add_contract_ordinal"
            )
            transformed[self.contract_output_column] = (
                transformed[self.contract_column]
                .map(self.contract_mapping)
                .astype("Int64")
            )

        if self.add_family_stability:
            _require_columns(
                transformed,
                [self.partner_column, self.dependents_column],
                "add_family_stability",
            )
            transformed[self.family_output_column] = (
                (transformed[self.partner_column] == "Yes")
                | (transformed[self.dependents_column] == "Yes")
            ).astype(int)

        if self.add_fiber_no_support:
            _require_columns(
                transformed,
                [self.internet_service_column, self.tech_support_column],
                "add_fiber_no_support",
            )
            transformed[self.fiber_output_column] = (
                (transformed[self.internet_service_column] == self.fiber_value)
                & (transformed[self.tech_support_column] == self.no_support_value)
            ).astype(int)

        if self.add_support_gap_count:
            _require_columns(
                transformed,
                self.service_columns,
                "add_support_gap_count",
            )
            transformed[self.support_gap_output_column] = sum(
                (transformed[column] != self.engagement_positive_value).astype(int)
                for column in self.service_columns
            )

        if self.add_payment_automatic_flag:
            _require_columns(
                transformed,
                [self.payment_method_column],
                "add_payment_automatic_flag",
            )
            transformed[self.payment_automatic_output_column] = (
                transformed[self.payment_method_column].isin(
                    self.automatic_payment_values
                )
            ).astype(int)

        if self.add_electronic_check_flag:
            _require_columns(
                transformed,
                [self.payment_method_column],
                "add_electronic_check_flag",
            )
            transformed[self.electronic_check_output_column] = (
                transformed[self.payment_method_column] == self.electronic_check_value
            ).astype(int)

        if self.add_paperless_echeck_flag:
            _require_columns(
                transformed,
                [self.paperless_billing_column, self.payment_method_column],
                "add_paperless_echeck_flag",
            )
            transformed[self.paperless_echeck_output_column] = (
                (
                    transformed[self.paperless_billing_column]
                    == self.paperless_positive_value
                )
                & (
                    transformed[self.payment_method_column]
                    == self.electronic_check_value
                )
            ).astype(int)

        if self.add_price_pressure_ratio:
            _require_columns(
                transformed,
                [self.monthly_charges_column, self.tenure_column],
                "add_price_pressure_ratio",
            )
            transformed[self.price_pressure_output_column] = transformed[
                self.monthly_charges_column
            ] / (transformed[self.tenure_column] + 1)

        return transformed




class GeoTransformer(BaseEstimator, TransformerMixin):
    """
    Geographic feature transformer used during experimentation.
    
    
    Apply a single geographic strategy inside an sklearn pipeline.
    
    """

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
