import pandas as pd
import pytest

from src.features import (
    ClusterEncoder,
    FeatureEngineerTransformer,
    FrequencyEncoder,
    GeoClusterEncoder,
    GeoTransformer,
    RiskBandEncoder,
    TargetEncoder,
    ZipRegionEncoder,
)


def make_sample_df():
    return pd.DataFrame(
        {
            "City": ["A", "A", "B", "C"],
            "Zip Code": [90003, 94105, 93721, 99999],
            "Latitude": [33.97, 37.79, 36.74, None],
            "Longitude": [-118.27, -122.39, -119.77, None],
            "Lat Long": [
                "33.97, -118.27",
                "37.79, -122.39",
                "36.74, -119.77",
                None,
            ],
            "Churn Score": [70, 60, 80, 50],
            "Tenure Months": [1, 8, 15, 30],
            "Contract": ["Month-to-month", "One year", "Two year", "One year"],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No"],
            "Internet Service": ["Fiber optic", "DSL", "Fiber optic", "Fiber optic"],
            "Tech Support": ["No", "Yes", "No", "Yes"],
            "Online Security": ["Yes", "No", "Yes", "No"],
            "Online Backup": ["Yes", "Yes", "No", "No"],
            "Device Protection": ["No", "Yes", "Yes", "No"],
            "Payment Method": [
                "Electronic check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Mailed check",
            ],
            "Paperless Billing": ["Yes", "No", "Yes", "Yes"],
            "Monthly Charges": [50.0, 100.0, 80.0, 20.0],
        }
    )


def test_frequency_encoder_replaces_city_with_frequency():
    df = make_sample_df()

    transformed = FrequencyEncoder(column="City").fit_transform(df)

    assert "City" not in transformed.columns
    assert "City_Frequency" in transformed.columns
    assert transformed["City_Frequency"].tolist() == [0.5, 0.5, 0.25, 0.25]


def test_target_encoder_uses_smoothing_and_global_mean_for_unknown_city():
    df = make_sample_df()
    y = pd.Series([1, 0, 1, 0])

    encoder = TargetEncoder(column="City").fit(df, y)
    transformed = encoder.transform(pd.DataFrame({"City": ["B", "Z"]}))

    assert "City" not in transformed.columns
    assert transformed["City_Target"].iloc[0] == pytest.approx(
        (0.5 * 20 + 1.0 * 1) / 21
    )
    assert transformed["City_Target"].iloc[1] == pytest.approx(0.5)


def test_target_encoder_requires_target_on_fit():
    df = make_sample_df()

    with pytest.raises(ValueError):
        TargetEncoder(column="City").fit(df, None)


def test_cluster_encoder_maps_city_to_named_clusters():
    df = make_sample_df()

    transformed = ClusterEncoder(
        column="City",
        cluster_map={"A": "north", "B": "south"},
        default_value="other",
    ).fit_transform(df)

    assert "City" not in transformed.columns
    assert transformed["City_Cluster"].tolist() == ["north", "north", "south", "other"]


def test_risk_band_encoder_creates_risk_labels_and_handles_unknown():
    df = make_sample_df()
    y = pd.Series([1, 1, 0, 0])

    encoder = RiskBandEncoder(column="City").fit(df, y)
    transformed = encoder.transform(pd.DataFrame({"City": ["A", "B", "Z"]}))

    assert "City" not in transformed.columns
    assert set(transformed["City_Risk_Band"]).issubset(
        {"low_risk", "mid_risk", "high_risk"}
    )
    assert transformed["City_Risk_Band"].iloc[2] == "mid_risk"


def test_zip_region_encoder_maps_zip_centroids_to_regions():
    df = make_sample_df()

    encoder = ZipRegionEncoder().fit(df)
    transformed = encoder.transform(df)
    unseen = encoder.transform(pd.DataFrame({"Zip Code": [12345]}))

    assert "Zip Code" not in transformed.columns
    assert transformed["Geo_Region"].tolist() == ["socal", "norcal", "central", "other"]
    assert unseen["Geo_Region"].iloc[0] == "other"


def test_geo_cluster_encoder_creates_cluster_labels_and_missing_bucket():
    df = make_sample_df()

    encoder = GeoClusterEncoder(max_clusters=3).fit(df)
    transformed = encoder.transform(df)

    assert "Latitude" not in transformed.columns
    assert "Longitude" not in transformed.columns
    assert "Geo_Cluster" in transformed.columns
    assert transformed["Geo_Cluster"].iloc[3] == "cluster_missing"
    assert encoder.n_clusters_ in {2, 3}


def test_feature_engineer_transformer_applies_expected_features():
    df = make_sample_df()

    transformer = FeatureEngineerTransformer(
        drop_churn_score=True,
        add_engagement_score=True,
        add_tenure_group=True,
        add_tenure_log=True,
        add_contract_ordinal=True,
        add_family_stability=True,
        add_fiber_no_support=True,
        add_support_gap_count=True,
        add_payment_automatic_flag=True,
        add_electronic_check_flag=True,
        add_paperless_echeck_flag=True,
        add_price_pressure_ratio=True,
    )

    transformed = transformer.fit_transform(df)

    assert "Churn Score" not in transformed.columns
    assert "City" in transformed.columns
    assert transformed["service_score"].tolist() == [2, 3, 2, 1]
    assert transformed["Tenure_Group_Ordinal"].tolist() == [2, 1, 0, 0]
    assert transformed["Contract_Ordinal"].tolist() == [2, 1, 0, 1]
    assert transformed["Family_Stability"].tolist() == [1, 0, 1, 1]
    assert transformed["Fiber_No_Support"].tolist() == [1, 0, 1, 0]
    assert transformed["Support_Gap_Count"].tolist() == [2, 1, 2, 3]
    assert transformed["Payment_Automatic_Flag"].tolist() == [0, 1, 1, 0]
    assert transformed["Electronic_Check_Flag"].tolist() == [1, 0, 0, 0]
    assert transformed["Paperless_ECheck_Flag"].tolist() == [1, 0, 0, 0]
    assert transformed["Price_Pressure_Ratio"].tolist() == [
        25.0,
        100.0 / 9,
        5.0,
        20.0 / 31,
    ]


def test_geo_transformer_drop_strategy_removes_all_geo_columns():
    df = make_sample_df()

    transformed = GeoTransformer(strategy="drop").fit_transform(df)

    for column in ["City", "Zip Code", "Latitude", "Longitude", "Lat Long"]:
        assert column not in transformed.columns


def test_geo_transformer_frequency_strategy_encodes_city_and_removes_geo_columns():
    df = make_sample_df()

    transformed = GeoTransformer(strategy="frequency").fit_transform(df)

    assert transformed["City_Frequency"].tolist() == [0.5, 0.5, 0.25, 0.25]
    for column in ["City", "Zip Code", "Latitude", "Longitude", "Lat Long"]:
        assert column not in transformed.columns


def test_geo_transformer_target_strategy_encodes_city_with_smoothing():
    df = make_sample_df()
    y = pd.Series([1, 0, 1, 0])

    transformed = GeoTransformer(strategy="target").fit_transform(df, y)

    assert "City" not in transformed.columns
    assert transformed["City_Target"].iloc[0] == pytest.approx(
        (0.5 * 20 + 0.5 * 2) / 22
    )
    assert transformed["City_Target"].iloc[2] == pytest.approx(
        (0.5 * 20 + 1.0 * 1) / 21
    )


def test_geo_transformer_risk_band_strategy_creates_risk_band():
    df = make_sample_df()
    y = pd.Series([1, 1, 0, 0])

    transformed = GeoTransformer(strategy="risk_band").fit_transform(df, y)

    assert "City_Risk_Band" in transformed.columns
    assert set(transformed["City_Risk_Band"]).issubset(
        {"low_risk", "mid_risk", "high_risk"}
    )


def test_geo_transformer_zip_region_strategy_creates_geo_region():
    df = make_sample_df()

    transformed = GeoTransformer(strategy="zip_region").fit_transform(df)

    assert transformed["Geo_Region"].tolist() == ["socal", "norcal", "central", "other"]
    for column in ["City", "Zip Code", "Latitude", "Longitude", "Lat Long"]:
        assert column not in transformed.columns


def test_geo_transformer_geo_cluster_strategy_creates_geo_cluster():
    df = make_sample_df()

    transformed = GeoTransformer(
        strategy="geo_cluster", geo_cluster_max_clusters=3
    ).fit_transform(df)

    assert "Geo_Cluster" in transformed.columns
    assert transformed["Geo_Cluster"].iloc[3] == "cluster_missing"
    for column in ["City", "Zip Code", "Latitude", "Longitude", "Lat Long"]:
        assert column not in transformed.columns
