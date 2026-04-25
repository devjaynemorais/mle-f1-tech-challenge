"""Schemas Pydantic para entrada e saída da API de inferência batch."""

from typing import List

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Features de um cliente para predição de churn."""

    gender: str = Field(..., examples=["Male", "Female"])
    senior_citizen: str = Field(..., alias="Senior Citizen", examples=["Yes", "No"])
    partner: str = Field(..., examples=["Yes", "No"])
    dependents: str = Field(..., examples=["Yes", "No"])
    tenure_months: float = Field(..., alias="Tenure Months", ge=0, examples=[12])
    phone_service: str = Field(..., alias="Phone Service", examples=["Yes", "No"])
    multiple_lines: str = Field(
        ..., alias="Multiple Lines", examples=["Yes", "No", "No phone service"]
    )
    internet_service: str = Field(
        ..., alias="Internet Service", examples=["DSL", "Fiber optic", "No"]
    )
    online_security: str = Field(
        ..., alias="Online Security", examples=["Yes", "No", "No internet service"]
    )
    online_backup: str = Field(
        ..., alias="Online Backup", examples=["Yes", "No", "No internet service"]
    )
    device_protection: str = Field(
        ..., alias="Device Protection", examples=["Yes", "No", "No internet service"]
    )
    tech_support: str = Field(
        ..., alias="Tech Support", examples=["Yes", "No", "No internet service"]
    )
    streaming_tv: str = Field(
        ..., alias="Streaming TV", examples=["Yes", "No", "No internet service"]
    )
    streaming_movies: str = Field(
        ..., alias="Streaming Movies", examples=["Yes", "No", "No internet service"]
    )
    contract: str = Field(..., examples=["Month-to-month", "One year", "Two year"])
    paperless_billing: str = Field(
        ..., alias="Paperless Billing", examples=["Yes", "No"]
    )
    payment_method: str = Field(
        ...,
        alias="Payment Method",
        examples=["Electronic check", "Mailed check", "Bank transfer (automatic)"],
    )
    monthly_charges: float = Field(..., alias="Monthly Charges", ge=0, examples=[65.0])
    total_charges: float = Field(..., alias="Total Charges", ge=0, examples=[780.0])
    cltv: float = Field(..., alias="CLTV", ge=0, examples=[3500])

    model_config = {"populate_by_name": True}


class PredictionResult(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label: int = Field(..., ge=0, le=1)


class BatchPredictRequest(BaseModel):
    records: List[CustomerFeatures] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    model: str
    n_records: int
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_path: str
