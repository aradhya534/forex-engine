from datetime import date
from functools import lru_cache
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline

Currency = Literal["USD", "GBP", "JPY", "CHF", "AUD", "CAD", "LKR"]


class PredictRequest(BaseModel):
    currency: Currency = Field(..., description="Currency code. Rates are units of this currency per 1 EUR.")


class PredictResponse(BaseModel):
    currency: str
    as_of_date: date = Field(..., description="Last trading day in the data")
    for_trading_day: date = Field(..., description="The trading day the signal refers to")
    horizon: str
    prob_up: float = Field(..., ge=0, le=1, description="Calibrated probability that the rate is higher on for_trading_day")
    direction: Literal["up", "down"]
    strength: Literal["high", "medium", "low"] = Field(..., description="How far prob_up is from 50/50, in tiers fixed from validation data")
    historical_hit_rate: float = Field(..., description="Out-of-sample share of correct calls in this strength tier")
    historical_sample_size: int = Field(..., description="Number of test days behind historical_hit_rate")
    model_version: str
    disclaimer: str


class ModelInfoResponse(BaseModel):
    model_version: str
    data_through: date
    test_from: str
    test_accuracy: float
    test_roc_auc: float


class HealthResponse(BaseModel):
    status: str


class CurrenciesResponse(BaseModel):
    currencies: List[str]


TAGS = [
    {"name": "forex", "description": "Next trading day direction signals and model information"},
    {"name": "system", "description": "Service health"},
]

app = FastAPI(
    openapi_tags=TAGS,
    title="Forex Direction API",
    version="0.2.0",
    description="Next trading day direction signal for EUR vs selected currencies. Educational project, not financial advice.",
)
app.openapi_version = "3.0.2"   # WSO2 APIM expects OpenAPI 3.0.x


@lru_cache(maxsize=1)
def get_pipeline() -> PredictPipeline:
    return PredictPipeline()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Liveness check. Returns ok when the service is running."""
    return {"status": "ok"}


@app.get("/currencies", response_model=CurrenciesResponse, tags=["forex"])
def currencies():
    return {"currencies": list(Currency.__args__)}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["forex"])
def model_info():
    """Model version, the date the data runs to, and out-of-sample test accuracy and AUC."""
    try:
        return get_pipeline().model_info()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="model artifacts not found; run the training pipeline")


@app.post("/predict", response_model=PredictResponse, tags=["forex"])
def predict(req: PredictRequest):
    """Probability that the EUR to currency rate is higher on the next trading day, with signal strength and its historical hit rate."""
    try:
        return get_pipeline().predict(req.currency)
    except FileNotFoundError:
        logging.error("model artifacts not found")
        raise HTTPException(status_code=503, detail="model artifacts not found; run the training pipeline")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))




