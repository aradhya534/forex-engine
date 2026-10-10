from locale import currency
from datetime import date
from typing import List, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

Currency = Literal['USD', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'LKR']
DISCLAIMER = "Educational Project. Not financial advice."

class PredictRequest(BaseModel):
    currency: Currency = Field(..., description="Currency code. Rates are units of this currency per 1 EUR")

class PredictResponse(BaseModel):
    currency: str
    as_of_date: date
    prob_up: float = Field(..., ge=0, le=1, description="Estimated probability the rate rises next trading day")
    signal: Literal["up", "down", "neutral"]
    model_version: str
    disclaimer: str

class HealthResponse(BaseModel):
    status: str

class CurrenciesResponse(BaseModel):
    currencies: List[str]

app = FastAPI(
    title="Forex Direction API",
    version="0.1.0",
    description="Next trading day direction probability for EUR vs selected currencies",
)

app.openapi_version = "3.0.2"

def predict_prob_up(currency:str)->float:
    return 0.5

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return {"status":"ok"}

@app.get("/currencies", response_model=CurrenciesResponse, tags=["forex"])

def currencies():
    return {"currencies": list(Currency.__args__)}

@app.post("/predict", response_model=PredictResponse, tags=["forex"])

def predict(req: PredictRequest):
    p = predict_prob_up(req.currency)
    signal = "up" if p >= 0.55 else "down" if p <= 0.45 else "neutral"
    return PredictResponse(currency=req.currency, as_of_date=date.today(), prob_up=p,
    signal=signal, model_version="stub-0", disclaimer=DISCLAIMER)

