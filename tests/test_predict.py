import json
from datetime import date

from fastapi.testclient import TestClient

from src.api.main import app, Currency
from src.pipeline.predict_pipeline import PredictPipeline, next_trading_day


def test_next_trading_day_skips_weekend():
    assert next_trading_day(date(2026, 10, 9)) == date(2026, 10, 12)    # Friday -> Monday
    assert next_trading_day(date(2026, 10, 7)) == date(2026, 10, 8)     # Wednesday -> Thursday


def test_pipeline_returns_valid_signal_for_every_currency():
    pipe = PredictPipeline()
    for c in pipe.currencies:
        out = pipe.predict(c)
        assert 0 < out["prob_up"] < 1
        assert out["direction"] == ("up" if out["prob_up"] >= 0.5 else "down")
        assert out["strength"] in {"high", "medium", "low"}
        assert out["historical_sample_size"] > 0


def test_api_currencies_match_the_trained_model():
    pipe = PredictPipeline()
    assert set(Currency.__args__) == set(pipe.currencies)


def test_api_predict_and_validation():
    client = TestClient(app)
    ok = client.post("/predict", json={"currency": "LKR"})
    assert ok.status_code == 200 and ok.json()["currency"] == "LKR"
    assert client.post("/predict", json={"currency": "XXX"}).status_code == 422
    assert client.get("/model-info").status_code == 200


def test_openapi_is_3_0_without_null_types():
    spec = TestClient(app).get("/openapi.json").json()
    assert spec["openapi"].startswith("3.0")
    assert '"null"' not in json.dumps(spec)