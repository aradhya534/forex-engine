from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.gateway_client import GatewayClient, GatewayError

SIGNAL = {"currency": "LKR", "as_of_date": "2026-10-07", "for_trading_day": "2026-10-08", "horizon": "next trading day",
          "prob_up": 0.4731, "direction": "down", "strength": "medium", "historical_hit_rate": 0.5113,
          "historical_sample_size": 1723, "model_version": "lgbm-1", "disclaimer": "Educational project. Not financial advice."}
INFO = {"model_version": "lgbm-1", "data_through": "2026-10-07", "test_from": "2022-05-23", "test_accuracy": 0.5079, "test_roc_auc": 0.5147}


def _resp(status=200, body=None):
    r = MagicMock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.json.return_value = body if body is not None else {}
    return r


def _client():
    return GatewayClient(base_url="https://gw/forex/1.0.0", token_url="https://gw/token", key="k", secret="s", verify_tls=True)


def test_missing_credentials_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("WSO2_CONSUMER_KEY", raising=False)      # a real .env on your machine must not affect this test
    monkeypatch.delenv("WSO2_CONSUMER_SECRET", raising=False)
    with pytest.raises(GatewayError, match="not set"):
        GatewayClient(token_url="https://gw/token")._get_token()


def test_credentials_can_come_from_a_dotenv_file(tmp_path, monkeypatch):
    from dotenv import load_dotenv
    monkeypatch.delenv("WSO2_CONSUMER_KEY", raising=False)
    monkeypatch.delenv("WSO2_CONSUMER_SECRET", raising=False)
    env = tmp_path / ".env"
    env.write_text("WSO2_CONSUMER_KEY=abc123\nWSO2_CONSUMER_SECRET=s3cret\nVERIFY_TLS=false\n")
    load_dotenv(env, override=True)
    c = GatewayClient()
    assert (c.key, c.secret, c.verify_tls) == ("abc123", "s3cret", False)


def test_token_is_cached_between_calls():
    with patch("requests.post", return_value=_resp(200, {"access_token": "tok", "expires_in": 3600})) as post, \
         patch("requests.request", return_value=_resp(200, SIGNAL)):
        c = _client()
        c.predict("LKR"); c.predict("LKR")
    assert post.call_count == 1


def test_expired_token_is_refreshed_once_on_401():
    with patch("requests.post", return_value=_resp(200, {"access_token": "tok", "expires_in": 3600})) as post, \
         patch("requests.request", side_effect=[_resp(401), _resp(200, SIGNAL)]):
        out = _client().predict("LKR")
    assert out["currency"] == "LKR" and post.call_count == 2


@pytest.mark.parametrize("status,text", [(429, "Rate limit"), (503, "not available"), (404, "no such resource")])
def test_error_statuses_become_friendly_messages(status, text):
    with patch("requests.post", return_value=_resp(200, {"access_token": "tok", "expires_in": 3600})), \
         patch("requests.request", return_value=_resp(status)):
        with pytest.raises(GatewayError, match=text):
            _client().predict("LKR")


def test_unreachable_gateway_is_reported():
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(GatewayError, match="Cannot reach"):
            _client().predict("LKR")


def _run_app():
    st.cache_data.clear(); st.cache_resource.clear()
    return AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app" / "main.py"), default_timeout=20)


def test_app_shows_the_signal():
    with patch.object(GatewayClient, "currencies", return_value={"currencies": ["USD", "LKR"]}), \
         patch.object(GatewayClient, "model_info", return_value=INFO), \
         patch.object(GatewayClient, "predict", return_value=SIGNAL):
        at = _run_app().run()
        assert not at.exception
        at.button[0].click().run()
        assert not at.exception
        labels = [m.label for m in at.metric]
        assert labels == ["Direction", "Probability up", "Strength"]
        assert "47.3%" in at.metric[1].value


def test_app_shows_a_friendly_message_when_throttled():
    with patch.object(GatewayClient, "currencies", return_value={"currencies": ["USD", "LKR"]}), \
         patch.object(GatewayClient, "model_info", return_value=INFO), \
         patch.object(GatewayClient, "predict", side_effect=GatewayError(429, "Rate limit reached for this subscription plan.")):
        at = _run_app().run()
        at.button[0].click().run()
        assert not at.exception
        assert any("Rate limit" in e.value for e in at.error)