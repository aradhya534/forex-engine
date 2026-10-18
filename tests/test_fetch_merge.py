from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.components.data_fetch import DataFetch
from src.components.data_ingestion import DataIngestion


def _raw(dates, rates, cur="USD"):
    return pl.DataFrame({"currency": cur, "base_currency": "EUR", "currency_name": "name",
                         "exchange_rate": rates, "date": dates}).with_columns(pl.col("date").str.to_date())


def _recent_file(tmp_path, dates, rates, cur="USD"):
    p = tmp_path / "recent.csv"
    pl.DataFrame({"currency": cur, "base_currency": "EUR", "currency_name": None,
                  "exchange_rate": rates, "date": dates}).write_csv(p)
    return str(p)


def _ingestion(recent_path, currencies=("USD",)):
    ing = DataIngestion()
    ing.params["data"]["currencies"] = list(currencies)
    ing.params["fetch"].update(enabled=True, recent_path=recent_path)
    return ing


RAW_DATES = ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"]
RAW_RATES = [1.1300, 1.1250, 1.1200, 1.1210, 1.1230]


def test_merge_appends_only_dates_after_the_raw_file(tmp_path):
    # the API runs about 0.3% above the raw file on the overlapping days
    api_dates = RAW_DATES + ["2026-10-08", "2026-10-09"]
    api_rates = [r * 1.003 for r in RAW_RATES] + [1.1260, 1.1280]
    out = _ingestion(_recent_file(tmp_path, api_dates, api_rates)).merge_recent(_raw(RAW_DATES, RAW_RATES))
    assert out.height == 7
    assert out["date"].max() == date(2026, 10, 9)
    assert out["date"].n_unique() == out.height          # no duplicate dates


def test_merge_splices_so_there_is_no_jump_at_the_seam(tmp_path):
    api_dates = RAW_DATES + ["2026-10-08"]
    api_rates = [r * 1.003 for r in RAW_RATES] + [1.1260]
    out = _ingestion(_recent_file(tmp_path, api_dates, api_rates)).merge_recent(_raw(RAW_DATES, RAW_RATES)).sort("date")
    seam_return = out["exchange_rate"][-1] / out["exchange_rate"][-2]
    api_return = 1.1260 / (1.1230 * 1.003)
    assert seam_return == pytest.approx(api_return, rel=1e-9)     # the new day's move is the API's own move


def test_merge_rejects_inverted_rates(tmp_path):
    api_dates = RAW_DATES + ["2026-10-08"]
    api_rates = [1 / r for r in RAW_RATES] + [1 / 1.126]
    with pytest.raises(ValueError, match="disagree"):
        _ingestion(_recent_file(tmp_path, api_dates, api_rates)).merge_recent(_raw(RAW_DATES, RAW_RATES))


def test_merge_needs_overlapping_days_to_splice(tmp_path):
    with pytest.raises(ValueError, match="share no dates"):
        _ingestion(_recent_file(tmp_path, ["2026-10-08", "2026-10-09"], [1.12, 1.13])).merge_recent(_raw(RAW_DATES, RAW_RATES))


def test_merge_without_recent_file_returns_the_raw_data(tmp_path):
    raw = _raw(RAW_DATES, RAW_RATES)
    assert _ingestion(str(tmp_path / "missing.csv")).merge_recent(raw).equals(raw)


def test_merge_when_already_up_to_date_adds_nothing(tmp_path):
    raw = _raw(RAW_DATES, RAW_RATES)
    out = _ingestion(_recent_file(tmp_path, RAW_DATES, [r * 1.001 for r in RAW_RATES])).merge_recent(raw)
    assert out.equals(raw)


def _api_response(status, body=None):
    r = MagicMock(); r.status_code = status; r.text = "error body"; r.json.return_value = body
    return r


def test_fetch_parses_the_frankfurter_format():
    f = DataFetch(); f.currencies = ["USD", "GBP"]
    records = [{"date": "2026-10-08", "base": "EUR", "quote": "USD", "rate": 1.1206},
               {"date": "2026-10-08", "base": "EUR", "quote": "GBP", "rate": 0.8700},
               {"date": "2026-10-09", "base": "EUR", "quote": "USD", "rate": 1.1216},
               {"date": "2026-10-09", "base": "EUR", "quote": "GBP", "rate": 0.8710}]
    with patch("requests.get", return_value=_api_response(200, records)) as get:
        out = f.fetch(date(2026, 9, 23))
    assert out.columns == ["currency", "base_currency", "currency_name", "exchange_rate", "date"]
    assert out["date"].dtype == pl.Date and out.height == 4
    assert get.call_args.kwargs["params"] == {"from": "2026-09-23", "base": "EUR", "quotes": "USD,GBP"}


def test_fetch_retries_then_fails_clearly_on_503():
    f = DataFetch(); f.currencies = ["USD"]
    with patch("requests.get", return_value=_api_response(503)) as get, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="could not fetch"):
            f.fetch(date(2026, 9, 23))
    assert get.call_count == f.params["fetch"]["retries"]


def test_fetch_fails_when_a_currency_is_missing():
    f = DataFetch(); f.currencies = ["USD", "LKR"]
    records = [{"date": "2026-10-08", "base": "EUR", "quote": "USD", "rate": 1.1206}]
    with patch("requests.get", return_value=_api_response(200, records)):
        with pytest.raises(RuntimeError, match="LKR"):
            f.fetch(date(2026, 9, 23))