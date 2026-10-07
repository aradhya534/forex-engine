# Forex Exchange Rate Prediction Engine

An end-to-end MLOps project: ingest daily forex rates, engineer time-series features, train and track models, serve predictions through an enterprise API gateway, and expose them in a web portal.

> **Status:** Phase 0 (EDA and model prototype) in progress. Sections marked _planned_ describe the target design, not finished work.

## Stack

| Layer | Tools |
|---|---|
| Data processing | Polars, DuckDB |
| Data and pipeline versioning | DVC |
| Modelling and tracking | LightGBM, scikit-learn, MLflow (registry) |
| Prediction service | FastAPI, Pydantic v2, Python 3.12 |
| API management | WSO2 API Manager 4.x (OAuth2, throttling, developer portal) |
| Frontend | Streamlit, Plotly |
| Orchestration | Docker, Docker Compose, GitHub Actions |

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | EDA and model prototype (`notebooks/`) | In progress |
| 1 | Ingestion, schema checks, DVC (`src/data/ingest.py`, `dvc.yaml`) | Planned |
| 2 | Polars feature engine (`src/features/build_features.py`) | Planned |
| 3 | LightGBM training, time-series CV, MLflow (`src/models/train.py`) | Planned |
| 4 | FastAPI prediction service (`src/api/main.py`) | Planned |
| 5 | WSO2 API Manager integration (OpenAPI 3.0, OAuth2, rate limits) | Planned |
| 6 | Streamlit app via the WSO2 gateway (`app/main.py`) | Planned |
| 7 | Multi-container orchestration (`docker-compose.yml`) | Planned |

## Repository layout

```
forex-engine/
├── data/
│   ├── raw/            # daily_forex_rates.csv (DVC-tracked from Phase 1)
│   └── processed/
├── notebooks/
│   ├── 00_eda.ipynb              # exploration, cleaning rules, baselines
│   └── 01_model_prototype.ipynb  # throwaway model prototype
├── src/
│   ├── data/  features/  models/  api/
├── app/                # Streamlit (planned)
├── tests/
└── README.md
```

## Quickstart (Windows PowerShell)

```powershell
git clone <repo-url>; cd forex-engine
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install polars duckdb pyarrow plotly jupyterlab ipykernel lightgbm scikit-learn mlflow dvc pandas
python -m ipykernel install --user --name forex-engine
```

Download the dataset manually from Kaggle and place the CSV at `data/raw/daily_forex_rates.csv`:
<https://www.kaggle.com/datasets/asaniczka/forex-exchange-rate-since-2004-updated-daily>

```powershell
jupyter lab
```

Docker Desktop is needed from Phase 5 onward.

## Data

- **File:** `daily_forex_rates.csv`, 505,223 rows, columns `currency, base_currency, currency_name, exchange_rate, date`
- **Base currency:** always EUR (each row is the price of 1 EUR in that currency)
- **Range:** 2004-08-30 to the present, 176 currency codes. The key is `(currency, date)`; `currency_name` is null for 6 codes, so never key on it.
- **No** nulls in rates, no rates <= 0, no duplicate keys.

### Data findings that drive the design

1. **Calendar changes in 2023.** Until about 2022, only weekdays are present. From late 2023 many currencies also have weekend rows. We keep **trading days only** so a lag always means "previous trading day".
2. **Coverage is uneven.** 9 currencies go back to 2004; most start around late 2013/2014; some start in 2023 or later and cannot be modelled per currency.
3. **Rate scales vary enormously** (about 1e-5 to millions), so the target is a **log return**, not the raw rate.
4. **Some currencies are not modelling candidates:** pegged currencies (almost all zero returns), hyperinflation and redenominated currencies, discontinued codes, metals and other non-currencies.
5. **Outliers are of three kinds:** one-day data glitches that snap back (repaired by rule), permanent regime breaks (kept, training target clipped), and ordinary fat tails (left alone).

## Methodology

### Target and framing
Each row is a (currency, trading day d). **Target:** the log return on day d. **Features:** only information through day d-1.

### Leakage rules
- Fixed rules (drop weekends, flag glitches) may run before the split.
- Anything fitted from data (eligibility statistics, clip bounds) is fitted on the **training window only**, and inside each cross-validation fold on that fold's training part only.
- Split by time: 80% of trading dates for training, the rest for testing. The test set is looked at once.
- Evaluation uses the **raw, unclipped** target.

### Cleaning rules
- **Glitch rule:** a daily move above 5% that is at least 70% reversed the next day is flagged (`is_glitch`), blanked and forward-filled. Raw values are kept in `rate_raw` / `ret_raw`.
- **Target clipping (training only):** per-currency 0.5% / 99.5% quantiles of the training target.
- **Eligibility:** active in the last 7 days, at least 1,000 training rows, at least 100 test rows, zero-return share under 50%, and not in the exclusion list. This gives 125 eligible currencies.

### Features (prototype)
Lags 1-30 of the daily log return, rolling mean and standard deviation over 5/10/20/30 days (computed on the series shifted by one day), a short/long volatility ratio, day of week, month, and currency as a categorical. All windows are computed per currency.

### Validation
Expanding-window time-series cross-validation over unique trading dates, with a final single evaluation on the held-out test window.

## Baselines and prototype results

Next-day return, test window starting 2022-05-06, 7 prototype currencies (USD, GBP, JPY, CHF, AUD, CAD, LKR):

| Predictor | MAE | Notes |
|---|---|---|
| Zero return | 0.0033 | the baseline to beat |
| Persistence (yesterday's return) | worse than zero for every currency | |
| LightGBM, L1 loss (prototype) | about 0.0033 (ratio to zero: 1.003) | directional accuracy about 50% |

**Honest reading:** daily forex returns are close to noise. In the prototype, LightGBM does not beat the zero predictor on next-day return, and directional accuracy is at chance. The value of this project is the reproducible, versioned, served pipeline, not trading alpha. A next-day **volatility** target looks more learnable (a quick single-run check gave about 7% lower MAE than a rolling-volatility baseline); this is under evaluation and not yet part of the pipeline.

_Update this section with MLflow-tracked results once Phase 3 is done._

## API, gateway, portal, deployment

_Planned. These sections will document the FastAPI endpoints and schemas (Phase 4), the WSO2 API definition, OAuth2 flow and throttling tiers (Phase 5), the Streamlit app (Phase 6), and `docker-compose` usage (Phase 7)._

## Disclaimer

Educational project. Not financial advice, and not intended for trading.