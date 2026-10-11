import json
import time
from datetime import date, timedelta

import joblib
import numpy as np
import polars as pl
import yaml

from src.components.modeling_utils import assign_tier, make_X, to_logit
from src.logger import logging


def next_trading_day(d: date) -> date:
    """The data has weekdays only, so Friday's next trading day is Monday (public holidays are ignored)."""
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


class PredictPipeline:
    def __init__(self, params_path: str = "params.yaml"):
        with open(params_path) as f:
            self.params = yaml.safe_load(f)

        bundle = joblib.load(self.params["paths"]["model_path"])
        self.model = bundle["model"]
        self.calibrator = bundle["calibrator"]
        self.feature_cols = bundle["feature_cols"]
        self.currencies = bundle["currencies"]
        self.config = bundle["signal_config"]
        with open(self.params["paths"]["metrics_path"]) as f:
            self.metrics = json.load(f)

        feats = pl.read_parquet(self.params["features"]["features_path"])
        self.latest = feats.filter(pl.col("date") == pl.col("date").max().over("currency"))
        logging.info("predict pipeline ready | model %s | %d currencies | latest data %s",
                     self.config["model_version"], self.latest.height, self.latest["date"].max())

    def predict(self, currency: str) -> dict:
        row = self.latest.filter(pl.col("currency") == currency)
        if row.height != 1:
            raise ValueError(f"no data available for {currency}")

        t0 = time.perf_counter()
        p_raw = self.model.predict_proba(make_X(row, self.feature_cols, self.currencies))[:, 1]
        prob_up = float(self.calibrator.predict_proba(to_logit(p_raw))[:, 1][0])

        tiers = self.config["tiers"]
        strength = str(assign_tier(np.array([abs(prob_up - 0.5)]),
                                   tiers["high_min_confidence"], tiers["medium_min_confidence"])[0])
        stats = self.config["test_tier_hit_rates"][strength]
        as_of = row["date"][0]

        logging.info("predict %s | prob_up %.4f | strength %s | %.1f ms",
                     currency, prob_up, strength, (time.perf_counter() - t0) * 1000)
        return {
            "currency": currency,
            "as_of_date": as_of,
            "for_trading_day": next_trading_day(as_of),
            "horizon": "next trading day",
            "prob_up": round(prob_up, 4),
            "direction": "up" if prob_up >= 0.5 else "down",
            "strength": strength,
            "historical_hit_rate": stats["hit_rate"] if stats["hit_rate"] is not None else 0.0,
            "historical_sample_size": stats["n_days"],
            "model_version": self.config["model_version"],
            "disclaimer": "Educational project. Not financial advice. Historical hit rates are out-of-sample "
                          "but close to chance; 'high' means the model's most confident calls, not a likely win.",
        }

    def model_info(self) -> dict:
        return {
            "model_version": self.config["model_version"],
            "data_through": self.latest["date"].max(),
            "test_from": self.config["test_from"],
            "test_accuracy": self.metrics["accuracy"],
            "test_roc_auc": self.metrics["roc_auc"],
        }