import os
import sys
from dataclasses import dataclass

import polars as pl
import yaml

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    params_path: str = "params.yaml"


def get_feature_columns(lags: int, windows: list) -> list:
    """The single source of truth for feature names and order (training and serving both import this)."""
    cols = [f"lag_{k}" for k in range(1, lags + 1)]
    cols += [f"abs_lag_{k}" for k in range(1, 6)]
    for w in windows:
        cols += [f"roll_mean_{w}", f"roll_std_{w}", f"roll_abs_{w}", f"mom_{w}"]
    cols += ["ewm_vol_10", "ewm_vol_30", "dev_20", "dev_60", "vol_ratio",
             "dow", "month", "dom", "month_end", "mkt_lag1", "mkt_abs_lag1", "mkt_vol_5"]
    return cols


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        with open(self.config.params_path) as f:
            self.params = yaml.safe_load(f)
        self.lags = self.params["features"]["lags"]
        self.windows = self.params["features"]["windows"]

    def feature_columns(self) -> list:
        return get_feature_columns(self.lags, self.windows)

    def build_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Features at row t use information through day t only. All windows run per currency."""
        r = pl.col("log_ret")
        lrate = pl.col("rate").log()

        exprs = [r.shift(k - 1).over("currency").alias(f"lag_{k}") for k in range(1, self.lags + 1)]
        exprs += [r.shift(k - 1).abs().over("currency").alias(f"abs_lag_{k}") for k in range(1, 6)]
        for w in self.windows:
            exprs += [
                r.rolling_mean(window_size=w).over("currency").alias(f"roll_mean_{w}"),
                r.rolling_std(window_size=w).over("currency").alias(f"roll_std_{w}"),
                r.abs().rolling_mean(window_size=w).over("currency").alias(f"roll_abs_{w}"),
                r.rolling_sum(window_size=w).over("currency").alias(f"mom_{w}"),
            ]
        for span in (10, 30):
            exprs.append((r ** 2).ewm_mean(span=span, ignore_nulls=True).sqrt().over("currency").alias(f"ewm_vol_{span}"))
        for w in (20, 60):
            exprs.append(
                ((lrate - lrate.rolling_mean(window_size=w)) / lrate.rolling_std(window_size=w))
                .over("currency").alias(f"dev_{w}")
            )

        return (
            df.sort(["currency", "date"])
              .with_columns(exprs)
              .with_columns(
                  vol_ratio=pl.col("roll_std_5") / pl.col("roll_std_30"),
                  dow=pl.col("date").dt.weekday(),
                  month=pl.col("date").dt.month(),
                  dom=pl.col("date").dt.day(),
                  month_end=(pl.col("date").dt.day() >= 28).cast(pl.Int8),
              )
              .with_columns(
                  mkt_lag1=pl.col("lag_1").mean().over("date"),
                  mkt_abs_lag1=pl.col("abs_lag_1").mean().over("date"),
                  mkt_vol_5=pl.col("roll_std_5").mean().over("date"),
              )
        )

    def add_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """label = 1 if the NEXT day's return is positive, 0 if negative; null if flat or if there is no next day."""
        def direction(col: str) -> pl.Expr:
            nxt = pl.col(col).shift(-1).over("currency")
            return (pl.when(nxt > 0).then(1).when(nxt < 0).then(0).otherwise(None)).cast(pl.Int8)

        return df.with_columns(
            label=direction("log_ret"),         # from repaired returns: what we train on
            label_raw=direction("ret_raw"),     # from raw returns: what we evaluate on
        )

    def initiate_data_transformation(self) -> str:
        try:
            logging.info("data transformation started")
            clean = pl.read_parquet(self.params["data"]["clean_path"])
            logging.info("read %d rows from %s", clean.height, self.params["data"]["clean_path"])

            feats = self.add_labels(self.build_features(clean))
            cols = self.feature_columns()

            before = feats.height
            feats = (
                feats.drop_nulls(cols)                                   # warm-up rows
                     .filter(pl.all_horizontal(pl.col(cols).is_finite()))  # zero-variance windows give NaN/inf
            )
            logging.info("dropped %d warm-up/non-finite rows, %d remain", before - feats.height, feats.height)

            labelled = feats.drop_nulls("label")
            n_latest = feats.filter(pl.col("date") == pl.col("date").max().over("currency")).height
            n_flat = feats.height - labelled.height - n_latest
            logging.info("labelled rows: %d | latest day per currency (no label yet): %d | flat next day (no label): %d | share up: %.4f",
                         labelled.height, n_latest, n_flat, labelled["label"].mean())

            out = self.params["features"]["features_path"]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            feats.write_parquet(out)
            logging.info("wrote %s (%d rows, %d features)", out, feats.height, len(cols))
            return out
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataTransformation().initiate_data_transformation()