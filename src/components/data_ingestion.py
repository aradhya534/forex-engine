import os
import sys
from dataclasses import dataclass

import polars as pl
import yaml

from src.exception import CustomException
from src.logger import logging

EXPECTED_COLS = ["currency", "base_currency", "currency_name", "exchange_rate", "date"]


@dataclass
class DataIngestionConfig:
    params_path: str = "params.yaml"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        with open(self.config.params_path) as f:
            self.params = yaml.safe_load(f)

    def load_raw(self) -> pl.DataFrame:
        path = self.params["data"]["raw_path"]
        df = pl.read_csv(path, try_parse_dates=True)
        logging.info("read %d rows, %d columns from %s", df.height, df.width, path)
        return df

    @staticmethod
    def _fail(msg: str) -> None:
        logging.error("validation failed: %s", msg)
        raise ValueError(msg)

    def validate(self, df: pl.DataFrame) -> None:
        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        if missing:
            self._fail(f"missing columns: {missing}")
        logging.info("check passed: all expected columns present")

        bases = df["base_currency"].unique().to_list()
        if bases != ["EUR"]:
            self._fail(f"base_currency must be only EUR, found {bases}")
        logging.info("check passed: base currency is EUR")

        if df["date"].dtype != pl.Date:
            self._fail(f"date column was not parsed as a date, dtype is {df['date'].dtype}")
        logging.info("check passed: date parsed as a date")

        n_nulls = sum(df.select(["currency", "date", "exchange_rate"]).null_count().row(0))
        if n_nulls:
            self._fail(f"{n_nulls} nulls in currency, date or exchange_rate")
        logging.info("check passed: no nulls in key columns")

        n_bad = df.filter(pl.col("exchange_rate") <= 0).height
        if n_bad:
            self._fail(f"{n_bad} rows with exchange_rate <= 0")
        logging.info("check passed: all rates are positive")

        n_dup = df.height - df.select(["currency", "date"]).n_unique()
        if n_dup:
            self._fail(f"{n_dup} duplicate (currency, date) rows")
        logging.info("check passed: no duplicate (currency, date) rows")

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        thr = self.params["cleaning"]["glitch_threshold"]
        reversal = self.params["cleaning"]["glitch_reversal"]
        currencies = self.params["data"]["currencies"]

        df = (
            df.filter(pl.col("currency").is_in(currencies))
              .filter(pl.col("date").dt.weekday() <= 5)          # trading days only (Mon=1 ... Fri=5)
              .sort(["currency", "date"])
              .select(["currency", "date", pl.col("exchange_rate").alias("rate_raw")])
              .with_columns(ret_raw=pl.col("rate_raw").log().diff().over("currency"))
              .with_columns(nxt=pl.col("ret_raw").shift(-1).over("currency"))
        )

        # glitch = a big move that is mostly reversed the next day
        glitch = (pl.col("ret_raw").abs() > thr) & (
            (pl.col("ret_raw") + pl.col("nxt")).abs() < reversal * pl.col("ret_raw").abs()
        )
        df = (
            df.with_columns(is_glitch=glitch.fill_null(False))
              .with_columns(
                  rate=pl.when(pl.col("is_glitch")).then(None)
                         .otherwise(pl.col("rate_raw"))
                         .fill_null(strategy="forward").over("currency")
              )
              .with_columns(log_ret=pl.col("rate").log().diff().over("currency"))
              .drop("nxt")
        )
        logging.warning("repaired %d glitch rows", df["is_glitch"].sum())
        logging.info("clean frame: %d rows, %d currencies", df.height, df["currency"].n_unique())
        return df

    def initiate_data_ingestion(self) -> str:
        try:
            logging.info("data ingestion started")
            df = self.load_raw()
            self.validate(df)
            clean = self.clean(df)

            out = self.params["data"]["clean_path"]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            clean.write_parquet(out)
            logging.info("wrote %s (%d rows)", out, clean.height)
            return out
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()