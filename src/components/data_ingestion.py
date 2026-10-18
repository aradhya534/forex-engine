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

    def merge_recent(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append API rates dated after the raw file ends. The API and the raw file are different sources, so
        (1) check they agree on the overlapping days, and (2) rescale the new rates by the ratio at the last
        common day, so no artificial jump appears at the seam."""
        fp = self.params["fetch"]
        path = fp["recent_path"]
        if not fp["enabled"] or not os.path.exists(path):
            logging.warning("no recent-rates file used; the data ends at the raw file's last date")
            return df
        recent = pl.read_csv(path, try_parse_dates=True, schema_overrides={"currency_name": pl.String})
        if recent.height == 0:
            logging.warning("recent-rates file is empty; the data ends at the raw file's last date")
            return df

        currencies = self.params["data"]["currencies"]
        base = df.filter(pl.col("currency").is_in(currencies)).select("currency", "date", "exchange_rate")
        rec = recent.select("currency", "date", pl.col("exchange_rate").alias("recent_rate"))

        both = (
            base.join(rec, on=["currency", "date"], how="inner")
                .with_columns(rel_diff=(pl.col("recent_rate") / pl.col("exchange_rate") - 1).abs())
                .sort("date")
        )
        stats = both.group_by("currency").agg(
            n_overlap=pl.len(), median_diff=pl.col("rel_diff").median(), max_diff=pl.col("rel_diff").max()
        ).sort("currency")
        no_overlap = sorted(set(currencies) - set(stats["currency"].to_list()))
        if no_overlap:
            self._fail(f"recent rates share no dates with the raw file for {no_overlap}; cannot splice")
        too_far = stats.filter(pl.col("median_diff") > fp["max_median_overlap_diff"])
        if too_far.height:
            self._fail(f"recent rates disagree with the raw file by more than "
                       f"{fp['max_median_overlap_diff']:.1%} (median) for {too_far['currency'].to_list()}")
        for row in stats.iter_rows(named=True):
            logging.info("overlap check %s | %d days | median diff %.4f%% | max diff %.4f%%",
                         row["currency"], row["n_overlap"], row["median_diff"] * 100, row["max_diff"] * 100)

        anchor = both.group_by("currency").agg(
            factor=(pl.col("exchange_rate") / pl.col("recent_rate")).last(), anchor_date=pl.col("date").last()
        )
        last_raw = base.group_by("currency").agg(last_date=pl.col("date").max())
        new = (
            rec.join(last_raw, on="currency").filter(pl.col("date") > pl.col("last_date"))
               .join(anchor, on="currency")
               .with_columns(exchange_rate=pl.col("recent_rate") * pl.col("factor"))
               .select(pl.col("currency"), pl.lit("EUR").alias("base_currency"),
                       pl.lit(None, dtype=pl.String).alias("currency_name"), pl.col("exchange_rate"), pl.col("date"))
        )
        if new.height == 0:
            logging.info("recent rates add nothing: the raw file is already up to date")
            return df
        logging.info("appended %d recent rows (%s to %s)", new.height, new["date"].min(), new["date"].max())
        return pl.concat([df, new.select(df.columns)], how="vertical_relaxed")


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
            df = self.merge_recent(self.load_raw())
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