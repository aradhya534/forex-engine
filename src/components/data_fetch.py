import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl
import requests
import yaml

from src.exception import CustomException
from src.logger import logging

OUT_SCHEMA = {"currency": pl.String, "base_currency": pl.String, "currency_name": pl.String,
              "exchange_rate": pl.Float64, "date": pl.Date}


@dataclass
class DataFetchConfig:
    params_path: str = "params.yaml"


class DataFetch:
    """Downloads recent daily rates (EUR base) from the Frankfurter API into data/raw/recent_rates.csv."""

    def __init__(self):
        self.config = DataFetchConfig()
        with open(self.config.params_path) as f:
            self.params = yaml.safe_load(f)
        self.currencies = self.params["data"]["currencies"]

    def latest_date_in_raw(self) -> date:
        raw = pl.read_csv(self.params["data"]["raw_path"], try_parse_dates=True)
        return raw.filter(pl.col("currency").is_in(self.currencies))["date"].max()

    def fetch(self, start: date) -> pl.DataFrame:
        fp = self.params["fetch"]
        query = {"from": start.isoformat(), "base": "EUR", "quotes": ",".join(self.currencies)}
        last_error = None
        for attempt in range(1, fp["retries"] + 1):
            try:
                r = requests.get(fp["base_url"], params=query, timeout=fp["timeout_seconds"],
                                 headers={"User-Agent": "forex-engine/0.1"})
                if r.status_code == 200:
                    break
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            except requests.exceptions.RequestException as e:
                last_error = str(e)
            logging.warning("fetch attempt %d/%d failed: %s", attempt, fp["retries"], last_error)
            time.sleep(2 * attempt)
        else:
            raise RuntimeError(f"could not fetch recent rates from {fp['base_url']}: {last_error}")

        records = r.json()
        if not records:
            raise RuntimeError("the API returned no records")
        df = pl.DataFrame(records)
        missing = set(self.currencies) - set(df["quote"].unique().to_list())
        if missing:
            raise RuntimeError(f"the API returned no data for: {sorted(missing)}")
        if df["base"].unique().to_list() != ["EUR"]:
            raise RuntimeError("the API returned a base other than EUR")

        out = df.select(
            pl.col("quote").alias("currency"),
            pl.col("base").alias("base_currency"),
            pl.lit(None, dtype=pl.String).alias("currency_name"),
            pl.col("rate").cast(pl.Float64).alias("exchange_rate"),
            pl.col("date").str.to_date(),
        )
        if out.filter(pl.col("exchange_rate") <= 0).height:
            raise RuntimeError("the API returned a non-positive rate")
        return out.sort(["currency", "date"])

    def initiate_data_fetch(self) -> str:
        try:
            fp = self.params["fetch"]
            if not fp["enabled"]:
                logging.warning("fetch is disabled in params.yaml; writing an empty recent-rates file")
                pl.DataFrame(schema=OUT_SCHEMA).write_csv(fp["recent_path"])
                return fp["recent_path"]

            last_raw = self.latest_date_in_raw()
            start = last_raw - timedelta(days=fp["overlap_days"])
            logging.info("raw file ends %s; fetching %s to today for %d currencies", last_raw, start, len(self.currencies))
            recent = self.fetch(start)
            recent.write_csv(fp["recent_path"])
            logging.info("wrote %s (%d rows, latest date %s)", fp["recent_path"], recent.height, recent["date"].max())
            return fp["recent_path"]
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    DataFetch().initiate_data_fetch()