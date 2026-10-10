import polars as pl

from src.components.data_transformation import DataTransformation


def _clean() -> pl.DataFrame:
    return pl.read_parquet("artifacts/clean.parquet")


def test_features_have_no_lookahead():
    """A feature at day t must be identical whether or not later days exist."""
    dt = DataTransformation()
    clean = _clean()
    cut = clean["date"].unique().sort()[-200]
    cols = ["currency", "date"] + dt.feature_columns()

    full = dt.build_features(clean).filter(pl.col("date") <= cut).sort(["currency", "date"]).select(cols)
    trunc = dt.build_features(clean.filter(pl.col("date") <= cut)).sort(["currency", "date"]).select(cols)
    assert full.equals(trunc)


def test_label_is_direction_of_next_day():
    dt = DataTransformation()
    usd = dt.add_labels(dt.build_features(_clean())).filter(pl.col("currency") == "USD").sort("date")
    nxt = usd["log_ret"].shift(-1).to_list()
    expected = [None if (x is None or x == 0) else int(x > 0) for x in nxt]
    assert usd["label"].to_list() == expected


def test_latest_day_has_no_label():
    dt = DataTransformation()
    f = dt.add_labels(dt.build_features(_clean()))
    last = f.filter(pl.col("date") == pl.col("date").max().over("currency"))
    assert last["label"].null_count() == last.height