import json

import joblib
import numpy as np
import polars as pl

from src.components.model_trainer import ModelTrainer, assign_tier, make_X
from src.components.data_transformation import DataTransformation



def test_tiers_are_ordered_and_assigned_correctly():
    conf = np.array([0.01, 0.05, 0.12])
    assert list(assign_tier(conf, high_min=0.10, medium_min=0.04)) == ["low", "medium", "high"]


def test_saved_bundle_predicts_valid_probabilities():
    trainer = ModelTrainer()
    bundle = joblib.load(trainer.params["paths"]["model_path"])
    feats = pl.read_parquet(trainer.params["features"]["features_path"])
    latest = feats.filter(pl.col("date") == pl.col("date").max().over("currency"))
    X = make_X(latest, bundle["feature_cols"], bundle["currencies"])
    p = bundle["model"].predict_proba(X)[:, 1]
    assert len(p) == len(trainer.currencies) and ((p > 0) & (p < 1)).all()


def test_signal_config_is_complete():
    cfg = json.load(open(ModelTrainer().params["paths"]["signal_config_path"]))
    t = cfg["tiers"]
    assert 0 < t["medium_min_confidence"] < t["high_min_confidence"] < 0.5
    assert set(cfg["test_tier_hit_rates"]) == {"high", "medium", "low"}

def test_split_is_strictly_time_ordered_and_labelled():
    dt = DataTransformation()
    feats = pl.read_parquet(dt.params["features"]["features_path"])
    fit, cal, test = dt.split(feats)
    assert fit["date"].max() < cal["date"].min()
    assert cal["date"].max() < test["date"].min()
    assert min(fit.height, cal.height, test.height) > 1000
    for part in (fit, cal, test):
        assert part["label"].null_count() == 0 and part["label_raw"].null_count() == 0


def test_saved_splits_cover_every_labelled_row_once():
    dt = DataTransformation()
    sp = dt.params["split"]
    feats = pl.read_parquet(dt.params["features"]["features_path"])
    n_labelled = feats.drop_nulls(["label", "label_raw"]).height
    total = sum(pl.read_parquet(sp[k]).height for k in ("fit_path", "calibration_path", "test_path"))
    assert total == n_labelled