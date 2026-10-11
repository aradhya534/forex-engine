import numpy as np
import pandas as pd
import polars as pl


def make_X(frame: pl.DataFrame, feature_cols: list, currencies: list) -> pd.DataFrame:
    """Model input. Shared by training and serving so both build X identically."""
    X = frame.select(["currency"] + feature_cols).to_pandas()
    X["currency"] = pd.Categorical(X["currency"], categories=sorted(currencies))
    return X


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def assign_tier(conf: np.ndarray, high_min: float, medium_min: float) -> np.ndarray:
    return np.where(conf >= high_min, "high", np.where(conf >= medium_min, "medium", "low"))