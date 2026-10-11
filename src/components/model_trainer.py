import json
import os
import sys
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.components.data_transformation import get_feature_columns
from src.exception import CustomException
from src.logger import logging

from src.components.modeling_utils import make_X, to_logit, assign_tier

@dataclass
class ModelTrainerConfig:
    params_path: str = "params.yaml"


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        with open(self.config.params_path) as f:
            self.params = yaml.safe_load(f)
        self.feature_cols = get_feature_columns(self.params["features"]["lags"], self.params["features"]["windows"])
        self.currencies = self.params["data"]["currencies"]

    def load_splits(self):
        """The split is made and saved by data transformation; the trainer only reads it."""
        sp = self.params["split"]
        fit = pl.read_parquet(sp["fit_path"])
        cal = pl.read_parquet(sp["calibration_path"])
        test = pl.read_parquet(sp["test_path"])
        logging.info("loaded splits | fit: %d | calibration: %d | test: %d", fit.height, cal.height, test.height)
        return fit, cal, test

    def initiate_model_training(self) -> dict:
        try:
            logging.info("model training started")
            fit, cal, test = self.load_splits()
            cal_start, cut = cal["date"].min(), test["date"].min()
            mp = self.params["model"]

            # 1. train on the fit part (labels from repaired returns)
            model = lgb.LGBMClassifier(objective="binary", verbose=-1, n_jobs=-1, **mp["lightgbm"])
            model.fit(make_X(fit, self.feature_cols, self.currencies), fit["label"].to_numpy())

            # 2. calibrate (Platt scaling) on the calibration part only
            p_cal_raw = model.predict_proba(make_X(cal, self.feature_cols, self.currencies))[:, 1]
            y_cal = cal["label"].to_numpy()
            calibrator = LogisticRegression(C=1e6).fit(to_logit(p_cal_raw), y_cal)
            p_cal = calibrator.predict_proba(to_logit(p_cal_raw))[:, 1]

            # 3. signal tiers from the calibration part's confidence, never from test
            conf_cal = np.abs(p_cal - 0.5)
            high_min = float(np.quantile(conf_cal, 1 - mp["tier_high_top_frac"]))
            medium_min = float(np.quantile(conf_cal, 1 - mp["tier_medium_top_frac"]))

            # 4. evaluate once on test (labels from RAW returns)
            y_test = test["label_raw"].to_numpy()
            p_raw = model.predict_proba(make_X(test, self.feature_cols, self.currencies))[:, 1]
            p_test = calibrator.predict_proba(to_logit(p_raw))[:, 1]
            base = float(np.mean(fit["label"].to_numpy()))
            majority_acc = np.mean(y_test) if base >= 0.5 else 1 - np.mean(y_test)   # always predict the TRAIN majority class

            metrics = {
                "test_rows": int(test.height),
                "test_share_up": round(float(np.mean(y_test)), 4),
                "accuracy": round(float(accuracy_score(y_test, p_test >= 0.5)), 4),
                "accuracy_always_train_majority": round(float(majority_acc), 4),
                "roc_auc": round(float(roc_auc_score(y_test, p_test)), 4),
                "log_loss_calibrated": round(float(log_loss(y_test, p_test)), 4),
                "log_loss_uncalibrated": round(float(log_loss(y_test, p_raw)), 4),
                "log_loss_base_rate_only": round(float(log_loss(y_test, np.full(len(y_test), base))), 4),
                "brier_calibrated": round(float(brier_score_loss(y_test, p_test)), 4),
            }

            tiers = assign_tier(np.abs(p_test - 0.5), high_min, medium_min)
            correct = (p_test >= 0.5).astype(int) == y_test
            tier_stats = {
                t: {"n_days": int((tiers == t).sum()),
                    "hit_rate": round(float(correct[tiers == t].mean()), 4) if (tiers == t).any() else None}
                for t in ("high", "medium", "low")
            }

            order = np.argsort(p_test)
            reliability = [
                {"mean_predicted_up": round(float(p_test[i].mean()), 4),
                 "actual_up_rate": round(float(y_test[i].mean()), 4), "n": int(len(i))}
                for i in np.array_split(order, 5)
            ]

            signal_config = {
                "model_version": mp["version"],
                "train_fit_until": str(cal_start),
                "calibration_until": str(cut),
                "test_from": str(cut),
                "direction_threshold": 0.5,
                "tiers": {"high_min_confidence": high_min, "medium_min_confidence": medium_min},
                "test_tier_hit_rates": tier_stats,
                "test_reliability_5_bins": reliability,
                "note": "Hit rates are out-of-sample on the test window; confidence = |calibrated prob - 0.5|.",
            }

            for key in ("model_path", "signal_config_path", "metrics_path"):
                os.makedirs(os.path.dirname(self.params["paths"][key]), exist_ok=True)
            joblib.dump({"model": model, "calibrator": calibrator, "feature_cols": self.feature_cols,
                         "currencies": self.currencies, "signal_config": signal_config},
                        self.params["paths"]["model_path"])
            with open(self.params["paths"]["signal_config_path"], "w") as f:
                json.dump(signal_config, f, indent=2)
            with open(self.params["paths"]["metrics_path"], "w") as f:
                json.dump(metrics, f, indent=2)

            # 5. track the run in MLflow
            mlflow.set_tracking_uri(self.params["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.params["mlflow"]["experiment"])
            with mlflow.start_run(run_name=mp["version"]):
                mlflow.log_params({**{f"lgbm_{k}": v for k, v in mp["lightgbm"].items()},
                                   "n_features": len(self.feature_cols), "n_currencies": len(self.currencies),
                                   "calibration_frac": self.params["split"]["calibration_frac"], "test_from": str(cut)})
                mlflow.log_metrics(metrics)
                for t, s in tier_stats.items():
                    if s["hit_rate"] is not None:
                        mlflow.log_metric(f"hit_rate_{t}", s["hit_rate"])
                mlflow.log_artifact(self.params["paths"]["signal_config_path"])
                mlflow.log_artifact(self.params["paths"]["model_path"])

            logging.info("test metrics: %s", metrics)
            logging.info("tier hit rates (test): %s", tier_stats)
            return metrics
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ModelTrainer().initiate_model_training()