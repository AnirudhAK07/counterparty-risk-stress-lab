"""XGBoost quantile training, honest holdout evaluation, and refit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .data_pipeline import (
    FEATURE_COLUMNS,
    HORIZON_OBSERVATIONS,
    DataSplits,
    historical_baseline,
)


QUANTILES = (0.05, 0.95)


def _dmatrix(frame: pd.DataFrame) -> xgb.DMatrix:
    labels = frame["future_log_return"] if "future_log_return" in frame else None
    return xgb.DMatrix(frame[FEATURE_COLUMNS], label=labels, feature_names=FEATURE_COLUMNS)


def _parameters(alpha: float) -> dict:
    return {
        "objective": "reg:quantileerror",
        "quantile_alpha": alpha,
        "tree_method": "hist",
        "eta": 0.04,
        "max_depth": 3,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "lambda": 8.0,
        "seed": 42,
        "nthread": 4,
    }


def _predict(booster: xgb.Booster, frame: pd.DataFrame, rounds: int) -> np.ndarray:
    # The current market row has no known future label. Prediction matrices
    # therefore carry features only, even for historical rows.
    matrix = xgb.DMatrix(frame[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
    return booster.predict(matrix, iteration_range=(0, rounds)).reshape(-1)


def pinball_loss(y: np.ndarray, prediction: np.ndarray, alpha: float) -> float:
    error = y - prediction
    return float(np.mean(np.maximum(alpha * error, (alpha - 1) * error)))


def interval_metrics(history: pd.DataFrame, prefix: str) -> dict:
    y = history["realized_return"].to_numpy()
    low = history[f"{prefix}q05"].to_numpy()
    high = history[f"{prefix}q95"].to_numpy()
    return {
        "observations": int(len(y)),
        "lower_exception_rate": float(np.mean(y < low)),
        "upper_exception_rate": float(np.mean(y > high)),
        "central_coverage": float(np.mean((y >= low) & (y <= high))),
        "average_interval_width": float(np.mean(high - low)),
        "q05_pinball_loss": pinball_loss(y, low, 0.05),
        "q95_pinball_loss": pinball_loss(y, high, 0.95),
    }


def train_evaluate_refit(
    labelled: pd.DataFrame, splits: DataSplits, latest: pd.DataFrame, model_dir: Path
) -> tuple[pd.DataFrame, dict, dict]:
    """Evaluate on untouched test dates, then refit for the latest forecast.

    The test history describes the original development models. Final models
    are separately refitted on every *known* target for the current forecast.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    trained: dict[float, tuple[xgb.Booster, int]] = {}
    val_predictions: dict[float, np.ndarray] = {}
    test_predictions: dict[float, np.ndarray] = {}
    train_matrix = _dmatrix(splits.train)
    val_matrix = _dmatrix(splits.validation)

    for alpha in QUANTILES:
        booster = xgb.train(
            _parameters(alpha),
            train_matrix,
            num_boost_round=400,
            evals=[(val_matrix, "validation")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        rounds = booster.best_iteration + 1
        trained[alpha] = (booster, rounds)
        val_predictions[alpha] = _predict(booster, splits.validation, rounds)
        test_predictions[alpha] = _predict(booster, splits.test, rounds)
        booster.save_model(model_dir / f"q{int(alpha * 100):02d}_evaluation.json")

    frames = []
    crossing_count = 0
    for split_name, frame, predictions in (
        ("validation", splits.validation, val_predictions),
        ("test", splits.test, test_predictions),
    ):
        raw_low, raw_high = predictions[0.05], predictions[0.95]
        crossing_count += int(np.count_nonzero(raw_low > raw_high))
        baseline = historical_baseline(labelled, frame.index)
        result = pd.DataFrame(
            {
                "origin_date": frame["date"].dt.strftime("%Y-%m-%d"),
                "spot": frame["spot"].to_numpy(),
                "realized_return": frame["future_log_return"].to_numpy(),
                "q05": np.minimum(raw_low, raw_high),
                "q95": np.maximum(raw_low, raw_high),
                "baseline_q05": baseline["baseline_q05"].to_numpy(),
                "baseline_q95": baseline["baseline_q95"].to_numpy(),
                "split": split_name,
            }
        )
        frames.append(result)
    history = pd.concat(frames, ignore_index=True)
    test_history = history.loc[history["split"] == "test"].copy()
    # Ten-observation forward returns overlap when forecast origins are daily.
    # This coarser view reduces dependence between checked outcomes; it is not
    # a formal significance test or a substitute for the full holdout report.
    test_nonoverlap = test_history.iloc[::HORIZON_OBSERVATIONS].copy()
    metrics = {
        "xgboost_test": interval_metrics(test_history, ""),
        "historical_baseline_test": interval_metrics(test_history, "baseline_"),
        "xgboost_test_every_10th_origin": interval_metrics(test_nonoverlap, ""),
        "historical_baseline_test_every_10th_origin": interval_metrics(
            test_nonoverlap, "baseline_"
        ),
        "quantile_crossings_before_ordering": crossing_count,
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "train_end": splits.train["date"].iloc[-1].strftime("%Y-%m-%d"),
        "validation_start": splits.validation["date"].iloc[0].strftime("%Y-%m-%d"),
        "validation_end": splits.validation["date"].iloc[-1].strftime("%Y-%m-%d"),
        "test_start": splits.test["date"].iloc[0].strftime("%Y-%m-%d"),
        "test_end": splits.test["date"].iloc[-1].strftime("%Y-%m-%d"),
        "chosen_boost_rounds": {
            f"q{int(a * 100):02d}": rounds for a, (_, rounds) in trained.items()
        },
    }

    # Refit with all outcomes actually known as of the latest market date.
    latest_row = latest.iloc[[-1]].copy()
    all_matrix = _dmatrix(labelled)
    latest_quantiles = {}
    for alpha in QUANTILES:
        rounds = trained[alpha][1]
        final = xgb.train(_parameters(alpha), all_matrix, num_boost_round=rounds, verbose_eval=False)
        final.save_model(model_dir / f"q{int(alpha * 100):02d}_final.json")
        latest_quantiles[alpha] = float(_predict(final, latest_row, rounds)[0])
    latest_crossing = latest_quantiles[0.05] > latest_quantiles[0.95]
    low, high = sorted((latest_quantiles[0.05], latest_quantiles[0.95]))
    spot = float(latest_row["spot"].iloc[0])
    forecast = {
        "origin_date": latest_row["date"].iloc[0].strftime("%Y-%m-%d"),
        "spot": spot,
        "horizon_observations": 10,
        "q05_log_return": low,
        "q95_log_return": high,
        "lower_spot": float(spot * np.exp(low)),
        "upper_spot": float(spot * np.exp(high)),
        "latest_quantile_crossing_before_ordering": latest_crossing,
        "final_training_label_end": labelled["date"].iloc[-1].strftime("%Y-%m-%d"),
    }
    return history, metrics, forecast
