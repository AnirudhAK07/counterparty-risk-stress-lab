"""Read the public FX series and create strictly backward-looking features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON_OBSERVATIONS = 10
FEATURE_COLUMNS = [
    "log_spot",
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "abs_return_1",
    "volatility_5",
    "volatility_20",
    "volatility_60",
    "distance_from_20_day_high",
    "distance_from_20_day_low",
]


@dataclass(frozen=True)
class DataSplits:
    """Indexes are positions in `labelled`, before boundary gaps are removed."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    first_validation_position: int
    first_test_position: int


def read_fx_csv(path: Path) -> pd.DataFrame:
    """Read FRED DEXUSEU, quoted as USD paid for one EUR.

    FRED includes blank market holidays. Removing them makes +10 mean ten
    *published observations*, not a fixed number of calendar days.
    """
    raw = pd.read_csv(path, parse_dates=["observation_date"])
    required = {"observation_date", "DEXUSEU"}
    if not required.issubset(raw.columns):
        raise ValueError(f"FX CSV must have columns {sorted(required)}")
    fx = raw.rename(columns={"observation_date": "date", "DEXUSEU": "spot"})
    fx["spot"] = pd.to_numeric(fx["spot"], errors="coerce")
    fx = fx.dropna(subset=["date", "spot"]).sort_values("date")
    if fx["date"].duplicated().any() or (fx["spot"] <= 0).any():
        raise ValueError("FX data contain duplicate dates or nonpositive rates")
    return fx[["date", "spot"]].reset_index(drop=True)


def make_feature_frame(fx: pd.DataFrame) -> pd.DataFrame:
    """All predictors use information available at the forecast origin."""
    frame = fx.copy()
    log_spot = np.log(frame["spot"])
    daily = log_spot.diff()

    frame["log_spot"] = log_spot
    frame["return_1"] = daily
    frame["return_5"] = log_spot.diff(5)
    frame["return_10"] = log_spot.diff(10)
    frame["return_20"] = log_spot.diff(20)
    frame["abs_return_1"] = daily.abs()
    for window in (5, 20, 60):
        frame[f"volatility_{window}"] = daily.rolling(window).std(ddof=0)
    frame["distance_from_20_day_high"] = log_spot - log_spot.rolling(20).max()
    frame["distance_from_20_day_low"] = log_spot - log_spot.rolling(20).min()

    # This is a label, never a feature. The final ten rows have no known label.
    frame["future_log_return"] = (
        log_spot.shift(-HORIZON_OBSERVATIONS) - log_spot
    )
    return frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def purged_time_split(labelled: pd.DataFrame) -> DataSplits:
    """Chronological 60/20/20 split with a ten-origin gap at each boundary.

    A label at origin t uses S[t+10]. We remove origins whose target would
    reach into the next block. No test row is used for early stopping.
    """
    n = len(labelled)
    first_val = int(n * 0.60)
    first_test = int(n * 0.80)
    h = HORIZON_OBSERVATIONS
    if first_val <= h or first_test - first_val <= h or n - first_test <= h:
        raise ValueError("Not enough labelled FX rows for purged split")
    train = labelled.iloc[: first_val - h].copy()
    validation = labelled.iloc[first_val : first_test - h].copy()
    test = labelled.iloc[first_test:].copy()
    assert train.index.max() + h < validation.index.min()
    assert validation.index.max() + h < test.index.min()
    return DataSplits(train, validation, test, first_val, first_test)


def historical_baseline(
    labelled: pd.DataFrame, positions: pd.Index, window: int = 500
) -> pd.DataFrame:
    """Rolling 5th/95th quantiles using only labels completed by origin t."""
    y = labelled["future_log_return"].to_numpy()
    rows = []
    for t in positions:
        last_completed_origin = int(t) - HORIZON_OBSERVATIONS
        first_origin = max(0, last_completed_origin - window + 1)
        history = y[first_origin : last_completed_origin + 1]
        if len(history) < 100:
            raise ValueError("Historical baseline has fewer than 100 known outcomes")
        rows.append(
            {
                "baseline_q05": float(np.quantile(history, 0.05)),
                "baseline_q95": float(np.quantile(history, 0.95)),
            }
        )
    return pd.DataFrame(rows, index=positions)
