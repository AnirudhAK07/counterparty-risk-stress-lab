"""Run the complete V1 data → model → C++ scenarios → SQLite workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import xgboost

from src.data_pipeline import (
    HORIZON_OBSERVATIONS,
    make_feature_frame,
    purged_time_split,
    read_fx_csv,
)
from src.engine import build_engine, run_engine
from src.model import train_evaluate_refit
from src.storage import save_run


ROOT = Path(__file__).resolve().parent
FX_SOURCE_URL = "https://fred.stlouisfed.org/series/DEXUSEU"
FUTURE_VALUATION_CALENDAR_DAYS = 14  # teaching approximation for ten observations
USD_RATE = 0.04  # illustrative, continuously compounded
EUR_RATE = 0.02  # illustrative, continuously compounded


def main() -> None:
    source = ROOT / "data" / "DEXUSEU.csv"
    if not source.exists():
        raise SystemExit("Missing data/DEXUSEU.csv; see README data download step")
    build_engine(ROOT)
    fx = read_fx_csv(source)
    feature_frame = make_feature_frame(fx)
    labelled = feature_frame.dropna(subset=["future_log_return"]).reset_index(drop=True)
    splits = purged_time_split(labelled)
    history, metrics, latest = train_evaluate_refit(
        labelled, splits, feature_frame, ROOT / "models"
    )
    metrics.update(
        {
            "data_source": FX_SOURCE_URL,
            "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "data_first_date": fx["date"].iloc[0].strftime("%Y-%m-%d"),
            "data_last_date": fx["date"].iloc[-1].strftime("%Y-%m-%d"),
            "data_observations": len(fx),
            "horizon_observations": HORIZON_OBSERVATIONS,
            "future_valuation_elapsed_calendar_days": FUTURE_VALUATION_CALENDAR_DAYS,
            "usd_rate_assumption": USD_RATE,
            "eur_rate_assumption": EUR_RATE,
            "xgboost_version": xgboost.__version__,
            "market_shock_assumption": "spot ±15%, illustrative; not a Federal Reserve scenario",
        }
    )
    model_dir = ROOT / "models"
    history.to_csv(model_dir / "forecast_history.csv", index=False)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "latest_forecast.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")

    spots = {
        "current": (latest["spot"], 0),
        "forecast_low": (latest["lower_spot"], FUTURE_VALUATION_CALENDAR_DAYS),
        "forecast_high": (latest["upper_spot"], FUTURE_VALUATION_CALENDAR_DAYS),
        "illustrative_stress_down_15pct": (
            latest["spot"] * 0.85,
            FUTURE_VALUATION_CALENDAR_DAYS,
        ),
        "illustrative_stress_up_15pct": (
            latest["spot"] * 1.15,
            FUTURE_VALUATION_CALENDAR_DAYS,
        ),
    }
    scenarios = {
        name: (
            spot,
            run_engine(
                ROOT,
                spot,
                elapsed_days=elapsed_days,
                usd_rate=USD_RATE,
                eur_rate=EUR_RATE,
            ),
        )
        for name, (spot, elapsed_days) in spots.items()
    }
    trades = pd.read_csv(ROOT / "data" / "trades.csv")
    netting_sets = pd.read_csv(ROOT / "data" / "netting_sets.csv")
    run_id = save_run(
        ROOT / "data" / "risk_lab.sqlite",
        fx,
        trades,
        netting_sets,
        history,
        metrics,
        latest,
        scenarios,
    )
    print(f"Run ID: {run_id}")
    print(f"Latest market date: {latest['origin_date']} (USD/EUR {latest['spot']:.4f})")
    print(
        "Forecast 10-observation FX range: "
        f"{latest['lower_spot']:.4f} to {latest['upper_spot']:.4f} USD/EUR"
    )
    for name, (_, frame) in scenarios.items():
        print(f"{name}: total unsecured scenario exposure USD {frame['exposure_usd'].sum():,.0f}")
    print("Test interval metrics are in models/metrics.json")


if __name__ == "__main__":
    main()
