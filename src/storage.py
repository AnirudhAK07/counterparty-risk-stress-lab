"""A small auditable SQLite record of market data, models, and scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pandas as pd


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS market_observations (
    date TEXT PRIMARY KEY,
    spot_usd_per_eur REAL NOT NULL CHECK (spot_usd_per_eur > 0)
);
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    netting_set TEXT NOT NULL,
    signed_eur_notional REAL NOT NULL,
    strike REAL NOT NULL,
    maturity_days INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS netting_sets (
    netting_set TEXT PRIMARY KEY,
    counterparty TEXT NOT NULL,
    collateral_usd REAL NOT NULL CHECK (collateral_usd >= 0)
);
CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    data_last_date TEXT NOT NULL,
    data_sha256 TEXT NOT NULL,
    horizon_observations INTEGER NOT NULL,
    metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_history (
    run_id TEXT NOT NULL REFERENCES model_runs(run_id),
    origin_date TEXT NOT NULL,
    split TEXT NOT NULL,
    spot REAL NOT NULL,
    realized_return REAL NOT NULL,
    q05 REAL NOT NULL,
    q95 REAL NOT NULL,
    baseline_q05 REAL NOT NULL,
    baseline_q95 REAL NOT NULL,
    PRIMARY KEY (run_id, origin_date)
);
CREATE TABLE IF NOT EXISTS scenario_exposures (
    run_id TEXT NOT NULL REFERENCES model_runs(run_id),
    scenario TEXT NOT NULL,
    spot REAL NOT NULL,
    counterparty TEXT NOT NULL,
    netting_set TEXT NOT NULL,
    mtm_usd REAL NOT NULL,
    collateral_usd REAL NOT NULL,
    exposure_usd REAL NOT NULL,
    PRIMARY KEY (run_id, scenario, netting_set)
);
"""


def save_run(
    db_path: Path,
    fx: pd.DataFrame,
    trades: pd.DataFrame,
    netting_sets: pd.DataFrame,
    history: pd.DataFrame,
    metrics: dict,
    forecast: dict,
    scenarios: dict[str, tuple[float, pd.DataFrame]],
) -> str:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT OR REPLACE INTO market_observations VALUES (?, ?)",
            ((row.date.strftime("%Y-%m-%d"), float(row.spot)) for row in fx.itertuples()),
        )
        connection.execute("DELETE FROM trades")
        connection.executemany(
            "INSERT INTO trades VALUES (?, ?, ?, ?, ?)",
            (tuple(row) for row in trades.itertuples(index=False, name=None)),
        )
        connection.execute("DELETE FROM netting_sets")
        connection.executemany(
            "INSERT INTO netting_sets VALUES (?, ?, ?)",
            (tuple(row) for row in netting_sets.itertuples(index=False, name=None)),
        )
        connection.execute(
            "INSERT INTO model_runs VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                forecast["origin_date"],
                metrics["data_sha256"],
                forecast["horizon_observations"],
                json.dumps(metrics, sort_keys=True),
            ),
        )
        connection.executemany(
            "INSERT INTO forecast_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    run_id,
                    row.origin_date,
                    row.split,
                    float(row.spot),
                    float(row.realized_return),
                    float(row.q05),
                    float(row.q95),
                    float(row.baseline_q05),
                    float(row.baseline_q95),
                )
                for row in history.itertuples()
            ),
        )
        for scenario, (spot, frame) in scenarios.items():
            connection.executemany(
                "INSERT INTO scenario_exposures VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        run_id,
                        scenario,
                        float(spot),
                        row.counterparty,
                        row.netting_set,
                        float(row.mtm_usd),
                        float(row.collateral_usd),
                        float(row.exposure_usd),
                    )
                    for row in frame.itertuples()
                ),
            )
    return run_id
