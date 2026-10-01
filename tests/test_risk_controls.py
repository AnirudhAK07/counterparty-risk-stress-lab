"""Focused checks for economic signs, netting boundaries, and time leakage."""

from __future__ import annotations

from pathlib import Path
import math
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_pipeline import (
    HORIZON_OBSERVATIONS,
    historical_baseline,
    make_feature_frame,
    purged_time_split,
)
from src.engine import build_engine


ROOT = Path(__file__).resolve().parents[1]


class PricingControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = build_engine(ROOT)

    def _call(
        self,
        trades: str,
        sets: str,
        elapsed_days: int = 0,
        usd_rate: float = 0.0,
        eur_rate: float = 0.0,
    ) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as directory:
            trade_path = Path(directory) / "trades.csv"
            set_path = Path(directory) / "netting_sets.csv"
            trade_path.write_text(trades, encoding="utf-8")
            set_path.write_text(sets, encoding="utf-8")
            return subprocess.run(
                [
                    str(self.engine), str(trade_path), str(set_path), "1.1",
                    str(elapsed_days), str(usd_rate), str(eur_rate),
                ],
                capture_output=True,
                text=True,
            )

    def test_netting_is_scoped_and_collateral_only_reduces_positive_value(self) -> None:
        trades = (
            "trade_id,netting_set,signed_eur_notional,strike,maturity_days\n"
            "A,NS1,1000000,1.0,365\n"
            "B,NS1,-250000,1.0,365\n"
            "C,NS2,-1000000,1.0,365\n"
        )
        sets = (
            "netting_set,counterparty,collateral_usd\n"
            "NS1,One Company,20000\n"
            "NS2,One Company,0\n"
        )
        result = self._call(trades, sets)
        self.assertEqual(result.returncode, 0, result.stderr)
        from io import StringIO

        rows = pd.read_csv(StringIO(result.stdout)).set_index("netting_set")
        self.assertAlmostEqual(rows.loc["NS1", "mtm_usd"], 75000, places=4)
        self.assertAlmostEqual(rows.loc["NS1", "exposure_usd"], 55000, places=4)
        self.assertAlmostEqual(rows.loc["NS2", "exposure_usd"], 0, places=4)

    def test_matured_trade_is_rejected(self) -> None:
        trades = (
            "trade_id,netting_set,signed_eur_notional,strike,maturity_days\n"
            "A,NS1,1000000,1.0,10\n"
        )
        sets = "netting_set,counterparty,collateral_usd\nNS1,One Company,0\n"
        result = self._call(trades, sets, elapsed_days=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("matured", result.stderr)

    def test_discounted_forward_value_and_position_direction(self) -> None:
        trades = (
            "trade_id,netting_set,signed_eur_notional,strike,maturity_days\n"
            "A,LONG,1000000,1.0,365\n"
            "B,SHORT,-1000000,1.0,365\n"
        )
        sets = (
            "netting_set,counterparty,collateral_usd\n"
            "LONG,Company A,0\n"
            "SHORT,Company B,0\n"
        )
        result = self._call(trades, sets, usd_rate=0.04, eur_rate=0.02)
        self.assertEqual(result.returncode, 0, result.stderr)
        from io import StringIO

        rows = pd.read_csv(StringIO(result.stdout)).set_index("netting_set")
        expected_long = 1_000_000 * (1.1 * math.exp(-0.02) - math.exp(-0.04))
        self.assertAlmostEqual(rows.loc["LONG", "mtm_usd"], expected_long, places=3)
        self.assertAlmostEqual(rows.loc["SHORT", "mtm_usd"], -expected_long, places=3)
        self.assertAlmostEqual(rows.loc["SHORT", "exposure_usd"], 0, places=3)


class TimeLeakageControls(unittest.TestCase):
    def test_features_do_not_change_when_future_price_changes(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=250)
        spot = 1.1 * np.exp(np.cumsum(np.full(250, 0.001)))
        original = pd.DataFrame({"date": dates, "spot": spot})
        revised = original.copy()
        revised.loc[110, "spot"] *= 1.5
        a = make_feature_frame(original)
        b = make_feature_frame(revised)
        columns = [name for name in a.columns if name not in ("date", "future_log_return")]
        # Features at origin 100 may not see the changed price at observation 110.
        row_a = a.loc[a["date"] == dates[100], columns].iloc[0]
        row_b = b.loc[b["date"] == dates[100], columns].iloc[0]
        pd.testing.assert_series_equal(row_a, row_b)
        label_a = a.loc[a["date"] == dates[100], "future_log_return"].iloc[0]
        label_b = b.loc[b["date"] == dates[100], "future_log_return"].iloc[0]
        self.assertNotEqual(label_a, label_b)

    def test_boundary_purge_and_baseline_use_only_completed_targets(self) -> None:
        n = 300
        frame = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=n),
            "future_log_return": np.arange(n, dtype=float),
        })
        splits = purged_time_split(frame)
        self.assertLess(
            splits.train.index.max() + HORIZON_OBSERVATIONS,
            splits.validation.index.min(),
        )
        self.assertLess(
            splits.validation.index.max() + HORIZON_OBSERVATIONS,
            splits.test.index.min(),
        )
        first = historical_baseline(frame, pd.Index([200])).iloc[0]
        modified = frame.copy()
        modified.loc[191:, "future_log_return"] = 1_000_000
        second = historical_baseline(modified, pd.Index([200])).iloc[0]
        self.assertEqual(first["baseline_q05"], second["baseline_q05"])
        self.assertEqual(first["baseline_q95"], second["baseline_q95"])


if __name__ == "__main__":
    unittest.main()
