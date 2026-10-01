"""Compile and call the separate C++ FX-forward exposure calculator."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


def executable_path(project_dir: Path) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return project_dir / "bin" / f"risk_engine{suffix}"


def build_engine(project_dir: Path) -> Path:
    source = project_dir / "src" / "risk_engine.cpp"
    executable = executable_path(project_dir)
    if executable.exists() and executable.stat().st_mtime >= source.stat().st_mtime:
        return executable
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required to compile src/risk_engine.cpp")
    executable.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-static-libgcc",
        "-static-libstdc++",
        str(source),
        "-o",
        str(executable),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"C++ build failed:\n{result.stderr}")
    return executable


def run_engine(
    project_dir: Path,
    spot: float,
    elapsed_days: int = 0,
    collateral_multiplier: float = 1.0,
    usd_rate: float = 0.04,
    eur_rate: float = 0.02,
) -> pd.DataFrame:
    """Value each netting set; collateral is cash *received* by the bank.

    The model rate inputs and 14-day future valuation convention are scenario
    assumptions for this teaching demo, not quotes from any bank.
    """
    if spot <= 0 or not 0 <= collateral_multiplier <= 2:
        raise ValueError("spot must be positive and collateral multiplier must be 0–2")
    executable = build_engine(project_dir)
    trades_file = project_dir / "data" / "trades.csv"
    sets_file = project_dir / "data" / "netting_sets.csv"

    with tempfile.TemporaryDirectory(prefix="risk_lab_") as temporary:
        if collateral_multiplier != 1.0:
            adjusted = pd.read_csv(sets_file)
            adjusted["collateral_usd"] *= collateral_multiplier
            sets_file = Path(temporary) / "netting_sets.csv"
            adjusted.to_csv(sets_file, index=False)
        command = [
            str(executable),
            str(trades_file),
            str(sets_file),
            f"{spot:.10f}",
            str(int(elapsed_days)),
            f"{usd_rate:.10f}",
            f"{eur_rate:.10f}",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return pd.read_csv(StringIO(result.stdout))
