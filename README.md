# Counterparty Risk Stress Lab

An interactive counterparty risk research project that connects a **real EUR/USD market series** to **illustrative FX forward trades**. XGBoost estimates lower and upper future FX returns; a C++ engine reprices each trade; SQL stores the model run and scenario results; Streamlit shows the analysis.

**Core question:** How does a change in EUR/USD change the amount a bank could be owed by each trading counterparty, after eligible netting and received cash collateral?

This is a research demonstration with fictional trades. It is not a production risk model, default prediction, regulatory potential future exposure (PFE), CCAR submission, or recommendation for collateral or trading.

## Quick start on Windows

Install Python 3.12 and a C++17 `g++` compiler. Clone or download this repository, open PowerShell in the project directory, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The app opens a local browser page. Walk through **Overview → Scenario explorer → Backtest & monitoring → Data & assumptions**. Saved model artifacts and a sample SQLite database are included; the C++ engine compiles from source on first use. The latest saved market observation is **25 September 2026**; the bundled FRED data snapshot may not include subsequent days.

To reproduce the model results and check the implementation:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The pipeline compiles the C++ engine with `g++` if it is absent or its source changes. After creating `.venv`, `.\start_app.ps1` is an optional launcher. If PowerShell blocks scripts, use the direct `python.exe` commands above. Training and app use the included `data/DEXUSEU.csv`; no API key is needed. The CSV comes from [FRED DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU), quoted as **US dollars per euro**, with missing market days removed by the pipeline. `models/metrics.json` records the exact CSV SHA-256 and package version used for the included results.

## What the model actually does

1. `src/data_pipeline.py` creates backward-looking market features. The label is the EUR/USD log return over the next **10 published FX observations**.
2. `src/model.py` trains XGBoost 5th- and 95th-percentile models. The train, validation, and test dates are chronological, with a gap at each boundary so targets cannot cross into the next period. A rolling historical quantile is the comparison model.
3. `src/risk_engine.cpp` values sample forwards for each illustrative netting set. Received USD cash collateral reduces positive net market value; trades in separate sets do not offset. Real netting requires enforceable legal agreements.
4. `run_pipeline.py` applies the forecast-boundary FX levels and an illustrative ±15% spot shock to the trades. `src/storage.py` saves data, model results, and scenarios to SQLite.
5. `app.py` provides an interactive dashboard for the forecast, scenario exploration, holdout backtest, assumptions, and data tables.

The future trade repricing uses a **14-calendar-day approximation** for 10 published FX observations. USD and EUR continuously compounded rates are fixed illustrative assumptions of **4% and 2%**. The manual scenario explorer changes current spot and collateral without advancing trade age.

### Included holdout result

On 1,377 untouched test origins, the XGBoost 5th–95th interval covered **93.8%** of subsequent 10-observation returns; the rolling historical interval covered **88.2%**. The XGBoost interval was wider (**0.0531** versus **0.0457** log-return units on average). The target central coverage is 90%. These 10-observation outcomes overlap across daily origins, so `metrics.json` also reports a coarser every-tenth-origin check. See the dashboard and docs for the full metrics and limitations. These are measurements on this historical series, not a guarantee of future performance.

## Reading path

Start with [docs/00-start-here.md](docs/00-start-here.md), then follow the build guide, finance concepts, XGBoost concepts, architecture, and code walkthrough in `docs/`. The documentation explains the formulas, build steps, tests, assumptions, and limitations.

## Project map

| Path | Purpose |
| --- | --- |
| `data/DEXUSEU.csv` | Public EUR/USD history bundled as a reproducible snapshot |
| `data/trades.csv`, `data/netting_sets.csv` | Fictional teaching portfolio and received collateral |
| `src/data_pipeline.py` | Data checks, features, labels, time split, comparison model |
| `src/model.py` | XGBoost training, evaluation, current refit |
| `src/risk_engine.cpp`, `src/engine.py` | Native C++ pricing and Python interface |
| `src/storage.py` | SQLite tables and saved runs |
| `run_pipeline.py` | One-command reproducible workflow |
| `app.py` | Streamlit frontend |
| `tests/test_risk_controls.py` | Economic sign, netting, maturity, and leakage checks |
| `models/` | Saved model files, forecast, backtest history, metrics |

## Data and source notes

- [FRED DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU), source: Federal Reserve H.10 foreign exchange rates.
- [XGBoost quantile regression parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
- [Federal Reserve model risk guidance SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) informs the emphasis on assumptions, outcome analysis, monitoring, and documentation. This independent research demo is not an independent model validation or bank compliance assessment.
