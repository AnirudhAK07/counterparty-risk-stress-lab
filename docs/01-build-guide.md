# Build guide: each step and why it exists

This is a teaching guide for **V1**, tied to the bundled market data and the first verified pipeline run. Start in [the project overview](00-start-here.md) if the banking words are unfamiliar.

## Run it yourself

Open PowerShell in the project root (the directory containing `run_pipeline.py`). Install Python 3.12 and a C++17 `g++` compiler, then create the environment and run the pipeline:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_pipeline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The optional launcher `.\start_app.ps1` uses the project's `.venv` after setup. The Streamlit command binds the app to local host `127.0.0.1`; stop it with Ctrl+C. If PowerShell blocks the launch script, use the direct `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1` form above. See the root [README](../README.md) for the current setup instructions.

## Step 0 — Define the question before writing code

**Question:** For illustrative EUR/USD FX forwards, how much positive value would remain uncovered after received USD collateral under a normal range of forecast FX moves and under a separate stress move?

**Inputs:** One dated EUR/USD market series; trades with signed EUR notional and contracted rate; counterparty and netting-set identifiers; received collateral; a chosen stress move.

**Outputs:** Cleaned data; trained models; chronological validation metrics; scenario FX rates; trade and netting-set values; uncovered exposure; saved run details; a browser dashboard.

**Why:** A narrow question makes the model's target and the pricing calculation testable. A good model score alone is not a banking decision.

## Step 1 — Obtain and inspect the market series

**File:** `data/DEXUSEU.csv`.

DEXUSEU is U.S. dollars per euro, from the Federal Reserve's H.10 release via [FRED](https://fred.stlouisfed.org/series/DEXUSEU). Check the date range, date order, duplicate dates, missing values, non-positive values, and publication gaps before calculating returns. Record the as-of date used by a model run. A downloaded data file is a reproducibility input; it must not silently change during one run.

**Command:** `python run_pipeline.py` loads the bundled CSV and reports the latest market date. The [FRED page](https://fred.stlouisfed.org/series/DEXUSEU) describes the original source if you later refresh it.

**Observed:** The bundled file produced 6,955 nonmissing published observations from 1999-01-04 through 2026-09-25. All model rows must be based on valid positive FX rates.

## Step 2 — Make a future target from past observations

For observation index `t`, define the 10-observation future log return:

```text
y[t] = ln(S[t+10] / S[t])
```

`S[t]` is the EUR/USD rate at row `t` after cleaning. Ten *published observations* are used; this is not guaranteed to be 10 calendar days. The last 10 rows cannot have a known target yet, so they can be scored but cannot be included in training or backtesting.

**Why:** The model learns a well-defined market outcome. It does not train on hypothetical counterparty defaults.

## Step 3 — Build features without seeing the future

`src/data_pipeline.py` builds 11 predictors: log spot; 1-, 5-, 10-, and 20-observation past log returns; the absolute 1-observation return; 5-, 20-, and 60-observation rolling volatility; and log-distance from the 20-observation high and low. Each calculation ends at or before `t`. Initial rows without enough history are dropped.

**Check:** For a row dated `t`, changing rates after `t` must not change that row's features. This catches data leakage.

## Step 4 — Split by time and purge overlapping targets

`src/data_pipeline.py` sets raw chronological boundaries at 60% and 80% of labelled rows: earlier rows for training, the next period for model selection/validation, and the final period as a held-out test. A target at `t` needs the rate at `t+10`; therefore, the last 10 origins before each boundary are removed from the earlier partition. This is the **purge**. It stops a training label from using market observations in the validation or test period.

**Why:** Randomly shuffling a time series makes the test less like a future deployment. A purged chronological split tests whether the model works on later market conditions.

**Check:** Confirm every training target ends before validation starts, and every validation target ends before test starts. Save the date boundaries with the metrics.

## Step 5 — Train a baseline and the XGBoost forecasts

The simple baseline uses up to 500 past historical target returns whose 10-observation outcomes were already known, and requires at least 100. The XGBoost models estimate the 5th and 95th conditional quantiles of the 10-observation return. Each XGBoost model is allowed up to 400 boosting rounds with validation early stopping after 30 rounds without improvement. The baseline provides a comparison; a more complex model should earn its place by improving measured behavior.

XGBoost's official parameters include [`reg:quantileerror` and `quantile_alpha`](https://xgboost.readthedocs.io/en/stable/parameter.html). The exact model parameters and seed are in `src/model.py`; selected boosting rounds and the XGBoost version are in `models/metrics.json`. Full old-run archival would also retain a separate copy of the parameters and model artifacts for every run.

**Command:** `python run_pipeline.py` trains and saves models in `models/`.

**Check:** The development models are scored on the untouched test period before final models are refitted on all labels known by the most recent market observation. The final refit uses the boosting-round counts chosen during validation and is saved separately from the evaluation models.

## Step 6 — Challenge the forecasts

Calculate quantile/pinball loss, lower-tail coverage, upper-tail coverage, and the share of observed returns inside the forecast band. Compare all of these against the baseline. Inspect failure periods, not only averages. A 5th/95th pair is intended to produce a roughly 90% return band across comparable out-of-sample cases; this is an empirical target, not a guarantee for the next observation.

**Observed held-out test:** 1,377 forecast origins dated 2021-03-15 through 2026-09-11. XGBoost's band covered 93.83% of realized returns; the historical baseline covered 88.24%. XGBoost's average band width was **0.05311 log-return units**, versus **0.04565** for the baseline. Wider bands can cover more often, so coverage alone does not prove XGBoost is better. A coarser every-10th-origin view had 138 cases: 92.75% versus 89.86% coverage, with widths 0.05320 versus 0.04565. That smaller check is descriptive, not a significance test.

## Step 7 — Revalue the illustrative FX forwards in C++

For each scenario rate, the C++ CLI calculates each trade's USD mark-to-market using the supplied annual USD and EUR interest rates and remaining maturity, groups trades by demo netting set, subtracts USD collateral received, and floors uncovered exposure at zero. Python sends validated inputs and parses the engine's netting-set output.

The engine uses continuously compounded discount factors and an ACT/365 approximation for remaining maturity. V1 assumes USD 4% and EUR 2% annual rates; these are illustrative fixed inputs, not rates calibrated from the market series. Zero rates are a useful special case for checking the calculation. See [finance concepts](02-finance-concepts.md) for the formula and what the model still omits.

**Check:** With both rates set to zero, a trade struck at the scenario spot rate has zero value. Reversing the trade direction reverses its mark-to-market. Netting is applied *within* a set, never across unrelated counterparties or sets.

## Step 8 — Add a deliberate stress scenario

The lower and upper XGBoost return forecasts represent modeled tails from historical patterns. The pipeline revalues those forecast scenarios **14 calendar days after the current observation**, a documented approximation for a ten-published-observation horizon. It also saves separate illustrative spot shocks of −15% and +15%, using the same 14-day future valuation convention. The dashboard's manual stress explorer changes spot and collateral **at the current valuation day**. Do not call either manually selected stress move a model prediction.

**Why:** A useful risk tool shows how an assumption changes exposure, even when that assumption lies outside the model's usual experience.

## Step 9 — Persist a run in SQLite

`src/storage.py` creates tables for market observations, current sample trades, current netting sets, model runs, forecast history, and scenario exposures. A model run records a UTC timestamp, market-data as-of date and SHA-256 file hash, horizon, and metrics JSON. The saved metrics include the illustrative rate and stress assumptions. This gives **run traceability**, not a complete archive of every historical run: trade and netting-set tables are replaced on rerun, and saved model JSON files are overwritten. Preserve the original files and model artifacts separately if reproducing an old result after changing inputs.

**Why:** Reproducibility and traceability are part of modeling work. If a number changes, you should be able to identify the data, parameters, and inputs that produced it.

## Step 10 — Present the result in Streamlit

The dashboard has Overview, Scenario explorer, Backtest & monitoring, and Data & assumptions pages. Overview displays observed and modeled FX scenarios, with forecast exposures valued at the documented +14 calendar-day approximation. The Scenario explorer moves spot by −20% to +20% and scales available collateral from 0 to 2 times its sample amount at the current valuation day. Its chart totals exposure by counterparty, while its detail table keeps **one row per netting set**, where the exposure floor is actually applied. Backtest compares the held-out results and displays misses. The dashboard reads saved model artifacts and calls the C++ engine; it does not retrain on page load. Interactive scenario changes are displayed but not saved as new database runs.

**Command:** `.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1` after portable setup, or `.\start_app.ps1` in this workspace.

## Step 11 — Run tests and make a model decision

Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` after portable setup, then compare XGBoost with the historical baseline on later data. In the verified development environment, **all five focused tests passed**. The measured XGBoost band has fewer misses but is also wider; it should be described as a trade-off, not unconditional superiority. The model decision for this teaching project is **use only as an illustrative FX scenario generator with clear caveats**. A real bank would require a stronger validation program before relying on it for risk decisions.

**Latest saved forecast from the first verified run:** 2026-09-25 spot `1.1400` USD/EUR; q05/q95 scenario rates `1.1146`/`1.1576` USD/EUR. The pipeline reported approximately `$58,340` uncovered exposure at observed spot, `$27,508` at forecast-low FX, and `$138,187` at forecast-high FX. Its illustrative −15%/+15% spot stresses gave approximately `$742,231`/`$868,483`. These are totals across the sample portfolio, not expected losses or regulatory PFE. Re-running after data or code changes can change them.

## Study each line after the demo works

Read in this order: `src/data_pipeline.py` → `src/model.py` → `src/risk_engine.cpp` → `src/engine.py` → `src/storage.py` → `run_pipeline.py` → `app.py` → `tests/`. At each function, ask: What enters? What leaves? What assumption is hidden? What test would fail if this function were wrong?
