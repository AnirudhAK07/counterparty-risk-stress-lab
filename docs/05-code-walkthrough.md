# Code walkthrough: read the project in a useful order

This guide follows the actual execution path. Read one function at a time, then look at its input and output before reading the next function. The formulas are in [finance concepts](02-finance-concepts.md), and the statistical ideas are in [XGBoost and validation](03-xgboost-and-validation.md).

## 1. Start at `run_pipeline.py`

`main()` is the top-level sequence. It does seven jobs:

1. Locate the bundled `data/DEXUSEU.csv` and compile the C++ executable if needed.
2. Clean FX history and make the feature/target table.
3. Split labelled observations by time.
4. Train/evaluate XGBoost and the historical baseline, then create a latest forecast.
5. Save model artifacts and metrics under `models/`.
6. Value the observed, forecast-low, forecast-high, and ±15% stress spots in C++.
7. Save the run in SQLite and print a short console summary.

The constants at the top are visible assumptions: 10-observation forecast horizon comes from `src/data_pipeline.py`; future scenarios are valued 14 calendar days later; USD/EUR rates are fixed at 4%/2%. The 14-day value is an approximation, because the number of calendar days spanned by 10 published FX observations varies with weekends and holidays.

**Pause and check:** Which values come from FRED, which are learned by XGBoost, and which were chosen by the developer? Say this aloud before moving on.

## 2. Read `src/data_pipeline.py`

### `read_fx_csv(path)`

Reads `observation_date` and `DEXUSEU`, renames them to `date` and `spot`, converts spot values to numbers, drops unavailable observations, sorts by date, and rejects duplicate dates or nonpositive rates. This is why the forecast horizon is measured in usable *published observations*.

### `make_feature_frame(fx)`

Converts spot to log spot, calculates past returns and rolling statistics, and creates the future label `log_spot[t+10] − log_spot[t]`. The future label is a separate column, never included in `FEATURE_COLUMNS`. Early rows lack enough history for the 60-observation volatility feature and are dropped. The last 10 rows lack a future label but remain available for latest-date forecasting.

**Pause and check:** Point to one feature that uses only past data and explain why `future_log_return` cannot appear in `FEATURE_COLUMNS`.

### `purged_time_split(labelled)`

Finds raw 60% and 80% boundaries, then omits the last 10 origins before each boundary from the earlier split. The assertions check that a retained training label ends before validation starts and a retained validation label ends before test starts. It returns the three frames and their boundary positions.

### `historical_baseline(labelled, positions, window=500)`

For each evaluated origin `t`, it takes at most 500 labels that had *finished* by `t`: the latest allowed label origin is `t − 10`. It estimates empirical 5th/95th return quantiles. This is a rolling baseline that updates with newly completed historical returns, not one fixed number learned on the entire series.

**Pause and check:** If today is origin 100, can the baseline use the outcome for origin 95? No; that outcome needs the rate at origin 105, which is still in the future at origin 100.

## 3. Read `src/model.py`

### `_parameters(alpha)`

Sets XGBoost's `reg:quantileerror` objective for the selected quantile `alpha`. It also controls tree complexity (`max_depth=3`, `min_child_weight=20`, regularization), learning speed (`eta=0.04`), sampling, and a fixed seed. It trains q05 and q95 as separate models.

### `_dmatrix(frame)` and `_predict(...)`

An XGBoost `DMatrix` carries the feature columns, and training matrices carry the known target. Prediction matrices deliberately carry features only, so a latest row with no known future outcome can still be scored. `_predict` uses the number of boosting rounds selected by validation.

### `pinball_loss(...)` and `interval_metrics(...)`

`pinball_loss` implements the asymmetric quantile error formula. `interval_metrics` counts lower/upper breaches, central coverage, average band width, and pinball loss. Read all of them together: a very wide band can have high coverage without being useful.

### `train_evaluate_refit(...)`

This is the model lifecycle in one function. It trains q05 and q95 on training rows, uses validation rows for early stopping, scores the untouched test rows, and saves the *evaluation models*. It then computes baseline forecasts at each validation/test date, checks for quantile crossing, and writes comparative metrics. Because 10-observation targets overlap when forecast origins are daily, it also reports a smaller every-10th-origin view of the test set. Finally, it refits separate *final models* on every outcome known by the latest market observation, using the validation-selected number of rounds, and predicts the latest feature row.

**Pause and check:** Why are there evaluation models *and* final models? The evaluation models provide an honest later-date performance record. The final models use more known history for the current forecast after that evaluation is complete.

## 4. Read `src/risk_engine.cpp`

The C++ program takes two CSV files and four numeric command-line arguments: scenario spot, elapsed calendar days, annual USD rate, and annual EUR rate.

### Input and validation helpers

`parse_csv_line` handles quoted text fields; `read_csv` checks exact headers and row widths; `parse_double` rejects malformed or nonfinite numbers; `parse_nonnegative_int` validates days; and `require_identifier` prevents empty/newline identifiers. These checks matter because a shifted CSV column could turn a trade into a different economic position without an obvious crash.

### Trade valuation loop in `main()`

For each trade, the program checks the netting-set ID and maturity, computes remaining years as `(maturity_days − elapsed_days)/365`, discounts the EUR and USD legs, and obtains signed MTM. It adds that MTM to its netting set.

```text
EUR leg = spot × exp(−EUR_rate × remaining_years)
USD leg = strike × exp(−USD_rate × remaining_years)
trade MTM = signed_EUR_notional × (EUR leg − USD leg)
```

The program then takes `max(netting_set_MTM − received_collateral_USD, 0)` for *each* set and writes one CSV row per set. The supplied sample has Northstar Trading in `NET-A` and `NET-D` on purpose: those sets must not be merged just because the counterparty name matches.

**Pause and check:** Why is the positive floor applied after summing trades within a netting set rather than to every trade separately? Because eligible positive and negative trade values offset within that set.

## 5. Read `src/engine.py`

`build_engine(project_dir)` compiles the C++ file with `g++` only if the executable is absent or older than the source. `run_engine(...)` calls the executable with the scenario rate and assumptions, reads its CSV output into a pandas table, and raises an error if the executable fails. The `collateral_multiplier` is a demo control: when it differs from 1, Python writes a temporary adjusted netting-set CSV for that one call. It does not modify the bundled input file.

**Pause and check:** Why keep the compiled executable behind a Python wrapper? It gives the modeling code and UI one stable function to call and one place to report compilation or subprocess errors.

## 6. Read `src/storage.py`

`SCHEMA` defines market observations, trades, netting sets, model runs, forecast history, and scenario exposures. `save_run(...)` opens SQLite, creates tables, writes the data and outputs, and returns a UTC-based run ID. SQL parameters (`?`) are used for values rather than concatenating values into SQL text. The data SHA-256 and metrics JSON help identify the source and assumptions of a model run.

The input trade and netting-set tables are **current snapshots**; `save_run` replaces their contents at each pipeline run. The saved `models/*.json` artifacts are replaced on rerun too. SQLite therefore gives run traceability, not complete old-run archival. To reproduce an earlier run after editing sample files, keep a separate copy of those files and the model artifacts. The dashboard's manual spot/collateral moves are not separate database records.

## 7. Read `app.py`

`main()` loads saved forecast, history, and metrics, then chooses among four pages. `_forecast_values` validates the forecast file. `_price` calls `src.engine.run_engine`, and `_by_counterparty` sums the engine's per-set rows for charts. The scenario explorer's detail table retains one row per netting set, because the exposure floor is applied within each set. The model and pricing source remain outside the UI.

- **Overview:** Shows current spot and q05/q95 spot endpoints, then values observed and +14-calendar-day forecast scenarios.
- **Scenario explorer:** Applies a user-selected simple percentage spot move and collateral multiplier at the current valuation day. This manual stress is not produced by XGBoost.
- **Backtest & monitoring:** Compares realized returns with model and baseline bands on later dates, with coverage and width visible together.
- **Data & assumptions:** Shows the fictional trade records, netting sets, SQLite table counts, and project limitations.

**Pause and check:** The overview's `+14` day forecast exposures and explorer's current-day manual sensitivities answer different questions. Make that distinction when showing screenshots.

## 8. Read `tests/`

`tests/test_risk_controls.py` has five focused controls. One checks that netting stays within its own set, including when two sets share a counterparty, and that collateral only reduces positive value. Another rejects a trade that has already matured. A third checks the nonzero-rate discounted forward formula and confirms that reversing signed notional reverses mark-to-market. A fourth changes a future FX rate and checks that an earlier row's features stay unchanged while its future label changes. The fifth checks the purge gaps and verifies that the rolling baseline ignores targets not yet completed. Run them with:

```powershell
python -m unittest discover -s tests -v
```

All five tests passed in the verified development environment. Passing tests mean these chosen behaviors worked there. The suite does not yet check every pricing convention, data-input error, or SQLite persistence path. It also cannot establish market forecast quality; that needs the held-out backtest and ongoing monitoring.
