# Start here: Counterparty Risk Stress Lab

## What you can explain in one sentence

This app uses historical EUR/USD moves to build two XGBoost forecasts, then revalues illustrative FX forwards under those moves to show how a counterparty's *uncovered exposure* changes after netting and collateral.

EUR/USD here means **U.S. dollars per one euro**. The market history is the Federal Reserve's DEXUSEU series, distributed through [FRED](https://fred.stlouisfed.org/series/DEXUSEU). The counterparties, trades, and collateral in the demo are illustrative.

## What problem does it solve?

A bank may have derivative contracts with a company. If the company's contracts are worth money to the bank and the company defaults, the bank may have to replace those contracts at a cost. The value can change when an exchange rate moves. This project lets you ask:

> If EUR/USD moves over the next 10 published observations, how might the value owed to the bank change, and how much would still be uncovered after the collateral already received?

The answer is an **illustrative scenario calculation**, not an estimate of default probability, expected loss, a regulatory capital number, or an approved potential future exposure (PFE) model.

## What this project demonstrates

The demo combines market-data preparation, quantile modeling, backtesting, FX-forward valuation, netting and collateral logic, SQL storage, and an interactive interface. It uses one transparent product and one market risk factor so each assumption can be inspected. It does not reproduce any bank's model or reporting process.

## What each part does

| Part | Purpose |
| --- | --- |
| FRED DEXUSEU CSV | Real, dated EUR/USD observations. Missing/non-numeric values are cleaned before modeling. |
| Python data pipeline | Creates past-only features and the future 10-observation return used as the learning target. |
| XGBoost | Predicts the lower and upper conditional return quantiles. |
| Historical baseline | Gives a simple comparison so the XGBoost result can be challenged. |
| C++ repricer | Turns a selected FX rate into each trade's value and sums values by netting set. |
| SQLite | Records the source-data hash, run metrics and scenario outputs, plus current sample trade and collateral inputs. |
| Streamlit | Makes the result and sensitivity to assumptions visible in a browser. |

## Reading order

1. [The build, step by step](01-build-guide.md) explains how the parts are assembled and run.
2. [Finance concepts](02-finance-concepts.md) explains every banking and FX term with a worked example.
3. [XGBoost and validation](03-xgboost-and-validation.md) explains the model, baseline, and testing method.
4. [Architecture and data](04-architecture-and-data.md) maps the files and data flow.
5. [Code walkthrough](05-code-walkthrough.md) follows each file and function in execution order.

## A boundary worth remembering

The [Federal Reserve's SR 26-2 revised model risk guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) discusses model development, validation, ongoing monitoring, and risk-based governance. We use those themes as a *learning framework*: record assumptions, compare against a benchmark, inspect out-of-sample behavior, and state limits. This project has not undergone a bank's independent validation or approval process.
