# XGBoost, quantiles, and model validation

## Why use a model here?

The future EUR/USD rate is unknown. A bank can choose hypothetical shocks, but an empirical model offers a repeatable way to estimate a *conditional range* of 10-observation moves from the historical information known today. XGBoost is used for the market move forecast; the C++ engine then translates moves into dollar values for trades.

## What is XGBoost?

XGBoost builds an ensemble of small decision trees. Each new tree is added to improve the model's objective on the training observations. A tree can learn conditions such as “recent movement was unusually large” and adjust the forecast for rows meeting that condition. The main controls balance fit and complexity: number of trees, learning rate, maximum tree depth, row/column sampling, and regularization.

Tree models can discover nonlinear patterns, but they can also fit noise. Historical FX returns are difficult to predict. This is why the baseline and untouched later-period test are essential.

## Why two quantiles instead of one average?

A mean forecast says where a typical return might be. A risk scenario also needs the tails. We fit a lower `q = 0.05` return and an upper `q = 0.95` return. If these are well calibrated over comparable future cases, about 5% of returns should fall below the lower forecast and about 5% should exceed the upper forecast. The two numbers form a nominal 90% return band.

XGBoost provides the [`reg:quantileerror` objective and `quantile_alpha` parameter](https://xgboost.readthedocs.io/en/stable/parameter.html). The learning target is the **10-published-observation EUR/USD log return**, not a default indicator, loan loss, exposure, or probability of default. Official XGBoost examples also caution that separately fitted quantiles can cross; if the reported lower value exceeds the upper value, that must be detected and handled explicitly ([XGBoost prediction-interval example](https://xgboost.readthedocs.io/en/latest/python/examples/prediction_intervals.html)).

### Pinball loss in plain English

The quantile objective uses an asymmetric error penalty. For a 95th-percentile forecast, predicting **too low** when a large return occurs is penalized more heavily than predicting the same distance too high. A 5th-percentile forecast reverses which side receives the larger penalty. For observed return `y` and forecast `q_hat`, at quantile level `alpha`:

```text
error = y − q_hat
pinball_loss = alpha × error                 if error >= 0
             = (alpha − 1) × error           if error < 0
```

Lower average pinball loss is better for that quantile. This metric is specific to quantile forecasting; ordinary mean squared error answers a different question.

## What goes into the model?

Only information already published by date `t` may be used. The implemented 11 predictors are log spot; past 1-, 5-, 10-, and 20-observation log returns; the absolute 1-observation return; 5-, 20-, and 60-observation rolling volatility; and log-distance from the 20-observation high and low. Do not include `S[t+10]`, future volatility, or statistics fitted using the whole dataset. Even scaling or imputation rules must be learned from the training portion where applicable.

The exact feature list should be reviewed in code. A named feature is not evidence that it causes FX moves; it may only be a historical association.

## Chronological, purged validation

For a row at `t`, the label needs `S[t+10]`. If a training row near a split date uses a rate from the validation period to calculate its label, the training set has seen part of the future evaluation period. The split therefore removes rows with labels that cross from an earlier partition into a later one. This is a **purged split**.

```text
older observations       later observations       latest observations
training     | purge |   validation   | purge |  held-out test
```

The code makes a chronological 60/20/20 split, with a 10-origin purge before the validation and test boundaries. Training estimates tree parameters. Validation supports model selection and early stopping. The test period should be examined only after choices are fixed. Do not choose hyperparameters, features, or stress definitions to make the test numbers look better. After that evaluation, the code refits separate final models on all outcomes known by the latest market date; those final models produce the current forecast. The saved evaluation models and saved final models have different roles.

## Why compare with a historical baseline?

The baseline estimates return quantiles from the latest 500 completed 10-observation returns available at each forecast date, requiring at least 100 completed outcomes. It is easy to explain and difficult to dismiss. If XGBoost does not improve tail coverage or pinball loss on later observations, the honest conclusion is that the extra complexity has not been justified by this test.

The baseline's completion check is in `src/data_pipeline.py`: at origin `t`, it includes target origins no later than `t − 10`.

## Metrics to read together

| Metric | What it asks | A warning sign |
| --- | --- | --- |
| Lower-tail coverage | What fraction of realized returns fell below the q05 forecast? | Far above 5% means the lower tail was underestimated. |
| Upper cumulative coverage | What fraction of realized returns were at or below q95? | Well below 95% means too many upper-tail breaches. |
| Interval coverage | What fraction of returns were between q05 and q95? | Well below 90% means the band missed too often. |
| Pinball loss | How large and asymmetric were quantile errors? | Worse than the baseline suggests the model added no measured value. |
| Band width | How wide is the q05–q95 range? | Huge width can give high coverage without useful precision. |

Compare these over the entire test and in different market periods. A single recent period may be too small to say much about rare tails. Also look at actual breaches and the dates on which they occurred.

### What the bundled backtest found

The untouched test period contains 1,377 forecast origins, from 2021-03-15 through 2026-09-11. Its measurements are saved in `models/metrics.json`:

| Sample and method | Central coverage | Average band width, log-return units |
| --- | ---: | ---: |
| All 1,377 origins, XGBoost | 93.83% | 0.05311 |
| All 1,377 origins, historical baseline | 88.24% | 0.04565 |
| Every 10th origin, 138 cases, XGBoost | 92.75% | 0.05320 |
| Every 10th origin, 138 cases, historical baseline | 89.86% | 0.04565 |

The XGBoost band missed less often, **and it was wider**. In the full test its q05/q95 pinball losses were `0.001499`/`0.001615`, versus `0.001542`/`0.001763` for the baseline. In the smaller every-10th-origin check, the baseline had slightly lower q05 pinball loss (`0.001496` versus XGBoost's `0.001536`), while XGBoost had lower q95 loss. This is a mixed result, not a claim of universal superiority. The coarser sample reduces target overlap but is not a formal uncertainty or significance analysis.

## Revalidation and monitoring questions

Use these questions when discussing the project:

1. Does the model remain calibrated on later dates?
2. Is it better than the simple benchmark, after considering uncertainty?
3. Do missing rates, stale data, changed volatility, or new market regimes break assumptions?
4. Did any code, data, trade, or model version change since the last run?
5. If it fails, should we retrain, choose a simpler baseline, widen stress assumptions, or restrict its use?

The [Federal Reserve's SR 26-2 guidance](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) emphasizes a risk-based approach to development, validation, and ongoing model management. This project illustrates those habits; it is not evidence of regulatory approval.

## Important statistical limits

- A forecast quantile is **not** a guarantee that tomorrow's outcome will stay inside the band.
- The model forecasts **FX returns**, not direct portfolio exposure. A return quantile does not map to the same exposure quantile for every mix of long and short trades.
- Observations overlap because each target spans 10 published rates. Reported coverage values are descriptive and do not imply independent trials.
- One FX pair and one historical series cannot represent an entire bank's market or counterparty risks.
- Tree models generally extrapolate poorly beyond the range of training examples. A user-entered severe stress scenario is therefore shown separately.
- The code detects quantile crossing and orders the lower/upper predictions before display and scoring. Ordering makes a usable band; frequent crossing would still be a model-quality concern to investigate.
