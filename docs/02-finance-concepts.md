# Finance concepts, from first principles

## 1. Who is the counterparty?

The **counterparty** is the company on the other side of a bank's contract. An FX forward is an agreement today to exchange currencies at a specified future rate. The bank can gain or lose value as the exchange rate changes. If the trade becomes valuable to the bank and the counterparty fails, the bank may lose that positive value when replacing the trade. This is one part of **counterparty credit risk**.

The [Basel framework's terminology](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/50/inforce/2019-12-15/published/2019-12-15) describes current exposure as the greater of zero and the current market value of a transaction or eligible netting set. This demo calculates a simpler illustrative version after collateral.

## 2. Read the exchange rate correctly

The FRED [DEXUSEU series](https://fred.stlouisfed.org/series/DEXUSEU) quotes **USD per EUR**. If `S = 1.10`, one euro costs $1.10. If `S` rises to `1.15`, the euro has strengthened against the dollar. A bank that will *receive euros and pay dollars* benefits from that rise; the opposite side loses value.

The model uses a **log return** over 10 published observations:

```text
r = ln(S_future / S_today)
S_scenario = S_today × exp(r_scenario)
```

For example, a 2% log-return shock multiplies the rate by `exp(0.02)`, about `1.0202`. It is close to, but not exactly, a 2% simple percentage move.

## 3. Simplified FX-forward value

Let:

- `N` = signed euro amount received by the bank. Positive `N` means the bank receives EUR and pays USD; negative `N` means the reverse.
- `K` = contracted USD/EUR exchange rate.
- `S` = scenario USD/EUR exchange rate.

The C++ engine uses continuously compounded annual EUR and USD rates. V1 assumes 2% EUR and 4% USD rates for its scenarios; these are illustrative inputs. With `tau = (maturity_calendar_days − elapsed_calendar_days) / 365`, its USD mark-to-market is:

```text
MTM_USD = N × [S × exp(−EUR_rate × tau) − K × exp(−USD_rate × tau)]
```

At zero EUR and USD rates, both discount factors equal one and the formula becomes `N × (S − K)`. The engine does account for remaining maturity through the discount factors, but it uses one constant rate per currency and a simple ACT/365 year fraction. The forecast and saved stress scenarios use an illustrative 14-calendar-day move forward from the latest observation; the interactive stress explorer values at the current day. A production valuation would require appropriate market curves, conventions, and contract details. `S = K` therefore gives zero value only in the zero-rate special case or when the two discount factors match.

### Worked example

A bank agreed to receive EUR 1,000,000 and pay USD at `K = 1.10`. Set both annual rates to zero for this easy arithmetic example. If the scenario rate is `S = 1.14`:

```text
MTM = 1,000,000 × (1.14 − 1.10) = +$40,000
```

The trade is worth $40,000 to the bank. If `S = 1.06`, the value is `−$40,000`: the bank owes value on this trade, so this trade alone does not create positive current exposure to this counterparty.

## 4. Netting is about agreements, not arithmetic convenience

Suppose the bank also has an opposite trade worth `−$25,000` with the **same counterparty** in the **same eligible netting set**. The net value is `$40,000 − $25,000 = $15,000`. The demo groups trades by netting set before applying the positive-exposure floor.

Real netting depends on enforceable legal agreements and jurisdiction. The demo's netting-set identifier is an illustrative input; it cannot establish legal eligibility. Never offset trades across different counterparties merely because the totals look similar.

## 5. Collateral reduces uncovered exposure

If the bank already holds `$8,000` of eligible USD collateral against that illustrative netting set, its simple uncovered exposure is:

```text
uncovered_exposure = max(net_MTM − received_USD_collateral, 0)
                   = max(15,000 − 8,000, 0)
                   = $7,000
```

This assumes the received collateral is available and maintains its USD value. It omits margin-call frequency, thresholds, minimum transfer amounts, haircuts, disputes, segregation, and the delay between default and close-out.

## 6. Why revalue in multiple scenarios?

The same portfolio can have low exposure if EUR/USD rises and high exposure if it falls, or vice versa. The direction depends on the **net signed EUR position** and trade strikes. Therefore, a 95th-percentile *FX return* is not necessarily a 95th-percentile *exposure*. The app revalues the portfolio under both lower and upper FX-return scenarios and displays the resulting exposure for each.

Those two endpoint exposures are **scenario outputs**. Even their maximum is not automatically a validated 95th-percentile exposure or regulatory PFE. A real PFE process would need an appropriate future distribution, portfolio revaluation across scenarios and horizons, collateral dynamics, and independent validation. The [Basel counterparty-risk guidance](https://www.bis.org/committees/bcbs/basel-consolidated-guidelines/module/cri/40) describes that broader setting.

## 7. Exposure is not loss

Exposure is a possible amount at risk **if** the counterparty defaults at a particular time. Loss also depends on whether default occurs and how much is recovered. This project does not estimate default probability, recovery rate, credit valuation adjustment, expected credit loss, or regulatory capital.
