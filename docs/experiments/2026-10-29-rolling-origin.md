# Rolling-origin baselines — 29 October 2026

Seasonal naive, SARIMA and the local GRU scored on the same expanding-window origins ([#15](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/15), [#16](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/16)). This is the reference table for the GRU. The earlier fixed-split numbers in [2026-10-29-real-data-smoke.md](2026-10-29-real-data-smoke.md) train once and early-stop on the whole validation year, so they are not in this table.

**These are assumed-lag sensitivity results on the validation origins, one seed, with no hyperparameter search.** They are not a real-time backtest and not a thesis claim. The test period was stored and not scored.

## Protocol

- **Horizon:** 1 month. Seasonal period: 12. Target: `target_gst_excluding_cess_crore`.
- **Expanding window:** for target month `m`, training uses only months before `m`. Scaling is refit on that slice.
- **Validation origins:** 2024-04 through 2025-03. Test origins from 2025-04 are in `folds.json` with `scored: false`.
- **GRU:** window of 6. The 6 months before `m` are the early-stopping holdout. Earlier windows are the fit. The origin month is in neither.
- **SARIMA:** one order per state, chosen by AIC on the contiguous history ending the month before 2024-04, then frozen. Each origin refits only the parameters. A failed fit would fall back to the seasonal naive; none did.
- **Seasonal naive:** the forecast for `m` is the observed target at `m-12`.
- **Eligibility:** at least 24 training months, a present seasonal lag, a complete GRU window and holdout, and the month immediately before the origin. Nothing is imputed.

## Command

```sh
python -m fl_experiment.baselines --source panel --allow-assumed-vintages \
    --panel data/reports/lagged45-2012/panel.csv \
    --output-dir fl_experiment/results/real-2026-10-29/rolling
```

The panel is the 45-day assumed-lag, CPI-base-2012 panel from the same-day data run (2,858 complete rows). Outputs stay out of Git: `folds.json`, `predictions.csv` and `metrics.json`. Wall time was 52 seconds on one CPU thread. The order grid is the full 12-candidate set.

## Coverage

- **32 states and union territories, 12 validation origins each, 384 scored.** Every state has the same count, so sample-weighted and equal-state RMSE, MAE and MAPE match.
- **384 test origins stored, 0 scored.**
- Arunachal Pradesh, undivided Jammu & Kashmir, and the legacy Daman & Diu and Dadra & Nagar Haveli labels have no complete month in or after 2024-04, so they are absent. That is the same set the fixed-split run excluded for short history.

## Validation results (INR crore)

| Model | RMSE | MAE | MAPE % | MASE, pooled | MASE, equal-state | Median state MAPE % |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal naive | 427.2 | 381.3 | 18.85 | 1.00 | 1.00 | 9.50 |
| SARIMA | 228.1 | 185.9 | 17.28 | 0.49 | 0.75 | 6.24 |
| Local GRU, all inputs | 479.7 | 317.3 | 11.76 | 0.83 | 1.03 | 8.99 |
| Local GRU, GST only | 440.2 | 312.6 | 11.18 | 0.82 | 1.00 | 8.49 |

MASE is MAE divided by the seasonal-naive MAE on the same origins. Pooled MASE weights every origin equally (large states dominate the absolute errors). Equal-state MASE is the mean of the 32 state MASE values.

## Observations

- **SARIMA is the accuracy reference.** It beats the seasonal naive on RMSE in 27 of 32 states, with no convergence fallbacks. Twenty-nine states selected `(0,1,1)(0,1,1,12)`. Goa, Rajasthan and Tamil Nadu selected `(1,1,1)(0,1,1,12)`. The pooled MASE of 0.49 against the equal-state MASE of 0.75 means the absolute-error gains are concentrated in the larger states.
- **The GRU does not beat the seasonal naive as a typical state.** GST-only has equal-state MASE 1.00 and pooled MASE 0.82. Its MAPE is lower than the naive MAPE in 21 of 32 states, but its RMSE is lower in only 11. It has a lower MAPE than SARIMA in 8 states. All-inputs is slightly worse than GST-only on every aggregate here, as in the fixed-split run.
- **Small territories still dominate percentage error.** Lakshadweep's seasonal-naive MAPE is 286% on targets of about 1 crore, and SARIMA is 304% there. The GST-only GRU is at 38% for Lakshadweep. That percentage improvement does not show up in RMSE, where the large states decide the aggregate.
- **No FedAvg row.** Rolling-origin FedAvg waits for #19, where the training budget can be matched to these baselines.

## Limitations

- Latest-vintage values with an assumed 45-day lag, not the values known at the time.
- One validation year, one GRU seed, and the GRU hyperparameters were not tuned. SARIMA orders were chosen by AIC before the validation year, not by validation error.
- A gap ends a SARIMA series at the last contiguous month. These 32 states had a long enough contiguous suffix, so that rule did not remove any scored origin.
- Federation of public data is a simulation and provides no privacy guarantee.
