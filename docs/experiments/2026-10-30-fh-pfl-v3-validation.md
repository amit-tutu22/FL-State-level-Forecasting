# FH-PFL v3 on the validation origins, 30 October 2026

Method: [FH-PFL v3, federated personalized residual correction of SARIMA](../methods/fh_pfl_v3.md), formalized in [fh_pfl_v3_methodology.md](../methods/fh_pfl_v3_methodology.md). It is scored here once, with the frozen settings in [2026-10-29-selection.json](2026-10-29-selection.json) and fold hash `47573acc…ddc7ae18`. The choices agreed on 30 October and the implementation details listed in the spec were fixed before this run. Nothing was tuned on these numbers. The setup:

- **Panel:** assumed 45-day lag, CPI base 2012. The residual GRU uses GST, the six E-Way Bill series, the month encoding and the SARIMA log-forecast. It does not use CPI.
- **Origins:** April 2024 to March 2025, with 32 states at all 12 origins, 384 origins in total. The test year was not read.
- **Seeds:** 0, 1 and 2. In-process wall time was 195 seconds.

```
python -m fl_experiment.run_residual --split val \
    --selection docs/experiments/2026-10-29-selection.json \
    --source panel --allow-assumed-vintages --panel data/reports/lagged45-2012/panel.csv
```

B0 reproduces the saved SARIMA validation predictions to within 5e-5 crore, which is the rounding in the saved file.

## Result

Equal-state figures, mean ± standard deviation across seeds. RMSE and MAE are in INR crore.

| Method | RMSE | RMSE by seed | MAE | MAPE | MASE |
|---|---:|---|---:|---:|---:|
| **B0, SARIMA** | **228.1** | 228.1 / 228.1 / 228.1 | 185.9 | 17.28 | 0.75 |
| B4, v3 with GST-only inputs | 230.9 ± 2.6 | 232.3 / 232.7 / 227.9 | 184.1 | 17.11 | 0.74 |
| **FH-PFL v3** (APFL, \(\lambda_i\) on the holdout) | **235.7 ± 4.0** | 236.5 / 239.2 / 231.4 | 188.9 | 17.23 | 0.76 |
| B2, FedAvg residual GRU | 236.6 ± 4.5 | 239.8 / 238.4 / 231.5 | 191.6 | 17.54 | 0.78 |
| B5, v3 with \(\lambda_i = 1\) | 239.9 ± 5.3 | 243.1 / 242.8 / 233.8 | 194.2 | 17.54 | 0.78 |
| B1, local residual GRU | 243.5 ± 4.0 | 243.6 / 239.5 / 247.5 | 197.4 | 17.53 | 0.78 |
| B3, per-state ridge | 244.9 | 244.9 / 244.9 / 244.9 | 199.6 | 17.74 | 0.78 |

## Decision rule

The rule was fixed in the spec before this run.

1. **"Residual correction beats SARIMA": not supported.** v3 is 7.6 crore worse than SARIMA, in all three seeds. No variant passes:
   - B4 comes closest. It is 2.9 crore worse on average and lower than SARIMA in only one seed (227.9 against 228.1).
   - By the rule's third clause, the thesis reports that **the GRU found no usable signal in SARIMA's one-step errors** at this data size.
2. **"Federation helps" and "deep learning is needed": not claimed.** Both are conditional on claim 1. Among the residual models alone:
   - v3 beats the local residual GRU (B1) by 7.8 crore, in 3 of 3 seeds.
   - It beats the ridge (B3) by 9.2 crore, in 3 of 3 seeds.
   - It beats FedAvg (B2) by only 0.9 crore, in 2 of 3 seeds, which is well inside the seed spread.

   None of these make a usable forecaster, because SARIMA alone is better.
3. **Shrinkage did what the theory says, but not enough.** Choosing \(\lambda_i\) on the holdout improved on \(\lambda_i = 1\) (B5) by 4.2 crore, in 3 of 3 seeds. This matches methodology Proposition 3: shrinkage moves the forecast toward SARIMA. The 6-month holdout is too short to choose \(\lambda_i\) reliably. \(\lambda = 1\) was still chosen in 52–61% of state-origins, depending on the seed, and \(\lambda = 0\) in 26–37%.
4. **E-Way Bills hurt.** v3 with E-Way Bills is worse than B4 without them, by 4.7 crore, in 3 of 3 seeds. That matches the weak correlations measured before the run (median +0.03 to +0.13).

On the secondary metrics, which were **not** pre-registered, B4 is slightly ahead of SARIMA: MAE 184.1 against 185.9, MAPE 17.11 against 17.28, MASE 0.74 against 0.75. These gaps are within seed noise, and the primary metric goes the other way. They are reported for completeness and support no claim.

## Where v3 loses

- Compared with SARIMA, v3 is better in 11 to 15 of the 32 states, depending on the seed. It is worse in the rest.
- The losses are large and fall in big states, and the biggest losers change with the seed:
  - Maharashtra, +147 crore of RMSE in seed 0 but −48 in seed 1;
  - Tamil Nadu, +68 to +116 in every seed;
  - West Bengal, +77 to +81 in seeds 1 and 2.
- The gains are smaller and also move with the seed. Examples: Haryana −116 in seed 2, Karnataka −25 and −84 in seeds 0 and 2, Andhra Pradesh −21 to −58 in every seed.

So the correction is mostly fitting noise. That fits the pre-run evidence: Ljung–Box rejects white noise in only 4 of 32 states, and the median lag-1 autocorrelation is 0.16.

## Coverage

- At 9 of the 384 origins, all of them Lakshadweep (about 1–3 crore a month), no residual fit was usable.
- At 8 of them, a holdout month's filtered SARIMA prediction was not positive, so the log ratio is undefined. At 2024-07, the SARIMA forecast itself was negative.
- Those origins get the plain SARIMA forecast, tagged `no_residual_fit`, in every variant.
- SARIMA never fell back to the seasonal naive.

The selected round ranged from 1 to 20.

## Flower agreement

See below. It is added when the Flower run finishes.

## What this does not say

- It is a retrospective run on an assumed-lag panel, not a real-time backtest.
- State GST is published, so "private" is a simulated setting. No privacy guarantee is made.
- The mechanism is prior work. Allassak, Trichni and Omary (2026) already federate an ARIMAX residual LSTM with FedAvg, which is B2 here. v3's additions (APFL, log-ratio residuals, shrinkage, rolling origins) did not produce a forecaster better than SARIMA.
- The test year and the April–August 2026 confirmation months were not scored. Claim 1 failed on validation, so they stay unopened, as for v2.
