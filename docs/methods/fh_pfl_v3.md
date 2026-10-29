# FH-PFL v3 — federated personalized residual correction of SARIMA (draft specification, #21)

> **Status: choices agreed on 30 October 2026 (see the end of this file). Formalized in [fh_pfl_v3_methodology.md](fh_pfl_v3_methodology.md). Not implemented, not scored.** This design is separate from [v1](fh_pfl.md) and [v2](fh_pfl_v2.md), which are unchanged.
>
> **Prior work.** Fitting a linear time-series model and training a neural network on its residuals is the classical hybrid of Zhang (2003, *Neurocomputing* 50:159–175), and many papers followed it. The federated, personalized version has **not been searched for yet**. Until that search is done, nothing about this design may be called new. At most it is "an ARIMA–neural residual hybrid trained with personalized FL across Indian states".

## Idea

SARIMA is the strongest method on record, at 228.1 crore validation equal-state RMSE. Every neural model so far is worse.

v3 keeps SARIMA as the forecast and lets a federated GRU learn only a multiplicative correction to it:

\[ \hat y_{i,m} = \hat y^{S}_{i,m}\,\exp\!\big(\lambda_i\,\hat r_{i,m}\big), \qquad r_{i,t} = \log y_{i,t} - \log \hat y^{S}_{i,t} \]

where:
- \(\hat y^{S}\) is the state's one-step SARIMA forecast with its frozen order;
- \(r\) is SARIMA's one-step error in logs;
- \(\hat r\) is the GRU's forecast of that error;
- \(\lambda_i \in [0,1]\) shrinks the correction toward plain SARIMA.

Why the log ratio: it has no units. A 5% miss is the same number in Maharashtra and in Lakshadweep, so the states' models can be averaged without one state's scale dominating. This is the "scale issue" that hurts the level GRUs.

## What can and cannot be proven

- **Identity, not a theorem.** The forecast error of v3 is the error of its residual forecast, scaled by the SARIMA forecast:

  \[ \hat y_{i,m} - y_{i,m} = \hat y^{S}_{i,m}\big(e^{\lambda_i \hat r_{i,m}} - e^{r_{i,m}}\big) \]

  So v3 beats SARIMA at a point if and only if \(\lambda_i\hat r_{i,m}\) is closer to \(r_{i,m}\) than 0 is. Squared error in log space has the same structure. With \(\lambda_i = 0\), v3 *is* SARIMA.

- **Shrinkage.** For a fixed residual forecast, the \(\lambda\) that minimises \(\mathrm{E}[(r - \lambda\hat r)^2]\) is \(\mathrm{E}[r\hat r]/\mathrm{E}[\hat r^2]\), clipped to \([0,1]\). The slope at \(\lambda = 0\) is \(-2\,\mathrm{E}[r\hat r]\). So whenever \(\mathrm{E}[r\hat r] > 0\), a small enough \(\lambda > 0\) improves on SARIMA *in expected squared log error*. That is a standard least-squares fact, not evidence. It also says nothing directly about RMSE in crore. Whether \(\mathrm{E}[r\hat r] > 0\) holds out of sample is an empirical question, and only the validation run can answer it.

- **No proof is available that v3 reaches any RMSE target.** The thesis must not state one.

## Evidence before designing (pre-validation data only)

For each of the 32 states, I took the frozen-order SARIMA fitted on months before April 2024 and computed its in-sample log-ratio residuals, after dropping the first 13 months.

| Check | Result |
|---|---|
| Ljung–Box at lag 12, \(p < 0.05\) | 4 of 32 states |
| Lag-1 autocorrelation | median 0.16 (interquartile range −0.01 to 0.31) |
| Residual standard deviation | median 0.108 (about 11%) |
| Correlation with the month-on-month change in each logged E-Way Bill series | median +0.03 to +0.13; \(\lvert r\rvert > 0.3\) in 1 to 6 states, depending on the series |

The residuals are mostly unpredictable from their own past. The E-Way Bill signal is weak. The realistic expectation is a small gain at best, so the design must be able to fall back to SARIMA (\(\lambda_i \to 0\)). These are in-sample residuals, which are optimistic.

## Method

At each origin \(m\), for each state \(i\):

1. **SARIMA base.** Two fits of the state's frozen order:
   - **Residual fit**, on months before the holdout starts (\(m-6\)). Its one-step-ahead filtered predictions give \(\hat y^{S}_{i,t}\) for the fit months. Running the same filter on through the holdout, with those parameters held fixed, gives \(\hat y^{S}_{i,t}\) for the holdout months.
   - **Forecast fit**, on months before \(m\), exactly as the SARIMA baseline. It gives \(\hat y^{S}_{i,m}\).
   - A filtered prediction for month \(t\) uses only observations before \(t\). Neither fit reads month \(m\), and no parameter used for a holdout residual has seen that holdout.
   - Drop the first 13 months of each contiguous run, while the filter is initialising.
2. **Residual targets.** \(r_{i,t} = \log y_{i,t} - \log \hat y^{S}_{i,t}\) for the fit and holdout months, normalised by the state's fit-month mean and standard deviation.
3. **Inputs.**
   - The existing window of 6 months: lagged GST, the six E-Way Bill series and the month encoding. CPI is excluded so that the April–August 2026 confirmation months stay scorable.
   - The SARIMA log-forecast is added as one more column.
   - Scaling is per state on the fit months, as for the other GRUs.
4. **Federated training.** The frozen APFL rule (step 1, \(\alpha\) starting at 0.5) on the residual targets, using the frozen architecture, epochs, rounds and patience. Round selection uses equal-state holdout RMSE **in crore** of the corrected forecast.
5. **Shrinkage \(\lambda_i\).** (See choice 2 below.)
6. **Forecast.** Apply the formula in "Idea" at the origin month.

No hyperparameter is searched. Everything is the frozen selection, except the choices below, which are fixed before any run.

## Comparators and ablations (same origins and seeds)

| ID | Variant | What it isolates |
|---|---|---|
| B0 | SARIMA (\(\hat r = 0\)) | The base. v3 must beat it |
| B1 | Local residual GRU, no federation | Whether federation helps |
| B2 | FedAvg residual GRU, no personalization | Whether personalization helps |
| B3 | Per-state ridge regression on the same inputs, last window step only | Whether a deep model is needed at all |
| B4 | v3 with GST-only inputs (no E-Way Bills) | Whether E-Way Bills carry the signal |
| B5 | v3 with \(\lambda_i = 1\) | What shrinkage contributes |

## Decision rule (fixed before any run)

1. **"Residual correction beats SARIMA"** only if v3 has lower equal-state RMSE than B0 on validation, in the seed mean and in at least 2 of 3 seeds. The same must hold on the confirmation months once the amendment exists.
2. **"Federation helps"** only if v3 also beats B1 under the same rule. **"Deep learning is needed"** only if v3 beats B3.
3. If v3 does not beat SARIMA, the thesis reports that the GRU found no usable signal in SARIMA's errors.
4. The test year is a second look only.

## Choices agreed on 30 October 2026

1. **Residual source: filtered one-step SARIMA predictions,** not monthly out-of-sample refits.
   - This gives about 24 fit targets per state at the first origin and 35 at the last (Lakshadweep: 14 at the first).
   - Fit-month residuals are slightly optimistic, because the residual fit's parameters have seen those months.
   - Refinement made while formalizing: the residual fit ends before the holdout (step 1 above), so the holdout residuals used to choose \(\lambda_i\) and the round are out of sample with respect to the parameters.
2. **Shrinkage:** \(\lambda_i\) is chosen per state on its 6 holdout months from {0, 0.5, 1}. B5 (\(\lambda_i = 1\)) is the ablation.
3. **Inputs:** GST, E-Way Bills and the SARIMA log-forecast column, with no CPI. B4 is the GST-only ablation.
4. **Federated rule:** the frozen APFL rule. B1 (local) and B2 (FedAvg) are the ablations.

Before implementation: run a literature search for federated or personalized hybrid statistical-plus-neural residual forecasting.

## Implementation outline (after agreement)

- `fl_experiment/residual.py`:
  - a per-state, per-origin SARIMA base that reuses `fit_sarima` and the frozen orders;
  - residual arrays in the same client-dict shape that `federated.py` already consumes. The existing FedAvg, FedProx and APFL code then runs unchanged on residual targets;
  - a decode step that applies \(\lambda_i\) and the SARIMA forecast.
- No change to `data_pipeline/`. Residuals must be recomputed at each origin from data before it. A precomputed residual matrix in the pipeline would leak later observations into earlier origins.
- `fl_experiment/run_residual.py`, tests (λ = 0 reproduces SARIMA exactly, no origin leakage, decode is the inverse of the residual), a validation note, and a Flower check through the existing `flower_rolling` client.
