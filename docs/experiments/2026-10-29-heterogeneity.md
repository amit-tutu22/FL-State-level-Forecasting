# State heterogeneity, 29 October 2026

Issue [#20](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/20). This note relates each state's GST series to which method does well there. It reuses the frozen validation run ([protocol-validation](2026-10-29-protocol-validation.md)) and the one-shot test run ([protocol-test](2026-10-29-protocol-test.md)). It does not train anything, and it does not change a selection.

Command:

```sh
python -m fl_experiment.heterogeneity --panel data/reports/lagged45-2012/panel.csv \
    --validation fl_experiment/results/real-2026-10-29/protocol/metrics.json \
    --test fl_experiment/results/real-2026-10-29/protocol-test/metrics.json
```

## Descriptors

All descriptors use the target series (`target_gst_excluding_cess_crore`) before April 2024.

| Descriptor | Definition |
|---|---|
| Mean level | Mean monthly GST, INR crore |
| Volatility | Standard deviation of month-on-month changes in log GST |
| Seasonal strength | Share of detrended log variance explained by month-of-year means |
| History | Months observed before April 2024 |
| 2020 dip | One minus April–June 2020 GST over April–June 2019 |

History runs from 61 to 64 months for all 32 states. With so little spread, it cannot explain a difference between states, and it is reported only for completeness.

## Outcomes

For each state, the three outcomes are relative RMSE differences, using the mean across seeds:

- **APFL against local**: APFL RMSE over local GST-only GRU RMSE, minus one. Negative means personalization helped.
- **FedAvg against local**: the same ratio for FedAvg.
- **SARIMA against best neural**: SARIMA RMSE over the lowest neural RMSE in that state, minus one. Negative means SARIMA was better.

## Spearman rank correlations, 32 states

These are descriptive. With 32 states, a coefficient near ±0.3 is easily produced by chance. No test is claimed.

| Descriptor | APFL vs local, val | test | FedAvg vs local, val | test | SARIMA vs best neural, val | test |
|---|---:|---:|---:|---:|---:|---:|
| Mean level | −0.18 | −0.61 | +0.10 | −0.31 | −0.59 | −0.48 |
| Volatility | −0.03 | +0.18 | −0.23 | +0.03 | +0.23 | +0.12 |
| Seasonal strength | +0.29 | +0.25 | +0.12 | +0.20 | −0.07 | −0.18 |
| History | −0.03 | −0.24 | +0.08 | −0.10 | −0.35 | −0.44 |
| 2020 dip | −0.05 | −0.26 | +0.01 | −0.16 | −0.09 | −0.17 |

## What the states show

- **SARIMA's lead is a large-state result.** The SARIMA-against-best-neural ratio falls with mean level in both years (−0.59 and −0.48). In the 16 largest states, the median ratio is −35% on validation and −34% on test. In the 16 smallest it is −17% and −22%. A neural method beats SARIMA in both years in one state only, Lakshadweep, whose level is about 2 crore.
- **APFL helps the larger states more consistently.** APFL beats the local GRU in both years in 11 states: Andhra Pradesh, Gujarat, Kerala, Lakshadweep, Madhya Pradesh, Maharashtra, Punjab, Telangana, Uttar Pradesh, Uttarakhand and West Bengal. It loses in both years in Assam, Delhi, Goa and Tripura. In the 16 largest states the median change is −6.6% in both years. In the 16 smallest it is −5.7% on validation and +6.5% on test. Himachal Pradesh, Goa, Puducherry, Andaman & Nicobar and Sikkim are each 16% to 28% worse than local on the test year.
- **FedAvg helps in a different, smaller set.** FedAvg beats the local GRU in both years in nine states, mostly the ones where APFL also helps, plus Chandigarh. Its correlation with level changes sign between years, so there is no stable pattern to report.
- **Seasonality goes the wrong way for personalization.** States with stronger month-of-year patterns gain less from APFL in both years (+0.29 and +0.25). Manipur, Nagaland and Mizoram have the strongest seasonal shares, 0.31 to 0.41, and are also among the smallest and most volatile series. That makes size, seasonality and volatility hard to separate with 32 states.
- **The 2020 dip does not separate states.** Its correlations are weak and change size between years.

## Limits

Assumed 45-day lag panel, CPI base 2012, and retrospective sensitivity data only. Each outcome compares two models that were each trained on noisy small samples. The seed spread for APFL is about 13 crore on validation and 30 on test, which is the same order as many state-level differences. These patterns are for describing the states, and they do not support choosing a method per state. Choosing that way after seeing the test year would be a second look.
