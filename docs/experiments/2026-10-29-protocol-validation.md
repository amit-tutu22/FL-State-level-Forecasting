# Matched validation comparison, 29 October 2026

Scored setup: [PROTOCOL.md](../PROTOCOL.md). Panel: assumed 45-day lag, CPI base 2012, target `target_gst_excluding_cess_crore`. Validation origins are April 2024 through March 2025, 32 states and 12 months, 384 origins. The test year was not read. Wall time 321 seconds. Seeds 0, 1 and 2. The standard deviation below is across those seeds. Seasonal naive and SARIMA are deterministic, so their standard deviation is zero, and their orders are the ones already frozen before April 2024 (29 states `(0,1,1)(0,1,1,12)`, Goa, Rajasthan and Tamil Nadu `(1,1,1)(0,1,1,12)`, no fallbacks).

The selection score is the mean, across seeds, of equal-state RMSE in INR crore. Each state has 12 origins, so the pooled and equal-state RMSE, MAE and MAPE are the same number. MASE is not: pooled MASE weights large states more than the average of the 32 state MASE values.

## What validation chose

These three choices are written in [2026-10-29-selection.json](2026-10-29-selection.json), with fold hash `47573acc0b043ecaa59dfcb8c59bca65915fbc12da63755f4bc64513ddc7ae18`.

- Feature set `gst_only`. The local GRU's equal-state RMSE is 446.9 with GST history and 470.4 with GST, E-Way Bills and CPI.
- FedProx `mu = 0.01`, the best of 0.01, 0.1 and 1. All three are worse than FedAvg, which is `mu = 0` and is not a second candidate.
- APFL step size 1.0, starting from `alpha = 0.5`. It beats step size 0.1 on the selection score (416.3 against 428.7).

## Validation table

Mean ± standard deviation. RMSE and MAE are INR crore. MAPE is a percent.

| Method | RMSE | MAE | MAPE | MASE, pooled | MASE, equal-state |
|---|---:|---:|---:|---:|---:|
| Seasonal naive | 427.2 | 381.3 | 18.85 | 1.00 | 1.00 |
| SARIMA | 228.1 | 185.9 | 17.28 | 0.49 | 0.75 |
| APFL, step 1 | 416.3 ± 13.1 | 293.6 ± 19.6 | 10.50 ± 0.23 | 0.77 ± 0.05 | 0.95 ± 0.04 |
| APFL, step 0.1 | 428.7 ± 14.7 | 288.4 ± 17.5 | 10.39 ± 0.14 | 0.76 ± 0.05 | 0.94 ± 0.02 |
| Local GRU, GST only | 446.9 ± 7.6 | 310.5 ± 5.6 | 11.23 ± 0.06 | 0.81 ± 0.01 | 1.00 ± 0.02 |
| FedAvg | 469.9 ± 7.7 | 342.4 ± 9.6 | 11.18 ± 0.13 | 0.90 ± 0.03 | 1.00 ± 0.02 |
| Local GRU, all inputs | 470.4 ± 8.0 | 310.4 ± 6.4 | 11.55 ± 0.23 | 0.81 ± 0.02 | 1.01 ± 0.02 |
| FedProx, μ = 0.01 | 474.6 ± 13.7 | 346.3 ± 15.1 | 11.22 ± 0.13 | 0.91 ± 0.04 | 1.00 ± 0.01 |
| FedProx, μ = 0.1 | 503.6 ± 7.2 | 363.6 ± 6.2 | 11.47 ± 0.15 | 0.95 ± 0.02 | 1.03 ± 0.02 |
| FedProx, μ = 1 | 521.0 ± 8.8 | 377.9 ± 5.4 | 11.69 ± 0.13 | 0.99 ± 0.01 | 1.06 ± 0.01 |
| Pooled GRU | 585.3 ± 19.7 | 444.7 ± 27.4 | 14.34 ± 0.35 | 1.17 ± 0.07 | 1.20 ± 0.01 |

SARIMA remains the RMSE reference. The best neural setting, APFL with step 1, is close to the seasonal naive on equal-state RMSE (416 against 427) and below it on equal-state MASE (0.95). FedAvg does not beat the local GST-only GRU. A larger FedProx penalty makes the error worse. Pooling every state into one GRU is the weakest of the set.

APFL's RMSE is lower than the seasonal naive in 15 of 32 states and lower than SARIMA in 5. Those five are Andaman & Nicobar, Chandigarh, Himachal Pradesh, Lakshadweep and Puducherry. The crore gains of SARIMA remain in the large states.

## Where the error sits

Maharashtra has the largest RMSE for every method: about 3,071 crore for APFL step 1, 3,192 for the seasonal naive, and 1,085 for SARIMA. Lakshadweep has the largest MAPE: about 44% for APFL, 286% for the seasonal naive, and 304% for SARIMA, on a level of about 1 crore. RMSE is decided by the large states. MAPE is decided by the small ones. The GRU loss is in log space, which is closer to a percentage goal, and that is why the neural MAPE can look better while the crore RMSE does not.

| State | Naive | SARIMA | Local GRU | Pooled | FedAvg | FedProx 0.01 | APFL step 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Andaman & Nicobar | 13 | 16 | 14 | 11 | 12 | 12 | 13 |
| Andhra Pradesh | 294 | 332 | 423 | 522 | 360 | 361 | 388 |
| Assam | 132 | 87 | 166 | 216 | 145 | 146 | 173 |
| Bihar | 226 | 143 | 288 | 364 | 295 | 296 | 271 |
| Chandigarh | 35 | 35 | 27 | 26 | 25 | 25 | 28 |
| Chhattisgarh | 307 | 246 | 314 | 498 | 326 | 326 | 296 |
| Delhi | 1,028 | 586 | 1,130 | 1,333 | 1,272 | 1,286 | 1,187 |
| Goa | 71 | 64 | 67 | 92 | 63 | 63 | 70 |
| Gujarat | 1,050 | 503 | 1,046 | 1,146 | 799 | 806 | 770 |
| Haryana | 1,440 | 788 | 980 | 1,084 | 1,412 | 1,427 | 980 |
| Himachal Pradesh | 67 | 81 | 82 | 122 | 102 | 102 | 74 |
| Jharkhand | 163 | 138 | 229 | 334 | 256 | 255 | 261 |
| Karnataka | 1,157 | 440 | 1,028 | 1,150 | 1,345 | 1,363 | 1,085 |
| Kerala | 239 | 153 | 273 | 279 | 212 | 216 | 242 |
| Lakshadweep | 6 | 6 | 1 | 2 | 1 | 1 | 1 |
| Madhya Pradesh | 258 | 195 | 385 | 450 | 344 | 346 | 339 |
| Maharashtra | 3,192 | 1,085 | 3,325 | 5,667 | 3,740 | 3,788 | 3,071 |
| Manipur | 13 | 11 | 17 | 18 | 18 | 18 | 20 |
| Meghalaya | 27 | 25 | 32 | 42 | 40 | 40 | 30 |
| Mizoram | 12 | 17 | 21 | 23 | 22 | 22 | 22 |
| Nagaland | 12 | 9 | 12 | 12 | 13 | 13 | 11 |
| Odisha | 492 | 221 | 307 | 610 | 464 | 470 | 316 |
| Puducherry | 24 | 16 | 16 | 23 | 11 | 12 | 15 |
| Punjab | 264 | 176 | 288 | 349 | 228 | 230 | 261 |
| Rajasthan | 431 | 266 | 461 | 591 | 390 | 391 | 406 |
| Sikkim | 50 | 37 | 46 | 54 | 42 | 42 | 39 |
| Tamil Nadu | 850 | 469 | 764 | 833 | 794 | 809 | 672 |
| Telangana | 347 | 275 | 434 | 450 | 426 | 430 | 425 |
| Tripura | 17 | 13 | 28 | 30 | 29 | 29 | 30 |
| Uttar Pradesh | 939 | 485 | 1,337 | 1,404 | 1,123 | 1,134 | 1,100 |
| Uttarakhand | 164 | 123 | 251 | 336 | 219 | 220 | 233 |
| West Bengal | 350 | 257 | 508 | 659 | 507 | 508 | 492 |

State RMSE is the mean across seeds, rounded to the nearest crore. The local GRU column is the GST-only model. The other neural columns use that same feature set.

## What this does not say

The panel is a retrospective sensitivity run, not a real-time backtest. The test months from April 2025 are still unscored. FH-PFL has no update rule and is not in this table. A Flower simulation on public series is not a privacy guarantee.
