# Test evaluation, 29 October 2026

This run reads the frozen file [2026-10-29-selection.json](2026-10-29-selection.json) and scores only target months from April 2025 through March 2026. The fold hash matches the validation freeze (`47573acc0b043ecaa59dfcb8c59bca65915fbc12da63755f4bc64513ddc7ae18`). The feature set, FedProx strength and APFL step were not searched again. Training for a test month may use earlier months, including the validation year. That is the expanding window.

The panel is still the assumed 45-day lag, CPI base 2012. There are 32 states and 12 test months, 384 origins. Neural numbers are the mean ± standard deviation across seeds 0, 1 and 2. Wall time 188 seconds. The validation table is [2026-10-29-protocol-validation.md](2026-10-29-protocol-validation.md).

## Test table

RMSE and MAE are INR crore. MAPE is a percent. Each state has 12 origins, so pooled and equal-state RMSE, MAE and MAPE match. MASE does not.

| Method | RMSE | MAE | MAPE | MASE, pooled | MASE, equal-state |
|---|---:|---:|---:|---:|---:|
| Seasonal naive | 392.0 | 328.3 | 15.13 | 1.00 | 1.00 |
| SARIMA | 287.6 | 228.6 | 15.64 | 0.70 | 0.89 |
| APFL, step 1 | 509.1 ± 29.7 | 371.4 ± 41.3 | 15.07 ± 0.40 | 1.13 ± 0.13 | 1.21 ± 0.07 |
| FedAvg | 536.5 ± 15.3 | 378.6 ± 12.3 | 16.00 ± 0.26 | 1.15 ± 0.04 | 1.23 ± 0.03 |
| Local GRU, GST only | 539.8 ± 5.4 | 379.2 ± 5.6 | 13.97 ± 0.07 | 1.15 ± 0.02 | 1.17 ± 0.02 |
| FedProx, μ = 0.01 | 540.3 ± 20.1 | 381.1 ± 15.5 | 15.99 ± 0.24 | 1.16 ± 0.05 | 1.23 ± 0.03 |
| Pooled GRU | 615.9 ± 30.7 | 460.6 ± 30.2 | 18.44 ± 0.26 | 1.40 ± 0.09 | 1.43 ± 0.01 |

SARIMA remains the RMSE reference, at 288 crore against 392 for the seasonal naive. Every neural method is worse than the seasonal naive on RMSE. APFL, which was the best neural method on validation and slightly ahead of the naive forecast there (416 against 427), is at 509 on the test year. Its equal-state MASE is 1.21. The validation gap does not carry over.

APFL's RMSE is lower than the seasonal naive in 6 of 32 states (Karnataka, Kerala, Lakshadweep, Puducherry, Sikkim, Tamil Nadu) and lower than SARIMA in one, Lakshadweep, where both errors are about 2 crore. FedAvg remains slightly ahead of FedProx and does not beat the local GRU. The pooled GRU is again the weakest.

The local GRU has the lowest MAPE of the table, 13.97%. That does not change the selection, which was fixed as equal-state RMSE before this run. The GRU is trained in log space. SARIMA is fit in crore.

## Where the error sits

Maharashtra again has the largest RMSE: about 3,447 crore for APFL, 2,559 for the seasonal naive, and 1,195 for SARIMA. Lakshadweep has the largest MAPE for every method, about 88% for APFL and 195% for SARIMA, on a level of about 1 crore.

| State | Naive | SARIMA | Local GRU | Pooled | FedAvg | FedProx 0.01 | APFL step 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Andaman & Nicobar | 10 | 6 | 13 | 11 | 13 | 13 | 16 |
| Andhra Pradesh | 260 | 308 | 329 | 462 | 360 | 360 | 320 |
| Assam | 150 | 146 | 190 | 251 | 194 | 194 | 205 |
| Bihar | 204 | 115 | 313 | 453 | 345 | 346 | 331 |
| Chandigarh | 42 | 47 | 51 | 58 | 48 | 48 | 50 |
| Chhattisgarh | 159 | 207 | 290 | 444 | 327 | 328 | 291 |
| Delhi | 871 | 871 | 1,446 | 1,666 | 1,541 | 1,549 | 1,540 |
| Goa | 47 | 44 | 59 | 78 | 66 | 66 | 73 |
| Gujarat | 869 | 721 | 1,406 | 1,085 | 982 | 990 | 1,075 |
| Haryana | 1,003 | 717 | 1,345 | 1,075 | 1,411 | 1,423 | 1,215 |
| Himachal Pradesh | 137 | 94 | 122 | 190 | 186 | 185 | 156 |
| Jharkhand | 218 | 203 | 335 | 369 | 355 | 353 | 310 |
| Karnataka | 1,462 | 640 | 1,468 | 1,299 | 1,676 | 1,694 | 1,379 |
| Kerala | 287 | 179 | 265 | 283 | 187 | 190 | 219 |
| Lakshadweep | 2 | 3 | 2 | 2 | 2 | 2 | 2 |
| Madhya Pradesh | 329 | 301 | 525 | 587 | 462 | 462 | 461 |
| Maharashtra | 2,559 | 1,195 | 3,684 | 5,582 | 3,828 | 3,867 | 3,447 |
| Manipur | 22 | 24 | 30 | 31 | 29 | 30 | 30 |
| Meghalaya | 48 | 41 | 47 | 64 | 58 | 58 | 52 |
| Mizoram | 13 | 11 | 15 | 15 | 15 | 15 | 15 |
| Nagaland | 19 | 14 | 18 | 19 | 22 | 22 | 20 |
| Odisha | 206 | 270 | 380 | 517 | 370 | 375 | 355 |
| Puducherry | 40 | 33 | 33 | 49 | 41 | 40 | 40 |
| Punjab | 236 | 224 | 295 | 308 | 245 | 248 | 278 |
| Rajasthan | 359 | 333 | 505 | 660 | 550 | 550 | 524 |
| Sikkim | 135 | 92 | 104 | 159 | 133 | 133 | 120 |
| Tamil Nadu | 1,046 | 789 | 933 | 934 | 903 | 917 | 944 |
| Telangana | 483 | 374 | 623 | 481 | 513 | 519 | 529 |
| Tripura | 18 | 21 | 27 | 28 | 31 | 31 | 29 |
| Uttar Pradesh | 592 | 600 | 1,351 | 1,327 | 1,218 | 1,225 | 1,261 |
| Uttarakhand | 248 | 228 | 337 | 354 | 344 | 343 | 321 |
| West Bengal | 471 | 351 | 733 | 871 | 712 | 714 | 684 |

State RMSE is the mean across seeds, rounded to the nearest crore.

## What this does not say

The settings were chosen on one validation year. This test pass does not retune them, and it is not a license to try another feature set or another `mu` on these months. The panel is a retrospective sensitivity run, not a real-time backtest. FH-PFL is not in the table. A Flower simulation on public series is not a privacy guarantee.
