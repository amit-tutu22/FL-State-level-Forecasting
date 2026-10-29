# FH-PFL v2 on the validation origins, 30 October 2026

Method: [FH-PFL v2, federated top-down shares](../methods/fh_pfl_v2.md). It is scored here once, with the frozen settings in [2026-10-29-selection.json](2026-10-29-selection.json) and fold hash `47573acc…ddc7ae18`. v2 adds no hyperparameter, and nothing was tuned on these numbers. The setup:

- **Panel:** assumed 45-day lag, CPI base 2012, GST-only inputs.
- **Origins:** April 2024 to March 2025, with the same 32 states at all 12 origins, 384 origins in total. The test year was not read.
- **Seeds:** 0, 1 and 2. In-process wall time was 146 seconds.

```
python -m fl_experiment.run_shares --split val \
    --selection docs/experiments/2026-10-29-selection.json \
    --source panel --allow-assumed-vintages --panel data/reports/lagged45-2012/panel.csv
```

## National anchor

The national SARIMA order was chosen by AIC before April 2024 and then frozen. It is `(0,1,1)(0,1,1,12)`, fitted on the shared months from July 2021, and it never fell back to the seasonal naive. Over the 12 origins its one-step error on \(N_m\) is:

| Anchor | RMSE (crore) | MAPE |
|---|---:|---:|
| SARIMA | 3,593 | 2.26% |
| Seasonal naive | 12,252 | — |

## Result

Equal-state figures, mean ± standard deviation across seeds. RMSE and MAE are in INR crore. The reference rows are the frozen validation numbers from [2026-10-29-protocol-validation.md](2026-10-29-protocol-validation.md).

| Method | RMSE | MAE | MAPE | MASE, equal-state |
|---|---:|---:|---:|---:|
| SARIMA (reference) | 228.1 | 185.9 | 17.28 | 0.75 |
| Oracle anchor: v2 shares × actual \(N_m\) (bound only) | 230.8 ± 11.0 | 183.2 | 11.42 | 0.72 |
| **A1, independent shares** | **240.1 ± 1.4** | 189.5 | 12.55 | 0.74 |
| **A2, post-hoc normalisation** | **240.1 ± 1.6** | 190.4 | 12.61 | 0.75 |
| A5, v2 without the national input | 253.3 ± 4.5 | 196.5 | 10.61 | 0.74 |
| **FH-PFL v2** | **255.9 ± 8.9** | 197.8 | 11.84 | 0.75 |
| A3, classical top-down (no learning) | 257.3 | 200.9 | 14.61 | 0.82 |
| APFL, step 1 (reference) | 416.3 ± 13.1 | 293.6 | 10.50 | 0.95 |
| Seasonal naive (reference) | 427.2 | 381.3 | 18.85 | 1.00 |
| A4, v2 with a seasonal-naive anchor | 433.2 ± 2.0 | 377.6 | 14.52 | 1.13 |
| Local GRU, GST only (reference) | 446.9 ± 7.6 | 310.5 | 11.23 | 1.00 |

## Decision rule

The rule was fixed in the spec before this run.

1. **Federation claim: not supported.** v2 is worse than A1 by 15.8 crore and worse than A2 by 15.8 crore. That holds in all three seeds; the rule needed v2 to be lower in at least two. v2 beats A1 in only 12 of the 32 states. Coupling the states' training through the secure sums did not help. Learning each state's share privately and normalising afterwards was at least as good, and it had less seed variance. Because the claim fails on validation, no confirmation result can rescue it.
2. **Usefulness against the references:** v2 does not beat state-level SARIMA, at 255.9 against 228.1. Neither does any ablation. A1 and A2 come closest, at 240.1. On equal-state MASE, A1 (0.74) and SARIMA (0.75) are level. Every share-based variant is far ahead of the earlier neural methods: local GRU 447, FedAvg 470, APFL 416.

## What drives the numbers

- **Most of the gain comes from the anchor, not from learning or federation.**
  - A3 multiplies the national SARIMA forecast by a 12-month average share. With no GRU and no federation, it reaches 257.3 crore. That is already level with v2 and 159 crore ahead of APFL.
  - With a seasonal-naive anchor instead (A4), the same shares give 433.2.
  - The national total is far easier to forecast than the states (2.3% MAPE), and the states inherit that.
- **Share errors limit what is reachable.** Even with the actual national total (the oracle row), v2's shares give 230.8, level with SARIMA.
- **The national input column does not clearly help.** A5 (253.3 ± 4.5) and v2 (255.9 ± 8.9) are within each other's seed spread.
- **Where v2 loses to A1:**
  - Gujarat accounts for most of the gap: 734 against 438 crore.
  - Tamil Nadu (587 against 512), Karnataka (453 against 394), West Bengal (374 against 330) and Bihar (248 against 206) also go against v2.
  - v2 is better in Haryana (554 against 600), Uttar Pradesh (558 against 586) and Jharkhand (163 against 184).
- **Untested explanation:** a round of v2 takes five Adam steps against \(c_t\) and \(E_t\) frozen at the start of the round, while all 32 states move at once. These stale coupled steps may overshoot. Nothing here tests that, and changing the update now would be tuning on validation.

Selected rounds (out of at most 20) were 3–20 for v2 and 6–20 for A1.

## Flower agreement

v2 was also run through Flower clients, one simulation per origin and seed (wall time about 10 minutes):

```
python -m fl_experiment.run_shares --split val --engine flower --variants v2 ...
```

| Path | Equal-state RMSE |
|---|---:|
| Flower | 255.3 ± 8.8 |
| In-process | 255.9 ± 8.9 |

The gap is 0.6 crore, inside the seed spread. The two paths differ only in dropout random numbers, and they match to `rtol 1e-5` at dropout 0 (`fl_experiment/tests/test_shares_flower.py`).

Every client upload in the Flower run was a one-dimensional vector with one value per month:
- 39 to 50 values for the fit months;
- 6 for the holdout months;
- 1 for the origin month.

No weights or gradients left a client. The sums were computed in the clear inside the simulation, so this shows what is sent, not that it is protected.

## What this does not say

- It is a retrospective run on an assumed-lag panel, not a real-time backtest.
- State GST is published. "Private" is a simulated setting.
- No secure-aggregation protocol was run, and no privacy guarantee is made.
- The test year and the April–August 2026 confirmation months were not scored. On 30 October 2026 it was decided not to use the confirmation months for v2, because its federation claim had already failed. They are kept unopened for a design that passes validation.
- Other papers already have top-down forecasting with learned proportions, but those methods are centralized. The only new element tested here, federated coupling during training, did not help.
