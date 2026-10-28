# Real-data smoke run — 29 October 2026

First end-to-end run of the GRU models on the official panel. It shows that the pipeline, the per-state windows, the local GRU (#17) and the Flower FedAvg GRU work on real data.

**These are assumed-lag sensitivity results on the validation period.** They are not real-time backtests, nothing has been tuned, and no number here is a thesis claim. The test period (target months from 2025-04) was not evaluated.

## Data

```sh
python -m data_pipeline.acquire --output data/raw/2026-10-29
python -m data_pipeline.normalize --raw data/raw/2026-10-29 --output data/processed/2026-10-29
python -m data_pipeline.panel --processed data/processed/2026-10-29 --output data/reports/strict-2012 --cpi-base 2012
python -m data_pipeline.panel --processed data/processed/2026-10-29 --output data/reports/lagged45-2012 \
    --cpi-base 2012 --mode assumed_lag_latest_vintage --lag-days 45
```

| Item | Value |
|---|---|
| Acquired | 10 GST workbooks, 9 E-Way Bill workbooks, CPI 2012 Current 5,697 / 2024 Current 777 / 2024 Back 144 records |
| Normalized rows | GST 4,264, E-Way Bill 3,702, CPI 6,618 (same as the [audit](../audits/2026-10-29.md)) |
| `manifest.json` SHA-256 | `f863aa75c5cd1cc617a09deb6bcea7023853d9843044525ffe100646c1659437` |
| Strict panel | 3,877 rows, **0 complete** (as expected without original-release vintages) |
| 45-day assumed-lag panel, CPI 2012 | 3,877 rows, **2,858 complete** (matches the audit) |
| Lagged `panel.csv` SHA-256 | `af0a3d2a0b4343108eecf915e2e8f84f847d5cf5ce76c2fc51fcead8989e4f21` |

The raw, processed and panel files are local only (ignored by Git). Rerunning the commands reproduces them from the official sources.

## Clients and windows

- **Setup:** window of 6 months, at least 24 training windows. Validation covers target months 2024-04 to 2025-03, and test starts at 2025-04.
- **Eligible:** 32 states and union territories. Each has 46–54 training windows, 12 validation windows and 12 test windows. Windows never cross a missing month (for example, the April–May 2020 CPI gap).
- **Excluded for too little training history:**
  - `arunachal_pradesh`: 0 training windows;
  - `daman_and_diu_legacy` and `dadra_and_nagar_haveli_legacy`: 10 each;
  - `jammu_and_kashmir_undivided`: 7.

## Runs

Code at commit `991da1a` plus the GRU change in this commit.

```sh
P="--source panel --panel data/reports/lagged45-2012/panel.csv --allow-assumed-vintages"
python -m fl_experiment.local_gru $P --features all
python -m fl_experiment.local_gru $P --features gst_only
python -m fl_experiment.run_fl_simulation $P --features all --rounds 30
```

(In zsh, write the options out instead of using `$P`.)

- **Model:** GRU with hidden size 16 and dropout 0.1, Adam with lr 0.01 and weight decay 1e-4. All runs use seed 0.
- **Local:** up to 500 epochs, with early stopping on validation loss (patience 30).
- **FedAvg:** 30 rounds of 5 local full-batch epochs, with all 32 clients in every round.
- **Compute:** CPU, 1 thread. Each local run took about 1.6 seconds of training time.

## Validation results (INR crore)

Every state has 12 validation windows, so the sample-weighted and equal-state means are the same.

| Model | Inputs | RMSE | MAE | MAPE % | Median state MAPE % |
|---|---|---:|---:|---:|---:|
| Local GRU per state | all (GST, 6 E-Way Bill, CPI) | 407.3 | 276.0 | 10.33 | 7.45 |
| Local GRU per state | GST only | 367.9 | 256.3 | 9.49 | 6.97 |
| FedAvg GRU, 30 rounds | all | 516.7 | 414.6 | 12.74 | 8.60 |

## Observations

- **Added inputs did not help.** GST-only beat all inputs on aggregate, and on MAPE in 22 of 32 states. This fits #17's warning that extra variables don't add independent monthly observations. It is one seed with no tuning, so it is a hypothesis for #26, not a finding.
- **FedAvg was undertrained.** Its validation RMSE was still falling at round 30 (670, 631, 584, 556, 527, 520, 517 at rounds 1, 6, 11, 16, 21, 26 and 30). Each round is 5 local epochs, so the training budget is not comparable to the early-stopped local models. #19 needs comparable, validation-tuned budgets before any local-versus-federated comparison.
- **Small territories have the largest percentage errors.** Lakshadweep (45–66% MAPE on targets of about 1 crore), Andaman & Nicobar, Mizoram, Tripura and Meghalaya are the weakest under every model. The local all-features Lakshadweep model's best validation epoch was epoch 1. These states need attention under #20.
- **No baseline yet.** Without seasonal naive or SARIMA (#16) these errors have no reference point. Each run's `predictions.csv` and `metrics.json`, or `run.json`, are under `fl_experiment/results/real-2026-10-29/` (local only).

## Limitations

- **Revision bias:** the inputs are latest-vintage values with an assumed 45-day lag, not the values that were actually available at the time.
- **Single validation period:** fixed cutoffs, one validation year and one seed. Rolling-origin folds (#15) and multiple seeds (#25) are still to come.
- **No privacy:** federating public data in a simulation provides no privacy guarantee.
