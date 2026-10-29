# Flower state-client experiments

A compact GRU forecaster trained two ways on identical states, windows and cutoffs:

- locally, as one model per state ([#17](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/17), `local_gru`);
- with Flower FedAvg, using one disjoint client per state ([#18](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/18), `run_fl_simulation`).

No result here is thesis evidence yet. Real-panel records: the [fixed-split smoke](../docs/experiments/2026-10-29-real-data-smoke.md) and the [rolling-origin comparison](../docs/experiments/2026-10-29-rolling-origin.md).

The GRU (`model.py`) has hidden size 16 and dropout 0.1. It reads a `(window, features)` sequence and predicts from its final hidden state.

## Setup

```sh
scripts/setup_venv.sh --fl
. .venv/bin/activate
python -m unittest discover -s fl_experiment/tests -t . -v
```

Run every command from the repository root. Pinned versions are in `requirements-fl.txt` (Flower 1.11.1, PyTorch 2.15.0, statsmodels 0.14.6, Python 3.11).

## Clients and splits

- **Ownership:** each eligible state is exactly one client and owns all of its rows. Nothing is resampled or shared, and `check_disjoint` fails the run if a state appears twice or its periods overlap.
- **Chronology:** rows are split by target month: train before `--val-start`, validation until `--test-start`, test after. Defaults are 2022-01 / 2023-01 for synthetic data and 2024-04 / 2025-04 for the real panel.
- **Windows:** the window for target month m stacks the input rows of the `--window` (default 6) target months ending at m. Each row only uses information available at its own forecast origin. A window must have every month present, so it never crosses a gap. Splits are assigned by the window's target month.
- **Preprocessing:** inputs and target are log-scaled, then standardized with the state's own training-period mean and standard deviation. Metrics are reported in INR crore after inverting this.
- **Inputs:** `--features all` (default) or `--features gst_only`, which keeps the lagged GST inputs plus the month encoding. This supports the GST-only versus multivariate comparison in #17.
- **Evaluation:** local early stopping and federated rounds both use validation. The test period is used only with `--final-test` in the simulation, which should wait until configurations are frozen (#25).
- **Exclusions:** states with fewer than `--min-train` (default 24) training windows, or no validation or test windows, are excluded and listed with a reason.
- **Aggregation:** the run record contains sample-weighted and equal-state means plus per-state metrics for every round.

### Data sources

- `--source synthetic` (default): the legacy `data/state_budget_gst.csv`, for software smoke tests only. Inputs are the previous month's and same-month-last-year GST plus a target-month encoding. The generator's `sgst_share`/`cgst_share` columns are target-derived and are not used.
- `--source panel --panel PATH`: a `data_pipeline.panel` output loaded through `data_pipeline.clients.load_clients`. Inputs are the lagged GST outcome, six E-Way Bill features and CPI. Strict panels currently have no complete rows. An assumed-lag panel needs `--allow-assumed-vintages` and is a retrospective sensitivity analysis only (see [REAL_DATA_PIPELINE.md](../docs/REAL_DATA_PIPELINE.md)).

The earlier IID, label-skew, feature-skew and overlapping quantity-skew splits were removed. The primary experiment uses natural state ownership. Any synthetic stress test added later must be a separately documented experiment.

## Local GRU per state

```sh
python -m fl_experiment.local_gru
python -m fl_experiment.local_gru --source panel \
    --panel data/reports/lagged45-2012/panel.csv --allow-assumed-vintages --features gst_only
```

Each state trains on its own windows with full-batch Adam, early stopping on validation loss (`--patience`, up to `--epochs`) and the best weights restored. The output directory contains:

- `predictions.csv`: train and validation predictions in INR crore;
- `metrics.json`: settings, seeds, best and run epochs, per-state and aggregate validation metrics, window counts and compute details.

## Rolling-origin baselines

```sh
python -m fl_experiment.baselines --order-grid small --epochs 20
python -m fl_experiment.baselines --source panel --allow-assumed-vintages \
    --panel data/reports/lagged45-2012/panel.csv
```

For each target month in the validation year, every model is refit using only earlier months. The test months are written into `folds.json` and not scored. SARIMA picks one order per state by AIC before the validation year (`--order-grid full` is 12 candidates; `small` is the fast check) and falls back to the seasonal naive if a fit fails. The local GRU early-stops on the six months before the origin. `--gru-features` defaults to both `all` and `gst_only`.

The output directory contains `folds.json`, `predictions.csv` (one row per model and origin) and `metrics.json` (RMSE, MAE, MAPE and MASE, per state and pooled). The first real-panel table is [the rolling-origin record](../docs/experiments/2026-10-29-rolling-origin.md).

## Simulation (single machine)

```sh
python -m fl_experiment.run_fl_simulation --rounds 5
python -m fl_experiment.run_fl_simulation --source panel \
    --panel data/reports/lagged45-2012/panel.csv --allow-assumed-vintages --rounds 30
```

Each run writes `fl_experiment/results/<source>-<timestamp>/`, or `--output-dir`. It contains `cache/` (client arrays and a manifest of states, months and exclusions) and `run.json` (configuration, data manifest, per-round losses and metrics). Results are ignored by Git.

## Separate server and client processes

```sh
python -m fl_experiment.run_fl_simulation --prepare-only --output-dir /tmp/fl-run
python -m fl_experiment.server --rounds 5 --min-clients 10 --address 127.0.0.1:8080
# in other terminals, one per state (partition ids 0..N-1 follow the manifest order)
python -m fl_experiment.client --cache /tmp/fl-run/cache --partition-id 0 --server 127.0.0.1:8080
```

The server waits until `--min-clients` clients connect, and every round uses all of them.

## Reference

`python -m fl_experiment.centralized_baseline` trains the same model on all states' pooled training rows and reports per-state validation error. Pooling raw rows is exactly what federation avoids, so this is a reference point only.

Federating public data is a simulation. It provides no formal privacy guarantee.
