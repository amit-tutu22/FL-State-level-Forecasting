# Flower state-client experiments

FedAvg over one Flower client per state ([#18](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/18)). This is the federated scaffolding for later baselines: the model is a small feed-forward network, not the GRU planned in #17, and no result here is thesis evidence.

## Setup

```sh
scripts/setup_venv.sh --fl
. .venv/bin/activate
python -m unittest discover -s fl_experiment/tests -t . -v
```

Run every command from the repository root. Pinned versions are in `requirements-fl.txt` (Flower 1.11.1, PyTorch 2.15.0, Python 3.11).

## Clients and splits

- **Ownership:** each eligible state is exactly one client and owns all of its rows. Nothing is resampled or shared, and `check_disjoint` fails the run if a state appears twice or its periods overlap.
- **Chronology:** rows are split by target month: train before `--val-start`, validation until `--test-start`, test after. Defaults are 2022-01 / 2023-01 for synthetic data and 2024-04 / 2025-04 for the real panel.
- **Preprocessing:** inputs and target are log-scaled, then standardized with the state's own training-period mean and standard deviation. Metrics are reported in INR crore after inverting this.
- **Evaluation:** rounds evaluate on validation. The test period is used only with `--final-test`, which should wait until configurations are frozen (#25).
- **Exclusions:** states with fewer than `--min-train` (default 24) training rows, or no validation or test rows, are excluded and listed with a reason.
- **Aggregation:** the run record contains sample-weighted and equal-state means plus per-state metrics for every round.

### Data sources

- `--source synthetic` (default): the legacy `data/state_budget_gst.csv`, for software smoke tests only. Inputs are the previous month's and same-month-last-year GST plus a target-month encoding. The generator's `sgst_share`/`cgst_share` columns are target-derived and are not used.
- `--source panel --panel PATH`: a `data_pipeline.panel` output loaded through `data_pipeline.clients.load_clients`. Inputs are the lagged GST outcome, six E-Way Bill features and CPI. Strict panels currently have no complete rows. An assumed-lag panel needs `--allow-assumed-vintages` and is a retrospective sensitivity analysis only (see [REAL_DATA_PIPELINE.md](../docs/REAL_DATA_PIPELINE.md)).

The earlier IID, label-skew, feature-skew and overlapping quantity-skew splits were removed. The primary experiment uses natural state ownership. Any synthetic stress test added later must be a separately documented experiment.

## Simulation (single machine)

```sh
python -m fl_experiment.run_fl_simulation --rounds 5
python -m fl_experiment.run_fl_simulation --source panel \
    --panel data/reports/lagged-2012/panel.csv --allow-assumed-vintages
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
