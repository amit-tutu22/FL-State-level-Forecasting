# State-level GST forecasting with federated learning

IIT Patna M.Tech research project: forecast monthly gross GST collections for Indian states and union territories, treating each state as a [Flower](https://flower.ai/) federated learning client, and study personalized federated learning (FedAvg, FedProx, APFL and the proposed FH-PFL) under state heterogeneity.

The work follows the 8-month roadmap in [#30](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/30). The original Semester 3 proposal is archived in [docs/PROPOSAL.md](docs/PROPOSAL.md).

## Status

| Milestone | State |
|---|---|
| M1 — Dataset feasibility | Official GST, E-Way Bill and CPI pipeline acquired, audited and tested ([#13](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/13) closed; [#14](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/14) remaining items open) |
| M2 — Forecasting baselines | Not started ([#15](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/15)–[#17](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/17)) |
| M3 — Flower federated baselines | Flower startup repaired with one disjoint client per state ([#18](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/18)); FedAvg/FedProx benchmarks not started |
| M4 onwards — review, personalization, evaluation | Not started |

`fl_experiment/` runs a FedAvg simulation over one client per state. So far it has run only on the legacy synthetic data from `data-generator/`, which is for software smoke tests and produces no thesis evidence. See [fl_experiment/README.md](fl_experiment/README.md).

No forecasting results or privacy guarantees are claimed yet. Federation over public data is a simulation, not a deployed privacy mechanism.

## Quickstart

Requires Python 3.11+ (pinned in `.python-version`) and `curl`. [uv](https://docs.astral.sh/uv/) is recommended; some Homebrew Python builds have a broken `pyexpat` that crashes pip.

```sh
scripts/setup_venv.sh
. .venv/bin/activate
python -m unittest discover -s tests -v
```

Rebuild the dataset from official sources (each command needs a new, empty output directory):

```sh
python -m data_pipeline.acquire --output data/raw/my-vintage
python -m data_pipeline.normalize --raw data/raw/my-vintage --output data/processed/my-vintage
python -m data_pipeline.panel --processed data/processed/my-vintage --output data/reports/strict --cpi-base 2012
```

Raw and processed exports stay out of Git under `data/`.

Federated experiments additionally need Flower and PyTorch:

```sh
scripts/setup_venv.sh --fl
python -m fl_experiment.run_fl_simulation --rounds 3
```

## Data definitions that matter

- **Target:** `target_gst_excluding_cess_crore` (CGST + SGST + IGST, domestic, gross of refunds, INR crore) is the historically consistent outcome. The published total includes cess only through FY2024–25, so it is kept separately and is not a consistent series.
- **Inputs:** six E-Way Bill features (counts and assessable values for intra-state, inter-state outward and inter-state inward flows) and CPI General Combined. CPI bases 2012 and 2024 are never spliced.
- **Availability:** strict mode only uses values known to be published by the forecast date. Because the downloaded files are current revisions without original release dates, strict mode currently yields zero complete historical rows. The opt-in `assumed_lag_latest_vintage` mode (45-day lag) yields 2,858 rows on CPI base 2012 and is a retrospective sensitivity analysis, not a real-time backtest.

Details: [docs/REAL_DATA_PIPELINE.md](docs/REAL_DATA_PIPELINE.md) and the [29 October 2026 audit](docs/audits/2026-10-29.md).

## Repository layout

```text
data_pipeline/          Official-source acquisition, normalization, geography mapping, panel joins, PDF checks
tests/                  Offline tests with official regression fixtures
docs/                   Pipeline guide, dated audits, archived proposal
fl_experiment/          Flower FedAvg simulation, one disjoint client per state
data-generator/         Legacy synthetic data generator (smoke tests only)
scripts/setup_venv.sh   Python 3.11 environment setup (--fl adds Flower/PyTorch)
```

CI runs the pipeline tests (`data-pipeline.yml`), Pylint on `data_pipeline/` and `tests/` (`pylint.yml`), and Pylint, tests and a two-round ten-state simulation for `fl_experiment/` (`fl-smoke.yml`).

## License

See [LICENSE](LICENSE).
