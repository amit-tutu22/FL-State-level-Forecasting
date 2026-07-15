# Technical Architecture Document

**Product:** Federated Learning Framework for Privacy-Preserving Public-Sector Forecasting & Leakage Detection  
**Repository:** FL-State-level-Forecasting  
**Document type:** Technical architecture (product + engineering)  
**Audience:** Product, ML architecture, research, GovTech stakeholders  
**Maturity:** Research prototype (Phases 0–5 complete)  
**Version:** 1.0  
**Date:** July 2026  

---

## 1. Executive summary

This product is a **privacy-preserving federated learning (FL) research platform** for Indian public-sector analytics. It enables collaborative model training across states (or analogous administrative units) **without centralizing raw fiscal, welfare, or education microdata**.

The platform delivers three product capabilities:

1. **Federated time-series forecasting** — next-period prediction of sector KPIs (GST revenue, school enrollment).
2. **Federated welfare leakage detection** — anomaly detection on synthetic DBT/MNREGA-style claims.
3. **Cross-sector reuse** — new government domains plug in via a `TaskSpec` config + dataset, without rewriting the FL loop.
4. **Privacy baseline** — threat model + client-side differential privacy (DP) on model updates, with measured privacy–utility trade-offs.

All experiments currently use **synthetic data** modeled on Indian federal structure, because real fiscal/welfare datasets are restricted. The architecture is designed so real data adapters can replace generators later without changing the FL core.

---

## 2. Product problem & use cases

### 2.1 Problem

India’s public finance and social programs are **federally distributed**. Centre, States, and districts hold sensitive data that cannot be freely pooled due to:

- Privacy and legal constraints (including future DPDP-aligned expectations)
- Political and institutional barriers to raw data sharing
- Non-IID distributions across states (different economies, schemes, seasonality)

Without collaboration, national forecasting and leakage detection remain weak. Centralized ML is often infeasible; purely local models miss national patterns.

### 2.2 Primary use cases

| ID | Use case | Actor | Value |
|----|----------|-------|-------|
| UC-1 | **State GST revenue forecast** | Finance departments / GST analysts | Collaborative next-month revenue prediction without sharing ledgers |
| UC-2 | **Compare FL vs centralized vs local** | Researchers / ML architects | Quantify FL effectiveness under IID and non-IID partitions |
| UC-3 | **Welfare leakage detection** | Scheme monitors (DBT / MNREGA-style) | Flag inflated claims, ghost beneficiaries, duplicate spikes |
| UC-4 | **Cross-sector forecast (education)** | Education planners | Same FL stack predicts enrollment using a different schema |
| UC-5 | **Add a new gov sector** | Product / data eng | Drop CSV + JSON `TaskSpec`; run same runner |
| UC-6 | **Privacy–utility analysis** | Privacy / compliance reviewers | Sweep DP noise vs forecast RMSE; document residual risks |

### 2.3 Non-goals (current release)

- Production deployment to Centre–State networks
- Certified (ε, δ)-DP guarantees or SecAgg/TLS
- Real GST/welfare system integration APIs
- LSTM/XGBoost production model zoo (documented as future)

---

## 3. Product architecture (logical)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Product surfaces                          │
│  Forecast runners │ Leakage runner │ Sector CLI │ Privacy sweep  │
└─────────────┬───────────────┬───────────────┬──────────┬────────┘
              │               │               │          │
┌─────────────▼───────────────▼───────────────▼──────────▼────────┐
│                     Experiment & metrics layer                    │
│   RMSE/MAE/MAPE │ Precision/Recall/F1/AUC │ DP ε (approx)       │
└─────────────┬───────────────┬───────────────┬───────────────────┘
              │               │               │
┌─────────────▼───────────────▼───────────────▼───────────────────┐
│                        FL runtime (Flower)                        │
│   FedAvg │ Ray simulation │ Partition / Sector / Leakage clients │
└─────────────┬───────────────┬───────────────┬───────────────────┘
              │               │               │
┌─────────────▼───────────────▼───────────────▼───────────────────┐
│                     Model & task layer                            │
│  Residual MLP (forecast) │ Autoencoder │ Isolation Forest       │
│  TaskSpec-driven features │ DP clip+noise on updates             │
└─────────────┬───────────────┬───────────────┬───────────────────┘
              │               │               │
┌─────────────▼───────────────▼───────────────▼───────────────────┐
│                     Data layer (synthetic)                        │
│  GST panel │ Welfare claims │ Education enrollment │ Sector JSON │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Trust boundary (product-critical)

```
 State / district node                         Aggregation server
 ┌──────────────────────┐                     ┌──────────────────┐
 │ Raw tables (private) │                     │ FedAvg strategy  │
 │ Local training       │ ── model updates ─► │ Global model     │
 │ Optional DP noise    │    (not raw data)   │ Metrics logs     │
 └──────────────────────┘                     └──────────────────┘
```

Raw records never leave the client by design. Updates may still leak information; Phase 5 DP reduces (does not eliminate) that risk. See `fl_experiment/privacy/THREAT_MODEL.md`.

---

## 4. Technology stack

| Layer | Technology | Role in product |
|-------|------------|-----------------|
| Language | **Python 3.9+** (CI on 3.10) | All experiments and generators |
| FL framework | **Flower (`flwr[simulation]` 1.8–1.11)** | FedAvg server strategy, `NumPyClient`, Ray-backed simulation |
| Distributed sim | **Ray** (via Flower simulation extra) | Virtual clients in one process/cluster for research runs |
| Deep learning | **PyTorch** | Residual MLP forecast model; reconstruction autoencoder |
| Classical ML | **scikit-learn** | `StandardScaler`, Isolation Forest, train/test utilities |
| Data | **pandas / NumPy** | Panels, partitions, metrics arrays |
| Persistence | **CSV / JSON / joblib** | Datasets, sector configs, experiment results, scalers |
| CI | **GitHub Actions** | Smoke pipeline (generate → baselines → FL → phases 2–5) |
| License | **MIT** | Open research reuse |

### 4.1 Explicitly not in stack (yet)

- FedML, TensorFlow Federated
- Opacus / TensorFlow Privacy (formal accountants)
- Secure Aggregation, TLS/mTLS certificates
- MLflow / W&B experiment tracking
- Production message bus / API gateway

---

## 5. System components & repository map

```
FL-State-level-Forecasting/
├── README.md                          # Product status + quickstart
├── requirements.txt                   # Runtime dependencies
├── docs/
│   └── TECHNICAL_ARCHITECTURE.md      # This document
├── data-generator/
│   ├── generate_state_budget_gst.py   # UC-1 synthetic GST panel
│   ├── generate_welfare_leakage.py    # UC-3 labeled anomalies
│   └── generate_education_enrollment.py # UC-4 education panel
├── fl_experiment/
│   ├── sectors/                       # TaskSpec JSON adapters (UC-5)
│   │   ├── public_finance_gst.json
│   │   ├── education_enrollment.json
│   │   └── README.md
│   ├── privacy/
│   │   └── THREAT_MODEL.md            # UC-6 threat model
│   ├── data/                          # Generated CSVs
│   ├── results/                       # Experiment artifacts
│   │
│   ├── # --- Forecast core (legacy + shared) ---
│   ├── model.py                       # Revenue residual MLP (GST path)
│   ├── utils.py                       # GST forecast frame + non-IID splits
│   ├── metrics.py                     # Regression + classification metrics
│   ├── client.py / server.py          # State FL path + optional server
│   ├── centralized_baseline.py        # Phase 1 gate vs seasonal naive
│   ├── run_fl_simulation.py           # Phase 1 state FedAvg sim
│   ├── partitions.py                  # IID / label / quantity skew
│   ├── partition_client.py            # Disk-backed FL client (Phase 2)
│   ├── train_eval.py                  # Shared train/eval helpers
│   ├── run_phase2_comparison.py       # UC-2 comparison harness
│   │
│   ├── # --- Sector-agnostic forecast (Phase 4) ---
│   ├── task_spec.py                   # TaskSpec contract loader
│   ├── sector_data.py                 # Spec-driven lag/horizon frames
│   ├── sector_models.py               # Generic ResidualMLP factory
│   ├── sector_client.py               # Spec + optional DP FL client
│   ├── run_sector_forecast.py         # UC-1 / UC-4 / UC-5 runner
│   │
│   ├── # --- Leakage detection (Phase 3) ---
│   ├── leakage_utils.py
│   ├── leakage_models.py              # Autoencoder
│   ├── leakage_train.py               # IF + AE training/thresholding
│   ├── leakage_client.py              # Federated AE client
│   ├── run_phase3_leakage.py          # UC-3 runner
│   │
│   ├── # --- Privacy (Phase 5) ---
│   ├── dp_utils.py                    # Clip + Gaussian noise + ε approx
│   └── run_phase5_privacy.py          # UC-6 privacy–utility sweep
└── .github/workflows/
    ├── smoke.yml                      # End-to-end research smoke tests
    └── pylint.yml                     # Style check
```

---

## 6. Data architecture

### 6.1 Datasets

| Dataset | Generator | Rows (approx) | Grain | Labels |
|---------|-----------|---------------|-------|--------|
| `state_budget_gst.csv` | `generate_state_budget_gst.py` | 1,080 | state × month (2016–2024) | Target: next-month GST |
| `welfare_leakage.csv` | `generate_welfare_leakage.py` | ~5.8k | state × district × month | `is_anomaly`, `anomaly_type` |
| `education_enrollment.csv` | `generate_education_enrollment.py` | 1,080 | state × month | Target: next-month enrollment |

### 6.2 GST / education forecast schema (TaskSpec-driven)

**Raw columns (examples):**

- Keys: `state_id`, `year`, `month`
- Value: `gst_revenue` or `enrollment_count`
- Covariates: sector-specific (`gdp_growth_pct`, `literacy_index`, …)

**Derived forecast frame:**

- `{value}_lag_1`, `{value}_lag_12`
- `target` = value at `t + horizon` (horizon = 1 month)
- Residual learning: model predicts `log1p(target) - log1p(lag_12)` so seasonal naive is the baseline

**Temporal holdout (product-standard):**

| Split | Years |
|-------|-------|
| Train | ≤ 2021 |
| Validation | 2022 |
| Test | ≥ 2023 |

### 6.3 Welfare leakage features

Engineered features include claim amount, beneficiaries, days worked, prior claim, urban ratio, month, claim-to-prior ratio, claim-per-beneficiary.

**Injected anomaly types (controllable ~8% rate):**

- `inflated_claim`
- `ghost_beneficiaries`
- `duplicate_spike`

### 6.4 Non-IID partition protocols (forecast research)

Applied to **train rows only**; test set stays shared for fair comparison:

| Protocol | Meaning |
|----------|---------|
| `by_state` | Natural federal clients |
| `iid` | Disjoint random shards |
| `label_skew` | Clients see different target bands |
| `quantity_skew` | Uneven client sizes (disjoint) |

---

## 7. ML architecture

### 7.1 Forecast model

- **Type:** Residual multilayer perceptron (PyTorch)
- **Structure:** `Linear → ReLU → Linear → ReLU → Linear(1)`
- **Init:** Final layer zero-initialized so predictions start at seasonal naive
- **Loss:** MSE on residual log-space targets
- **Decoding:** `expm1(log1p(lag_12) + residual)`
- **Baseline gate:** Must beat seasonal naive (`lag_12`) on test RMSE

### 7.2 Leakage models

| Model | Training | Scoring |
|-------|----------|---------|
| Isolation Forest | Normal-only train rows | Sklearn anomaly score / prediction |
| Autoencoder | Reconstruct normal-only features | Reconstruction MSE; threshold max F1 on val |
| Federated Autoencoder | FedAvg of AE weights across states | Same thresholding on shared test |

### 7.3 Federated learning algorithm

- **Aggregation:** FedAvg (Flower `fl.server.strategy.FedAvg`)
- **Client API:** `flwr.client.NumPyClient` (+ `.to_client()`)
- **Simulation:** `flwr.simulation.start_simulation` with Ray actors
- **Client data for Ray:** Written to disk cache (Ray workers do not share parent globals)

### 7.4 Differential privacy (update-level)

1. Client receives global weights `w`
2. Local train → `w'`
3. `δ = clip_L2(w' − w, C)`
4. `δ ← δ + N(0, (σ · C)² I)`
5. Upload `w + δ`

`C` is calibrated from median local update norms. `σ` is swept in Phase 5. Approximate ε is reported for communication—not a certification-grade accountant.

---

## 8. Runtime & experiment flows

### 8.1 End-to-end forecast (sector-agnostic)

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 data-generator/generate_state_budget_gst.py
python3 data-generator/generate_education_enrollment.py
cd fl_experiment
python3 run_sector_forecast.py --list
python3 run_sector_forecast.py --all --rounds 5
```

**Flow:** load TaskSpec → build lag frame → temporal split → fit scaler on train → centralized residual MLP → FedAvg simulation → write `results/phase4_sectors.csv`.

### 8.2 Non-IID comparison (UC-2)

```bash
python3 run_phase2_comparison.py --rounds 5
```

Compares `naive_seasonal`, `centralized`, `local_only`, `federated_fedavg` under each protocol; logs rounds-to-best and estimated communication bytes.

### 8.3 Leakage detection (UC-3)

```bash
python3 data-generator/generate_welfare_leakage.py
python3 run_phase3_leakage.py --rounds 5
```

Outputs Precision / Recall / F1 / ROC-AUC to `results/phase3_leakage.csv`.

### 8.4 Privacy sweep (UC-6)

```bash
python3 run_phase5_privacy.py --sector public_finance_gst --rounds 3
```

Sweeps noise multipliers; writes `results/phase5_privacy.csv`.

### 8.5 Adding a sector (UC-5)

1. Generate or place CSV under `fl_experiment/data/`
2. Add `fl_experiment/sectors/<sector_id>.json` (see existing specs)
3. Run `python3 run_sector_forecast.py --sector <sector_id>`
4. **Do not** edit `sector_client.py` / `run_sector_forecast.py`

---

## 9. TaskSpec contract (cross-sector product surface)

`TaskSpec` is the primary **product extension API**.

| Field | Purpose |
|-------|---------|
| `sector_id` | Stable ID / filename stem |
| `display_name` | Human-readable sector name |
| `task_type` | `forecast` (anomaly via separate leakage path today) |
| `data_path` | Relative path to CSV |
| `client_id_col` | Federation key (e.g. `state_id`) |
| `time_cols` | Temporal columns (`year`, `month`) |
| `value_col` | Series to forecast |
| `covariate_cols` | Extra features |
| `lag_months` / `horizon` | Lag structure and forecast horizon |
| `residual_on_seasonal_lag` | Use seasonal naive residual head |
| `train_end_year` / `val_year` / `test_start_year` | Holdout policy |

---

## 10. Metrics & success gates

| Capability | Metrics | Gate used in repo |
|------------|---------|-------------------|
| Forecast | RMSE, MAE, MAPE | Beat seasonal naive on test RMSE |
| Non-IID study | Same + bytes / rounds-to-best | Artifact table produced |
| Leakage | Precision, Recall, F1, ROC-AUC | Best F1 ≥ 0.3 |
| Cross-sector | Per-sector RMSE vs naive | Both GST & education pass |
| Privacy | RMSE vs σ; approx ε | Sweep completes; utility degrades with noise |

---

## 11. Change log — Phases 0 to 5 (product delivery history)

### Phase 0 — Runnable FL core

- Restored broken data loading; package-relative CSV paths
- Flower simulation runner (10 state clients)
- Removed target leakage features (`sgst`/`cgst` shares)
- `requirements.txt`, smoke CI, honest README status

### Phase 1 — Temporal forecasting

- Next-month forecast with lag-1 / lag-12
- Temporal train/val/test split
- Residual MLP vs seasonal naive gate
- Enriched synthetic GST (trend + covariate effects)

### Phase 2 — Non-IID comparison product

- Disjoint IID / label / quantity / by-state protocols
- Local vs centralized vs FedAvg on shared test
- Results: `phase2_comparison.csv`

### Phase 3 — Welfare leakage product line

- Labeled synthetic claims + three anomaly types
- Isolation Forest + Autoencoder + Federated AE
- Results: `phase3_leakage.csv`

### Phase 4 — Multi-sector research product

- `TaskSpec` + JSON sector adapters
- Education enrollment as second vertical
- Generic `run_sector_forecast.py` (no FL rewrite per sector)

### Phase 5 — Privacy credibility

- Threat model document
- Client-side DP on updates
- Privacy–accuracy sweep with calibrated clip norm
- Explicit non-claims: SecAgg, TLS, formal ε accounting

---

## 12. Security, privacy & compliance posture

| Control | Status | Product implication |
|---------|--------|---------------------|
| Raw data stays local | Implemented | Core value proposition |
| DP on updates | Research baseline | Enables privacy–utility narrative |
| Threat model | Documented | Prevents overselling |
| SecAgg | Not implemented | Blocker for hostile-server pilots |
| TLS | Not in simulation path | Required before multi-host pilots |
| Legal / DPIA | Out of band | Required for real Indian gov data |

**Product messaging rule:** Describe the system as a **privacy-aware FL research prototype**, not as “encrypted production-ready governance AI,” until SecAgg/TLS and formal accounting exist.

---

## 13. Quality & operability

- **Smoke CI** (`.github/workflows/smoke.yml`): regenerates data; runs centralized, FL sim, Phase 2 (short), Phase 3 (short), Phase 4 sectors, Phase 5 DP sweep (short).
- **Results artifacts:** CSV + JSON under `fl_experiment/results/` for reproducibility tables in papers/demos.
- **Determinism:** Fixed seeds (`42`, generator-specific seeds) for comparable runs.
- **Caches:** `_fl_*_cache/` directories are gitignored (Ray client materialization).

---

## 14. Deployment topology (current vs future)

### Current (research)

```
Single workstation / CI runner
  └── Flower simulation + Ray actors
        └── N virtual clients (states)
```

### Future (pilot — not built)

```
State VMs / on-prem nodes
  └── Flower clients (TLS)
        └── SecAgg + optional DP
              └── Centre aggregation service
                    └── Model registry + dashboards
```

---

## 15. Extensibility roadmap (post Phase 5)

| Priority | Item | Unlocks |
|----------|------|---------|
| P1 | SecAgg + TLS | Credible multi-host pilots |
| P1 | Real data adapters (GST / PFMS / education MIS) | Policy relevance |
| P2 | Formal DP accountant (Opacus) | Compliance-grade ε reporting |
| P2 | LSTM / XGBoost baselines in TaskSpec model factory | Stronger forecast paper claims |
| P2 | TaskSpec `task_type: anomaly` unified with leakage | One CLI for all tasks |
| P3 | XAI on global/local models | Policy explainability |
| P3 | Continuous auditing / risk scoring | 4th-semester product vision |

---

## 16. Stakeholder view (product designer summary)

| Stakeholder | What they get today | What to expect next |
|-------------|---------------------|---------------------|
| Research lead | Reproducible FL vs central tables; dual tasks; DP sweep | Real data + stronger privacy |
| Finance / education analyst | Demo forecasts on synthetic panels | Sector adapters for their schema |
| Welfare monitor | Labeled leakage detection demo | Federated AE on real claims |
| Privacy officer | Honest threat model + measurable DP trade-off | SecAgg/TLS before production language |
| Engineer | Clear module map + TaskSpec extension path | Hardening + packaging |

---

## 17. Related documents

| Document | Path |
|----------|------|
| Product status / quickstart | `README.md` |
| Threat model | `fl_experiment/privacy/THREAT_MODEL.md` |
| Sector adapter guide | `fl_experiment/sectors/README.md` |
| Phase results | `fl_experiment/results/phase*.csv` |

---

## 18. Glossary

| Term | Meaning |
|------|---------|
| FedAvg | Federated Averaging of client model weights |
| Non-IID | Client datasets not identically distributed |
| Residual forecast | Model predicts correction on top of seasonal naive |
| TaskSpec | JSON contract describing a government sector task |
| Update-level DP | Noise applied to weight deltas, not raw records |
| Seasonal naive | Predict next month ≈ value from 12 months earlier |

---

*End of technical architecture document.*
