Title

Federated Learning Framework for Privacy-Preserving Budget Forecasting and Welfare Leakage Detection in Indian Public Finance

⸻

## Current status (Phase 5)

Honest snapshot of what runs today vs the proposal below:

| Area | Status |
|------|--------|
| GST + education sector forecast (TaskSpec) | Implemented |
| Welfare leakage detection | Implemented |
| **Threat model** (`fl_experiment/privacy/THREAT_MODEL.md`) | Implemented |
| **Client-side DP on FedAvg updates** (clip + Gaussian noise) | Implemented |
| **Privacy–accuracy sweep** (`run_phase5_privacy.py`) | Implemented |
| Secure Aggregation / TLS | **Not implemented** (documented) |
| Formal (ε, δ) certification-grade accounting | Approximate only |
| LSTM / XGBoost | Not implemented |

### Privacy (read this before citing “privacy-preserving”)

- Simulation is local Ray — no network TLS.
- Phase 5 DP = clipped noisy **parameter updates**, not encryption and not SecAgg.
- ε from the sweep is a **rough composition estimate** for research communication.

### Quickstart

**Important on Apple Silicon Homebrew:** default `python3` may be 3.14 with a broken `pyexpat` (`Symbol not found: _XML_SetAllocTrackerActivationThreshold`). Prefer Apple’s `/usr/bin/python3` or run `bash scripts/setup_venv.sh` (auto-picks a working interpreter).

```bash
cd /path/to/FL-State-level-Forecasting

# Easiest: auto-select working Python + install deps
bash scripts/setup_venv.sh
source .venv/bin/activate

# Or manually with system Python (recommended if Homebrew 3.14 is broken):
#   /usr/bin/python3 -m venv .venv
#   source .venv/bin/activate
#   python3 -m pip install -r requirements.txt

# Generate data + run (use the venv interpreter)
.venv/bin/python3 data-generator/generate_state_budget_gst.py
.venv/bin/python3 data-generator/generate_welfare_leakage.py
.venv/bin/python3 data-generator/generate_education_enrollment.py
cd fl_experiment
../.venv/bin/python3 run_sector_forecast.py --all --rounds 3
../.venv/bin/python3 run_phase5_privacy.py --sector public_finance_gst --rounds 3
```

Do **not** run `python3 -m pip install ...` with Homebrew 3.14 until that expat issue is fixed (`brew reinstall expat python@3.14` or switch to 3.11/3.12). Use the project `.venv` instead.
Roadmap phases: **0–5 complete** for the research prototype arc (forecast + leakage + sectors + DP baseline).

### Documentation

- **Technical architecture (detailed):** [`docs/TECHNICAL_ARCHITECTURE.md`](docs/TECHNICAL_ARCHITECTURE.md) — product use cases, tech stack, data/ML/FL design, Phase 0–5 change log
- **Threat model:** [`fl_experiment/privacy/THREAT_MODEL.md`](fl_experiment/privacy/THREAT_MODEL.md)
- **Sector adapters:** [`fl_experiment/sectors/README.md`](fl_experiment/sectors/README.md)

⸻

1. Introduction & Motivation

India’s public finance system operates in a federal structure, where financial data related to budget allocation, GST revenue, and welfare schemes is distributed across the Centre, States, and districts.
Due to privacy, legal, and political constraints, raw fiscal data cannot be freely centralized, leading to:
	•	Inaccurate national revenue forecasts
	•	Delayed detection of welfare leakages
	•	Limited data-driven decision-making

Traditional centralized machine learning approaches are not suitable in this setting due to data sensitivity and compliance requirements.

Federated Learning (FL) enables collaborative model training without sharing raw data, making it a promising solution for Indian public finance analytics.

⸻

2. Problem Statement

How can we design a privacy-preserving machine learning framework that:
	1.	Accurately forecasts budget and GST revenue across Indian states
	2.	Detects welfare leakage patterns (DBT / MNREGA-style schemes)
	3.	Operates without centralized access to sensitive fiscal data

while handling non-IID data distributions inherent in India’s federal structure?

⸻

3. Objectives

The objectives of this project are:
	1.	To design a Federated Learning architecture suitable for Indian public finance data
	2.	To implement federated revenue forecasting models across states
	3.	To develop federated anomaly detection models for welfare leakage detection
	4.	To compare performance of federated vs centralized approaches
	5.	To analyze privacy–accuracy trade-offs in fiscal FL systems

⸻

4. Scope of Work

This project focuses on two core public finance tasks:

4.1 Federated Budget & GST Revenue Forecasting
	•	Each state acts as an independent FL client
	•	Local models are trained on state-level revenue data
	•	A central FL server aggregates model updates
	•	No raw fiscal data is shared

4.2 Federated Welfare Leakage Detection
	•	District/state-level nodes train local anomaly detection models
	•	Models identify unusual beneficiary or expenditure patterns
	•	Aggregated model learns national leakage signatures

⸻

5. Methodology

5.1 System Architecture

State / District Nodes
 ├─ Local Fiscal Data
 ├─ Local ML Models
 ↓ (Encrypted Model Updates)
Federated Aggregation Server
 ↓
Global Fiscal Intelligence Model

5.2 Machine Learning Models
	•	Revenue Forecasting:
	•	LSTM
	•	XGBoost
	•	Welfare Leakage Detection:
	•	Isolation Forest
	•	Autoencoders

5.3 Federated Learning Setup
	•	FL framework: Flower / FedML
	•	Aggregation: FedAvg
	•	Evaluation under non-IID data distributions

⸻

6. Dataset Description

Due to restricted access to real fiscal data, the project will use:
	•	Synthetic datasets modeled on:
	•	GST revenue trends
	•	State budget allocations
	•	Welfare beneficiary statistics

Dataset characteristics:
	•	State-wise heterogeneity
	•	Seasonal patterns
	•	Controlled anomaly injection

⸻

7. Evaluation Metrics

Revenue Forecasting
	•	RMSE
	•	MAE
	•	MAPE

Welfare Leakage Detection
	•	Precision
	•	Recall
	•	F1-score
	•	Anomaly detection rate

System Evaluation
	•	Communication overhead
	•	Convergence rate
	•	Privacy exposure analysis

⸻

8. Expected Outcomes
	1.	A working federated learning prototype for public finance analytics
	2.	Quantitative evidence showing FL effectiveness under non-IID fiscal data
	3.	Demonstration of privacy-preserving governance intelligence
	4.	Research insights applicable to Indian budget planning and monitoring
	5.	Foundation for 4th semester advanced auditing system

⸻

9. Tools & Technologies
	•	Programming Language: Python
	•	FL Framework: Flower / FedML
	•	ML Libraries: PyTorch / TensorFlow, Scikit-learn
	•	Visualization: Matplotlib, Seaborn
	•	Platform: Linux / Cloud VM

⸻

10. Timeline (3rd Semester)

Month
Activity
Month 1
Literature review, dataset design
Month 2
FL framework implementation
Month 3
Model training & experiments
Month 4
Evaluation, report writing

11. Expected Contribution

This project contributes to:
	•	Federated Learning research in public finance
	•	Privacy-aware governance analytics
	•	India-specific fiscal ML systems
	•	Foundation for PhD-level research and policy pilots

⸻

12. Future Extension (4th Semester)
	•	Federated continuous auditing
	•	Fiscal risk scoring
	•	Explainable FL for policy decisions
	•	Integration with GST, welfare, and budget systems

⸻

Keywords

Federated Learning, Public Finance, Indian Budget, GST, Welfare Leakage, Privacy-Preserving Machine Learning, GovTech



