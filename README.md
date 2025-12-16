Title

Federated Learning Framework for Privacy-Preserving Budget Forecasting and Welfare Leakage Detection in Indian Public Finance

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



