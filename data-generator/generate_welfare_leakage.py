"""Generate synthetic DBT / MNREGA-style welfare claims with labeled leakages."""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(7)

STATES = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat",
    "Uttar Pradesh", "Bihar", "West Bengal", "Rajasthan",
    "Madhya Pradesh", "Odisha",
]
YEARS = range(2018, 2025)
MONTHS = range(1, 13)
# Fraction of rows that are injected anomalies (controllable).
ANOMALY_RATE = 0.08

rows = []
row_id = 0

for state_id, state in enumerate(STATES, start=1):
    urban_ratio = np.random.uniform(20, 65)
    base_claim = np.random.uniform(800, 6000)
    base_beneficiaries = np.random.uniform(40, 400)
    n_districts = np.random.randint(4, 9)

    for district_id in range(1, n_districts + 1):
        district_scale = np.random.uniform(0.7, 1.3)
        for year in YEARS:
            for month in MONTHS:
                season = 1.15 if month in (3, 6, 12) else 1.0
                beneficiaries = max(
                    5,
                    base_beneficiaries
                    * district_scale
                    * np.random.normal(1.0, 0.12),
                )
                avg_days = float(np.clip(np.random.normal(14, 3), 4, 28))
                claim = max(
                    50.0,
                    base_claim
                    * district_scale
                    * season
                    * (beneficiaries / 100.0)
                    * (avg_days / 14.0)
                    * np.random.normal(1.0, 0.08),
                )
                prior = claim * np.random.normal(1.0, 0.1)
                scheme = np.random.choice(["MNREGA", "DBT_FOOD", "DBT_PENSION"], p=[0.5, 0.3, 0.2])

                is_anomaly = 0
                anomaly_type = "none"

                if np.random.rand() < ANOMALY_RATE:
                    is_anomaly = 1
                    kind = np.random.choice(
                        ["inflated_claim", "ghost_beneficiaries", "duplicate_spike"],
                        p=[0.4, 0.35, 0.25],
                    )
                    anomaly_type = kind
                    if kind == "inflated_claim":
                        claim *= np.random.uniform(2.5, 4.5)
                    elif kind == "ghost_beneficiaries":
                        beneficiaries *= np.random.uniform(2.0, 3.5)
                        claim *= np.random.uniform(1.8, 3.0)
                    else:  # duplicate_spike vs prior
                        claim = prior * np.random.uniform(2.2, 3.8)

                rows.append([
                    row_id,
                    state_id,
                    state,
                    district_id,
                    year,
                    month,
                    round(float(claim), 2),
                    int(round(beneficiaries)),
                    round(avg_days, 2),
                    round(float(prior), 2),
                    round(float(urban_ratio), 2),
                    scheme,
                    is_anomaly,
                    anomaly_type,
                ])
                row_id += 1

columns = [
    "record_id",
    "state_id",
    "state_name",
    "district_id",
    "year",
    "month",
    "claim_amount",
    "beneficiary_count",
    "avg_days_worked",
    "prior_month_claim",
    "urban_ratio",
    "scheme_type",
    "is_anomaly",
    "anomaly_type",
]

df = pd.DataFrame(rows, columns=columns)

out_dir = Path(__file__).resolve().parents[1] / "fl_experiment" / "data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "welfare_leakage.csv"
df.to_csv(out_path, index=False)

rate = df["is_anomaly"].mean()
print(f"{out_path} generated successfully ({len(df)} rows, anomaly_rate={rate:.3f})")
