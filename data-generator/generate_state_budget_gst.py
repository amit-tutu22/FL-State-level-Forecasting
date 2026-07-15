"""Generate synthetic monthly state GST / budget panel data."""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

states = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat",
    "Uttar Pradesh", "Bihar", "West Bengal", "Rajasthan",
    "Madhya Pradesh", "Odisha",
]

years = range(2016, 2025)
months = range(1, 13)

data = []

for state_id, state in enumerate(states, start=1):
    base_gst = np.random.uniform(500, 12000)
    population = np.random.uniform(10, 120)
    urban_ratio = np.random.uniform(20, 60)
    fiscal_deficit_base = np.random.uniform(2.5, 6.0)

    for year in years:
        for month in months:
            seasonality = 1.2 if month in [3, 12] else 0.9 if month in [4, 5] else 1.0
            # Slow growth trend + covariate effects (learnable beyond pure seasonality).
            year_trend = 1.0 + 0.015 * (year - 2016)
            gdp_growth = float(np.clip(np.random.normal(7.0, 1.8), 2.0, 14.0))
            fiscal_deficit = float(
                np.clip(fiscal_deficit_base + np.random.normal(0, 0.2), 1.0, 8.0)
            )
            gdp_effect = 1.0 + 0.025 * (gdp_growth - 7.0)
            deficit_effect = 1.0 - 0.02 * (fiscal_deficit - 4.0)
            noise = np.random.normal(1.0, 0.05)

            gst_revenue = max(
                50.0,
                base_gst * seasonality * year_trend * gdp_effect * deficit_effect * noise,
            )

            data.append([
                state_id,
                state,
                year,
                month,
                round(gst_revenue, 2),
                round(gst_revenue * 0.45, 2),
                round(gst_revenue * 0.55, 2),
                round(np.random.uniform(50000, 200000), 2),
                round(np.random.uniform(20000, 150000), 2),
                round(fiscal_deficit, 2),
                round(population, 2),
                round(urban_ratio, 2),
                round(gdp_growth, 2),
            ])

columns = [
    "state_id", "state_name", "year", "month",
    "gst_revenue", "sgst_share", "cgst_share",
    "budget_allocation", "actual_expenditure",
    "fiscal_deficit_pct", "population_million",
    "urban_ratio", "gdp_growth_pct",
]

df = pd.DataFrame(data, columns=columns)

out_dir = Path(__file__).resolve().parents[1] / "fl_experiment" / "data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "state_budget_gst.csv"
df.to_csv(out_path, index=False)

print(f"{out_path} generated successfully ({len(df)} rows)")
