import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
np.random.seed(42)

states = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat",
    "Uttar Pradesh", "Bihar", "West Bengal", "Rajasthan",
    "Madhya Pradesh", "Odisha"
]

years = range(2016, 2025)
months = range(1, 13)

data = []

for state_id, state in enumerate(states, start=1):
    base_gst = np.random.uniform(500, 12000)
    population = np.random.uniform(10, 120)

    for year in years:
        for month in months:
            seasonality = 1.2 if month in [3, 12] else 0.9 if month in [4, 5] else 1.0
            noise = np.random.normal(1, 0.1)

            gst_revenue = max(50, base_gst * seasonality * noise)

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
                round(np.random.uniform(2.5, 6.0), 2),
                round(population, 2),
                round(np.random.uniform(20, 60), 2),
                round(np.random.uniform(3, 12), 2)
            ])

columns = [
    "state_id", "state_name", "year", "month",
    "gst_revenue", "sgst_share", "cgst_share",
    "budget_allocation", "actual_expenditure",
    "fiscal_deficit_pct", "population_million",
    "urban_ratio", "gdp_growth_pct"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("state_budget_gst.csv", index=False)

print("state_budget_gst.csv generated successfully")