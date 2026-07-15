"""Generate synthetic state-level school enrollment panel (education sector)."""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(21)

states = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat",
    "Uttar Pradesh", "Bihar", "West Bengal", "Rajasthan",
    "Madhya Pradesh", "Odisha",
]

years = range(2016, 2025)
months = range(1, 13)
data = []

for state_id, state in enumerate(states, start=1):
    base_enrollment = np.random.uniform(5e4, 4e5)
    urban_ratio = np.random.uniform(20, 65)
    literacy_base = np.random.uniform(55, 90)
    teacher_base = np.random.uniform(15, 35)

    for year in years:
        for month in months:
            # Academic calendar seasonality (admissions peak mid-year).
            seasonality = 1.12 if month in (6, 7) else 0.95 if month in (4, 5) else 1.0
            year_trend = 1.0 + 0.012 * (year - 2016)
            literacy_index = float(np.clip(literacy_base + np.random.normal(0, 1.0), 40, 99))
            teacher_ratio = float(np.clip(teacher_base + np.random.normal(0, 0.8), 8, 45))
            literacy_effect = 1.0 + 0.004 * (literacy_index - 70)
            teacher_effect = 1.0 + 0.006 * (teacher_ratio - 25)
            noise = np.random.normal(1.0, 0.04)

            enrollment = max(
                1000.0,
                base_enrollment
                * seasonality
                * year_trend
                * literacy_effect
                * teacher_effect
                * noise,
            )

            data.append([
                state_id,
                state,
                year,
                month,
                round(enrollment, 2),
                round(teacher_ratio, 2),
                round(urban_ratio, 2),
                round(literacy_index, 2),
            ])

columns = [
    "state_id",
    "state_name",
    "year",
    "month",
    "enrollment_count",
    "teacher_ratio",
    "urban_ratio",
    "literacy_index",
]

df = pd.DataFrame(data, columns=columns)
out_dir = Path(__file__).resolve().parents[1] / "fl_experiment" / "data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "education_enrollment.csv"
df.to_csv(out_path, index=False)
print(f"{out_path} generated successfully ({len(df)} rows)")
