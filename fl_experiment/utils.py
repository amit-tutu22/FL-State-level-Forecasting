import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def load_state_data(state_id):
    df = pd.read_csv("data/state_budget_gst.csv")
    df = df[df["state_id"] == state_id]

    features = df[[
        "sgst_share", "cgst_share",
        "fiscal_deficit_pct",
        "urban_ratio",
        "gdp_growth_pct"
    ]].values

    target = df["gst_revenue"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32).view(-1, 1)
    )