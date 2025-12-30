import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

"""
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

"""

# Global variable to hold client data
# implement non-IID splits

CLIENT_DATA = {}

def set_client_data(split_data):
    global CLIENT_DATA
    CLIENT_DATA = split_data

def load_client_data(client_id):
    client_df = CLIENT_DATA[client_id]

    X = client_df[[
        "sgst_share", "cgst_share",
        "fiscal_deficit_pct",
        "urban_ratio", "gdp_growth_pct"
    ]].values

    y = client_df["gst_revenue"].values

    return X, y

"""
NON-IID — QUANTITY SKEW

Clients have different amounts of data. 
Client 0–2 → 100%
Client 3–5 → 50%
Client 6–9 → 20%

Usages:
ratios = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2]
clients = quantity_skew_split(df, ratios)
"""

def quantity_skew_split(df, ratios):
    clients = {}
    for i, ratio in enumerate(ratios):
        clients[i] = df.sample(frac=ratio, random_state=42+i)
    return clients



"""
NON-IID — LABEL DISTRIBUTION SKEW (MOST IMPORTANT)

Clients see different GST ranges.

Logic:

Sort by gst_revenue → split label space.

Result:
	•	Client 0 → low GST states
	•	Client N → high GST states

"""
def label_skew_split(df, num_clients):
    df_sorted = df.sort_values("gst_revenue")
    splits = np.array_split(df_sorted, num_clients)
    return {i: splits[i] for i in range(num_clients)}

""" NON-IID — FEATURE DISTRIBUTION SKEW

Clients dominated by specific features.

 Example:
	•	Urban states → high urban_ratio
	•	Rural states → high rural_ratio
    """

def feature_skew_split(df):
    clients = {}

    clients[0] = df[df["urban_ratio"] > 50]
    clients[1] = df[df["urban_ratio"] <= 50]

    return clients