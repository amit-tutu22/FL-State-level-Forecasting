import pandas as pd
import torch
import torch.optim as optim
from model import RevenueModel
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/state_budget_gst.csv")

X = df[[
    "sgst_share", "cgst_share",
    "fiscal_deficit_pct",
    "urban_ratio",
    "gdp_growth_pct"
]].values

y = df["gst_revenue"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

model = RevenueModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(50):
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()

print("Centralized MSE:", loss.item())