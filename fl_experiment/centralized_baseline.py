"""Centralized MLP baseline on pooled synthetic state GST data."""

from pathlib import Path

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from model import RevenueModel
from utils import FEATURE_COLS, TARGET_COL, load_dataframe, _to_tensors

df = load_dataframe()
X_raw = df[FEATURE_COLS].values
y_raw = df[TARGET_COL].values

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

X_train, y_train, scaler = _to_tensors(X_train_raw, y_train_raw, fit_scaler=True)
X_test, y_test, _ = _to_tensors(X_test_raw, y_test_raw, fit_scaler=False, scaler=scaler)

model = RevenueModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    train_mse = loss_fn(model(X_train), y_train).item()
    test_mse = loss_fn(model(X_test), y_test).item()

print(f"Centralized train MSE: {train_mse:.4f}")
print(f"Centralized test MSE:  {test_mse:.4f}")
print(f"Features: {FEATURE_COLS}")
print(f"Data: {Path(__file__).resolve().parent / 'data' / 'state_budget_gst.csv'}")
