"""Centralized next-month GST forecast baseline vs seasonal naive."""

import torch
import torch.optim as optim

from metrics import regression_metrics
from model import RevenueModel
from utils import (
    FEATURE_COLS,
    TARGET_COL,
    TEST_START_YEAR,
    TRAIN_END_YEAR,
    VAL_YEAR,
    build_forecast_frame,
    dataframe_to_xy,
    fit_feature_scaler,
    naive_seasonal_predict,
    predict_from_residual,
    temporal_masks,
)

df = build_forecast_frame()
train_mask, val_mask, test_mask = temporal_masks(df)
train_df = df.loc[train_mask]
val_df = df.loc[val_mask]
test_df = df.loc[test_mask]

x_scaler = fit_feature_scaler(train_df)
X_train, y_train = dataframe_to_xy(train_df, x_scaler)
X_val, y_val = dataframe_to_xy(val_df, x_scaler)
X_test, y_test = dataframe_to_xy(test_df, x_scaler)

model = RevenueModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

best_val = float("inf")
best_state = None
patience, bad_epochs = 40, 0
epoch = -1

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()
    if val_loss < best_val - 1e-9:
        best_val = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    train_pred = predict_from_residual(model(X_train), train_df["gst_lag_12"].values)
    val_pred = predict_from_residual(model(X_val), val_df["gst_lag_12"].values)
    test_pred = predict_from_residual(model(X_test), test_df["gst_lag_12"].values)

y_train_raw = train_df[TARGET_COL].values
y_val_raw = val_df[TARGET_COL].values
y_test_raw = test_df[TARGET_COL].values

train_m = regression_metrics(y_train_raw, train_pred)
val_m = regression_metrics(y_val_raw, val_pred)
test_m = regression_metrics(y_test_raw, test_pred)

naive_test = naive_seasonal_predict(test_df)
naive_m = regression_metrics(y_test_raw, naive_test)

beats_naive = test_m["rmse"] < naive_m["rmse"]

print("=== Phase 1 centralized next-month forecast ===")
print(f"Split: train year<={TRAIN_END_YEAR}, val={VAL_YEAR}, test>={TEST_START_YEAR}")
print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
print(f"Features: {FEATURE_COLS}")
print("Model: residual MLP on log1p(target) - log1p(seasonal naive)")
print(f"Epochs run: {epoch + 1} (early-stop on residual val MSE)")
print()
print(f"{'split':<12} {'RMSE':>12} {'MAE':>12} {'MAPE%':>12}")
print(
    f"{'train':<12} {train_m['rmse']:12.2f} {train_m['mae']:12.2f} {train_m['mape']:12.2f}"
)
print(f"{'val':<12} {val_m['rmse']:12.2f} {val_m['mae']:12.2f} {val_m['mape']:12.2f}")
print(
    f"{'test MLP':<12} {test_m['rmse']:12.2f} {test_m['mae']:12.2f} {test_m['mape']:12.2f}"
)
print(
    f"{'test naive':<12} {naive_m['rmse']:12.2f} {naive_m['mae']:12.2f} {naive_m['mape']:12.2f}"
)
print()
print(f"MLP beats seasonal naive on test RMSE: {'YES' if beats_naive else 'NO'}")
if not beats_naive:
    raise SystemExit(1)
