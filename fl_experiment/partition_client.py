"""Flower client backed by on-disk train partitions + shared global test.

Ray workers do not share parent-process module globals, so partitions and the
feature scaler are written to a cache directory before start_simulation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from flwr.common import Context

from metrics import regression_metrics
from model import RevenueModel
from utils import dataframe_to_xy, predict_from_residual

DEFAULT_CACHE = Path(__file__).resolve().parent / "results" / "_fl_partition_cache"


def cache_dir() -> Path:
    return Path(
        __import__("os").environ.get("FL_PARTITION_CACHE", str(DEFAULT_CACHE))
    )


def write_partition_cache(
    parts: Dict[int, pd.DataFrame],
    test_df: pd.DataFrame,
    x_scaler,
    local_epochs: int,
    directory: Optional[Path] = None,
) -> Path:
    directory = directory or DEFAULT_CACHE
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("train_*.csv"):
        path.unlink()
    for pid, df in parts.items():
        df.to_csv(directory / f"train_{pid}.csv", index=False)
    test_df.to_csv(directory / "test.csv", index=False)
    joblib.dump(x_scaler, directory / "x_scaler.joblib")
    meta = {
        "num_clients": len(parts),
        "local_epochs": local_epochs,
        "partition_ids": sorted(int(k) for k in parts.keys()),
    }
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory


class PartitionClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int):
        directory = cache_dir()
        meta = json.loads((directory / "meta.json").read_text())
        train_path = directory / f"train_{partition_id}.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing partition cache: {train_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(directory / "test.csv")
        x_scaler = joblib.load(directory / "x_scaler.joblib")
        self.local_epochs = int(meta["local_epochs"])

        self.partition_id = partition_id
        self.model = RevenueModel()
        self.X_train, self.y_train = dataframe_to_xy(train_df, x_scaler)
        self.X_test, self.y_test = dataframe_to_xy(test_df, x_scaler)
        self.lag12_test = test_df["gst_lag_12"].values.astype(float)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss_fn = torch.nn.MSELoss()

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.as_tensor(new_p, dtype=p.dtype, device=p.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        loss = None
        for _ in range(self.local_epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(self.X_train), self.y_train)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config), len(self.X_train), {"loss": float(loss.item())}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            resid = self.model(self.X_test)
            loss = self.loss_fn(resid, self.y_test)
            pred = predict_from_residual(resid, self.lag12_test)
            y_true = predict_from_residual(self.y_test, self.lag12_test)
            metrics = regression_metrics(y_true, pred)
        return float(loss.item()), len(self.X_test), metrics


def partition_client_fn(context: Context) -> fl.client.Client:
    pid = int(context.node_config["partition-id"])
    return PartitionClient(partition_id=pid).to_client()
