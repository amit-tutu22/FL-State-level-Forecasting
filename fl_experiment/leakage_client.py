"""Flower autoencoder client for federated welfare leakage detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from flwr.common import Context

from leakage_models import LeakageAutoencoder
from leakage_utils import FEATURE_COLS, normal_only, transform_X
from metrics import classification_metrics

DEFAULT_CACHE = Path(__file__).resolve().parent / "results" / "_fl_leakage_cache"


def write_leakage_cache(
    train_parts: Dict[int, pd.DataFrame],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_scaler,
    threshold: float,
    local_epochs: int,
    directory: Optional[Path] = None,
) -> Path:
    directory = directory or DEFAULT_CACHE
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("train_*.csv"):
        path.unlink()
    for pid, df in train_parts.items():
        # Federated AE trains on normal rows only.
        normal_only(df).to_csv(directory / f"train_{pid}.csv", index=False)
    val_df.to_csv(directory / "val.csv", index=False)
    test_df.to_csv(directory / "test.csv", index=False)
    joblib.dump(x_scaler, directory / "x_scaler.joblib")
    meta = {
        "local_epochs": local_epochs,
        "threshold": threshold,
        "n_features": len(FEATURE_COLS),
        "partition_ids": sorted(int(k) for k in train_parts.keys()),
    }
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory


class LeakageAEClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int):
        directory = DEFAULT_CACHE
        meta = json.loads((directory / "meta.json").read_text())
        train_df = pd.read_csv(directory / f"train_{partition_id}.csv")
        test_df = pd.read_csv(directory / "test.csv")
        scaler = joblib.load(directory / "x_scaler.joblib")

        self.local_epochs = int(meta["local_epochs"])
        self.threshold = float(meta["threshold"])
        self.model = LeakageAutoencoder()
        self.X_train = torch.tensor(transform_X(train_df, scaler), dtype=torch.float32)
        self.X_test = torch.tensor(transform_X(test_df, scaler), dtype=torch.float32)
        self.y_test = test_df["is_anomaly"].values.astype(int)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.as_tensor(new_p, dtype=p.dtype, device=p.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if len(self.X_train) == 0:
            return self.get_parameters(config), 1, {"loss": 0.0}
        self.model.train()
        loss = None
        for _ in range(self.local_epochs):
            self.optimizer.zero_grad()
            recon = self.model(self.X_train)
            loss = self.loss_fn(recon, self.X_train)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config), len(self.X_train), {"loss": float(loss.item())}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            scores = self.model.reconstruction_error(self.X_test).cpu().numpy()
            loss = float(torch.mean(self.model.reconstruction_error(self.X_test)).item())
        pred = (scores >= self.threshold).astype(int)
        metrics = classification_metrics(self.y_test, pred, scores=scores)
        return loss, len(self.X_test), metrics


def leakage_client_fn(context: Context) -> fl.client.Client:
    pid = int(context.node_config["partition-id"])
    # partition-id is 0..n-1; map to sorted state keys via meta
    directory = DEFAULT_CACHE
    meta = json.loads((directory / "meta.json").read_text())
    state_ids = meta["partition_ids"]
    state_id = state_ids[pid]
    return LeakageAEClient(partition_id=state_id).to_client()
