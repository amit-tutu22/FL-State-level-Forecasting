"""Disk-backed Flower client for any TaskSpec forecast sector."""

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

from dp_utils import apply_dp_to_update
from metrics import regression_metrics
from sector_data import dataframe_to_xy, predict_from_model_output
from sector_models import build_forecast_model
from task_spec import TaskSpec, load_task_spec

DEFAULT_CACHE = Path(__file__).resolve().parent / "results" / "_fl_sector_cache"


def write_sector_cache(
    spec: TaskSpec,
    parts: Dict[int, pd.DataFrame],
    test_df: pd.DataFrame,
    x_scaler,
    local_epochs: int,
    directory: Optional[Path] = None,
    dp_clip_norm: float = 0.0,
    dp_noise_multiplier: float = 0.0,
    dp_seed: int = 42,
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
        "sector_id": spec.sector_id,
        "local_epochs": local_epochs,
        "n_features": len(spec.feature_cols),
        "num_clients": len(parts),
        "dp_clip_norm": float(dp_clip_norm),
        "dp_noise_multiplier": float(dp_noise_multiplier),
        "dp_seed": int(dp_seed),
    }
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory


class SectorForecastClient(fl.client.NumPyClient):
    def __init__(self, partition_id: int):
        directory = DEFAULT_CACHE
        meta = json.loads((directory / "meta.json").read_text())
        spec = load_task_spec(meta["sector_id"])
        train_df = pd.read_csv(directory / f"train_{partition_id}.csv")
        test_df = pd.read_csv(directory / "test.csv")
        x_scaler = joblib.load(directory / "x_scaler.joblib")

        self.spec = spec
        self.partition_id = partition_id
        self.test_df = test_df
        self.local_epochs = int(meta["local_epochs"])
        self.dp_clip_norm = float(meta.get("dp_clip_norm", 0.0))
        self.dp_noise_multiplier = float(meta.get("dp_noise_multiplier", 0.0))
        self.dp_seed = int(meta.get("dp_seed", 42))
        self.model = build_forecast_model(len(spec.feature_cols))
        self.X_train, self.y_train = dataframe_to_xy(spec, train_df, x_scaler)
        self.X_test, self.y_test = dataframe_to_xy(spec, test_df, x_scaler)
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

        local_params = self.get_parameters(config)
        if self.dp_noise_multiplier > 0 and self.dp_clip_norm > 0:
            rng = np.random.default_rng(self.dp_seed + self.partition_id)
            local_params = apply_dp_to_update(
                parameters,
                local_params,
                clip_norm=self.dp_clip_norm,
                noise_multiplier=self.dp_noise_multiplier,
                rng=rng,
            )
            # Keep local model aligned with what we upload.
            self.set_parameters(local_params)

        return local_params, len(self.X_train), {"loss": float(loss.item())}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.X_test)
            loss = self.loss_fn(out, self.y_test)
            pred = predict_from_model_output(self.spec, out, self.test_df)
            y_true = self.test_df["target"].values.astype(float)
            metrics = regression_metrics(y_true, pred)
        return float(loss.item()), len(self.X_test), metrics


def sector_client_fn(context: Context) -> fl.client.Client:
    pid = int(context.node_config["partition-id"])
    return SectorForecastClient(partition_id=pid).to_client()
