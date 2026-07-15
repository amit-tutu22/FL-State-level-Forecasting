"""Flower NumPyClient for per-state next-month GST forecasting."""

from __future__ import annotations

from typing import List

import flwr as fl
import numpy as np
import torch
import torch.optim as optim
from flwr.common import Context

from metrics import regression_metrics
from model import RevenueModel
from utils import load_state_data, predict_from_residual


class StateClient(fl.client.NumPyClient):
    def __init__(self, state_id: int, local_epochs: int = 10):
        self.state_id = state_id
        self.local_epochs = local_epochs
        self.model = RevenueModel()
        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.lag12_test,
        ) = load_state_data(state_id)
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
            # y_test is residual; reconstruct true target for metrics
            y_true = predict_from_residual(self.y_test, self.lag12_test)
            metrics = regression_metrics(y_true, pred)
        return float(loss.item()), len(self.X_test), metrics


def client_fn(context: Context) -> fl.client.Client:
    partition_id = int(context.node_config["partition-id"])
    return StateClient(state_id=partition_id + 1).to_client()
