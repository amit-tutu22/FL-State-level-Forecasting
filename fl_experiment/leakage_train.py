"""Train / score helpers for Phase 3 leakage detection."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.ensemble import IsolationForest

from leakage_models import LeakageAutoencoder
from metrics import classification_metrics


def train_isolation_forest(
    X_train_normal: np.ndarray,
    contamination: float = 0.08,
    seed: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train_normal)
    return model


def score_isolation_forest(model: IsolationForest, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Higher score => more normal in sklearn; invert for anomaly score.
    raw = -model.score_samples(X)
    pred = (model.predict(X) == -1).astype(int)
    return pred, raw


def train_autoencoder(
    X_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    epochs: int = 80,
    lr: float = 1e-3,
    patience: int = 15,
) -> LeakageAutoencoder:
    model = LeakageAutoencoder()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best = float("inf")
    best_state = None
    bad = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        recon = model(X_train)
        loss = loss_fn(recon, X_train)
        loss.backward()
        opt.step()

        if X_val is None or len(X_val) == 0:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), X_val).item()
        if val_loss < best - 1e-9:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def reconstruction_scores(model: LeakageAutoencoder, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.reconstruction_error(X).cpu().numpy()


def choose_threshold(scores_val: np.ndarray, y_val: np.ndarray) -> float:
    """Pick threshold maximizing F1 on validation labels."""
    if len(scores_val) == 0:
        return float("inf")
    candidates = np.quantile(scores_val, np.linspace(0.5, 0.99, 30))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        pred = (scores_val >= t).astype(int)
        f1 = classification_metrics(y_val, pred)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def predict_with_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(int)


def eval_scores(y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return classification_metrics(y_true, pred, scores=scores)
