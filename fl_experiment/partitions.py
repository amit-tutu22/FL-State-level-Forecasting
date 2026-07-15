"""Partition protocols for Phase 2 non-IID experiments."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from utils import (
    feature_skew_split,
    iid_split,
    label_skew_split,
    quantity_skew_split,
)

DEFAULT_QUANTITY_RATIOS = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2]

PROTOCOLS = ("by_state", "iid", "label_skew", "quantity_skew")


def partition_train(
    train_df: pd.DataFrame,
    protocol: str,
    num_clients: int = 10,
) -> Dict[int, pd.DataFrame]:
    """
    Disjoint train partitions. Temporal holdout is applied before this call;
    only training rows are partitioned.
    """
    if protocol == "by_state":
        parts = {}
        for i, state_id in enumerate(sorted(train_df["state_id"].unique())):
            parts[i] = train_df[train_df["state_id"] == state_id].reset_index(drop=True)
        return parts

    if protocol == "iid":
        return iid_split(train_df, num_clients)

    if protocol == "label_skew":
        return label_skew_split(train_df, num_clients)

    if protocol == "quantity_skew":
        ratios = DEFAULT_QUANTITY_RATIOS[:num_clients]
        if len(ratios) < num_clients:
            ratios = ratios + [0.2] * (num_clients - len(ratios))
        return quantity_skew_split(train_df, ratios)

    if protocol == "feature_skew":
        # Always 2 clients; pad empties not used — caller should set num_clients=2.
        return feature_skew_split(train_df)

    raise ValueError(f"Unknown protocol: {protocol}. Choose from {PROTOCOLS}")


def partition_sizes(parts: Dict[int, pd.DataFrame]) -> List[int]:
    return [len(parts[i]) for i in sorted(parts)]


def describe_partitions(parts: Dict[int, pd.DataFrame]) -> str:
    sizes = partition_sizes(parts)
    return (
        f"clients={len(parts)}, sizes={sizes}, "
        f"min={min(sizes) if sizes else 0}, max={max(sizes) if sizes else 0}, "
        f"total={sum(sizes)}"
    )
