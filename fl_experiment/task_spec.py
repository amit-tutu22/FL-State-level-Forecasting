"""Sector-agnostic TaskSpec contract (Phase 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

SECTORS_DIR = Path(__file__).resolve().parent / "sectors"
FL_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TaskSpec:
    """Config contract for plugging a government sector into the FL forecast runner."""

    sector_id: str
    display_name: str
    task_type: str
    data_path: str
    client_id_col: str
    time_cols: List[str]
    value_col: str
    covariate_cols: List[str]
    lag_months: List[int] = field(default_factory=lambda: [1, 12])
    horizon: int = 1
    residual_on_seasonal_lag: bool = True
    train_end_year: int = 2021
    val_year: int = 2022
    test_start_year: int = 2023

    def resolve_data_path(self) -> Path:
        path = Path(self.data_path)
        if not path.is_absolute():
            path = FL_ROOT / path
        return path

    @property
    def lag_feature_cols(self) -> List[str]:
        return [f"{self.value_col}_lag_{lag}" for lag in self.lag_months]

    @property
    def feature_cols(self) -> List[str]:
        # month from time_cols if present, else all covariates + lags
        feats = list(self.lag_feature_cols)
        if "month" in self.time_cols:
            feats.append("month")
        feats.extend(self.covariate_cols)
        return feats

    @property
    def seasonal_lag_col(self) -> Optional[str]:
        if 12 in self.lag_months:
            return f"{self.value_col}_lag_12"
        return None


def load_task_spec(sector_id: str) -> TaskSpec:
    path = SECTORS_DIR / f"{sector_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in SECTORS_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"Unknown sector '{sector_id}'. Available: {available}. "
            f"Expected config at {path}"
        )
    raw = json.loads(path.read_text())
    return TaskSpec(
        sector_id=raw["sector_id"],
        display_name=raw.get("display_name", raw["sector_id"]),
        task_type=raw.get("task_type", "forecast"),
        data_path=raw["data_path"],
        client_id_col=raw["client_id_col"],
        time_cols=list(raw["time_cols"]),
        value_col=raw["value_col"],
        covariate_cols=list(raw.get("covariate_cols", [])),
        lag_months=list(raw.get("lag_months", [1, 12])),
        horizon=int(raw.get("horizon", 1)),
        residual_on_seasonal_lag=bool(raw.get("residual_on_seasonal_lag", True)),
        train_end_year=int(raw.get("train_end_year", 2021)),
        val_year=int(raw.get("val_year", 2022)),
        test_start_year=int(raw.get("test_start_year", 2023)),
    )


def list_sectors() -> List[str]:
    return sorted(p.stem for p in SECTORS_DIR.glob("*.json"))
