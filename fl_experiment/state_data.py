"""One disjoint, chronologically split Flower client per state.

Each state owns all of its rows. Rows are split by target month into train,
validation and test periods, and every transform is fitted on that state's
training rows only. Targets are never used as same-month inputs.
"""

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_pipeline.clients import load_clients
from data_pipeline.panel import EWB_FEATURES

SYNTHETIC_PATH = Path(__file__).resolve().parent / "data" / "state_budget_gst.csv"
PANEL_TARGET = "target_gst_excluding_cess_crore"
PANEL_FEATURES = (
    ["feature_gst_excluding_cess_crore"]
    + ["feature_" + name for name in EWB_FEATURES]
    + ["feature_cpi_index"]
)
SYNTHETIC_FEATURES = ["gst_previous_month", "gst_same_month_last_year"]
DEFAULT_CUTOFFS = {"synthetic": ("2022-01", "2023-01"), "panel": ("2024-04", "2025-04")}
SPLITS = ("train", "val", "test")


def shift_month(month, offset):
    """Return the YYYY-MM label `offset` months after `month`."""
    year, number = map(int, month.split("-"))
    index = year * 12 + number - 1 + offset
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def load_synthetic_records(path=SYNTHETIC_PATH):
    """Build next-month records from the legacy synthetic CSV (smoke tests only).

    Only GST history is used: the generator's other columns include
    target-derived shares and contemporaneous values that leak the label.
    """
    series = defaultdict(dict)
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            month = f"{int(row['year']):04d}-{int(row['month']):02d}"
            series[row["state_id"]][month] = float(row["gst_revenue"])
    records = []
    for state, values in series.items():
        for month, target in values.items():
            previous = values.get(shift_month(month, -1))
            last_year = values.get(shift_month(month, -12))
            if previous is not None and last_year is not None:
                records.append((state, month, [previous, last_year], target))
    return records, list(SYNTHETIC_FEATURES)


def load_panel_records(path, allow_assumed_vintages=False):
    """Load complete rows of a `data_pipeline.panel` output through its guarded loader."""
    records = []
    for state, rows in load_clients(path, allow_assumed_vintages).items():
        for row in rows:
            if row[PANEL_TARGET] == "":
                continue
            features = [float(row[name]) for name in PANEL_FEATURES]
            records.append((state, row["target_month"], features, float(row[PANEL_TARGET])))
    return records, list(PANEL_FEATURES)


def _month_angle(month):
    return 2.0 * math.pi * (int(month[5:7]) - 1) / 12.0


def _raw_inputs(features, month):
    """Log-scale nonnegative inputs and add a cyclic encoding of the target month."""
    logged = [math.log1p(value) for value in features]
    angle = _month_angle(month)
    return logged + [math.sin(angle), math.cos(angle)]


def _safe_std(values):
    std = values.std(axis=0)
    return np.where(std < 1e-8, 1.0, std)


@dataclass
class StateClientData:  # pylint: disable=too-many-instance-attributes
    """Normalized arrays for one state, plus what is needed to decode predictions."""

    state_id: str
    months: dict
    x: dict
    y: dict
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @property
    def n_features(self):
        """Number of model inputs."""
        return int(self.x["train"].shape[1])

    def decode(self, normalized):
        """Convert normalized log predictions back to INR crore."""
        values = np.asarray(normalized, dtype=float).reshape(-1)
        return np.expm1(values * self.y_std + self.y_mean)

    def summary(self):
        """Counts and date ranges per split."""
        return {
            "state_id": self.state_id,
            **{
                split: {
                    "rows": len(self.months[split]),
                    "first": self.months[split][0] if self.months[split] else None,
                    "last": self.months[split][-1] if self.months[split] else None,
                }
                for split in SPLITS
            },
        }


def _split_name(month, val_start, test_start):
    if month < val_start:
        return "train"
    if month < test_start:
        return "val"
    return "test"


def _split_rows(rows, val_start, test_start):
    """Group log-scaled inputs and targets by chronological split."""
    months = {split: [] for split in SPLITS}
    raw_x = {split: [] for split in SPLITS}
    raw_y = {split: [] for split in SPLITS}
    for _, month, features, target in rows:
        split = _split_name(month, val_start, test_start)
        months[split].append(month)
        raw_x[split].append(_raw_inputs(features, month))
        raw_y[split].append(math.log1p(target))
    width = len(raw_x["train"][0])
    raw_x = {k: np.asarray(v, dtype=float).reshape(-1, width) for k, v in raw_x.items()}
    raw_y = {k: np.asarray(v, dtype=float).reshape(-1, 1) for k, v in raw_y.items()}
    return months, raw_x, raw_y


def _build_one(state, rows, val_start, test_start):
    months, raw_x, raw_y = _split_rows(rows, val_start, test_start)
    x_mean, x_std = raw_x["train"].mean(axis=0), _safe_std(raw_x["train"])
    y_mean = float(raw_y["train"].mean())
    y_std = float(_safe_std(raw_y["train"])[0])
    x = {split: ((raw_x[split] - x_mean) / x_std).astype(np.float32) for split in SPLITS}
    y = {split: ((raw_y[split] - y_mean) / y_std).astype(np.float32) for split in SPLITS}
    return StateClientData(state, months, x, y, x_mean, x_std, y_mean, y_std)


def build_state_clients(records, val_start, test_start, min_train=24):
    """Return (clients, excluded): one client per eligible state, sorted by state id."""
    if not val_start < test_start:
        raise ValueError("Validation start must precede test start")
    by_state = defaultdict(list)
    for record in records:
        by_state[record[0]].append(record)
    clients, excluded = [], {}
    for state in sorted(by_state, key=lambda s: (len(s), s)):
        rows = sorted(by_state[state], key=lambda r: r[1])
        if len({r[1] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate target month for state {state}")
        counts = {split: 0 for split in SPLITS}
        for row in rows:
            counts[_split_name(row[1], val_start, test_start)] += 1
        if counts["train"] < min_train:
            excluded[state] = f"{counts['train']} training rows < {min_train}"
        elif counts["val"] == 0 or counts["test"] == 0:
            excluded[state] = "no validation or test rows"
        else:
            clients.append(_build_one(state, rows, val_start, test_start))
    check_disjoint(clients)
    return clients, excluded


def check_disjoint(clients):
    """Fail unless each state is one client and its splits are ordered and disjoint."""
    states = [client.state_id for client in clients]
    if len(states) != len(set(states)):
        raise ValueError("A state is assigned to more than one client")
    for client in clients:
        ordered = [month for split in SPLITS for month in client.months[split]]
        if ordered != sorted(set(ordered)):
            raise ValueError(f"Overlapping or unordered splits for state {client.state_id}")


def write_cache(clients, directory, manifest):
    """Persist clients so Ray worker processes can load them by partition id."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, client in enumerate(clients):
        arrays = {f"x_{split}": client.x[split] for split in SPLITS}
        arrays.update({f"y_{split}": client.y[split] for split in SPLITS})
        np.savez(
            directory / f"client_{index}.npz",
            x_mean=client.x_mean,
            x_std=client.x_std,
            y_scale=np.asarray([client.y_mean, client.y_std]),
            **arrays,
        )
    payload = dict(manifest, clients=[client.summary() for client in clients])
    payload["months"] = [client.months for client in clients]
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory


def load_cached_client(directory, partition_id):
    """Load the client written at `partition_id` by `write_cache`."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with np.load(directory / f"client_{partition_id}.npz") as arrays:
        y_mean, y_std = arrays["y_scale"]
        return StateClientData(
            state_id=manifest["clients"][partition_id]["state_id"],
            months=manifest["months"][partition_id],
            x={split: arrays[f"x_{split}"] for split in SPLITS},
            y={split: arrays[f"y_{split}"] for split in SPLITS},
            x_mean=arrays["x_mean"],
            x_std=arrays["x_std"],
            y_mean=float(y_mean),
            y_std=float(y_std),
        )


def add_data_arguments(parser):
    """Register the shared data-source command-line options."""
    parser.add_argument("--source", choices=["synthetic", "panel"], default="synthetic")
    parser.add_argument("--panel", help="panel.csv produced by data_pipeline.panel")
    parser.add_argument(
        "--allow-assumed-vintages",
        action="store_true",
        help="accept assumed-lag latest-vintage panels (retrospective sensitivity only)",
    )
    parser.add_argument("--val-start", help="first validation target month, YYYY-MM")
    parser.add_argument("--test-start", help="first test target month, YYYY-MM")
    parser.add_argument("--min-train", type=int, default=24)


def clients_from_arguments(args):
    """Return (clients, excluded, manifest) for parsed `add_data_arguments` options."""
    if args.source == "panel":
        if not args.panel:
            raise SystemExit("--panel is required with --source panel")
        records, features = load_panel_records(args.panel, args.allow_assumed_vintages)
    else:
        records, features = load_synthetic_records()
    default_val, default_test = DEFAULT_CUTOFFS[args.source]
    val_start = args.val_start or default_val
    test_start = args.test_start or default_test
    clients, excluded = build_state_clients(records, val_start, test_start, args.min_train)
    if not clients:
        raise SystemExit(f"No eligible state clients; excluded: {excluded}")
    manifest = {
        "source": args.source,
        "synthetic": args.source == "synthetic",
        "panel": args.panel,
        "features": features + ["target_month_sin", "target_month_cos"],
        "val_start": val_start,
        "test_start": test_start,
        "min_train": args.min_train,
        "excluded_states": excluded,
    }
    return clients, excluded, manifest
