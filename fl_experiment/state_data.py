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
        """Number of inputs per time step."""
        return int(self.x["train"].shape[2])

    def decode(self, normalized):
        """Convert normalized log predictions back to INR crore."""
        values = np.asarray(normalized, dtype=float).reshape(-1)
        return np.expm1(values * self.y_std + self.y_mean)

    def summary(self):
        """Window counts and target-month ranges per split."""
        return {
            "state_id": self.state_id,
            **{
                split: {
                    "windows": len(self.months[split]),
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


def window_targets(months, window):
    """Target months whose `window` input months are all present (no gap is crossed)."""
    present = set(months)
    return [
        month
        for month in sorted(present)
        if all(shift_month(month, -offset) in present for offset in range(window))
    ]


def _build_one(state, rows, val_start, test_start, window):
    """Normalize with training-period rows, then stack gap-free input windows.

    The window for target month m holds the input rows of target months
    m-window+1 .. m; each row only uses information available at its own
    forecast origin, so the whole window is known before m.
    """
    inputs = {month: _raw_inputs(features, month) for _, month, features, _ in rows}
    targets = {month: math.log1p(target) for _, month, _, target in rows}
    x_mean, x_std, y_mean, y_std = _training_scales(inputs, targets, val_start)

    months = {split: [] for split in SPLITS}
    for month in window_targets(inputs, window):
        months[_split_name(month, val_start, test_start)].append(month)
    x = {
        split: ((_stack(inputs, months[split], window, len(x_mean)) - x_mean) / x_std)
        .astype(np.float32)
        for split in SPLITS
    }
    y = {
        split: ((np.asarray([targets[m] for m in months[split]]).reshape(-1, 1) - y_mean) / y_std)
        .astype(np.float32)
        for split in SPLITS
    }
    return StateClientData(state, months, x, y, x_mean, x_std, y_mean, y_std)


def _training_scales(inputs, targets, val_start):
    """Mean/std of inputs and log targets over training-period rows only."""
    train_months = [month for month in inputs if month < val_start]
    train_x = np.asarray([inputs[m] for m in train_months], dtype=float)
    train_y = np.asarray([targets[m] for m in train_months], dtype=float).reshape(-1, 1)
    return (
        train_x.mean(axis=0),
        _safe_std(train_x),
        float(train_y.mean()),
        float(_safe_std(train_y)[0]),
    )


def _stack(inputs, target_months, window, width):
    """Return a (n, window, width) array of the input rows ending at each target month."""
    stacked = [
        [inputs[shift_month(month, offset - window + 1)] for offset in range(window)]
        for month in target_months
    ]
    return np.asarray(stacked, dtype=float).reshape(-1, window, width)


def select_features(records, names, feature_set):
    """Keep all inputs, or only GST-history inputs for `gst_only`."""
    if feature_set == "all":
        return records, list(names)
    if feature_set != "gst_only":
        raise ValueError("feature_set must be 'all' or 'gst_only'")
    keep = [index for index, name in enumerate(names) if "gst" in name]
    selected = [(s, m, [f[i] for i in keep], y) for s, m, f, y in records]
    return selected, [names[i] for i in keep]


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def build_state_clients(records, val_start, test_start, min_train=24, window=6):
    """Return (clients, excluded): one client per eligible state, sorted by state id.

    Eligibility is counted in usable windows per split.
    """
    if not val_start < test_start:
        raise ValueError("Validation start must precede test start")
    if window < 1:
        raise ValueError("window must be at least 1")
    by_state = defaultdict(list)
    for record in records:
        by_state[record[0]].append(record)
    clients, excluded = [], {}
    for state in sorted(by_state, key=lambda s: (len(s), s)):
        rows = sorted(by_state[state], key=lambda r: r[1])
        if len({r[1] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate target month for state {state}")
        counts = {split: 0 for split in SPLITS}
        for month in window_targets([r[1] for r in rows], window):
            counts[_split_name(month, val_start, test_start)] += 1
        if counts["train"] < min_train:
            excluded[state] = f"{counts['train']} training windows < {min_train}"
        elif counts["val"] == 0 or counts["test"] == 0:
            excluded[state] = "no validation or test windows"
        else:
            clients.append(_build_one(state, rows, val_start, test_start, window))
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
    parser.add_argument("--min-train", type=int, default=24, help="minimum training windows")
    parser.add_argument("--window", type=int, default=6, help="input months per GRU window")
    parser.add_argument(
        "--features",
        choices=["all", "gst_only"],
        default="all",
        help="gst_only keeps the lagged GST inputs (plus the month encoding)",
    )


def records_from_arguments(args):
    """Return (records, feature names, validation start, test start)."""
    if args.source == "panel":
        if not args.panel:
            raise SystemExit("--panel is required with --source panel")
        records, features = load_panel_records(args.panel, args.allow_assumed_vintages)
    else:
        records, features = load_synthetic_records()
    records, features = select_features(records, features, args.features)
    default_val, default_test = DEFAULT_CUTOFFS[args.source]
    val_start = args.val_start or default_val
    test_start = args.test_start or default_test
    return records, features, val_start, test_start


def clients_from_arguments(args):
    """Return (clients, excluded, manifest) for parsed `add_data_arguments` options."""
    records, features, val_start, test_start = records_from_arguments(args)
    clients, excluded = build_state_clients(
        records, val_start, test_start, args.min_train, args.window
    )
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
        "window": args.window,
        "feature_set": args.features,
        "excluded_states": excluded,
    }
    return clients, excluded, manifest
