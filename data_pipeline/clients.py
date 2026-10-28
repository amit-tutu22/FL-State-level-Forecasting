"""Validated per-state model inputs; synthetic and unreviewed panels are refused."""
import csv
from collections import defaultdict
from pathlib import Path


def load_clients(panel_path, allow_assumed_vintages=False):
    clients = defaultdict(list)
    with Path(panel_path).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if str(row.get('synthetic', '')).lower() in {'true', '1'}:
            raise ValueError('Synthetic rows are not permitted in research inputs')
        mode = row.get('availability_mode')
        if mode not in {'strict', 'assumed_lag_latest_vintage'}:
            raise ValueError('Missing or unsupported provenance/availability mode')
        if mode != 'strict' and not allow_assumed_vintages:
            raise ValueError('Assumed-lag latest-vintage analysis requires explicit opt-in')
        if row['complete_features'] != 'True':
            continue
        clients[row['state_id']].append(row)
    if not clients:
        raise ValueError('No complete eligible state inputs; do not replace with synthetic data')
    for state, records in clients.items():
        records.sort(key=lambda r: r['target_month'])
        if len({r['target_month'] for r in records}) != len(records):
            raise ValueError(f'Duplicate target month for {state}')
    return dict(clients)
