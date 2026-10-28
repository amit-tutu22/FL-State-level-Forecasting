"""Coverage audits and publication-aware joins; never backdate latest vintages."""
import argparse
import calendar
from collections import Counter, defaultdict
import csv
from datetime import date, timedelta
import json
from pathlib import Path

from .normalize import write_csv

EWB_FEATURES = [f'ewb_{s}_{m}' for s in ['intra', 'outward', 'inward']
                for m in ['bills', 'assessable_value_crore']]


def month_end(month):
    year, number = map(int, month.split('-'))
    return date(year, number, calendar.monthrange(year, number)[1])


def month_distance(later, earlier):
    y1, m1 = map(int, later.split('-'))
    y0, m0 = map(int, earlier.split('-'))
    return 12 * (y1 - y0) + m1 - m0


def available(row, mode, lag_days):
    if mode == 'strict':
        return date.fromisoformat(row['available_at'][:10])
    if mode != 'assumed_lag_latest_vintage' or lag_days < 1:
        raise ValueError('Unknown availability mode or invalid lag')
    return month_end(row['reference_month']) + timedelta(days=lag_days)


def select_asof(rows, state_id, origin, target_month, mode='strict', lag_days=45,
                max_age_months=3):
    """Choose latest known vintage of the latest eligible reference month."""
    cutoff = date.fromisoformat(origin)
    if cutoff >= date.fromisoformat(target_month + '-01'):
        raise ValueError('Forecast origin must precede target month')
    candidates = []
    for row in rows:
        if row['state_id'] != state_id or row['geography_status'] != 'validated_name':
            continue
        age = month_distance(target_month, row['reference_month'])
        if age < 1 or age > max_age_months:
            continue
        known = available(row, mode, lag_days)
        if known <= cutoff:
            candidates.append((row['reference_month'], known, row))
    candidates.sort(key=lambda item: item[:2])
    return candidates[-1][2] if candidates else None


def read_sources(folder):
    result = {}
    for name in ['gst', 'ewb', 'cpi']:
        with (Path(folder) / f'{name}.csv').open() as stream:
            result[name] = list(csv.DictReader(stream))
    return result


def usable(row, source):
    fields = {'gst': ['gst_excluding_cess_crore'], 'ewb': EWB_FEATURES,
              'cpi': ['cpi_index']}[source]
    return row['geography_status'] == 'validated_name' and all(row.get(f, '') != '' for f in fields)


def audit(data):
    summary = {'sources': {}, 'overlap_by_cpi_base': {}, 'geography_exclusions': {}}
    for source, rows in data.items():
        months = sorted({r['reference_month'] for r in rows})
        summary['sources'][source] = {'rows': len(rows), 'first_month': months[0],
                                     'last_month': months[-1], 'reference_months': len(months),
                                     'usable_rows': sum(usable(r, source) for r in rows)}
        summary['geography_exclusions'][source] = dict(Counter(r['geography_status'] for r in rows if r['geography_status'] != 'validated_name'))
    gst = {(r['state_id'], r['reference_month']) for r in data['gst'] if usable(r, 'gst')}
    ewb = {(r['state_id'], r['reference_month']) for r in data['ewb'] if usable(r, 'ewb')}
    for base in sorted({r['cpi_base'] for r in data['cpi']}):
        cpi = {(r['state_id'], r['reference_month']) for r in data['cpi'] if usable(r, 'cpi') and r['cpi_base'] == base}
        joint = gst & ewb & cpi
        states = defaultdict(list)
        for state, month in sorted(joint):
            states[state].append(month)
        summary['overlap_by_cpi_base'][base] = {
            'joint_state_months': len(joint),
            'states': {state: {'first': months[0], 'last': months[-1], 'months': len(months),
                               'internal_missing_months': month_distance(months[-1], months[0]) + 1 - len(months)}
                       for state, months in states.items()}}
    summary['limitations'] = [
        'Overlap is reference-month overlap, not proof of real-time availability.',
        'CPI bases are never spliced. Official 2024 back series may be All India only.',
        'Latest downloaded files are revised snapshots; exact historical publication vintages are unavailable.',
        'Strict joins use retrieval date unless supported by a verified vintage registry.',
        'Gross GST components exclude imports; source totals change cess coverage in FY2025-26.',
        'Ambiguous post-reorganisation geographies are retained but quarantined from model joins.']
    return summary


def apply_registry(data, path):
    """Explicit source-hash/period evidence for the availability of this exact vintage.

    A release calendar alone is insufficient: an old release date cannot authorize
    backdating a revised file. The evidence must identify the exact archived vintage.
    """
    with Path(path).open() as stream:
        registry = list(csv.DictReader(stream))
    entries = {}
    for entry in registry:
        if entry.get('exact_vintage_verified') != 'true' or not entry.get('evidence_url'):
            raise ValueError('Registry requires verified exact-vintage evidence')
        key = (entry['source_sha256'], entry['reference_month'])
        if key in entries:
            raise ValueError('Duplicate registry evidence')
        released = date.fromisoformat(entry['available_at'])
        if released < month_end(entry['reference_month']):
            raise ValueError('Monthly observation cannot be available before month end')
        entries[key] = entry
    matched = set()
    for rows in data.values():
        for row in rows:
            key = (row['source_sha256'], row['reference_month'])
            if key in entries:
                entry = entries[key]
                if entry['available_at'] > row['retrieved_at'][:10]:
                    raise ValueError('Evidence date is after source retrieval')
                row['available_at'] = entry['available_at']
                row['availability_basis'] = 'verified_exact_vintage'
                matched.add(key)
    if matched != set(entries):
        raise ValueError('Registry contains unmatched source hashes/months')


def build(data, base, mode='strict', lag_days=45):
    sources = {source: [r for r in rows if usable(r, source) and
                        (source != 'cpi' or r['cpi_base'] == base)]
               for source, rows in data.items()}
    # Index by state to avoid repeated full-table scans.
    indexed = {source: defaultdict(list) for source in sources}
    for source, rows in sources.items():
        for row in rows:
            indexed[source][row['state_id']].append(row)
    result = []
    for target in sources['gst']:
        month, state = target['reference_month'], target['state_id']
        origin = (date.fromisoformat(month + '-01') - timedelta(days=1)).isoformat()
        row = {'state_id': state, 'target_month': month, 'forecast_origin': origin,
               'target_gst_excluding_cess_crore': target['gst_excluding_cess_crore'],
               'target_source_sha256': target['source_sha256'],
               'availability_mode': mode, 'assumed_lag_days': lag_days if mode != 'strict' else '',
               'cpi_base': base}
        for source, fields in [('gst', ['gst_excluding_cess_crore']), ('ewb', EWB_FEATURES), ('cpi', ['cpi_index'])]:
            selected = select_asof(indexed[source][state], state, origin, month, mode, lag_days)
            row[f'{source}_reference_month'] = selected['reference_month'] if selected else ''
            row[f'{source}_available_at'] = available(selected, mode, lag_days).isoformat() if selected else ''
            row[f'{source}_source_sha256'] = selected['source_sha256'] if selected else ''
            for field in fields:
                row['feature_' + field] = selected[field] if selected else ''
        row['complete_features'] = all(row[f'{s}_reference_month'] for s in sources)
        result.append(row)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpi-base', choices=['2012', '2024'], default='2012')
    parser.add_argument('--mode', choices=['strict', 'assumed_lag_latest_vintage'], default='strict')
    parser.add_argument('--lag-days', type=int, default=45)
    parser.add_argument('--release-registry')
    args = parser.parse_args()
    data = read_sources(args.processed)
    if args.release_registry:
        apply_registry(data, args.release_registry)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report = audit(data)
    rows = build(data, args.cpi_base, args.mode, args.lag_days)
    report['join'] = {'mode': args.mode, 'cpi_base': args.cpi_base, 'rows': len(rows),
                      'complete_rows': sum(r['complete_features'] for r in rows)}
    write_csv(out / 'panel.csv', rows)
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['join']))
