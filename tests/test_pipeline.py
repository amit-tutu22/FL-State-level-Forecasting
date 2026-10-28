"""Synthetic unit cases are software smoke tests, never empirical training data."""
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from data_pipeline.clients import load_clients
from data_pipeline.geography import resolve, check_gst_code
from data_pipeline.normalize import monetary, month_header, write_csv, parse_gst, validate_archive
from data_pipeline.panel import select_asof, apply_registry, audit, EWB_FEATURES
from data_pipeline.pdf import collect_monthly_table, parse_table


def observation(month='2020-01', available='2020-02-12', **kwargs):
    row = {'state_id': 'bihar', 'state': 'Bihar', 'reference_month': month,
           'geography_status': 'validated_name', 'available_at': available,
           'availability_basis': 'verified_exact_vintage', 'source_sha256': 'test-only',
           'retrieved_at': '2026-10-29T00:00:00Z'}
    row.update(kwargs)
    return row


class AsOfTests(unittest.TestCase):
    def test_before_and_after_release(self):
        row = observation()
        self.assertIsNone(select_asof([row], 'bihar', '2020-01-31', '2020-02'))
        self.assertIs(select_asof([row], 'bihar', '2020-02-29', '2020-03'), row)

    def test_later_revision_not_used_early(self):
        old = observation(value='100')
        revised = observation(available='2020-04-12', value='200')
        self.assertEqual(select_asof([old, revised], 'bihar', '2020-02-29', '2020-03')['value'], '100')

    def test_latest_vintage_and_reference(self):
        rows = [observation(value='100'), observation(available='2020-02-20', value='101'),
                observation('2020-02', '2020-03-12', value='110')]
        self.assertEqual(select_asof(rows, 'bihar', '2020-02-29', '2020-03')['value'], '101')
        self.assertEqual(select_asof(rows, 'bihar', '2020-03-31', '2020-04')['value'], '110')

    def test_snapshot_not_backdated(self):
        row = observation(available='2026-10-29', availability_basis='retrieval_only')
        self.assertIsNone(select_asof([row], 'bihar', '2020-02-29', '2020-03'))
        self.assertIsNotNone(select_asof([row], 'bihar', '2020-02-29', '2020-03', 'assumed_lag_latest_vintage', 20))

    def test_excludes_wrong_state_boundary_and_stale_data(self):
        row = observation()
        self.assertIsNone(select_asof([row], 'kerala', '2020-02-29', '2020-03'))
        self.assertIsNone(select_asof([row], 'bihar', '2020-08-31', '2020-09'))
        row['geography_status'] = 'boundary_definition_unverified'
        self.assertIsNone(select_asof([row], 'bihar', '2020-02-29', '2020-03'))

    def test_forecast_origin_cannot_use_target_month(self):
        with self.assertRaises(ValueError):
            select_asof([observation()], 'bihar', '2020-03-15', '2020-03')

    def test_registry_requires_exact_vintage_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'registry.csv'
            record = {'source_sha256': 'test-only', 'reference_month': '2020-01',
                      'available_at': '2020-02-12', 'exact_vintage_verified': 'false',
                      'evidence_url': 'https://example.test/archive'}
            write_csv(path, [record])
            with self.assertRaises(ValueError): apply_registry({'cpi': [observation()]}, path)
            record['exact_vintage_verified'] = 'true'
            write_csv(path, [record])
            row = observation(available='2026-10-29')
            apply_registry({'cpi': [row]}, path)
            self.assertEqual(row['available_at'], '2020-02-12')


class GeographyTests(unittest.TestCase):
    def test_aliases_and_code_namespaces(self):
        self.assertEqual(resolve('NCT of Delhi', '2026-01', 'cpi_2024')[0], 'delhi')
        self.assertEqual(resolve('ANDAMAN & NICOBAR', '2026-01', 'ewb')[0], 'andaman_and_nicobar_islands')
        check_gst_code('37', 'Andhra Pradesh')
        with self.assertRaises(ValueError): check_gst_code('28', 'Andhra Pradesh')

    def test_jk_split_is_not_silently_spliced(self):
        self.assertEqual(resolve('Jammu & Kashmir', '2019-09', 'gst')[0], 'jammu_and_kashmir_undivided')
        self.assertEqual(resolve('Jammu & Kashmir', '2019-10', 'gst')[2], 'boundary_transition_month')
        self.assertEqual(resolve('Jammu & Kashmir', '2021-01', 'cpi_2012')[2], 'boundary_definition_unverified')
        self.assertEqual(resolve('Ladakh', '2019-09', 'gst')[2], 'pre_formation_placeholder')

    def test_merged_territories_and_unknown(self):
        self.assertEqual(resolve('Daman & Diu', '2024-01', 'ewb')[2], 'boundary_definition_unverified')
        self.assertEqual(resolve('The Dadra And Nagar Haveli And Daman And Diu', '2026-01', 'cpi_2024')[2], 'validated_name')
        self.assertEqual(resolve('invented state', '2026-01', 'gst')[2], 'unmapped_name')


class SourceTests(unittest.TestCase):
    def test_months_and_missing_values(self):
        self.assertEqual(month_header(datetime(2018, 5, 1)), '2018-05')
        self.assertEqual(month_header('Apr-18'), '2018-04')
        self.assertIsNone(month_header('Financial Year 2018-19'))
        self.assertIsNone(monetary(None))
        self.assertEqual(monetary(0), 0)
        for value in ['NaN', '-1', 'Infinity']:
            with self.assertRaises(ValueError): monetary(value)

    def test_cpi_bases_stay_separate_in_overlap(self):
        gst = observation(gst_excluding_cess_crore='10')
        ewb = observation(**{f: '1' for f in EWB_FEATURES})
        cpi = [observation(cpi_base='2012', cpi_index='100'),
               observation('2025-01', cpi_base='2024', cpi_index='100')]
        result = audit({'gst': [gst], 'ewb': [ewb], 'cpi': cpi})
        self.assertEqual(result['overlap_by_cpi_base']['2012']['joint_state_months'], 1)
        self.assertEqual(result['overlap_by_cpi_base']['2024']['joint_state_months'], 0)

    def test_official_workbook_component_layouts_and_string_codes(self):
        for year, expected in [('2019-20', 'included'), ('2026-27', 'not_in_layout')]:
            name = f'statewise_GST_collection_{year}.json'
            sample = json.loads((Path(__file__).parent / 'fixtures' / name).read_text(),
                                object_hook=lambda obj: datetime.fromisoformat(obj['datetime']) if set(obj) == {'datetime'} else obj)
            sheet = SimpleNamespace(values=sample['rows'], title='Collections-Statewise')
            class Workbook(dict):
                def close(self):
                    pass
            with patch('data_pipeline.normalize.load_workbook', return_value=Workbook({'Collections-Statewise': sheet})):
                rows, checks = parse_gst(Path(name), sample['meta'])
            self.assertTrue(any(r['state_id'] == 'himachal_pradesh' for r in rows))
            self.assertTrue(all(r['cess_in_published_total'] == expected for r in rows))
            self.assertEqual(len(checks), 12 if year == '2019-20' else 5)

    def test_partial_api_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            name = 'ewb-data-2018-19.xlsx'
            (raw/'gst_statistics_index.html').write_text(f'<a href="{name}">export</a>')
            manifest = [{'file': name}]
            for base, series in [('2012','Current'),('2024','Current'),('2024','Back')]:
                filename = f'cpi_{base}_{series}_001.json'
                (raw/filename).write_text(json.dumps({'statusCode':True,'data':[{}],
                    'meta_data':{'page':1,'totalPages':1,'totalRecords':1}}))
                manifest.append({'file':filename})
            validate_archive(raw, manifest)
            last=raw/manifest[-1]['file']
            broken=json.loads(last.read_text());broken['meta_data']['totalPages']=2
            last.write_text(json.dumps(broken))
            with self.assertRaises(ValueError): validate_archive(raw, manifest)


class OfficialPdfRegressionTests(unittest.TestCase):
    def test_three_official_layout_samples(self):
        cases = [('pdf-dec-2023-final', '2023-12', '26814'),
                 ('pdf-sep-2023', '2023-09', '25137'),
                 ('pdf-sep-2026', '2026-09', '29986')]
        for filename, month, maharashtra in cases:
            with self.subTest(month=month):
                sample = json.loads((Path(__file__).parent / 'fixtures' / (filename + '.json')).read_text())
                table, pages = collect_monthly_table(sample['pages'], month)
                rows, checks = parse_table(table, month)
                current = [r for r in rows if r['period_role'] == 'current']
                self.assertEqual(len(current), 38)
                self.assertEqual(sum(r['row_role'] == 'state_or_ut' for r in current), 36)
                self.assertEqual(next(r['amount_inr_crore'] for r in current if r['state_source'] == 'Maharashtra'), maharashtra)
                self.assertEqual(len(checks), 2)
                self.assertTrue(pages['Grand Total'] >= pages['Jammu and Kashmir'])


class ClientTests(unittest.TestCase):
    def test_synthetic_and_assumed_vintage_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'panel.csv'
            row = {'state_id': 'bihar', 'target_month': '2020-03', 'complete_features': True,
                   'availability_mode': 'assumed_lag_latest_vintage', 'synthetic': False}
            write_csv(path, [row])
            with self.assertRaises(ValueError): load_clients(path)
            self.assertEqual(len(load_clients(path, allow_assumed_vintages=True)['bihar']), 1)
            row['synthetic'] = True
            write_csv(path, [row])
            with self.assertRaises(ValueError): load_clients(path, allow_assumed_vintages=True)

    def test_empty_inputs_are_not_replaced_with_synthetic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'panel.csv'
            write_csv(path, [{'state_id': 'bihar', 'target_month': '2020-03',
                             'complete_features': False, 'availability_mode': 'strict'}])
            with self.assertRaises(ValueError): load_clients(path)


if __name__ == '__main__':
    unittest.main()
