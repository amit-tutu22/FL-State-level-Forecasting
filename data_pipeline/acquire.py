"""Download official, public exports with immutable provenance manifests."""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin

GST_INDEX = 'https://www.gst.gov.in/download/gststatistics'
CPI_ROOT = 'https://api.mospi.gov.in'


def acquire(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Use an empty directory to preserve source vintages')
    manifest = []

    def fetch(url, name):
        data = subprocess.check_output(['curl', '--fail', '--silent', '--show-error',
                                        '--location', '--retry', '3', '--max-time', '60', url])
        (destination / name).write_bytes(data)
        manifest.append({'file': name, 'url': url,
                         'retrieved_at': datetime.now(timezone.utc).isoformat(),
                         'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                         'publication_date': None, 'historical_vintage': False})
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        return data

    html = fetch(GST_INDEX, 'gst_statistics_index.html').decode()
    links = sorted(set(re.findall(r'href="([^"]+(?:statewise_GST_collection_|ewb-data-)[^"]+\.xlsx)"', html)))
    if not links:
        raise ValueError('No official annual export links found')
    for link in links:
        url = urljoin(GST_INDEX, link)
        if not url.startswith('https://tutorial.gst.gov.in/'):
            raise ValueError(f'Unexpected export host: {url}')
        response = fetch(url, url.rsplit('/', 1)[-1])
        if not response.startswith(b'PK'):
            raise ValueError('Expected XLSX ZIP signature')
        print(f'Downloaded {manifest[-1]["file"]}', flush=True)
    for base, series in [('2012', 'Current'), ('2024', 'Current'), ('2024', 'Back')]:
        filters = {'base_year': base, 'level': 'Group', 'series_code': series}
        fetch(CPI_ROOT + '/api/cpi/getCpiFilterByLevelAndBaseYear?' + urlencode(filters),
              f'cpi_filters_{base}_{series}.json')
        params = {'base_year': base, 'series': series, 'sector_code': '3',
                  'limit': 200 if base == '2012' else 100, 'page': 1, 'Format': 'JSON'}
        params['group_code' if base == '2012' else 'division_code'] = '0'
        endpoint = '/api/cpi/getCPIIndex' if base == '2012' else '/api/cpi/getCPIData'
        count, expected, pages = 0, None, 1
        while params['page'] <= pages:
            payload = json.loads(fetch(CPI_ROOT + endpoint + '?' + urlencode(params),
                                       f'cpi_{base}_{series}_{params["page"]:03d}.json'))
            if payload.get('statusCode') is not True:
                raise ValueError(f'CPI API failed: {payload}')
            metadata = payload['meta_data']
            if expected is not None and expected != metadata['totalRecords']:
                raise ValueError('CPI record count changed during pagination; retry new vintage')
            expected = metadata['totalRecords']
            pages = metadata['totalPages']
            if int(metadata['page']) != params['page']:
                raise ValueError('CPI response page does not match request')
            count += len(payload['data'])
            params['page'] += 1
        if count != expected:
            raise ValueError(f'CPI incomplete: {count} / {expected}')
        print(f'CPI {base} {series}: {count} records, {pages} pages', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    acquire(parser.parse_args().output)
