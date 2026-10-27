"""Acquire official source tile 09LD2812 for the focused Hibiya check.

Run with .venv/bin/python src/preprocessing/acquire_hibiya_lowpoint_sources.py.
Existing originals are verified, never replaced. No Git or DVC operations.
"""
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/tokyo_lidar_dem/2024'
OUT = ROOT / 'data/exp/data-briefing/hibiya_lowpoint_verification'
BASE = 'https://gic-tokyo.s3.ap-northeast-1.amazonaws.com/2024/dig/'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    resources = []
    entries = [
        (BASE + 'lp/09LD2812.zip', 'pilot_hibiya/09LD2812_lp.zip'),
        (BASE + 'Vectortile/23ku/ortho/18/232839/103235.pbf', 'source_documents/ortho_index_18_232839_103235.pbf'),
        (BASE + 'ortho/09LD2812.zip', 'pilot_hibiya/09LD2812_ortho.zip'),
        ('https://catalog.data.metro.tokyo.lg.jp/api/3/action/package_show?id=t000029d0000000024', 'source_documents/tokyo_catalog_lowpoint_check.json'),
        ('https://www.driveplaza.com/assets/pdf/hibiya_parking/floormap.pdf', 'source_documents/hibiya_parking_floormap.pdf'),
    ]
    for url, relative in entries:
        path = RAW / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        entry = {'url': url, 'path': str(path.relative_to(ROOT)),
                 'bytes': path.stat().st_size,
                 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError('CRC failed: ' + str(path))
                entry['zip_crc_verified'] = True
                entry['members'] = [{'name': i.filename, 'bytes': i.file_size,
                                     'zip_member_timestamp': list(i.date_time)} for i in archive.infolist()]
                for member in archive.namelist():
                    expected = {'09LD2812.las'} if '_lp' in path.name else {'09LD2812.tif', '09LD2812.tfw'}
                    if member not in expected:
                        raise ValueError('Unexpected archive member: ' + member)
                    target = path.parent / member if '_lp' in path.name else path.parent / 'orthophoto' / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    original = archive.read(member)
                    if target.exists():
                        if target.read_bytes() != original:
                            raise ValueError('Existing source differs: ' + str(target))
                    else:
                        target.write_bytes(original)
        resources.append(entry)
    (OUT / 'source_provenance.json').write_text(json.dumps({
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'tile': '09LD2812', 'resources': resources,
        'discovery': 'Tokyo catalog links a now-missing G-spatial orthophoto resource. The same official S3 vector index under ortho returns the exact tile archive URL; the index is preserved.',
        'timing_limit': 'ZIP and LAS creation timestamps are product dates, not flight dates.'
    }, ensure_ascii=False, indent=2) + '\n')
    print('Sources preserved; both ZIPs passed full CRC; SHA256 provenance saved.')


if __name__ == '__main__':
    main()
