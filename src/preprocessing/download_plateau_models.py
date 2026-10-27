"""Acquire original PLATEAU CityGML bundles; retain provenance and verify ZIPs.

Run from the project root with Python's standard library and curl:
    python3 src/preprocessing/download_plateau_models.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
YEAR = 2025
WARDS = ("13101-chiyoda-ku", "13102-chuo-ku", "13103-minato-ku")
REPORT = ROOT / "data/exp/plateau_inventory/download_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_report(records: list[dict]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(REPORT)


def verify(path: Path) -> dict:
    print(f"Verifying SHA-256 and ZIP CRC: {path.name}", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failed: {path}: {bad}")
        entries = archive.infolist()
        return {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "zip_crc_verified": True,
            "zip_entries": len(entries),
            "uncompressed_bytes": sum(item.file_size for item in entries),
        }


def main() -> None:
    records = json.loads(REPORT.read_text()) if REPORT.exists() else []
    for ward in WARDS:
        package = f"plateau-{ward}-{YEAR}"
        api_url = f"https://www.geospatial.jp/ckan/api/3/action/package_show?id={package}"
        destination = ROOT / "data/raw/plateau" / str(YEAR) / ward
        destination.mkdir(parents=True, exist_ok=True)
        catalog_path = destination / "catalog.json"
        if not catalog_path.exists():
            payload = urllib.request.urlopen(api_url, timeout=60).read()
            catalog_path.write_bytes(payload)
        payload = json.loads(catalog_path.read_bytes())
        if not payload.get("success"):
            raise RuntimeError(f"Catalog request failed for {package}")
        catalog = payload["result"]
        selected = [r for r in catalog["resources"] if r["name"].startswith("CityGML")]
        if len(selected) != 1:
            raise ValueError(f"Expected one CityGML resource for {package}")
        # Small companion bundle retains the administrative boundary and related layers.
        selected += [r for r in catalog["resources"] if r["name"].startswith("関連データセット")]
        for resource in selected:
            url = resource["url"]
            filename = url.rsplit("/", 1)[-1]
            if not filename.endswith(".zip") or "/" in filename:
                raise ValueError(f"Unexpected resource filename: {filename}")
            path = destination / filename
            partial = path.with_suffix(".zip.part")
            if not path.exists():
                request = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(request, timeout=60) as response:
                    length = response.headers.get("Content-Length")
                    expected_bytes = int(length) if length else None
                remaining = max(0, (expected_bytes or 0) - (partial.stat().st_size if partial.exists() else 0))
                if expected_bytes and shutil.disk_usage(destination).free < remaining + 1024**3:
                    raise RuntimeError(f"Insufficient free space for {filename}")
                print(f"Downloading {filename}; expected bytes={expected_bytes}", flush=True)
                command = ["curl", "--silent", "--show-error", "--fail", "--location", "--retry", "3", "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), url]
                process = subprocess.Popen(command)
                while process.poll() is None:
                    time.sleep(10)
                    size = partial.stat().st_size if partial.exists() else 0
                    percent = f" ({size / expected_bytes:.1%})" if expected_bytes else ""
                    print(f"  {filename}: {size / 1024**2:.1f} MiB{percent}", flush=True)
                if process.returncode:
                    raise RuntimeError(f"Download failed with exit code {process.returncode}; partial file retained")
                if expected_bytes and partial.stat().st_size != expected_bytes:
                    raise ValueError(f"Download size mismatch for {filename}")
                partial.replace(path)
            existing = next((r for r in records if r["local_path"] == str(path.relative_to(ROOT))), None)
            if existing and existing.get("zip_crc_verified") and existing.get("bytes") == path.stat().st_size:
                print(f"Previously downloaded and verified: {filename}", flush=True)
                continue
            record = {
                "dataset": catalog["title"],
                "dataset_id": catalog["id"],
                "catalog_url": f"https://www.geospatial.jp/ckan/dataset/{package}",
                "catalog_api_url": api_url,
                "catalog_modified": catalog["metadata_modified"],
                "license_title": catalog.get("license_title"),
                "license_url": catalog.get("license_url"),
                "resource_id": resource["id"],
                "resource_name": resource["name"],
                "source_url": url,
                "downloaded_at": utc_now(),
                "local_path": str(path.relative_to(ROOT)),
                **verify(path),
            }
            records = [r for r in records if r["local_path"] != record["local_path"]] + [record]
            save_report(records)
            print(f"Verified: {filename}; {record['bytes']} bytes; {record['zip_entries']} entries", flush=True)
    print(f"Complete. Provenance: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
