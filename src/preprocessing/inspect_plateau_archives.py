"""Inventory downloaded PLATEAU ZIPs without unpacking the full models.

The geometry/attribute examples are the first ten objects in the first sorted
mesh file for each model family, not district-wide completeness statistics.
"""

from __future__ import annotations

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/plateau_inventory"
GML = "http://www.opengis.net/gml"
CORE = "http://www.opengis.net/citygml/2.0"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("Unknown text encoding")


def inspect_sample(archive: zipfile.ZipFile, filename: str, family: str) -> dict:
    with archive.open(filename) as stream:
        header = stream.read(65536).decode("utf-8-sig")
    sample = {
        "file": filename,
        "srs_names_in_header": sorted(set(re.findall(r'srsName="([^"]+)"', header))),
        "envelope_lower": re.findall(r"<gml:lowerCorner>(.*?)</gml:lowerCorner>", header),
        "envelope_upper": re.findall(r"<gml:upperCorner>(.*?)</gml:upperCorner>", header),
    }
    if family == "dem":
        sample["geometry_tags_in_header"] = sorted(set(re.findall(r"<(?:gml|dem):([\w]+)", header)))
        return sample
    objects = []
    with archive.open(filename) as stream:
        for event, element in ET.iterparse(stream, events=("end",)):
            if element.tag != f"{{{CORE}}}cityObjectMember":
                continue
            feature = next(iter(element), None)
            if feature is not None:
                descendants = list(feature.iter())
                tags = {local_name(item.tag) for item in descendants}
                attributes = {}
                interesting = {"buildingID", "measuredHeight", "usage", "function", "storeysAboveGround", "storeysBelowGround", "buildingStructureType", "yearOfConstruction", "surveyYear", "creationDate", "lodType", "name", "width"}
                for item in descendants:
                    name = local_name(item.tag)
                    if name in interesting and item.text and item.text.strip():
                        attributes.setdefault(name, []).append({"value": item.text.strip(), **item.attrib})
                lod_z = {}
                for item in descendants:
                    name = local_name(item.tag)
                    if not re.match(r"lod\d", name):
                        continue
                    heights = []
                    for pos in item.iter(f"{{{GML}}}posList"):
                        values = [float(x) for x in (pos.text or "").split()]
                        heights.extend(values[2::3])
                    if heights:
                        lod_z[name] = {"min_z": min(heights), "max_z": max(heights), "vertex_count": len(heights)}
                objects.append({
                    "feature_type": local_name(feature.tag),
                    "gml_id": feature.get(f"{{{GML}}}id"),
                    "lod_geometry_tags": sorted(t for t in tags if re.match(r"lod\d", t)),
                    "has_ground_surface": "GroundSurface" in tags,
                    "has_door": "Door" in tags,
                    "attributes": attributes,
                    "sample_lod_coordinate_heights": lod_z,
                })
            element.clear()
            if len(objects) >= 10:
                break
    sample["objects"] = objects
    sample["sampling_note"] = "First ten city objects in one sorted mesh file; not representative of the entire ward. Height parsing assumes 3D posList coordinates."
    return sample


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted((moga_raw(ROOT, 'plateau')).rglob("*citygml*.zip")):
        ward = path.parent.name
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            gml_files = sorted(item.filename for item in infos if item.filename.startswith("udx/") and item.filename.endswith(".gml"))
            count = Counter(name.split("/")[1] for name in gml_files)
            report = {
                "ward": ward,
                "archive": moga_source(path, ROOT),
                "compressed_bytes": path.stat().st_size,
                "uncompressed_bytes": sum(item.file_size for item in infos),
                "gml_file_counts_by_family": dict(sorted(count.items())),
                "gml_files": gml_files,
                "metadata_members": [item.filename for item in infos if item.filename.startswith("metadata/") and not item.is_dir()],
                "samples": {},
            }
            evidence = OUTPUT / ward
            evidence.mkdir(exist_ok=True)
            for item in infos:
                if item.filename == "README.md" or (item.filename.startswith("metadata/") and item.filename.endswith((".xml", ".csv"))):
                    text = decode(archive.read(item))
                    # Copies of source text for inspection; original archive remains intact.
                    name = Path(item.filename).name + ".txt"
                    (evidence / name).write_text(text, encoding="utf-8")
            report["readme_text_copy"] = moga_source(evidence / "README.md.txt", ROOT)
            for family in ("bldg", "tran", "dem"):
                candidates = [name for name in gml_files if name.startswith(f"udx/{family}/")]
                if candidates:
                    report["samples"][family] = inspect_sample(archive, candidates[0], family)
            reports.append(report)
            print(ward, "GML files:", dict(count), flush=True)
    destination = OUTPUT / "archive_inventory.json"
    destination.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Inventory: {destination}", flush=True)


if __name__ == "__main__":
    main()
