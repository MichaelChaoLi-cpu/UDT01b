"""Read every packaged DEM GML; audit sources, CRS, and terrain mesh scale.

Uses only Python's standard library. Original ZIPs remain unchanged. Triangle
statistics use a seeded reservoir of up to 10,000 triangles per GML file;
counts and attribute inventories use a full scan. Horizontal lengths use a
local WGS84 ellipsoidal tangent approximation and are not accuracy estimates.
"""

from __future__ import annotations

import csv
import json
import math
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/exp/data-briefing/plateau_terrain"
GML = "http://www.opengis.net/gml"
ATTRIBUTES = {"creationDate", "lod", "surveyYear", "geometrySrcDescLod1", "thematicSrcDesc", "srcScaleLod1", "publicSurveySrcDescLod1"}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def quantiles(values: list[float]) -> dict:
    values = sorted(values)
    def q(fraction: float) -> float:
        index = fraction * (len(values) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)
    if not values:
        return {}
    return {"n": len(values), "min": values[0], "p05": q(.05), "p25": q(.25), "median": q(.5), "p75": q(.75), "p95": q(.95), "max": values[-1]}


def triangle_geometry(points: list[tuple]) -> tuple[list[float], float]:
    lat = math.radians(sum(p[0] for p in points) / 3)
    eccentricity2 = 6.69437999014e-3
    base = 1 - eccentricity2 * math.sin(lat)**2
    north_radius = 6378137 * (1 - eccentricity2) / base**1.5
    east_radius = 6378137 / math.sqrt(base)
    xy = [(math.radians(p[1] - points[0][1]) * east_radius * math.cos(lat), math.radians(p[0] - points[0][0]) * north_radius) for p in points]
    lengths = [math.dist(xy[i], xy[(i + 1) % 3]) for i in range(3)]
    area = abs((xy[1][0] - xy[0][0]) * (xy[2][1] - xy[0][1]) - (xy[2][0] - xy[0][0]) * (xy[1][1] - xy[0][1])) / 2
    return lengths, area


def code_lists(archive: zipfile.ZipFile) -> dict:
    names = ["DataQualityAttribute_geometrySrcDesc", "DataQualityAttribute_thematicSrcDesc", "PublicSurveyDataQualityAttribute_geometrySrcDesc", "PublicSurveyDataQualityAttribute_srcScale"]
    result = {}
    for name in names:
        root = ET.fromstring(archive.read(f"codelists/{name}.xml"))
        codes = {}
        for element in root.iter(f"{{{GML}}}Definition"):
            codes[element.findtext(f"{{{GML}}}name")] = element.findtext(f"{{{GML}}}description")
        result[name] = codes
    return result


def inspect_gml(archive: zipfile.ZipFile, name: str) -> dict:
    rng = random.Random(13101)
    counts = Counter()
    attrs = defaultdict(Counter)
    crs = Counter()
    envelopes = {}
    reservoir = []
    triangles = 0
    malformed_triangles = 0
    triangle_depth = 0
    relief_names = []
    with archive.open(name) as stream:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            tag = local_name(element.tag)
            if event == "start":
                if tag == "Triangle":
                    triangle_depth += 1
                if element.get("srsName"):
                    crs[element.get("srsName")] += 1
                continue
            if tag in ("ReliefFeature", "TINRelief", "RasterRelief", "MassPointRelief", "BreaklineRelief", "Triangle"):
                counts[tag] += 1
            text = (element.text or "").strip()
            if tag in ATTRIBUTES and text:
                attrs[tag][text] += 1
            if tag in ("lowerCorner", "upperCorner") and text:
                envelopes[tag] = text
            if element.tag == f"{{{GML}}}name" and text:
                relief_names.append(text)
            if tag == "posList" and triangle_depth:
                values = [float(v) for v in text.split()]
                triangles += 1
                if len(values) != 12 or values[:3] != values[-3:]:
                    malformed_triangles += 1
                else:
                    points = [tuple(values[i:i + 3]) for i in (0, 3, 6)]
                    if len(reservoir) < 10000:
                        reservoir.append(points)
                    else:
                        index = rng.randrange(triangles)
                        if index < len(reservoir):
                            reservoir[index] = points
            if tag == "Triangle":
                triangle_depth -= 1
            element.clear()
    edges, short_edges, long_edges, areas, heights = [], [], [], [], []
    for points in reservoir:
        lengths, area = triangle_geometry(points)
        edges.extend(lengths)
        short_edges.append(min(lengths))
        long_edges.append(max(lengths))
        areas.append(area)
        heights.extend(p[2] for p in points)
    return {
        "gml_member": name,
        "uncompressed_bytes": archive.getinfo(name).file_size,
        "element_counts": dict(counts),
        "attribute_counts": {key: dict(value) for key, value in attrs.items()},
        "srs_names": dict(crs),
        "envelope": envelopes,
        "relief_names": sorted(set(relief_names)),
        "triangle_poslists_scanned": triangles,
        "nonstandard_triangle_poslists": malformed_triangles,
        "sampled_triangles": len(reservoir),
        "edge_length_m": quantiles(edges),
        "shortest_edge_per_triangle_m": quantiles(short_edges),
        "longest_edge_per_triangle_m": quantiles(long_edges),
        "triangle_area_m2": quantiles(areas),
        "sample_vertex_height_m": quantiles(heights),
        "sample_edge_fraction_4_8_to_5_2_m": sum(4.8 <= v <= 5.2 for v in edges) / len(edges) if edges else None,
        "sample_edge_fraction_6_8_to_7_3_m": sum(6.8 <= v <= 7.3 for v in edges) / len(edges) if edges else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wards = []
    table = []
    for path in sorted((ROOT / "data/raw/plateau").rglob("*citygml*.zip")):
        ward = {"ward": path.parent.name, "archive": str(path.relative_to(ROOT))}
        with zipfile.ZipFile(path) as archive:
            metadata = next(name for name in archive.namelist() if name.startswith("metadata/") and name.endswith("_2025_op.xml"))
            root = ET.fromstring(archive.read(metadata))
            ward["metadata_member"] = metadata
            ward["lineage_statements"] = [(e.text or "").strip() for e in root.iter() if local_name(e.tag) == "statement"]
            ward["metadata_crs_labels"] = [(e.text or "").strip() for e in root.iter() if local_name(e.tag) == "code" and "JGD" in (e.text or "")]
            ward["quality_results"] = []
            for element in root.iter():
                if local_name(element.tag) == "DQ_Element" and any("P-dem-01" in (e.text or "") for e in element.iter()):
                    ward["quality_results"].append({local_name(e.tag): (e.text or "").strip() for e in element.iter() if local_name(e.tag) in ("evaluationMethodDescription", "explanation", "pass")})
            ward["code_lists"] = code_lists(archive)
            ward["terrain_files"] = []
            for name in archive.namelist():
                if name.startswith("udx/dem/") and name.endswith(".gml"):
                    print(f"Scanning {path.parent.name}: {name}", flush=True)
                    result = inspect_gml(archive, name)
                    ward["terrain_files"].append(result)
                    table.append({
                        "ward": ward["ward"],
                        "gml_member": name,
                        "relief_features": result["element_counts"].get("ReliefFeature", 0),
                        "triangles": result["element_counts"].get("Triangle", 0),
                        "sampled_triangles": result["sampled_triangles"],
                        "median_short_edge_m": result["shortest_edge_per_triangle_m"].get("median"),
                        "median_long_edge_m": result["longest_edge_per_triangle_m"].get("median"),
                        "median_triangle_area_m2": result["triangle_area_m2"].get("median"),
                    })
                    print(f"  {result['element_counts']}; shortest edge median={table[-1]['median_short_edge_m']:.3f} m", flush=True)
        wards.append(ward)
    report = {
        "scope": "All eight DEM GML files in the three downloaded 2025 PLATEAU ward archives.",
        "method": "Full streaming count and source/CRS scan; seed=13101, up to 10000 reservoir-sampled triangles per file for geometric distributions. Triangle lengths do not measure terrain accuracy, original acquisition density, or hydraulic fitness. Sampling includes packaged boundary meshes rather than administrative-clipped wards.",
        "wards": wards,
    }
    (OUT / "terrain_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "terrain_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved terrain_audit.json and terrain_summary.csv under {OUT}", flush=True)


if __name__ == "__main__":
    main()
