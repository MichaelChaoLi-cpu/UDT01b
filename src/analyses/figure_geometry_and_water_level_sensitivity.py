#!/usr/bin/env python3
"""Geometry and Water-Level Sensitivity.
Plan: four panels of matched geometry/stage differences; numerical sensitivity,
not observed accuracy, simultaneous closure, or independent storm replication.
Framework: AnaSOP Sections 5-7: original common masks and fine area weights,
peak-depth interval contrasts, registered rain and stage controls.
Run from the repository root with the project's uv Python environment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/udt01b-article-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pyproj import Transformer
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
TITLE = "Geometry and Water-Level Sensitivity"
CACHE = ROOT / "data/processed/three_ward_formal_transport_experiments_v1/factor_comparison_v1"
REVIEW = ROOT / "data/exp/estimation/three_ward_formal_transport_experiments_v1/B2_B3_forcing_geometry_comparison"
AUDIT = ROOT / "data/exp/figure-table-generation/three_ward_transport_v1"
OUTPUT = ROOT / "data/results/figures/Figure_geometry_and_water_level_sensitivity.png"
EXPORT_DPI = 450


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    anasop = ROOT / "docs/AnaSOP.md"
    before_doc = sha(anasop)
    contract = json.loads((ROOT / "data/exp/research-design/three_ward_figure_table_plan_v1/confirmed_plan.json").read_text())
    item = next(x for x in contract["figures"] if x["title"] == TITLE)
    assert item["panels"] == 4 and contract["status"] == "confirmed"
    framework = json.loads((ROOT / "data/exp/research-design/three_ward_estimation_framework_v1/validation.json").read_text())
    assert framework["passed"]
    original_gate = json.loads((REVIEW / "validation.json").read_text())
    assert original_gate["all_passed"]

    support_path = CACHE / "common_support.npz"
    extent_path = REVIEW / "case_peak_extent.csv"
    ward_path = ROOT / "data/processed/three_ward_five_event_temporal_analysis_v1/wards_6677.geojson"
    source_paths = [support_path, extent_path, ward_path]
    source_paths += [CACHE / f"{rain}__{stage}__{arm}_peak_bounds.npz"
                     for rain in ("R0", "R1", "R125")
                     for stage in ("L", "H") for arm in ("T", "U")]
    source_paths += [REVIEW / "validation.json", ROOT / "data/exp/research-design/three_ward_estimation_framework_v1/validation.json"]
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    with np.load(support_path) as z:
        common = z["common_admitted"].copy()
        ward_code = z["ward_code"].copy()
        region = common & (ward_code > 0)
        weights = {"ground": z["ground_area"].copy(), "road": z["road_area"].copy()}
    assert region.shape == (1012, 770)
    extents = pd.read_csv(extent_path)
    lookup = {}
    audit = []
    peaks = {}
    for rain in ("R0", "R1", "R125"):
        for stage in ("L", "H"):
            for arm in ("T", "U"):
                case = f"{rain}__{stage}__{arm}"
                with np.load(CACHE / f"{case}_peak_bounds.npz") as z:
                    if rain == "R1":
                        peaks[(stage, arm)] = (z["ground_lower"].copy(), z["ground_upper"].copy())
                    for domain, threshold in (("ground", .1), ("road", .3)):
                        lower, upper = z[f"{domain}_lower"], z[f"{domain}_upper"]
                        use = region & (weights[domain] > 0)
                        assert np.isfinite(lower[use]).all() and np.isfinite(upper[use]).all()
                        assert (lower[use] <= upper[use]).all()
                        # Preserve the original geometry receipt's strict > comparator.
                        # Record exact ties; do not substitute the road-closure >= rule.
                        lo = float(weights[domain][use & (lower > threshold)].sum() / 1e6)
                        hi = float(weights[domain][use & (upper > threshold)].sum() / 1e6)
                        row = extents[(extents.case_id == case) &
                                      (extents.scope == "Three wards") &
                                      (extents.support == domain) &
                                      np.isclose(extents.threshold_m, threshold)]
                        assert len(row) == 1
                        assert np.isclose(lo, row.certain_km2.iloc[0], rtol=0, atol=1e-9)
                        assert np.isclose(hi, row.possible_km2.iloc[0], rtol=0, atol=1e-9)
                        ties = int(np.count_nonzero(use & ((lower == threshold) | (upper == threshold))))
                        lookup[(stage, rain, domain, arm)] = (lo, hi)
                        audit.append({"case": case, "support": domain, "threshold_m": threshold,
                                      "certain_km2": lo, "possible_km2": hi,
                                      "exact_threshold_ties": ties})
    # Source threshold comparisons are unaltered and the framework's >= display
    # has identical areas only if there are no exact threshold ties on this support.
    assert sum(x["exact_threshold_ties"] for x in audit) == 0

    dem = ROOT / "data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif"
    dem_before = {"size": dem.stat().st_size, "mtime_ns": dem.stat().st_mtime_ns}
    with rasterio.open(dem) as ds:
        bounds = ds.bounds
        assert ds.shape == (20240, 15400)
    extent = [bounds.left / 1000, bounds.right / 1000,
              bounds.bottom / 1000, bounds.top / 1000]
    wards = [shape(x["geometry"]) for x in json.loads(ward_path.read_text())["features"]]
    # Transform every native cell corner, not just tick labels. Values and
    # quality masks stay on their original metric grid without resampling.
    to_lonlat = Transformer.from_crs(6677, 4326, always_xy=True)
    to_metric = Transformer.from_crs(4326, 6677, always_xy=True)
    east_edges = np.linspace(bounds.left, bounds.right, region.shape[1] + 1)
    north_edges = np.linspace(bounds.top, bounds.bottom, region.shape[0] + 1)
    east_mesh, north_mesh = np.meshgrid(east_edges, north_edges)
    longitude, latitude = to_lonlat.transform(east_mesh, north_mesh)
    geo_extent = [float(longitude.min()), float(longitude.max()),
                  float(latitude.min()), float(latitude.max())]
    mid_latitude = (geo_extent[2] + geo_extent[3]) / 2
    geo_aspect = 1 / np.cos(np.deg2rad(mid_latitude))
    test_east, test_north = to_metric.transform(longitude[::253, ::192],
                                                latitude[::253, ::192])
    coordinate_error_m = float(max(np.max(abs(test_east - east_mesh[::253, ::192])),
                                   np.max(abs(test_north - north_mesh[::253, ::192]))))
    assert coordinate_error_m < 1e-5

    sns.set_theme(style="ticks", context="paper", font="DejaVu Sans", font_scale=1.05)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                        "savefig.facecolor": "white"})
    # Give each column one fixed physical width. A dedicated colorbar axis
    # must not steal width only from the upper row. Choose map height from
    # the geographic aspect so equal-aspect maps retain the same x edges
    # as the bar panels below them.
    fig = plt.figure(figsize=(12.5, 13.5))
    column_left = (.075, .545)
    column_width = .375
    bar_height_in = 4.5  # Previously 2.875 in: increase vertical plotting space.
    bar_bottom = .065
    bar_height = bar_height_in / fig.get_figheight()
    map_bottom = bar_bottom + bar_height + .08
    map_height = (column_width * fig.get_figwidth() / fig.get_figheight()
                  * (geo_extent[3] - geo_extent[2])
                  / (geo_extent[1] - geo_extent[0]) * geo_aspect)
    axes = np.empty((2, 2), dtype=object)
    for j, left in enumerate(column_left):
        axes[0, j] = fig.add_axes([left, map_bottom, column_width, map_height])
        axes[1, j] = fig.add_axes([left, bar_bottom, column_width, bar_height])
    colorbar_ax = fig.add_axes([.945, map_bottom + .03, .017, map_height - .06])
    for j, stage in enumerate(("L", "H")):
        ax = axes[0, j]
        tl, tu = peaks[(stage, "T")]
        ul, uu = peaks[(stage, "U")]
        diff_lower, diff_upper = ul - tu, uu - tl
        conservative = np.where(diff_lower > 0, diff_lower,
                                np.where(diff_upper < 0, diff_upper, 0.))
        image = ax.pcolormesh(longitude, latitude,
                              np.ma.masked_where(~region, conservative),
                              shading="flat", antialiased=False, cmap="RdBu_r",
                              norm=TwoSlopeNorm(vmin=-.30, vcenter=0, vmax=.30),
                              rasterized=True)
        for geom in wards:
            pieces = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for piece in pieces:
                x, y = piece.exterior.xy
                lon_boundary, lat_boundary = to_lonlat.transform(x, y)
                ax.plot(lon_boundary, lat_boundary, color="#53616b", lw=.65)
        ax.set_xlim(geo_extent[0], geo_extent[1])
        ax.set_ylim(geo_extent[2], geo_extent[3])
        ax.set_aspect(geo_aspect, adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}°E"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}°N"))
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.8)
            spine.set_edgecolor("#303030")
        # Geographic axes: constant-longitude/latitude lines above the raster,
        # below the ward outlines, with the existing degree-labeled ticks.
        ax.set_axisbelow("line")
        ax.grid(True, which="major", axis="both", color="#75838b",
                linewidth=.55, linestyle=(0, (3, 3)), alpha=.55)
    fig.colorbar(image, cax=colorbar_ax,
                 extend="both", label="Conservative urban minus terrain peak-depth contrast (m)")

    categories = [(rain, domain) for domain in ("ground", "road")
                  for rain in ("R0", "R1", "R125")]
    labels = {"R0": "No rain", "R1": "Reference", "R125": "1.25 x rain"}
    for j, stage in enumerate(("L", "H")):
        ax = axes[1, j]
        xpos = np.arange(len(categories))
        for arm, offset, color, label in (("T", -.19, "#4477aa", "Terrain only"),
                                           ("U", .19, "#ee7733", "Urban constraints")):
            lower = np.array([lookup[(stage, rain, domain, arm)][0] for rain, domain in categories])
            upper = np.array([lookup[(stage, rain, domain, arm)][1] for rain, domain in categories])
            ax.bar(xpos + offset, lower, width=.36, color=color, alpha=.9,
                   label=label, yerr=np.vstack([np.zeros_like(lower), upper-lower]),
                   capsize=2, error_kw={"elinewidth": 1.0, "ecolor": "#263743"})
        ax.set_xticks(xpos, [labels[rain] + "\n" + ("Ground" if domain == "ground" else "Road area")
                            for rain, domain in categories], fontsize=8)
        ax.set_ylabel("Wet area (km²)")
        ax.set_ylim(0, max(v[1] for k, v in lookup.items()) * 1.18)
        ax.axvline(2.5, color="#b5c0c8", lw=.8, ls="--")
        ax.grid(axis="y", color="#dce2e6", lw=.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    panel_stage_labels = {
        "a": "a: Low stage", "b": "b: Raised stage",
        "c": "c: Low stage", "d": "d: Raised stage",
    }
    for label, ax in zip("abcd", axes.flat):
        ax.text(-.10, 1.05, panel_stage_labels[label], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
    assert all("stage" not in ax.get_ylabel().lower() for ax in axes.flat)
    assert all(not ax.get_title() for ax in axes.flat)
    assert fig._suptitle is None
    fig.canvas.draw()
    graticule_checks = {}
    for label, ax in zip("ab", axes[0]):
        graticule_checks[label] = {
            "longitude_gridlines_visible": int(sum(line.get_visible() for line in ax.get_xgridlines())),
            "latitude_gridlines_visible": int(sum(line.get_visible() for line in ax.get_ygridlines())),
        }
        assert all(count > 0 for count in graticule_checks[label].values())
    map_frames = {label: {side: bool(spine.get_visible())
                          for side, spine in ax.spines.items()}
                  for label, ax in zip("ab", axes[0])}
    assert all(all(sides.values()) for sides in map_frames.values())
    panel_positions = {label: list(ax.get_position().bounds)
                       for label, ax in zip("abcd", axes.flat)}
    alignment = {}
    for top, bottom in (("a", "c"), ("b", "d")):
        top_box = axes[0, 0 if top == "a" else 1].get_position()
        bottom_box = axes[1, 0 if bottom == "c" else 1].get_position()
        error = max(abs(top_box.x0 - bottom_box.x0),
                    abs(top_box.x1 - bottom_box.x1))
        assert error < 1e-9, f"Panel edges do not align: {top}/{bottom}"
        alignment[f"{top}_{bottom}"] = {
            "left_edges_equal": bool(abs(top_box.x0 - bottom_box.x0) < 1e-9),
            "right_edges_equal": bool(abs(top_box.x1 - bottom_box.x1) < 1e-9),
            "max_horizontal_error_px": float(error * fig.get_figwidth() * EXPORT_DPI),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)

    assert all(sha(ROOT / key) == value for key, value in bindings.items())
    assert dem_before == {"size": dem.stat().st_size, "mtime_ns": dem.stat().st_mtime_ns}
    assert sha(anasop) == before_doc
    AUDIT.mkdir(parents=True, exist_ok=True)
    receipt = {"title": TITLE, "status": "generated_awaiting_user_confirmation",
               "passed": True, "panels": 4, "dpi": EXPORT_DPI, "no_figure_or_axes_titles": True,
               "panel_stage_annotations": panel_stage_labels,
               "stage_removed_from_y_axes": True,
               "panel_labels": list("abcd"), "three_ward_common_support": True,
               "column_alignment": alignment, "panel_positions": panel_positions,
               "colorbar_has_dedicated_axis": True,
               "map_geographic_aspect_preserved": True,
               "map_axis_crs": "EPSG:4326", "source_grid_crs": "EPSG:6677",
               "geographic_graticules": graticule_checks,
               "four_sided_map_frames": map_frames,
               "graticules_above_raster_below_boundaries": True,
               "coordinate_roundtrip_max_error_m": coordinate_error_m,
               "geographic_extent": geo_extent,
               "native_cells_reprojected_without_value_resampling": True,
               "lower_panel_height_in": bar_height_in,
               "previous_lower_panel_height_in": 2.875,
               "lower_panel_height_increase_percent": 100 * (bar_height_in / 2.875 - 1),
               "area_reconstructions_checked": len(audit) * 2,
               "threshold_ties": sum(x["exact_threshold_ties"] for x in audit),
               "unmodified_geometry_comparator": ">",
               "road_wet_area_not_motor_road_length": True,
               "peak_envelope_not_simultaneous": True,
               "source_values_and_masks_unchanged": True,
               "anaSOP_unchanged_by_generator": True,
               "new_solver_runs": 0, "new_training": 0,
               "source_sha256": bindings, "terrain_metadata": dem_before,
               "script_sha256": sha(Path(__file__)), "output_sha256": sha(OUTPUT),
               "output": str(OUTPUT.relative_to(ROOT)), "data_checks": audit}
    (AUDIT / "geometry_and_water_level_sensitivity_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    caption = """# Geometry and Water-Level Sensitivity

Status: generated; awaiting human output confirmation.

Panels a and b show the reference-rain, low- and raised-stage difference between urban-constrained and terrain-only peak ground-depth envelopes on the original all-twelve-case common three-ward support. These are cellwise peak envelopes, not simultaneous flooding. Positive conservative contrast indicates deeper urban-model peaks; negative contrast indicates shallower peaks. The neutral color includes exact zero and overlapping numerical bounds; it is not evidence of equality. White outside the map is outside common reporting support. The color scale saturates at +/-0.30 m. Water-stage descriptions appear after each panel letter as a: Low stage, b: Raised stage, c: Low stage and d: Raised stage; vertical axes contain only Latitude or Wet area (km²). Map panels retain complete four-sided frames. Dashed geographic graticules follow the degree-labeled longitude and latitude ticks. Map axes display WGS84 longitude and latitude; native JGD2011 zone IX cell corners and ward outlines are transformed with no resampling of scientific values. Column alignment is retained. The lower-panel plotting height is 4.5 inches, increased from 2.875 inches, with unchanged vertical data limits.

Panels c and d compare original fine-area-weighted peak-envelope wet areas for no rain, reference rain and a 1.25-times scaled descendant, at low and raised prescribed stage. Ground wetness uses the existing strict threshold >0.10 m; classified ground-road wetness uses >0.30 m. Road wet area is not the length of closed motor roads. Bars show the lower area bound, with whiskers to the upper bound; these are numerical interval bounds, not statistical confidence intervals. Where bounds coincide, the whisker has zero height. Zero bars are retained, including dry no-rain low-stage controls.

The original geometry comparator is preserved. No sample lies exactly on the selected thresholds on admitted support, so the >= notation in the framework yields the same areas in these panels. Water levels remain scenarios; 1.25-times rain is not an independent historical storm or a return-period estimate. Differences in model output do not establish improved observed accuracy from urban information. Source masks, weights and results are unchanged; no solver, traffic fit, training or sealed-test access is initiated.
"""
    (AUDIT / "geometry_and_water_level_sensitivity_caption.md").write_text(caption)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Validated: {len(audit)*2} original area bounds; four labeled panels; {EXPORT_DPI} dpi; source data unchanged")


if __name__ == "__main__":
    main()
