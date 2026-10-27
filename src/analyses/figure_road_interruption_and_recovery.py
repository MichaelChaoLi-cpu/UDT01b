#!/usr/bin/env python3
"""Road Interruption and Recovery.
Plan: five-parent road trajectories, nine-hour length-time exposure and one
simultaneous frozen-anchor map. Framework: AnaSOP Sections 5-7, original lower
depth branch and quality support. Never fits or runs a hydraulic/traffic model.
User-approved style: 450 dpi, descriptive letter labels, aligned columns,
geographic graticules and complete map frame. Run with project uv Python.
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
import rasterio
import seaborn as sns
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[2]
TITLE = "Road Interruption and Recovery"
BASE = ROOT / "data/processed/three_ward_five_event_temporal_analysis_v1"
AUDIT = ROOT / "data/exp/figure-table-generation/three_ward_transport_v1"
OUTPUT = ROOT / "data/results/figures/Figure_road_interruption_and_recovery.png"
DPI = 450
EVENTS = ["E20240618", "E20240731", "E20240830", "E20250710", "E20250911"]
COLORS = ["#4477aa", "#ee7733", "#228833", "#aa3377", "#cc3311"]

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def date(e: str) -> str:
    return f"{e[1:5]}-{e[5:7]}-{e[7:9]}"

def main() -> None:
    doc = ROOT / "docs/AnaSOP.md"
    doc_before = sha(doc)
    contract = json.loads((ROOT / "data/exp/research-design/three_ward_figure_table_plan_v1/confirmed_plan.json").read_text())
    item = next(x for x in contract["figures"] if x["title"] == TITLE)
    assert item["panels"] == 4 and contract["status"] == "confirmed"
    assert f"| {TITLE} |" in doc.read_text()
    framework = json.loads((ROOT / "data/exp/research-design/three_ward_estimation_framework_v1/validation.json").read_text())
    assert framework["passed"]
    admission_path = ROOT / "data/exp/estimation/three_ward_five_event_temporal_analysis_v1/A1_validation.json"
    admission = json.loads(admission_path.read_text())
    assert admission["passed"] and admission["hydraulic_cases"] == 10
    sources = [BASE / name for name in ("road_trajectories.csv", "road_temporal_summary.csv",
               "reference_states.csv", "clipped_road_lengths.npz", "clipped_roads_6677.geojson",
               "wards_6677.geojson", "exposure/E20250911__H__U.npz")]
    sources += [admission_path,
                ROOT / "data/exp/research-design/three_ward_figure_table_plan_v1/confirmed_plan.json"]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    all_rows = pd.read_csv(BASE / "road_trajectories.csv")
    data = all_rows[(all_rows.bound == "lower") & (all_rows.arm == "flood") &
                    (all_rows.scope == "Three wards")].copy()
    summary = pd.read_csv(BASE / "road_temporal_summary.csv")
    summary = summary[(summary.bound == "lower") & (summary.arm == "flood") &
                      (summary.scope == "Three wards")]
    assert len(data) == 1090 and len(summary) == 10
    with np.load(BASE / "clipped_road_lengths.npz") as z:
        qualified = z["qualified"].copy()
        weights = z["union_length_m"].copy()
    known_km = float(weights[qualified].sum() / 1000)
    unknown_km = float(weights[~qualified].sum() / 1000)
    assert np.allclose(data.qualified_total_km, known_km, rtol=0, atol=1e-9)
    assert np.allclose(data.unknown_exposure_km, unknown_km, rtol=0, atol=1e-9)
    assert np.allclose(data.qualified_open_km + data.qualified_slow_km +
                       data.qualified_closed_km, data.qualified_total_km, rtol=0, atol=1e-7)
    curves = {}
    exposure = np.empty((5, 2))
    checks = []
    for ei, event in enumerate(EVENTS):
        for si, stage in enumerate(("L", "H")):
            g = data[(data.event_id == event) & (data.stage == stage)].sort_values("frame")
            assert len(g) == 109 and np.array_equal(g.frame, np.arange(109))
            times = g.analysis_elapsed_s.to_numpy()
            actual = g.actual_elapsed_s.to_numpy()
            assert times[0] == 0 and times[-1] == 32400
            assert np.all(np.diff(times) > 0) and np.max(abs(times-actual)) <= .001
            integral = float(np.trapezoid(g.qualified_closed_km, times) / 3600)
            row = summary[summary.case == f"{event}__{stage}__U"]
            assert len(row) == 1
            assert np.isclose(integral, row.closed_km_hours.iloc[0], rtol=0, atol=1e-8)
            assert np.isclose(g.qualified_closed_km.max(), row.peak_closed_km.iloc[0], rtol=0, atol=1e-8)
            curves[(event, stage)] = g
            exposure[ei, si] = integral
            checks.append({"event": event, "stage": stage, "frames": len(g),
                           "closed_km_hours": integral,
                           "peak_closed_km": float(g.qualified_closed_km.max()),
                           "closed_at9h_km": float(g.qualified_closed_km.iloc[-1]),
                           "recovery_right_censored": bool(g.qualified_closed_km.iloc[-1] > 0)})
    refs = pd.read_csv(BASE / "reference_states.csv")
    anchor = refs[(refs.event_id == "E20250911") &
                  refs.selection_reasons.str.contains("H_noaction_closed_length_peak")]
    assert len(anchor) == 1 and not bool(anchor.selected_using_intervention_benefit.iloc[0])
    frame = int(anchor.frame.iloc[0])
    with np.load(BASE / "exposure/E20250911__H__U.npz") as z:
        indices = z["qualified_road_indices"].copy()
        lower = z["lower"].copy()
        upper_frame = z["upper"][frame].copy()
        map_time = float(z["analysis_times"][frame])
    assert np.array_equal(indices, np.flatnonzero(qualified))
    assert np.isfinite(lower).all() and (lower[frame] <= upper_frame).all()
    state = np.where(lower >= .30, 2, np.where(lower >= .10, 1, 0)).astype("u1")
    g = curves[("E20250911", "H")]
    for code, name in ((0, "qualified_open_km"), (1, "qualified_slow_km"), (2, "qualified_closed_km")):
        reconstructed = (state == code) @ (weights[indices] / 1000)
        assert np.allclose(reconstructed, g[name], rtol=0, atol=1e-7)
    assert frame == 29 and abs(map_time - 8700) <= .001
    road_state = np.full(len(weights), -1, dtype="i1")
    road_state[indices] = state[frame]
    geodata = json.loads((BASE / "clipped_roads_6677.geojson").read_text())
    features = geodata["features"]
    feature_indices = np.array([int(f["properties"]["road_index"]) for f in features])
    assert len(feature_indices) == len(np.unique(feature_indices))
    assert set(feature_indices) == set(np.flatnonzero(weights > 0))
    to_geo = Transformer.from_crs(6677, 4326, always_xy=True)
    segments = {code: [] for code in (-1, 0, 1, 2)}
    geometric_lengths = []
    for f, idx in zip(features, feature_indices):
        geom = shape(f["geometry"])
        geometric_lengths.append(geom.length)
        parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for part in parts:
            xy = np.asarray(part.coords)
            lon, lat = to_geo.transform(xy[:, 0], xy[:, 1])
            segments[int(road_state[idx])].append(np.column_stack([lon, lat]))
    assert np.allclose(geometric_lengths, weights[feature_indices], rtol=0, atol=1e-6)
    dem = ROOT / "data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif"
    dem_stat = (dem.stat().st_size, dem.stat().st_mtime_ns)
    with rasterio.open(dem) as ds:
        b = ds.bounds
    lon, lat = to_geo.transform([b.left, b.left, b.right, b.right],
                                [b.bottom, b.top, b.bottom, b.top])
    extent = [min(lon), max(lon), min(lat), max(lat)]
    aspect = 1 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2))
    wards = [shape(x["geometry"]) for x in json.loads((BASE / "wards_6677.geojson").read_text())["features"]]

    sns.set_theme(style="ticks", context="paper", font="DejaVu Sans", font_scale=1.05)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                        "savefig.facecolor": "white"})
    # User-selected mosaic: add / bdd / cdd. The map occupies two
    # column widths and spans all three left-hand rows without distortion.
    fig = plt.figure(figsize=(13.0, 11.0))
    left, width = .085, .26
    map_left, map_width = .425, .52
    map_bottom = .13
    map_height = map_width * fig.get_figwidth() / fig.get_figheight() * (extent[3]-extent[2]) / (extent[1]-extent[0]) * aspect
    row_gap = .075
    row_height = (map_height - 2*row_gap) / 3
    c_panel = fig.add_axes([left,map_bottom,width,row_height])
    b_panel = fig.add_axes([left,map_bottom+row_height+row_gap,width,row_height])
    a_panel = fig.add_axes([left,map_bottom+2*(row_height+row_gap),width,row_height])
    d_panel = fig.add_axes([map_left,map_bottom,map_width,map_height])
    axes = np.array([[a_panel,b_panel],[c_panel,d_panel]])
    ymax = np.ceil(data.qualified_closed_km.max() / 25) * 25
    for si, stage in enumerate(("L", "H")):
        ax = axes[0, si]
        for event, color in zip(EVENTS, COLORS):
            g = curves[(event, stage)]
            ax.plot(g.analysis_elapsed_s / 3600, g.qualified_closed_km,
                    color=color, lw=1.6, label=date(event))
        ax.set(xlim=(0, 9), ylim=(0, ymax), xlabel="Elapsed model time (h)", ylabel="Closed road length (km)")
        ax.set_xticks([0, 3, 6, 9])
        ax.grid(axis="y", color="#dce2e6", lw=.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    paired = axes[1, 0]
    ypos = np.arange(len(EVENTS))[::-1]
    paired.hlines(ypos, exposure[:, 0], exposure[:, 1], color="#8a98a1", linewidth=1.6, zorder=2)
    paired.scatter(exposure[:, 0], ypos, color="#4477aa", s=36, label="Low stage", zorder=3)
    paired.scatter(exposure[:, 1], ypos, color="#ee7733", s=36, label="Raised stage", zorder=3)
    for i, y in enumerate(ypos):
        paired.annotate(f"{exposure[i,0]:.1f}", (exposure[i,0], y), xytext=(0,-13),
                        textcoords="offset points", ha="center", va="top", fontsize=8.5, color="#345f88")
        paired.annotate(f"{exposure[i,1]:.1f}", (exposure[i,1], y), xytext=(0,9),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#b95a24")
    paired.set_yticks(ypos, [date(e) for e in EVENTS], fontsize=9)
    paired.set(xlabel="Closure exposure (km h)", ylabel="Rainfall parent",
               xlim=(0, np.ceil(exposure.max()*1.08/50)*50), ylim=(-.85,len(EVENTS)-.15))
    paired.grid(axis="x", color="#dce2e6", lw=.6)
    paired.set_axisbelow(True)
    paired.legend(loc="upper center", bbox_to_anchor=(left+width/2,.065),
                  bbox_transform=fig.transFigure, ncol=2, frameon=False, fontsize=9)
    ax = axes[1, 1]
    for geom in wards:
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            x, y = part.exterior.xy
            lon, lat = to_geo.transform(x, y)
            ax.plot(lon, lat, color="#53616b", lw=.65, zorder=4)
    map_colors = {0: "#b8c8d4", 1: "#ee7733", 2: "#bb2233", -1: "#6e7478"}
    map_widths = {0: .32, 1: .65, 2: .8, -1: .35}
    for code in (0, -1, 1, 2):
        collection = LineCollection(segments[code], colors=map_colors[code],
                                    linewidths=map_widths[code],
                                    linestyles="dashed" if code == -1 else "solid",
                                    rasterized=True, zorder=3 if code in (1,2) else 2)
        ax.add_collection(collection)
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_aspect(aspect, adjustable="box")
    ax.set(xlabel="Longitude", ylabel="Latitude")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}°N"))
    ax.tick_params(labelsize=9)
    ax.set_axisbelow("line")
    ax.grid(color="#75838b", lw=.55, linestyle=(0,(3,3)), alpha=.55)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(.8); spine.set_edgecolor("#303030")
    legend_names = {0:"Open by rule", 1:"Restricted by rule", 2:"Closed by rule", -1:"Unknown exposure"}
    handles = [Line2D([0],[0],color=map_colors[c],lw=1.5,
                       ls="--" if c == -1 else "-",label=legend_names[c]) for c in (0,1,2,-1)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(map_left+map_width/2,.065),
              bbox_transform=fig.transFigure, ncol=2, frameon=False, fontsize=8)
    labels = ["a: Low stage", "b: Raised stage", "c: Nine-hour closure exposure", "d: 2025-09-11, raised stage"]
    for index, (label, panel) in enumerate(zip(labels, axes.flat)):
        # Equal physical label offsets for the narrow left plots and wide map.
        x_offset = -.10 * width / panel.get_position().width
        y_offset = 1 + .28 / (panel.get_position().height * fig.get_figheight())
        panel.text(x_offset,y_offset,label,transform=panel.transAxes,fontsize=12,fontweight="bold",va="top")
        assert not panel.get_title()
    fig.canvas.draw()
    boxes = {label: ax.get_position() for label, ax in zip("abcd",axes.flat)}
    alignment_errors = {
        "a_b_left_column": max(abs(boxes["a"].x0-boxes["b"].x0),abs(boxes["a"].x1-boxes["b"].x1)),
        "c_left_matches_a": abs(boxes["c"].x0-boxes["a"].x0),
        "c_right_matches_a": abs(boxes["c"].x1-boxes["a"].x1),
        "a_d_top": abs(boxes["a"].y1-boxes["d"].y1),
        "c_d_bottom": abs(boxes["c"].y0-boxes["d"].y0),
        "d_two_column_width": abs(boxes["d"].width-2*boxes["a"].width),
    }
    alignment = {}
    for key, err in alignment_errors.items():
        assert err < 1e-9, key
        alignment[key] = {"aligned":bool(err < 1e-9),"max_error_px":float(err*fig.get_figwidth()*DPI)}
    panel_positions = [list(a.get_position().bounds) for a in axes.flat]
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUTPUT,dpi=DPI,bbox_inches="tight")
    plt.close(fig)
    assert sha(doc) == doc_before
    assert all(sha(ROOT/k)==h for k,h in bindings.items())
    assert (dem.stat().st_size,dem.stat().st_mtime_ns)==dem_stat
    with Image.open(OUTPUT) as im:
        png = {"width_px":im.width,"height_px":im.height,"dpi":list(im.info["dpi"])}
    assert all(abs(d-DPI)<.1 for d in png["dpi"])
    AUDIT.mkdir(parents=True,exist_ok=True)
    map_counts = {legend_names[c]:float(weights[road_state==c].sum()/1000) for c in (0,1,2,-1)}
    receipt = {"title":TITLE,"passed":True,"status":"generated_awaiting_user_confirmation",
               "research_question":"RQ1","framework":"matched historical no-action cases; original lower-depth road classification and trapezoidal length-time exposure",
               "rain_parents":5,"event_stage_cases":10,"trajectory_frames":1090,
               "numerical_admission_receipt_unchanged":True,"independent_storm_count":5,
               "main_depth_branch":"lower","qualified_road_length_km":known_km,
               "unknown_road_length_km":unknown_km,"map_anchor_frame":frame,
               "map_anchor_elapsed_s":map_time,"map_road_state_lengths_km":map_counts,
               "all_109_anchor_case_frames_reconstructed":True,
               "all_clipped_geometries_match_unique_length_weights":True,
               "map_feature_count":len(features),"map_axis_crs":"EPSG:4326",
               "four_sided_map_frame":True,"map_graticules":True,"column_alignment":alignment,
               "panel_c_type":"horizontal paired points", "panel_c_original_values":exposure.tolist(),
               "layout":"a/b/c stacked left;d spans all three rows and two right columns",
               "layout_grid":["add","bdd","cdd"],"layout_decision":"MILI-D-20261020-012",
               "layout_status":"user_specified_arrangement_output_pending_review", "panel_c_revision_decision":"MILI-D-20261020-010",
               "panel_positions":panel_positions,"panel_labels":labels,"dpi":DPI,"png_metadata":png,
               "source_data_unchanged":True,"AnaSOP_unchanged":True,
               "new_simulations":0,"new_training":0,"new_traffic_fits":0,
               "style_decision":"MILI-D-20261020-008","source_sha256":bindings,
               "script_sha256":sha(Path(__file__)),"output_sha256":sha(OUTPUT),
               "output":str(OUTPUT.relative_to(ROOT)),"case_checks":checks}
    (AUDIT/"road_interruption_and_recovery_validation.json").write_text(json.dumps(receipt,indent=2)+"\n")
    caption = """# Road Interruption and Recovery

Status: generated; awaiting human output confirmation. Four panels, 450 dpi.

Panels a and b show qualified closed motor-road length at simultaneous five-minute model states for five historical rainfall parents under low and raised receiving-water assumptions. The main road-depth lower branch is used throughout. The original scenario rule closes a qualified road when mapped depth reaches 0.30 m; these are not observed road closures or safety thresholds. Identical qualified length and explicit unknown coverage are retained. Time is elapsed model time within each registered nine-hour dry-start window, not synchronized historical clock time across storms. End-window closure is right-censored: recovery after nine hours is unknown.

Panel c compares nine-hour closed-length exposure in km h using horizontal paired points, computed by trapezoidal integration of the same trajectories on their approved analysis times. Each rainfall parent has a blue low-stage point and an orange raised-stage point joined by a gray connector; endpoint positions show cumulative exposure and their separation shows the prescribed-stage contrast. The ten original values are retained and labeled. This is length multiplied by time, not a count of unique roads or exact closure duration. Parent dates and water-stage descendants do not create additional independent storms. The user-selected three-row mosaic stacks a, b and c in the left column; d occupies both right-hand columns and spans all three rows. The map keeps geographic proportions, while the original ten cumulative values and their units remain unchanged.

Panel d is the original frozen high-stage no-action closure-peak state for the 2025-09-11 rainfall parent, frame 29, approximately 2 h 25 min after model start. It is a simultaneous road state, not a map of every road's separate peak. Open by rule means lower mapped depth below 0.10 m, restricted means at least 0.10 m but below 0.30 m, and closed means at least 0.30 m. Unknown structural or hazard exposure is gray dashed and is not treated as safe or flooded. Vector geometries are the existing union-clipped source roads, transformed from JGD2011 zone IX to geographic coordinates for display only. This parent is the disclosed corridor-selection context, not a typical or independent validation event.

The map and curves cover 836.958 km of qualified roads and retain 277.920 km of unknown exposure as a separate denominator. No census population, stranded-person count, traffic queue or observed congestion is inferred here. Upper-depth, ward and other-road/protected strata remain existing supporting comparisons. Source masks, road weights, forcing, timestamps, thresholds and results are unchanged. No hydraulic simulation, traffic assignment or training is run by this generator.
"""
    (AUDIT/"road_interruption_and_recovery_caption.md").write_text(caption)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print("Validated: ten case trajectories and exposure integrals;109 anchor-case road-state reconstructions;map geometry and original support;450 dpi")
    print(json.dumps({"map_reference_closed_km":map_counts["Closed by rule"],"case_exposure_km_h":exposure.tolist()}))

if __name__ == "__main__":
    main()
