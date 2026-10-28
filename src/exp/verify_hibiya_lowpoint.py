"""Verify the previously detected Hibiya low point using original LAS and ortho.

Run .venv/bin/python src/exp/verify_hibiya_lowpoint.py. Requires the already
approved numpy, rasterio, matplotlib, scipy. Does not change source values.
"""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
import os
for key in ['PROJ_LIB', 'PROJ_DATA', 'GDAL_DATA']:
    os.environ.pop(key, None)
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/udt01b-mpl')
import csv
import json
import struct
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = moga_raw(ROOT, 'tokyo_lidar_dem/2024/pilot_hibiya')
PILOT = ROOT / 'data/exp/data-briefing/plateau_terrain_pilot'
OUT = ROOT / 'data/exp/data-briefing/hibiya_lowpoint_verification'
BOTTOM = np.array([-6918.25, -36324.25])
OTHER = np.array([-6922.25, -36327.75])
# Focused 70 x 48 m window; chosen around the previous DEM minimum.
BOUNDS = (-6953.25, -36348.25, -6883.25, -36300.25)
# A straight longitudinal check through the visible ramp; not a surveyed axis.
START = np.array([-6939.0, -36349.0])
AXIS = (BOTTOM - START) / np.linalg.norm(BOTTOM - START)


def read_local_las():
    path = RAW / '09LD2812.las'
    with path.open('rb') as stream:
        header = stream.read(227)
    assert header[:4] == b'LASF' and header[24:26] == bytes([1, 2])
    assert header[104] == 3 and struct.unpack_from('<H', header, 105)[0] == 34
    offset = struct.unpack_from('<I', header, 96)[0]
    count = struct.unpack_from('<I', header, 107)[0]
    assert path.stat().st_size == offset + count * 34
    scale = np.array(struct.unpack_from('<3d', header, 131))
    origin = np.array(struct.unpack_from('<3d', header, 155))
    dtype = np.dtype({'names': ['x', 'y', 'z', 'class', 'source', 'gps'],
                     'formats': ['<i4', '<i4', '<i4', 'u1', '<u2', '<f8'],
                     'offsets': [0, 4, 8, 15, 18, 20], 'itemsize': 34})
    points = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,))
    x = points['x'] * scale[0] + origin[0]
    y = points['y'] * scale[1] + origin[1]
    keep = (x >= BOUNDS[0]) & (x <= BOUNDS[2]) & (y >= BOUNDS[1]) & (y <= BOUNDS[3])
    return dict(x=x[keep], y=y[keep], z=points['z'][keep] * scale[2] + origin[2],
                classification=points['class'][keep] & 31, raw_class=points['class'][keep],
                source=points['source'][keep], gps=points['gps'][keep]), count


def sample(name, xy):
    with rasterio.open(PILOT / name) as dataset:
        return np.array([value[0] for value in dataset.sample(xy)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, total = read_local_las()
    np.savez_compressed(OUT / 'local_points.npz', **a)
    xy = np.column_stack((a['x'], a['y']))
    ground = a['classification'] == 2
    radius = np.linalg.norm(xy - BOTTOM, axis=1)
    rows = []
    for r in [0.5, 1, 2, 5]:
        selected = ground & (radius <= r)
        low = selected & (a['z'] < 0)
        rows.append({'radius_m': r, 'class2_count': int(selected.sum()),
                     'class2_below_zero_count': int(low.sum()),
                     'min_z_m': float(a['z'][selected].min()),
                     'median_all_class2_z_m': float(np.median(a['z'][selected])),
                     'max_z_m': float(a['z'][selected].max())})
    dem_sites = sample('tokyo_dem_native_0_50m.tif', [BOTTOM, OTHER])
    plateau_sites = sample('plateau_tin_on_0_50m_grid.tif', [BOTTOM, OTHER])
    nearest_low = np.flatnonzero(ground & (a['z'] < 0))
    nearest_low = nearest_low[np.argmin(radius[nearest_low])]
    s = (xy - START) @ AXIS
    cross = (xy - START) @ np.array([-AXIS[1], AXIS[0]])
    strip = (np.abs(cross) < 1.25) & (s >= -5) & (s <= 40)
    distances = np.arange(-5, 40.01, 0.25)
    line_xy = START + distances[:, None] * AXIS
    dem = sample('tokyo_dem_native_0_50m.tif', line_xy)
    pla = sample('plateau_tin_on_0_50m_grid.tif', line_xy)
    with (OUT / 'longitudinal_raster_profile.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['distance_m', 'easting_m', 'northing_m', 'tokyo_dem_z_m', 'plateau_z_m'])
        writer.writerows(zip(distances, line_xy[:, 0], line_xy[:, 1], dem, pla))
    with (OUT / 'ground_points_profile.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['distance_m', 'cross_offset_m', 'easting_m', 'northing_m', 'z_m', 'class', 'point_source_id', 'gps_week_time'])
        m = strip & ground
        writer.writerows(zip(s[m], cross[m], a['x'][m], a['y'][m], a['z'][m], a['classification'][m], a['source'][m], a['gps'][m]))
    low = ground & (a['z'] < 0)
    counts = {str(k): int(v) for k, v in zip(*np.unique(a['classification'], return_counts=True))}
    summary = {'tile_point_count': total, 'local_bounds_epsg6677': BOUNDS,
               'local_point_count': len(xy), 'local_class_counts': counts,
               'bottom_lonlat': [139.75691043875514, 35.67256765208067],
               'bottom_xy': BOTTOM.tolist(), 'plateau_minimum_xy': OTHER.tolist(),
               'minimum_location_separation_m': float(np.linalg.norm(BOTTOM - OTHER)),
               'tokyo_z_at_two_sites_m': dem_sites.tolist(), 'plateau_z_at_two_sites_m': plateau_sites.tolist(),
               'difference_of_separate_minima_m': float(plateau_sites[1] - dem_sites[0]),
               'plateau_minus_tokyo_at_dem_minimum_m': float(plateau_sites[0] - dem_sites[0]),
               'nearest_below_zero_class2_point': {key: float(a[key][nearest_low]) for key in ['x', 'y', 'z', 'gps']},
               'nearest_below_zero_class2_distance_m': float(radius[nearest_low]),
               'radius_checks': rows, 'local_below_zero_class2_count': int(low.sum()),
               'local_below_minus_half_class2_count': int((ground & (a['z'] < -0.5)).sum()),
               'below_zero_point_source_counts': {str(k): int(v) for k, v in zip(*np.unique(a['source'][low], return_counts=True))},
               'below_zero_gps_weektime_range': [float(a['gps'][low].min()), float(a['gps'][low].max())],
               'profile_start_xy': START.tolist(), 'profile_axis': AXIS.tolist(), 'profile_half_width_m': 1.25,
               'classification': 'Real engineered descending vehicle ramp at Hibiya underground parking portal; isolated DEM outlier unlikely. Exact height accuracy and upper/lower surface selection remain unresolved.',
               'limits': ['Class 2 is a classification, not proof of correct hydraulic surface selection.', 'LAS and DEM likely share observations; agreement is source-support checking, not independent accuracy validation.', 'Raster nearest-cell profile and minimum comparisons are exploratory, not flood depths.', 'Flight date and named vertical datum are not established by this LAS header or orthophoto tags.']}
    (OUT / 'verification_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = f'''Hibiya low-point verification — exploratory source audit

Conclusion
The feature at 35.672567652 N, 139.756910439 E is an engineered descending
vehicle ramp at the Hibiya underground parking portal, beside Hibiya Avenue.
The official 0.20 m Tokyo orthophoto shows the curved vehicle ramp, retaining
edges and a road crossing over its northern portal. The parking operator's
B1 floor map independently identifies vehicle entrance/exit ramps here.
Matching these sources supports the feature identification; the floor map is
schematic, not a coordinate survey. An isolated erroneous DEM pixel is unlikely.

Measured support
Full source tile: {total:,} LAS points, original Class 1/2/3 labels retained.
Within 2 m of the DEM minimum: {rows[2]['class2_count']} Class 2 points, including
{rows[2]['class2_below_zero_count']} below zero in the source elevation system.
The closest below-zero Class 2 point is {radius[nearest_low]:.3f} m away,
with elevation {a['z'][nearest_low]:.3f} m, versus DEM {dem_sites[0]:.3f} m.
Within the 70 x 48 m window, {low.sum()} Class 2 points are below zero;
they form a spatially continuous lower ramp, with point-source IDs 88, 89, 90.
These IDs do not by themselves identify surveyed flight dates or establish
independent accuracy. Upper and lower returns both carry Class 2 near the portal.
The nearest 0.5 m radius contains Class 2 heights from {rows[0]['min_z_m']:.3f}
to {rows[0]['max_z_m']:.3f} m. This is a multilevel site, not a flat surface.

Correction of the earlier 0.68 m description
Tokyo DEM minimum: {dem_sites[0]:.4f} m at E=-6918.25, N=-36324.25.
PLATEAU minimum: {plateau_sites[1]:.4f} m at E=-6922.25, N=-36327.75.
Their minima differ by {plateau_sites[1]-dem_sites[0]:.4f} m, but their locations
are {np.linalg.norm(BOTTOM-OTHER):.3f} m apart. This is not a same-location bias.
At the Tokyo DEM minimum XY, PLATEAU is {plateau_sites[0]:.4f} m: a same-location
difference of {plateau_sites[0]-dem_sites[0]:.4f} m. At the PLATEAU minimum XY,
Tokyo DEM is {dem_sites[1]:.4f} m. The coarse TIN misses/repositions the lowest
visible ramp portion and transitions to the upper surface earlier. The maps
and profiles support resolution/surface-representation effects; the exact
production algorithm responsible has not been established.

Implication for pluvial-flood simulation
Retain this as a real engineered ramp candidate; do not automatically fill it
as noise. Do not treat its raster depression depth as actual floodwater depth
or closed storage volume. The lower surface continues into an underground
facility that airborne data cannot fully survey. A single elevation per XY
cannot encode both the crossing road and the ramp. A hydraulic implementation
would need the entrance/portal and sill geometry, walls and surface connectivity,
plus drainage/pump and underground boundary information. None are assumed here.

Timing, datum and confidence
LAS creation date/ZIP timestamp: 2023-10-30 (processing, not flight date).
DEM ZIP member timestamp from the previous acquisition: 2023-12-08.
Orthophoto TIFF/TFW ZIP timestamps: 2024-11-21 (packaging, not photography date).
Official Tokyo publication describes FY2022–FY2023 acquisition/preparation.
Exact acquisition dates for this sheet remain unknown. Horizontal CRS of LAS
and ortho is JGD2011 / Japan Plane Rectangular CS IX (EPSG:6677); no coordinate
shift or elevation correction was applied. A named vertical datum and measured
absolute vertical accuracy are not established by these source headers.
Confidence is high that the descending engineered feature is real; confidence
in its exact hydraulic bottom/sill height and present-day condition is pending.
DEM/LAS agreement is source consistency, not independent validation.

Reproduction and evidence
1. .venv/bin/python src/preprocessing/acquire_hibiya_lowpoint_sources.py
2. .venv/bin/python src/exp/verify_hibiya_lowpoint.py
The profile samples existing raster cell values at 0.25 m steps and displays
all Class 2 points within 1.25 m of its straight axis. This exploratory axis
crosses adjacent surfaces before reaching the northern descending segment;
it is not a surveyed curved ramp centerline. Original data are not edited.
Both acquired ZIP archives passed full CRC and have SHA256 provenance.
Source URLs and timestamps are recorded in source_provenance.json.
Official parking map: https://www.driveplaza.com/assets/pdf/hibiya_parking/floormap.pdf
Official Tokyo catalog: https://catalog.data.metro.tokyo.lg.jp/dataset/t000029d0000000024
Official release: https://www.my.metro.tokyo.lg.jp/w/000-20241031-65748882
'''
    (OUT / 'assessment.txt').write_text(report)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    extent = (BOUNDS[0] - BOTTOM[0], BOUNDS[2] - BOTTOM[0], BOUNDS[1] - BOTTOM[1], BOUNDS[3] - BOTTOM[1])
    with rasterio.open(RAW / 'orthophoto/09LD2812.tif') as ds:
        window = from_bounds(*BOUNDS, ds.transform).round_offsets().round_lengths()
        rgb = ds.read(window=window).transpose(1, 2, 0)
        b = ds.window_bounds(window)
        image_extent = (b[0] - BOTTOM[0], b[2] - BOTTOM[0], b[1] - BOTTOM[1], b[3] - BOTTOM[1])
        axes[0, 0].imshow(rgb, extent=image_extent)
        ortho_metadata = {'crs': ds.crs.to_string(), 'resolution': list(ds.res), 'tags': ds.tags()}
    axes[0, 0].set_title('Official Tokyo 0.20 m orthophoto: vehicle ramp')
    sc = axes[0, 1].scatter(a['x'][ground] - BOTTOM[0], a['y'][ground] - BOTTOM[1], c=a['z'][ground], s=1, cmap='terrain', vmin=-1, vmax=4, rasterized=True)
    axes[0, 1].set_title('Original LAS Class 2 points: continuous lower ramp')
    fig.colorbar(sc, ax=axes[0, 1], label='Source elevation (m; datum not resolved)')
    for ax in axes[0]:
        ax.plot(0, 0, 'x', color='red', ms=9, mew=2, label='Tokyo DEM minimum')
        ax.plot(*(OTHER - BOTTOM), '+', color='cyan', ms=9, mew=2, label='PLATEAU minimum')
        ax.plot(line_xy[:, 0] - BOTTOM[0], line_xy[:, 1] - BOTTOM[1], color='magenta', lw=1, label='Profile axis')
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect('equal')
        ax.set_xlabel('Easting from DEM minimum (m)'); ax.set_ylabel('Northing from DEM minimum (m)')
    axes[0, 0].legend(loc='upper left', fontsize=7)
    ax = axes[1, 0]
    other = strip & ~ground
    ax.scatter(s[other], a['z'][other], s=2, color='0.75', label='Class 1/3', rasterized=True)
    m = strip & ground
    ax.scatter(s[m], a['z'][m], s=3, color='black', label='Class 2', rasterized=True)
    ax.plot(distances, dem, color='tab:blue', lw=1.5, label='Tokyo 0.50 m DEM')
    ax.plot(distances, pla, color='tab:orange', lw=1.5, label='PLATEAU TIN on same grid')
    ax.axvline(np.linalg.norm(BOTTOM - START), color='red', ls=':', lw=1)
    ax.set_ylim(-1.3, 4.5); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    ax.set_xlabel('Distance along straight profile (m)'); ax.set_ylabel('Source elevation (m)')
    ax.set_title('Straight profile across adjacent surfaces and lower ramp')
    ax = axes[1, 1]
    nearby = radius <= 5
    ax.scatter(radius[nearby & ~ground], a['z'][nearby & ~ground], s=3, color='0.75', label='Class 1/3')
    ax.scatter(radius[nearby & ground], a['z'][nearby & ground], s=4, color='black', label='Class 2')
    ax.axhline(dem_sites[0], color='tab:blue', label=f'DEM at minimum: {dem_sites[0]:.2f} m')
    ax.axhline(plateau_sites[0], color='tab:orange', label=f'PLATEAU same XY: {plateau_sites[0]:.2f} m')
    ax.set_ylim(-1.3, 4.5); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    ax.set_xlabel('Horizontal distance from DEM minimum (m)'); ax.set_ylabel('Source elevation (m)')
    ax.set_title('Portal vicinity: Class 2 contains two elevation levels')
    fig.suptitle('Hibiya low-point verification | 09LD2812 | 35.672568 N, 139.756910 E', fontsize=13)
    fig.text(0.5, -0.014, 'Sources: Tokyo Digital Twin official ortho/LAS, PLATEAU Chiyoda 2025. No height correction. Class labels retained.', ha='center', fontsize=8)
    fig.savefig(OUT / 'lowpoint_verification.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'orthophoto_metadata.json').write_text(json.dumps(ortho_metadata, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
