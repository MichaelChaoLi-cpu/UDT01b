"""Focused road terrain audit; preserves source LAS, DEM and PLATEAU values.

Run: .venv/bin/python src/exp/verify_hibiya_road.py
The selected road corridor and screening thresholds are exploratory choices,
not a final study area, engineering tolerance or independent accuracy test.
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
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = moga_raw(ROOT, 'tokyo_lidar_dem/2024/pilot_hibiya')
PILOT = ROOT / 'data/exp/data-briefing/plateau_terrain_pilot'
OUT = ROOT / 'data/exp/data-briefing/hibiya_road_verification'
START = np.array([-6956., -36465.])
END = np.array([-6923., -36403.])
LENGTH = float(np.linalg.norm(END - START))
U = (END - START) / LENGTH
V = np.array([-U[1], U[0]])  # positive to west/park; negative to east/buildings
STATIONS = [15., 35., 55.]
TO_LONLAT = Transformer.from_crs(6677, 6668, always_xy=True)


def position(s, c):
    return START + np.asarray(s)[..., None] * U + np.asarray(c)[..., None] * V


def read_points():
    path = RAW / '09LD2812.las'
    with path.open('rb') as f:
        h = f.read(227)
    if h[:4] != b'LASF' or h[104] != 3 or struct.unpack_from('<H', h, 105)[0] != 34:
        raise ValueError('Expected uncompressed LAS 1.2 point format 3')
    n = struct.unpack_from('<I', h, 107)[0]
    offset = struct.unpack_from('<I', h, 96)[0]
    if path.stat().st_size != offset + 34 * n:
        raise ValueError('LAS length mismatch')
    scales = np.array(struct.unpack_from('<3d', h, 131))
    origin = np.array(struct.unpack_from('<3d', h, 155))
    dtype = np.dtype({'names': ['x', 'y', 'z', 'classification', 'source'],
                     'formats': ['<i4', '<i4', '<i4', 'u1', '<u2'],
                     'offsets': [0, 4, 8, 15, 18], 'itemsize': 34})
    a = np.memmap(path, mode='r', dtype=dtype, offset=offset, shape=(n,))
    x = a['x'] * scales[0] + origin[0]
    y = a['y'] * scales[1] + origin[1]
    # Bounding box enclosing the 48 m wide corridor plus context.
    keep = (x > -7005) & (x < -6880) & (y > -36490) & (y < -36380)
    xy = np.column_stack([x[keep], y[keep]])
    return dict(xy=xy, z=a['z'][keep] * scales[2] + origin[2],
                classification=a['classification'][keep] & 31, source=a['source'][keep],
                s=(xy - START) @ U, c=(xy - START) @ V)


def raster_samples(name, xy):
    with rasterio.open(PILOT / name) as d:
        return np.array([z[0] for z in d.sample(xy)])


def statistics(x):
    x = np.asarray(x)
    return {'n': len(x), 'mean_m': float(np.mean(x)), 'median_m': float(np.median(x)),
            'mae_m': float(np.mean(abs(x))), 'rms_m': float(np.sqrt(np.mean(x*x))),
            'p95_abs_m': float(np.percentile(abs(x), 95))} if len(x) else {'n': 0}


def write_csv(path, fields, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = read_points()
    ground = a['classification'] == 2
    corridor = (a['s'] >= 0) & (a['s'] <= LENGTH) & (abs(a['c']) <= 24)
    # Visible carriageways; exclude raised median and edge/sidewalk transitions.
    road = corridor & ground & (((a['c'] >= -12) & (a['c'] <= 0)) |
                                ((a['c'] >= 6) & (a['c'] <= 15)))
    np.savez_compressed(OUT / 'corridor_points.npz', **{k: v[corridor] for k, v in a.items()})
    with rasterio.open(PILOT / 'tokyo_dem_native_0_50m.tif') as d:
        dem = d.read(1); transform = d.transform
        rr, cc = rasterio.transform.rowcol(transform, a['xy'][road, 0], a['xy'][road, 1])
    with rasterio.open(PILOT / 'plateau_tin_on_0_50m_grid.tif') as d:
        pla = d.read(1)
    rr, cc = np.asarray(rr), np.asarray(cc)
    cell_id = rr * dem.shape[1] + cc
    order = np.argsort(cell_id)
    unique, first, count = np.unique(cell_id[order], return_index=True, return_counts=True)
    road_z = a['z'][road]
    cells = []
    for cid, begin, n in zip(unique, first, count):
        z = road_z[order[begin:begin+n]]
        r, c = divmod(int(cid), dem.shape[1]); med = float(np.median(z))
        spread = float(np.percentile(z, 90) - np.percentile(z, 10))
        x, y = rasterio.transform.xy(transform, r, c)
        s, cross = (np.array([x, y]) - START) @ U, (np.array([x, y]) - START) @ V
        cells.append({'easting_m': x, 'northing_m': y, 'station_m': s, 'cross_m': cross,
                      'class2_count': int(n), 'p90_p10_z_m': spread, 'ground_median_z_m': med,
                      'dem_z_m': float(dem[r,c]), 'plateau_z_m': float(pla[r,c]),
                      'dem_minus_ground_m': float(dem[r,c]-med),
                      'plateau_minus_ground_m': float(pla[r,c]-med),
                      'supported_single_surface': int(n >= 3 and spread <= .1)})
    write_csv(OUT / 'road_cell_support.csv', list(cells[0]), cells)
    supported = [x for x in cells if x['supported_single_surface']]
    # Longitudinal profile in an unobstructed traffic lane, offset -3 m.
    lane = corridor & ground & (abs(a['c'] + 3) <= .4)
    longitudinal = []
    for s in np.arange(.5, LENGTH, 1):
        m = lane & (abs(a['s']-s) <= .5)
        xy = position(s, -3)
        d = float(raster_samples('tokyo_dem_native_0_50m.tif', [xy])[0])
        p = float(raster_samples('plateau_tin_on_0_50m_grid.tif', [xy])[0])
        longitudinal.append({'station_m': float(s), 'easting_m': xy[0], 'northing_m': xy[1],
                             'class2_count': int(m.sum()), 'ground_median_z_m': float(np.median(a['z'][m])) if m.any() else np.nan,
                             'dem_z_m': d, 'plateau_z_m': p})
    write_csv(OUT / 'longitudinal_profile.csv', list(longitudinal[0]), longitudinal)
    valid = [r for r in longitudinal if r['class2_count'] >= 5]
    slopes = {}
    for key in ['ground_median_z_m', 'dem_z_m', 'plateau_z_m']:
        coef = np.polyfit([r['station_m'] for r in valid], [r[key] for r in valid], 1)
        slopes[key] = {'linear_slope_percent': float(coef[0]*100),
                       'fitted_delta_over_segment_m': float(coef[0]*LENGTH),
                       'first_three_bins_median_z_m': float(np.median([r[key] for r in valid[:3]])),
                       'last_three_bins_median_z_m': float(np.median([r[key] for r in valid[-3:]]))}
    # Adjacent surface bands: height contrasts, NOT exact vertical kerb faces.
    bands = {'east_road': (-13., -12.), 'east_sidewalk_candidate': (-18.5, -17.),
             'west_road': (15., 16.), 'west_sidewalk_candidate': (18.5, 20.5)}
    sections = []; section_samples = []; band_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for ax, station in zip(axes, STATIONS):
        strip = corridor & (abs(a['s']-station) <= .4)
        g = strip & ground
        nonground = strip & ~ground
        cc = np.arange(-24, 24.001, .1); q = position(station, cc)
        dz = raster_samples('tokyo_dem_native_0_50m.tif', q)
        pz = raster_samples('plateau_tin_on_0_50m_grid.tif', q)
        for cross, xy, d, p in zip(cc, q, dz, pz):
            section_samples.append(dict(station_m=station, cross_m=cross, easting_m=xy[0], northing_m=xy[1], dem_z_m=float(d), plateau_z_m=float(p)))
        result = {'station_m': station, 'bands': {}}
        for name, (lo, hi) in bands.items():
            m = g & (a['c'] >= lo) & (a['c'] <= hi)
            raster_band = (cc >= lo) & (cc <= hi)
            result['bands'][name] = {'class2_count': int(m.sum()), 'ground_median_z_m': float(np.median(a['z'][m])) if m.any() else None,
                                     'ground_p90_p10_m': float(np.percentile(a['z'][m],90)-np.percentile(a['z'][m],10)) if m.any() else None,
                                     'dem_median_z_m': float(np.median(dz[raster_band])), 'plateau_median_z_m': float(np.median(pz[raster_band]))}
            band_rows.append({'station_m': station, 'band': name, 'cross_lo_m': lo, 'cross_hi_m': hi, **result['bands'][name]})
        for side in ['east', 'west']:
            b1, b2 = result['bands'][side+'_road'], result['bands'][side+'_sidewalk_candidate']
            result[side+'_height_contrast_m'] = {key: (b2[key]-b1[key]) if b1[key] is not None and b2[key] is not None else None for key in ['ground_median_z_m','dem_median_z_m','plateau_median_z_m']}
        sections.append(result)
        ax.scatter(a['c'][nonground], a['z'][nonground], s=3, color='.8', label='Class 1/3')
        ax.scatter(a['c'][g], a['z'][g], s=6, color='black', label='Class 2 ground')
        ax.plot(cc, dz, lw=1.4, label='Tokyo 0.50 m DEM')
        ax.plot(cc, pz, lw=1.4, label='PLATEAU terrain')
        for name, (lo, hi) in bands.items():
            ax.axvspan(lo, hi, color='grey', alpha=.1)
        ax.set_title(f'Cross-section at station {station:.0f} m; point strip +/-0.4 m')
        ax.set_ylim(2.15, 3.25); ax.grid(alpha=.25); ax.set_ylabel('Source elevation (m)')
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel('Cross offset (m): east/buildings < 0; west/park > 0')
    axes[-1].set_xticks(np.arange(-24,25,2))
    fig.suptitle('Hibiya Avenue | adjacent-surface heights and DEM smoothing', fontsize=13)
    fig.savefig(OUT / 'cross_sections.png', dpi=180); plt.close(fig)
    write_csv(OUT / 'cross_section_raster_samples.csv', list(section_samples[0]), section_samples)
    write_csv(OUT / 'surface_band_heights.csv', list(band_rows[0]), band_rows)
    summary = {'selection': {'start_xy': START.tolist(), 'end_xy': END.tolist(), 'length_m': LENGTH,
                             'start_lonlat': list(TO_LONLAT.transform(*START)), 'end_lonlat': list(TO_LONLAT.transform(*END)),
                             'axis_unit_vector': U.tolist(), 'positive_cross_direction': 'west/park',
                             'minimum_axis_distance_to_known_parking_lowpoint_m': float(np.linalg.norm(END-np.array([-6918.25,-36324.25]))),
                             'reason': 'Ordinary visible carriageways between the park and eastern buildings; exclude known portal and intersections. Sidewalk edges still require local structural checks.'},
               'class2_road_points': int(road.sum()), 'occupied_road_cells': len(cells),
               'supported_road_cells': len(supported), 'supported_cell_rule': 'At least 3 Class 2 points and within-cell P90-P10 <=0.10 m; exploratory screening only.',
               'dem_minus_ground_supported_cells': statistics([r['dem_minus_ground_m'] for r in supported]),
               'plateau_minus_ground_supported_cells': statistics([r['plateau_minus_ground_m'] for r in supported]),
               'longitudinal_bin_count': len(longitudinal), 'longitudinal_supported_bins': len(valid), 'longitudinal_slope': slopes,
               'cross_sections': sections,
               'limits': ['No independent height accuracy test; DEM and LAS share the survey source.', 'No height correction applied; named vertical datum and exact acquisition date still unresolved.', 'Height contrasts compare adjacent surface bands with road crossfall and within-band slopes; they are not surveyed kerb-face heights.', 'Ground Class 2 can omit raised structures; source classification retained. Native nearest-cell raster values are repeated at 0.10 m steps for display, not new 0.10 m resolution.', 'PLATEAU terrain result does not evaluate building elevations or all three wards.']}
    (OUT / 'verification_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    # Lane profile and ground-to-grid support map.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    s = np.array([r['station_m'] for r in longitudinal])
    axes[0].scatter(a['s'][lane], a['z'][lane], color='black', s=2, label='Class 2 ground')
    for key, label in [('dem_z_m','Tokyo DEM'), ('plateau_z_m','PLATEAU terrain')]:
        axes[0].plot(s, [r[key] for r in longitudinal], label=label)
    axes[0].set_xlabel('Station SW to NE (m)'); axes[0].set_ylabel('Source elevation (m)')
    axes[0].set_title('Traffic-lane longitudinal profile (cross offset -3 m)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    sc = axes[1].scatter([r['station_m'] for r in cells], [r['cross_m'] for r in cells], c=[r['dem_minus_ground_m'] for r in cells], s=5, cmap='RdBu_r', vmin=-.1, vmax=.1)
    axes[1].set_xlabel('Station SW to NE (m)'); axes[1].set_ylabel('Cross offset (m)')
    axes[1].set_title('Native 0.50 m cells: DEM minus Class 2 median')
    fig.colorbar(sc, ax=axes[1], label='Source consistency difference (m)')
    fig.savefig(OUT / 'longitudinal_and_cell_support.png', dpi=180); plt.close(fig)
    # Review candidates are identified from the ortho and all three profiles.
    # They mark where geometry needs attention; source rasters are not repaired.
    candidates = [
        ('K1',15.,17.,'PLATEAU edge representation correction candidate',
         'West adjacent-surface contrast: Class 2 0.2125 m, DEM 0.2000 m, PLATEAU 0.07225 m.',
         'Use finer ground terrain for this transition; survey a kerb breakline/crest before enforcing a hydraulic barrier.'),
        ('K2',35.,18.,'Supplemental survey: near-level western verge under tree cover',
         'West adjacent-band contrast is 0.0025 m in Class 2 and 0.0050 m in DEM. Tree cover obscures parts of the edge; the band is not a confirmed continuous sidewalk.',
         'Distinguish vegetated ground, an access opening/lowered kerb, and an unresolved narrow kerb. Survey the actual hard edge and drainage; do not infer a continuous raised kerb or a confirmed opening.'),
        ('K3',55.,-15.5,'Supplemental survey: eastern kerb transition',
         'East Class 2 surface-band contrast 0.1570 m vs DEM 0.1500 m; the narrow transition is sparsely represented relative to the broad surfaces.',
         'Measure kerb face/breakline and any gutter/inlets; broad surface heights do not establish centimetre-scale lip geometry.'),
        ('K4',35.,2.5,'Classification and supplemental geometry review: central divider',
         'Class 2 coverage is discontinuous at the raised/vegetated divider; other-class returns lie above the interpolated ground surface.',
         'Separate vegetation/objects from physical divider edges and verify barrier height/continuity; do not interpret interpolated bare ground as its top.')
    ]
    issues = []; features = []
    for ident, s0, c0, status, evidence, action in candidates:
        xy = position(s0,c0); lon,lat=TO_LONLAT.transform(*xy)
        prop = {'id':ident,'station_m':s0,'cross_m':c0,'easting_m':xy[0],'northing_m':xy[1],
                'longitude':lon,'latitude':lat,'status':status,'evidence':evidence,'recommended_action':action,
                'decision_status':'Agent screening candidate; no terrain values changed.'}
        issues.append(prop); features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[lon,lat]},'properties':prop})
    write_csv(OUT/'correction_and_survey_candidates.csv',list(issues[0]),issues)
    (OUT/'correction_and_survey_candidates.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,9),constrained_layout=True)
    bounds=(-7000,-36485,-6880,-36385)
    with rasterio.open(RAW/'orthophoto/09LD2812.tif') as d:
        win=from_bounds(*bounds,d.transform).round_offsets().round_lengths();rgb=d.read(window=win).transpose(1,2,0);bb=d.window_bounds(win)
    ax.imshow(rgb,extent=(bb[0],bb[2],bb[1],bb[3]))
    poly=np.array([position(0,-24),position(LENGTH,-24),position(LENGTH,24),position(0,24),position(0,-24)])
    ax.plot(poly[:,0],poly[:,1],color='cyan',lw=1)
    line=position(np.array([0,LENGTH]),-3)
    ax.plot(line[:,0],line[:,1],color='yellow',lw=2,label='Longitudinal traffic lane')
    for s0 in STATIONS:
        line=position(s0,np.array([-24,24]));ax.plot(line[:,0],line[:,1],color='magenta',lw=1)
        ax.text(*position(s0,-25),f'S{s0:.0f}',fontsize=9,color='magenta',bbox={'facecolor':'white','alpha':.8,'pad':1})
    for prop in issues:
        color='red' if prop['id']=='K1' else 'orange'
        ax.plot(prop['easting_m'],prop['northing_m'],'o',color=color,ms=7)
        ax.annotate(prop['id'],(prop['easting_m'],prop['northing_m']),xytext=(6,7),textcoords='offset points',fontweight='bold',bbox={'facecolor':'white','alpha':.85,'pad':1})
    ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_aspect('equal')
    ax.set_xlabel('Easting (m; EPSG:6677)');ax.set_ylabel('Northing (m; EPSG:6677)')
    ax.set_title('Hibiya ordinary road | 70.24 m | correction/survey candidates')
    ax.legend(loc='upper left',fontsize=8)
    fig.text(.5,-.015,'K1: coarse edge | K2: near-level verge | K3: eastern kerb | K4: divider/classification\nOfficial Tokyo 0.20 m ortho. Locations are screening candidates, not survey marks.',ha='center',fontsize=9)
    fig.savefig(OUT/'road_selection_and_review_map.png',dpi=180,bbox_inches='tight');plt.close(fig)
    provenance=[]
    for path in [RAW/'09LD2812.las',RAW/'09LD2812.tif',RAW/'orthophoto/09LD2812.tif',PILOT/'plateau_tin_on_0_50m_grid.tif']:
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        provenance.append({'path':moga_source(path, ROOT),'bytes':path.stat().st_size,'sha256':sha})
    (OUT/'input_provenance.json').write_text(json.dumps({'inputs':provenance,'official_sources_record':'data/exp/data-briefing/hibiya_lowpoint_verification/source_provenance.json','plateau_interpolation_script':'src/exp/compare_hibiya_terrain.py'},indent=2)+'\n')
    report=f'''Hibiya ordinary-road terrain check — exploratory evidence

Selection
Hibiya Avenue between the park and eastern buildings. Reference axis is
{LENGTH:.2f} m, from E/N {-6956}/{-36465} to {-6923}/{-36403} (EPSG:6677).
The known parking-ramp low point is at least 78.89 m from this axis. This
distance is to the verified low point, not an exhaustive entrance clearance.
Ortho review excludes visible portals/intersections from the ordinary
carriageway. All possible underground structures below the road are not known.
The longitudinal profile is in a traffic lane at cross offset -3 m; positive
cross offsets point west toward the park. Cross-sections are at 15, 35, 55 m.
Class 2 point strips are +/-0.4 m around each axis, original labels retained.
An agent-selected pilot segment, not a final human study-area decision.

Longitudinal slope
All {len(longitudinal)} one-metre profile bins contain at least five ground points.
Linear slopes SW to NE: ground {slopes['ground_median_z_m']['linear_slope_percent']:.4f}%,
0.50 m DEM {slopes['dem_z_m']['linear_slope_percent']:.4f}%, PLATEAU
{slopes['plateau_z_m']['linear_slope_percent']:.4f}%. The source ground profile
falls overall toward NE, with fitted change about -0.174 m over this segment;
first/last three-bin ground medians differ by -0.145 m (profile is not perfectly
linear). This describes slope; it is not a full drainage-route calculation.

Carriageway consistency
{int(road.sum()):,} ground points occupy {len(cells):,} native 0.50 m road cells;
{len(supported):,} occupied cells meet the exploratory screening rule of >=3
points and within-cell P90-P10 <=0.10 m. This fraction is among occupied cells,
not total road coverage. Raised median and sidewalk transitions are excluded
from this interior-road statistic. DEM versus Class 2 cell medians: MAE
{summary['dem_minus_ground_supported_cells']['mae_m']:.4f} m, RMS difference
{summary['dem_minus_ground_supported_cells']['rms_m']:.4f} m. PLATEAU versus
the same ground medians: MAE {summary['plateau_minus_ground_supported_cells']['mae_m']:.4f} m,
RMS difference {summary['plateau_minus_ground_supported_cells']['rms_m']:.4f} m,
median {summary['plateau_minus_ground_supported_cells']['median_m']:.4f} m.
These small differences are not millimetre absolute-accuracy claims: DEM and
LAS share source observations; the ground cell median can differ from the
height at its centre on a slope. No independent benchmark was used.

Cross-section surface contrasts
Compare fixed adjacent ROAD and SIDEWALK-CANDIDATE bands, not kerb faces.
Station | side | Class 2 contrast | DEM contrast | PLATEAU contrast (metres)
'''
    for section in sections:
        for side in ['east','west']:
            h=section[side+'_height_contrast_m']
            report+=f"{section['station_m']:5.0f} | {side:4} | {h['ground_median_z_m']:.4f} | {h['dem_median_z_m']:.4f} | {h['plateau_median_z_m']:.4f}\n"
    report+='''
The DEM resolves broad raised adjacent surfaces at the checked sections,
including east contrasts about 0.09–0.16 m and west contrasts about 0.13–0.21 m
at S15/S55. The west S35 contrast is essentially zero in both ground and DEM;
the canopy-obscured band could represent vegetated ground, an access opening/
lowered edge, or ground behind an unresolved narrow kerb. It is not a confirmed
continuous sidewalk, and a small surface contrast does not prove no barrier.
The PLATEAU TIN reduces the west S15/S55 contrasts to about 0.07/0.05 m.
Adjacent bands are metres apart: contrasts include crossfall, gutter and
sidewalk slope. They must not be reported as exact kerb vertical-face heights.
0.50 m cells do not prove narrow lips, kerb widths or inlet sills are resolved.
The displayed 0.10 m raster-profile steps repeat native cells; no new resolution
is created. No height shift, ground reconstruction or source-data repair applied.

Correction / supplemental-survey locations
K1 west S15: coarse PLATEAU edge representation needs finer geometry; DEM retains
the broad contrast, but a surveyed breakline is needed for exact hydraulic edges.
K2 west S35: distinguish vegetated verge, lowered edge/opening and unresolved
narrow kerb, and survey actual hard-edge/drainage geometry. Do not raise this
location merely to match typical kerbs at neighbouring sections.
K3 east S55: survey narrow kerb/gutter transition and any inlet elevations.
K4 central divider S35: discontinuous Class 2 support; distinguish vegetation
and physical barriers and check barrier top/continuity. A DEM of bare ground
does not necessarily represent the hydraulic top of a raised divider.
These four are illustrative review targets, not an exhaustive deficiency map.
Coordinates and evidence are in correction_and_survey_candidates.csv/.geojson.

Does the 3D city model have an elevation problem?
For this ordinary road, PLATEAU terrain follows the measured longitudinal
trend and broad cross-road form; no substantial uniform offset is evident.
At kerbs/dividers and underground portals, coarse/single-valued terrain
representation can suppress or mix hydraulically important surfaces. The
previous portal mismatch and these ordinary-road checks support local
representation limitations, not a demonstrated citywide absolute-height bias.
Building model elevations have not been tested. Results from this one segment
cannot certify all PLATEAU Tokyo roads or all three wards.

Remaining limits
Exact survey/photography dates, named vertical datum and height-revision status
are still unresolved. Original LAS/DEM heights are preserved. For drainage
elevations or decimetre-sensitive flood barriers, use independent surveyed
checkpoints and structural/inlet geometry; source consistency alone is
insufficient. Present-day road and sidewalk modifications are not checked.

Reproduce
.venv/bin/python src/exp/verify_hibiya_road.py
input_provenance.json preserves hashes of the four inputs. Original source
URLs/ZIP checks are in the previous low-point source_provenance.json. Scripts,
CSV summaries, GeoJSON review positions and scientific figures run locally.
No AnaSOP, Git or DVC update performed. Human workflow decision:
MILI-D-20261013-003. Selection coordinates and review labels are agent choices.
'''
    (OUT/'assessment.txt').write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
