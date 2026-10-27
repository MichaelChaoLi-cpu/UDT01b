#!/usr/bin/env python3
"""Read-only hydraulic label feasibility audit. Never runs a solver or creates ML tensors.

Requires existing numpy and netCDF4. All outputs are exploratory audit reports.
Run: /private/tmp/udt01b-sfincs-runner/bin/python src/analyses/audit_surrogate_reference_labels.py
"""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import netCDF4

ROOT = Path(__file__).resolve().parents[2]
CASES = ['rain60_low', 'rain100_low', 'rain60_raised', 'dry_raised']
GRIDS = ['corrected_10m', 'adaptive_10_5m']

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def arr(v):
    return np.ma.filled(np.ma.asarray(v, dtype=np.float64), np.nan)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'data/exp/feasibility-check/ml-surrogate'
    out.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(p.relative_to(root))
    read_json = lambda p: json.loads(p.read_text(encoding='utf-8'))
    base = root / 'data/processed/three_ward_sfincs_framework'
    audit_dir = root / 'data/exp/data-briefing/three_ward_adaptive_sensitivity'
    protected = [root / 'docs/AnaSOP.md', root / 'data/exp/research-design/ml-surrogate/design-contract.json']
    protected_before = {rel(p): digest(p) for p in protected}
    findings = read_json(audit_dir / 'findings.json')
    budgets = {x['run']: x for x in read_json(audit_dir / 'water_budgets.json')}
    datasets, variables, runs, excluded, evidence = [], [], [], [], []
    hash_cache = {}
    # Hard-linked geometry is hashed once per physical file, even when reused by many runs.
    def cached_digest(p):
        s = p.stat()
        key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if key not in hash_cache:
            hash_cache[key] = digest(p)
        return hash_cache[key]
    def register(p, role):
        s = p.stat()
        datasets.append({'source_file': rel(p), 'role': role, 'readable': True,
                         'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns})
    def nc_inventory(p, role):
        register(p, role)
        with netCDF4.Dataset(p) as d:
            for k, v in d.variables.items():
                variables.append({'source_file': rel(p), 'variable': k,
                                  'dimensions': json.dumps(v.dimensions), 'shape': json.dumps(v.shape),
                                  'units_in_file': str(getattr(v, 'units', '')),
                                  'long_name_in_file': str(getattr(v, 'long_name', ''))})
            return {k: len(v) for k, v in d.dimensions.items()}
    for case in CASES:
        for grid in GRIDS:
            name = case + '_' + grid
            folder = base / ('adaptive_connectivity_comparison' if case == CASES[0] else 'adaptive_sensitivity') / name
            ex = read_json(folder / 'execution.json')
            contract = read_json(folder / 'scenario_contract.json')
            checks = {f: cached_digest(folder / f) == h for f, h in ex['input_sha256'].items()}
            for f in ['execution.json', 'scenario_contract.json', 'sfincs.inp', 'sfincs.bzs', 'flux_sections.json']:
                p = folder / f
                evidence.append({'source': rel(p), 'sha256': cached_digest(p)})
            mapdims = nc_inventory(folder / 'sfincs_map.nc', 'native hydraulic state and extrema')
            hisdims = nc_inventory(folder / 'sfincs_his.nc', 'station histories and selected section fluxes')
            nc_inventory(folder / 'roof_surface_precip.nc', 'effective source rain; not raw meteorological rain')
            with netCDF4.Dataset(folder / 'sfincs_map.nc') as d:
                ts = arr(d['time'][:])
                active = np.asarray(d['msk'][:]) == 1
                peak = arr(d['zsmax'][0])
                frames = arr(d['zs'][:])
                finite = np.isfinite(frames)
                comparable = active & np.isfinite(peak) & finite.any(axis=0)
                sampled = np.where(finite, frames, -np.inf).max(axis=0)
                gaps = peak[comparable] - sampled[comparable]
                k = int(np.argmin(abs(ts - 7200)))
                h, zmin = arr(d['h'][k]), arr(d['zb'][:])
                ok = active & finite[k] & np.isfinite(h)
                h_semantic_error = float(np.max(abs(h[ok] + zmin[ok] - frames[k][ok])))
                final_status = float(d['status'][0])
                dynamic_vars = [v for v in ['zs', 'h', 'u', 'v', 'subgrid_volume'] if v in d.variables]
                full_face_flux = any(v in d.variables for v in ['q', 'qx', 'qy', 'uv', 'face_discharge'])
                tpeak_exported = 't_zsmax' in d.variables
            with netCDF4.Dataset(folder / 'sfincs_his.nc') as d:
                ht = arr(d['time'][:])
                hdt = np.diff(ht)
                history_status = float(d['status'][0])
                sections = len(d.dimensions['crosssections'])
                stations = len(d.dimensions['stations'])
            with netCDF4.Dataset(folder / 'roof_surface_precip.nc') as d:
                rain = d['Precipitation']
                forcing_times = arr(d['time'][:]).tolist()
                forcing_units = str(getattr(rain, 'units', ''))
                forcing_unit_singular = str(getattr(rain, 'unit', ''))
                forcing_fill = float(rain._FillValue)
                # Solver-specific input: zero is both a real dry forcing and NetCDF fill.
                # CF auto-masking would erase all dry frames; inspect raw stored values.
                rain.set_auto_mask(False)
                forcing_time_units = str(d['time'].units)
                rain_max = [float(np.nanmax(arr(rain[i]))) for i in range(len(d['time']))]
            stages = np.loadtxt(folder / 'sfincs.bzs')
            register(folder / 'sfincs.bzs', 'prescribed water levels; first column seconds')
            cache = root / 'data/processed/three_ward_adaptive_sensitivity' / (name + '.npz')
            register(cache, 'peak-level expansion and 7200s/20m velocity cache; not primary target')
            with np.load(cache) as d:
                cache_fields = {k: {'shape': list(d[k].shape), 'dtype': str(d[k].dtype)} for k in d.files}
            log = (folder / 'solver_stdout.txt').read_text(errors='replace')
            completed = (ex['returncode'] == 0 and ex['completion_marker'] and
                         ('Simulation finished' in log or 'Simulation is finished' in log) and
                         final_status == 0 and history_status == 0 and ts[-1] == 14400 and ht[-1] == 14400)
            budget = budgets[name]
            q = dict(run=name, case=case, grid=grid, completed=bool(completed),
                     inputs_hashes_passed=all(checks.values()), input_hash_checks=checks,
                     map_dimensions=mapdims, history_dimensions=hisdims,
                     map_frames=len(ts), map_start_s=float(ts[0]), map_end_s=float(ts[-1]),
                     map_step_s=np.unique(np.diff(ts)).tolist(), dynamic_fields=dynamic_vars,
                     history_frames=len(ht), history_start_s=float(ht[0]), history_end_s=float(ht[-1]),
                     history_nominal_step_s=0.4, history_step_min_s=float(hdt.min()), history_step_max_s=float(hdt.max()),
                     station_count=stations, boundary_section_count=sections,
                     full_internal_face_flux_exported=full_face_flux, peak_time_exported=tpeak_exported,
                     native_active_cells=int(active.sum()), peak_censored_active_cells=int((active & ~np.isfinite(peak)).sum()),
                     dynamic_censored_active_cell_frames=int((~finite[:, active]).sum()),
                     temporal_peak_diagnostic=dict(comparable_native_cells=int(comparable.sum()),
                         native_cells_missed_by_more_than_0_01m=int((gaps > .01).sum()),
                         water_level_peak_minus_snapshot_peak_p95_m=float(np.quantile(gaps, .95)),
                         water_level_peak_minus_snapshot_peak_max_m=float(gaps.max()),
                         interpretation='Native water-level sampling diagnostic, not common-cell depth error or a measure of terrain accuracy.'),
                     h_equals_zs_minus_native_min_max_abs_error_m=h_semantic_error,
                     effective_precip_time_values=forcing_times, effective_precip_time_units=forcing_time_units,
                     effective_precip_units_attribute=forcing_units, effective_precip_unit_singular=forcing_unit_singular,
                     effective_precip_fill_value=forcing_fill, effective_precip_maxima=rain_max,
                     stage_rows=int(len(stages)), stage_points=int(stages.shape[1]-1),
                     stage_increment_range_m=[float(np.min(stages[:,1:]-stages[0,1:])),float(np.max(stages[:,1:]-stages[0,1:]))],
                     physical_rain_mm_h=contract['rain_mm_h'], runtime_seconds=ex.get('runtime_seconds'),
                     budget_passed=budget['passed'], relative_budget_error=budget['relative_error'],
                     strict_nonnegative=budget['strict_nonnegative'], building_overtopped_fine_area_m2=budget['building_overtopped_fine_area_m2'],
                     cached_fields=cache_fields, primary_label_status=(
                         'reconstructible_native_aligned_10m_after_support_censoring_and_quality_contract' if grid == GRIDS[0]
                         else 'conditional_bounds_only_where_common_10m_contains_multiple_native_cells'),
                     training_admitted=False, sealed_test_admitted=False)
            runs.append(q)
            print('AUDITED', name, 'complete', completed, 'hashes', all(checks.values()), flush=True)
    # Deliberately record, but never admit, short tests and interrupted copies.
    for parent in [base/'adaptive_connectivity_comparison', base/'adaptive_sensitivity/interrupted_runs']:
        if not parent.exists():
            continue
        for p in sorted(parent.iterdir()):
            if not p.is_dir() or not (p/'sfincs_map.nc').exists() or p.name in {r['run'] for r in runs}:
                continue
            with netCDF4.Dataset(p/'sfincs_map.nc') as d:
                t = arr(d['time'][:])
                end = float(t[-1]) if len(t) else None
            excluded.append({'source_directory': rel(p), 'end_time_s': end,
                             'reason': '60-second geometry/forcing pilot, not a four-hour scenario' if p.name in ['pilot_corrected_10m', 'pilot_adaptive_10_5m'] else 'interrupted/rejected; no accepted full-event result',
                             'label_admitted': False})
    static = [
        ('data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif', 'bare-ground support'),
        ('data/processed/three_ward_sfincs_scenarios/active_mask_20m.tif', 'existing common interior domain'),
        ('data/processed/three_ward_urban_constraints/constraints_0_50m.tif', 'building and road class mask'),
        ('data/processed/three_ward_urban_constraints/hydraulic_obstacle_0_50m.tif', 'solver obstacle surface'),
        ('data/processed/three_ward_urban_constraints/manning_0_50m.tif', 'roughness source'),
        ('data/processed/three_ward_urban_constraints/constraint_vectors_6677.geojson', 'candidate building and road vectors; eligible road lengths not yet constructed'),
        ('data/processed/three_ward_adaptive_connectivity/adaptive_cell_geometry.npz', 'native cell mapping'),
        ('data/processed/three_ward_adaptive_connectivity/regular_face_geometry.npz', 'regular face connectivity'),
        ('data/processed/three_ward_adaptive_connectivity/adaptive_face_geometry.npz', 'adaptive face connectivity'),
        ('data/processed/three_ward_adaptive_sensitivity/common_ground_support.npz', 'current 20m reporting support, not proposed 10m target'),
    ]
    for s, role in static:
        register(root/s, role)
    gaps = [
        ('P0','Ground elevation / eligible area','present; packaging required','0.50m ground + building classes + existing interior mask','Freeze EPSG:6677 grid origin/extent; build eligible area and conservative native-to-common-10m mapping; exclude boundary/buildings/unknown terrain. Existing support cache is 20m.','label construction'),
        ('P0','Building / road hydraulic connections','present; encode exact geometry','native face geometry + thindams + class vectors','Carry separation, blocked faces, conveyance and roughness; occupancy fraction alone does not represent narrow-passage connectivity.','label construction'),
        ('P0','Rainfall','physical scalar and effective spatial forcing present','scenario_contract.json + roof_surface_precip.nc','Declare physical mm/h vs roof-redistributed effective source, interpolation, one-second ramps and lower-left/half-cell reader convention. Precipitation has singular unit=mm.hr-1 but no CF units; _FillValue=0 also denotes real dry forcing. Read raw zeros under the solver contract rather than auto-masking them as missing; do not double count roof runoff.','input contract'),
        ('P0','River stage','synthetic point forcing present','sfincs.bnd + sfincs.bzs + boundary_points.csv','Preserve points/order/groups and spatial low-stage offsets; document working datum and terrain epoch; +1.5m is a uniform synthetic increment, not observed T.P. stage.','input contract'),
        ('P0','Initial state / parameters','dry-start configuration and t=0 fields present','sfincs.inp + first map frame + subgrid/roughness files','Record zsini=-100 initialization as solver dry-start convention, not physical water level. Fix or explicitly condition roughness, roof rule, drainage and initial state; include restart state if varied later.','input contract'),
        ('P0','Provenance / groups / runtime','run metadata and execution times present','execution.json + source audits','Create parent-event ID and geometry/solver/forcing/output versions before splitting; timings currently omit shared preparation and are Docker amd64 emulation, not end-to-end ML speedup.','sample registry'),
        ('P1','Measured rainfall/stage and response','not matched to a validation event','local historical footprints and modeled hazard references only','Independent event pairing needed for real-world accuracy; scenario-emulation work may proceed with explicit conditional scope. Model hazard maps are not observations.','observational validation'),
        ('P1','Domain/epoch/river bed','unresolved physical gaps','existing provisional contributing area + water polygons','Complete cross-ward terrain/urban constraints and height-epoch alignment before claiming coverage; river polygons do not supply underwater channel geometry. No pipe optimization required.','reference scope'),
        ('P2','Active measures','no hydraulic response library','no verified assets/capacity/receiving links','After milestones 1–4, obtain action locations, capacities/storage, receiving-water connectivity, schedules and matching physical source/sink accounting; baseline surrogate cannot infer unseen actions.','later measures'),
    ]
    input_rows = [dict(zip(['priority','input','availability','evidence','required_work','phase'], r)) for r in gaps]
    temporal = [
        ('Full-domain level/velocity/storage','present: 49 frames / 0–14400s / 300s','All eight accepted runs','Reusable for coarse evolution; interval not qualified for peak timing, wet duration or path evolution. Denser output/online target accumulator requires later authorization.','P0'),
        ('Common-10m Peak Mean Depth','not saved in designed form','Regular10m zsmax can be remapped over fine eligible ground; adaptive5m children may peak asynchronously','Regular native-aligned cells can use monotonic storage at zsmax. Adaptive common cells: synchronous-frame maximum is a lower bound after censoring treatment; weighted local maxima an upper bound. Quantify bracket before deciding reruns.','P0'),
        ('Instantaneous h versus hmax','different semantics despite depth names','Pinned sfincs_ncoutput.F90 lines2379–2392,2825–2840,3541–3569,3736–3767','h=zs-zmin; hmax=table volume(zsmax)/whole hydraulic cell area with storehmean=1. Neither is directly eligible-open-ground mean. Use water level plus declared ground support.','P0'),
        ('Dry/shallow/unknown masks','peak and dynamic zs are threshold-censored','huthresh=0.01; map msk and zb; old source audit','Retain interval/censor flags; masked zs with known active cell is bounded shallow state, not unknown or exact zero. Adaptive h also censored; regular h dry states written.','P0'),
        ('Boundary exchange','present: 36001 timestamps; nominal 0.4s','crosssection_discharge at 7062 sections; float32 time jitter','Useful for existing budget audit; group/sign definitions retained. Selected boundary sections are not all interior faces or actual channel discharge.','P1'),
        ('High-frequency whole-domain depth','absent at history interval','History has one station, not a spatial state cube','Cannot use 0.4s history as full-domain dynamic labels. Export denser full fields or common-grid online summaries if that task is later approved.','P1'),
        ('Internal face flux / source-sink time budgets','not exported as complete dynamic balance','Cell-centered u/v and final cumulative precipitation exist','Future dynamic conservation/control tasks require face exchanges, source/sink increments and restart-compatible states; cell-center velocities cannot uniquely recover face fluxes.','P2'),
        ('Peak time, onset, duration, recession tail','no dedicated validated fields','t_zsmax absent; fixed four-hour window','Define horizon-specific target; remaining final storage is not a simulation-completion error. Extend/drain-to-criterion only if full-event recession is required.','P1'),
    ]
    temporal_rows = [dict(zip(['output','availability','evidence','required_work','priority'], r)) for r in temporal]
    scenario_specs = [
        ('P0','Current controls','4 prescribed-forcing combinations × 2 grids','60/100mm/h low stage;60mm/h raised;dry raised','Open development controls only. All share fixed duration/initial-state/geometry; eight configurations are not eight independent storms.'),
        ('P0','Compound extreme','missing matched 100mm/h + raised-stage pair','No completed run in accepted set','First future contrast after label/output contract; use identical geometry, no-action forcing, and both grids where needed.'),
        ('P0','Dry low-stage negative control','not in current matched pair set','Older20m dry_low exists','Generate or qualify matched current-geometry control later; do not silently promote old20m geometry to current labels.'),
        ('P1','Rain temporal/spatial diversity','missing','One one-hour plateau rain from hour1 to2','Vary total/duration/peak shape, storm timing and spatial pattern, with scenario envelopes; roof-weight variation is not spatial storm diversity.'),
        ('P1','Receiving-water hydrographs and phase','missing diversity','One uniform additive stage pulse','Vary segment-specific stage levels/rise/fall/lag; compound and withheld combinations need separate families.'),
        ('P1','Initial-state / physical uncertainty','no designed ensemble','One dry start; fixed surface assumptions','If varied, condition on them or hold fixed. Group grid/output-step/parameter variants with parent forcing family to prevent leakage.'),
        ('P1','Sealed evaluation','none eligible among already inspected controls','Prior design proposes240 parents plus48 challenge parents','Budget and envelope remain proposals. Assign families before tiles/time windows; do not count existing inspected scenarios as sealed tests.'),
        ('P2','Active actions','absent','No-action only','Later paired action/no-action hydraulic scenarios, including receiving-stage and outside-target risk transfer; group all variants by parent forcing.'),
    ]
    scenario_rows = [dict(zip(['priority','family','availability','evidence','required_work'], r)) for r in scenario_specs]
    anasop = protected[0].read_text()
    central = next(x.split(':',1)[1].strip() for x in anasop.splitlines() if x.startswith('- Research question:'))
    supporting = [x.split('|')[1].strip() for x in anasop.splitlines() if x.startswith('| How ') or x.startswith('| Can surrogate')]
    questions = [central] + supporting
    assert len(questions) == 5
    statuses = ['partly-testable', 'partly-testable', 'not-yet-testable', 'weakly-testable', 'not-yet-testable']
    reasons = [
        'Reference-emulation concept partly supported; no trained model or unseen-event library; real-event and action validity not established.',
        'Matched grid/forcing and budgets exist; no formal convergence, complete upstream coverage or observational validation.',
        'Insufficient parent-event diversity; no sealed evaluation events or end-to-end surrogate runtime.',
        'Intensity and stage contrasts exist, but strongest-rain-plus-raised-stage and varied timing/spatial patterns absent.',
        'No action-response simulations or verified measure inputs; conditional later milestone.'
    ]
    question_rows = [dict(id='RQ'+str(i+1), question=q, status=s, rationale=r) for i,(q,s,r) in enumerate(zip(questions,statuses,reasons))]
    for p in [*protected, audit_dir/'findings.json', audit_dir/'water_budgets.json',
              root/'src/exp/review_three_ward_adaptive_sensitivity.py',
              root/'data/processed/three_ward_sfincs_scenarios/solver_provenance/sfincs_ncoutput.F90',
              root/'src/analyses/audit_surrogate_reference_labels.py']:
        evidence.append({'source':rel(p),'sha256':cached_digest(p)})
    for filename, rows in [('dataset_availability.csv',datasets),('variable_inventory.csv',variables),
                           ('question_feasibility.csv',question_rows),('input_gaps.csv',input_rows),
                           ('temporal_output_gaps.csv',temporal_rows),('scenario_gaps.csv',scenario_rows),
                           ('excluded_runs.csv',excluded)]:
        table(out/filename, rows)
    table(out/'run_label_admission.csv', [{k:q[k] for k in ['run','case','grid','completed','inputs_hashes_passed',
          'map_frames','map_end_s','history_frames','station_count','boundary_section_count','budget_passed',
          'relative_budget_error','peak_censored_active_cells','primary_label_status','training_admitted','sealed_test_admitted']} for q in runs])
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),root=str(root),
                   scope='Three wards and existing provisional contributing area; file/semantics feasibility, not new model execution',
                   completed_reference_runs=len(runs), forcing_combinations=len(CASES), inspected_excluded_runs=len(excluded),
                   detailed_inventory_files=len(datasets), variable_occurrences=len(variables),
                   questions_found=len(questions),status_counts=dict(Counter(statuses)),
                   common_open_ground_area_km2=findings['common_open_support_km2'],
                   training_ready=False, sealed_test_ready=False, real_event_validation=False,
                   primary_target_constructed=False,new_simulations=False,training_executed=False,internet_search=False,
                   target='max_t sum_fine_eligible_area(a*max(eta_native(t)-z,0))/sum_fine_eligible_area(a), common aligned10m',
                   regular10m_reconstruction='Native-aligned common10m has one water-level state per cell: max_t f(eta(t)) = f(max_t eta(t)) for monotone eligible-ground storage f. Use zsmax, exact fine area and censor intervals; qualify quality masks separately.',
                   adaptive_reconstruction='Unrefined native10m cells same as regular case. Where common10m contains native5m children, max_t(sum_j f_j(eta_j(t))) is generally below sum_j f_j(max_t eta_j(t)); only synchronized aggregation defines the target. Existing 300s fields give sampled estimates and bounds after censoring; no exact continuous-time peak guarantee.',
                   peak_wet_area='Existing fine-ground local-peak exceedance footprints remain useful conditional reference products; they are neither simultaneous flood extents nor thresholded common-cell Peak Mean Depth.',
                   sparse_sampling_diagnostic='Differences between native maximum level and maximum written level are a sampling screen, not common-cell mean-depth error. No local anomaly diagnosis undertaken.',
                   output_content_hash_policy='NetCDF outputs inventoried and read, recorded by size/mtime; no full-output content hashes. All run input hashes rechecked with inode cache. Small evidence files SHA256 bound.',
                   prior_numerical_flags={k:findings[k] for k in ['all_water_budgets_passed','formal_grid_convergence_established','event_validated','domain_is_provisional_not_certified_complete','external_cross_ward_building_road_constraints_complete','height_epoch_alignment_complete','strict_nonnegative_all']},
                   runs=runs,excluded=excluded)
    dump(out/'audit.json', summary)
    dump(out/'evidence-register.json', evidence)
    max_budget = max(q['relative_budget_error'] for q in runs)
    body = f'''# 三区代理模型参考标签可行性核查

现有结果可作为有条件的模型仿真参考与开发控制组；尚不能直接视为完整训练库或真实积水观测。核查范围覆盖三区及既有跨区候选汇水范围，不做逐点核验。本轮只读数据，未训练、未运行新情景、未检索文献，也未修改 AnaSOP 或 processed 数据。

核查依据：`docs/AnaSOP.md`；共5个研究问题（中心问题1个、支持问题4个）。详细清单包含{len(datasets)}个文件、{len(variables)}次NetCDF变量记录；状态：{dict(Counter(statuses))}。另行保存的技能基础表名扫描发现84个表格/文本文件、227次列记录，只识别中心问题且仅按名称匹配，不能据此判断水动力标签合格。

## 结论与准入

- 正式结果：4种强迫组合 × 2种网格 = 8次完整运行。实读 map/his 终点均为14400秒，求解器状态0，日志完成标记、退出码及本次重算的输入SHA256均通过。短试跑和中断副本见 excluded_runs.csv，全部排除。
- 数值水量平衡沿用已完成审计，8次通过原1%探索门槛，最大闭合误差{max_budget:.3%}。本轮核验输入未变，未重算水量积分。局部负记账体积、建筑假设障碍超越和瞬态漏采问题仍保留，不能用全域闭合替代局部峰值质量。
- 共同可分析开敞地面{summary['common_open_ground_area_km2']:.6f} km²，是模型分析支撑面积，不是三区总面积；现存 support 缓存是20m汇总，尚无冻结的10m标签包。正式网格收敛、完整外缘覆盖、高程历元统一和真实事件验证均未完成。
- 4个现有强迫组合共享固定的一小时降雨过程/干初态等设置，不能当作8场独立暴雨，也不能进入封存测试集。

## 标签定义必须对齐

目标为同一时刻先把细地形上的水量汇总至共同10m开敞地面，再沿时间取峰值：
`H[c] = max_t(sum_j a[j] * max(eta[native(j),t] - z[j], 0) / A_open[c])`。

全域规则10m：共同单元与原生单元对齐时，单元内只有一个水位状态，地面储水量随水位单调变化，可用原生 zsmax 与0.50m地形、开敞面积重建该目标，并保留浅水删失区间。这是可重建性结论，本轮未构造标签。

10/5m加密：未加密单元同上；一个共同10m单元包含多个5m子单元时，各子单元的峰值可能不同步。先平均各自峰值会给出上界，不能当作同一时刻的最大平均水深。现有300秒同步帧可形成采样估计；结合浅水区间与原生峰值可构造上下界，尚不能保证精确连续时峰值。先量化上下界差，再决定是否需要更密输出或计算中在线累积共同网格目标，避免一上来重跑所有结果。

现有积水范围与区域“平均峰值水深”主要是细地面各自峰值的汇总，仍可支持对应口径的对照；不能改名后充当共同单元目标。对共同单元平均水深做阈值，也不会自动得到细地面峰值超阈面积，必须保留两种定义。

## 实际时序与字段含义

- 全域 zs/h/u/v/subgrid_volume：每300秒，0–14400秒，共49帧。
- history：名义0.4秒，共36001条时间；但只有1个观测站，7062个选定边界交换断面，绝不是0.4秒全域水深。
- `h` 实际为水位减原生单元最低地面；`storehmean=1`下 `hmax` 为峰值水位对应查表储水量除整个原生单元面积。二者虽单位都是m，含义不同，且均不等于设计中的开敞面积平均水深。已用源码及7200秒数值等式核对。
- 降雨输入还有一个导出陷阱：文件使用单数 `unit=mm.hr-1`，而标准 `units` 为空；`_FillValue=0` 同时用于真实无雨值。通用NetCDF自动掩膜会把整帧无雨读成缺失。本审计按既有求解器输入约定读取原始0值，后续导出需明确区分；有效源雨量含屋面重分配，峰值可能远大于物理60/100mm/h。
- 瞬时水位和峰值水位受0.01m阈值删失；规则网格瞬时h保留干单元值，加密网格瞬时h也有删失。浅水、未知、无效建筑/边界必须分别编码。
- 各运行原生水位峰值与快照峰值的差见 audit.json。多数单元差小，但存在尾部漏采，不能因5分钟序列完整就断言峰值准确。这些差不是共同单元平均深度误差，也不等于地形误差。
- 没有完整内部面流量时序，也没有专门的峰值时刻/湿润历时输出。4小时是当前情景窗口；末时仍蓄水，不表示运行未完成，但不能称完整消退事件。

## 缺口与优先顺序

1. **P0：先整理现有数据，形成标签导出约定。** 固定共同10m原点/坐标/支撑、原生映射、开敞面积、删失和质量掩膜；从 zsmax 重建规则10m示例，给加密结果计算同步峰值上下界。输入同时固定物理雨量与屋面重分配有效雨量的区别、水位点顺序/基准、初态与参数版本。现有道路面积不能直接当道路长度。
2. **P0：先确认输出约定，再补最小诊断情景。** 未来优先补相同几何的100mm/h+抬升水位，以及低水位无雨控制；旧20m无雨结果只能辅助比较。是否加密时序由上下界/漏采筛查决定，本轮没有启动。
3. **P1：再设计多样情景与独立分组。** 降雨历时/峰形/空间移动、河段差异与时滞、复合工况；初态/参数若变化则显式作为输入。按父事件族划分后才裁块、切时间窗；网格/参数/控制变体同组。240+48是已有设计提案，尚未批准执行。
4. **P1：真实预测验证另补事件配对。** 现有历史积水范围和官方模拟风险图不自动提供配对实测标签；需要同一事件的降雨、水位和积水响应。河床、跨区覆盖与高程历元问题限定物理解释，不妨碍明确条件下先做仿真代理。
5. **P2：前四阶段后再接主动措施。** 先具备位置、容量、受纳连通、调度与源汇核算，再生成同一事件的无措施/措施配对结果。基线标签不能支撑未见措施效果，也不能验证时变反馈控制。

建议下一技能为 `data-preprocessing`，范围仅限统一输入/标签导出与质量掩膜；研究目标可保持，无须因暂缺情景而重写。动态模型、真实事件验证和主动措施各自保留门槛。本轮没有修改任何候选变量为最终变量。

可复现命令：`/private/tmp/udt01b-sfincs-runner/bin/python src/analyses/audit_surrogate_reference_labels.py --root /Volumes/Extension/Research/UDT01b`。脚本依赖已有numpy、netCDF4；运行环境路径可替换为安装这些库的本地Python。没有新增依赖。
'''
    (out/'README.md').write_text(body, encoding='utf-8')
    def html_table(rows, cols):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[c]))+'</td>' for c in cols)+'</tr>' for r in rows)+'</tbody></table></div>'
    content = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区代理参考标签核查</title><style>body{max-width:1100px;margin:32px auto;padding:0 24px;font:16px/1.8 system-ui;color:#203342}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;border-bottom:1px solid #d6e0e7;text-align:left;vertical-align:top}.scroll{overflow:auto}p{white-space:pre-line}a{color:#076898}.note{background:#eaf3f7;padding:18px}</style><h1>三区代理模型：现有标签能否复用？</h1><p class="note">结论：可作有条件的仿真参考，尚未准入训练。8次完整运行仅4种强迫组合。规则10m峰值可重建；加密子单元须同步汇总后取峰值，现有5分钟快照需先评估误差范围。</p>'
    for block in body.split('\n\n')[1:]:
        if block.startswith('## '):
            content += '<h2>'+html.escape(block[3:])+'</h2>'
        else:
            content += '<p>'+html.escape(block)+'</p>'
    content += '<h2>逐运行准入</h2>' + html_table(runs, ['run','completed','inputs_hashes_passed','map_frames','peak_censored_active_cells','primary_label_status'])
    content += '<h2>输入缺口</h2>'+html_table(input_rows,['priority','input','availability','required_work'])
    content += '<h2>输出缺口</h2>'+html_table(temporal_rows,['priority','output','availability','required_work'])
    content += '<h2>情景缺口</h2>'+html_table(scenario_rows,['priority','family','availability','required_work'])
    content += '<p>'+ ' · '.join('<a href="'+n+'">'+html.escape(n)+'</a>' for n in ['README.md','run_label_admission.csv','input_gaps.csv','temporal_output_gaps.csv','scenario_gaps.csv','audit.json','validation.json'])+'</p></html>'
    (out/'review.html').write_text(content, encoding='utf-8')
    checks = dict(eight_completed=len(runs)==8 and all(q['completed'] for q in runs),
                  four_forcings=len({q['case'] for q in runs})==4,
                  input_hashes_all_match=all(q['inputs_hashes_passed'] for q in runs),
                  all_maps_same_49_frames=all(q['map_frames']==49 and q['map_step_s']==[300.] for q in runs),
                  all_histories_36001_frames_one_station=all(q['history_frames']==36001 and q['station_count']==1 for q in runs),
                  h_semantics_numerically_supported=all(q['h_equals_zs_minus_native_min_max_abs_error_m']<1e-5 for q in runs),
                  five_questions=len(questions)==5,
                  source_design_unchanged=all(digest(root/p)==h for p,h in protected_before.items()),
                  no_training_admission=all(not q['training_admitted'] for q in runs),
                  no_sealed_test_admission=all(not q['sealed_test_admitted'] for q in runs),
                  two_short_pilots_excluded=sum('60-second' in q['reason'] for q in excluded)==2,
                  interrupted_outputs_excluded=sum('interrupted' in q['reason'] for q in excluded)>=2)
    dump(out/'validation.json',dict(valid=all(checks.values()), checks=checks, protected_sha256=protected_before,
                                   note='Audit invariants only; valid=true does not certify hydraulic labels or approve training.'))
    assert all(checks.values()), checks
    print(json.dumps({k:v for k,v in summary.items() if k not in ['runs','excluded']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
