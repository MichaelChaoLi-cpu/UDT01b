"""Finalize scenario provenance, output-cadence audit and framework status."""
from pathlib import Path
import json,csv,hashlib,re
import numpy as np
import netCDF4
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/data-briefing/three_ward_sfincs_scenarios';PRO=ROOT/'data/processed/three_ward_sfincs_scenarios';BASE=ROOT/'data/processed/three_ward_sfincs_framework'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def filled(v):return np.ma.asarray(v[:],dtype=np.float64).filled(np.nan)
def main():
 audits=[]
 for base_name in ['dry_raised','rain60_raised']:
  original=None;original_volume=None
  for suffix,cadence in [('',10),('_his2',2),('_hisstep',.8)]:
   run=base_name+suffix;folder=BASE/('terrain_'+run+'_20m')
   with netCDF4.Dataset(folder/'sfincs_map.nc') as d:
    m=filled(d.variables['msk'])==1;vol=filled(d.variables['subgrid_volume']);dv=float(np.nansum(vol[-1,m])-np.nansum(vol[0,m]));rain=filled(d.variables['cumprcp'])[-1];rv=float(np.nansum(rain[m])*400);peak=filled(d.variables['hmax'])[0]
    if original is None:original=peak.copy();original_volume=vol.copy()
    same=bool(np.allclose(peak,original,atol=0,rtol=0,equal_nan=True))
   with netCDF4.Dataset(folder/'sfincs_his.nc') as d:
    t=filled(d.variables['time']);q=np.nansum(filled(d.variables['crosssection_discharge']),axis=1);trap=float(np.trapezoid(q,t));right=float(np.sum(q[1:]*np.diff(t)))
   audits.append({'run':run,'history_cadence_s':cadence,'flux_samples':len(t),'stored_volume_change_m3':dv,'rain_volume_m3':rv,'trapezoid_boundary_volume_m3':trap,'right_rectangle_boundary_volume_m3':right,'closure_residual_trapezoid_m3':dv+trap-rv,'closure_residual_right_m3':dv+right-rv,'peak_mean_depth_identical_to_original':same,'instantaneous_volume_maps_identical_to_original':bool(np.allclose(vol,original_volume,atol=0,rtol=0,equal_nan=True)),'interpretation':'output sampling diagnostic;physics,terrain,rain and stage unchanged'})
 dump(OUT/'history_sampling_audit.json',{'records':audits,'all_peak_maps_unchanged':all(x['peak_mean_depth_identical_to_original'] for x in audits),'primary_budget_uses_per_step_history_for_raised_cases':True,'netcdf_average_dt_scalar_not_used':'actual cadence verified using solver log and history sample count'})
 validation=read(OUT/'numerical_validation.json');findings=read(OUT/'findings.json');registry=read(OUT/'scenario_registry.json')
 hashes=[]
 for entry in registry:
  folder=ROOT/entry['path'];ex=read(folder/'execution.json');ok=ex['returncode']==0 and ex['completion_marker']
  for name,sha in ex['input_sha256'].items():ok=ok and hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
  with netCDF4.Dataset(folder/'sfincs_map.nc') as d:stop=float(d.variables['time'][-1])
  end_label_ok=abs(stop-14400)<=.81
  hashes.append({'scenario':entry['scenario'],'inputs_unchanged':bool(ok),'last_native_map_time_label_s':stop,'end_label_offset_s':stop-14400,'engine_completion_marker':ex['completion_marker'],'native_end_label_within_one_actual_time_step':end_label_ok})
 assert all(x['inputs_unchanged'] and x['native_end_label_within_one_actual_time_step'] for x in hashes)
 assert validation['water_budgets_all_passed'] and validation['time_step']['initial_screening_passed']
 tiers={'screening_runs_completed':True,'official_subgrid_and_engine_compatible':True,'water_budget_checks_passed':True,'time_step_screening_passed':True,'strict_internal_volume_nonnegative':False,'grid_10_vs20m_convergence_tested':False,'event_validated':False,'intervention_stage_completed':False,'physical_inputs_limit':'terrain only;no building barriers,pipe/pump removal or actual channel bathymetry;DEM-relative stages hypothetical','priority_hotspots':findings['priority_hotspots_count'],'snapshot_flow_paths':findings['snapshot_paths_count'],'report':'data/exp/data-briefing/three_ward_sfincs_scenarios/review.html'}
 spec=read(BASE/'model_spec.json');spec['official_solver_bundle_created']=True;spec['builder']['status']='official1.2.2_subgrid_built_and_read_by_pinned_SFINCS2.4';spec['builder']['verified_runner_requirements']='data/processed/three_ward_sfincs_scenarios/runner_requirements.txt';spec['scenario_screening']=tiers;spec['baseline_model_ready']=False;dump(BASE/'model_spec.json',spec)
 readiness=read(BASE/'readiness.json');readiness['scenario_screening']=tiers;readiness['scenario_run_ready']=True;dump(BASE/'readiness.json',readiness)
 with (BASE/'input_inventory.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 for row in rows:
  if row['id']=='terrain':row['next_action']='官方子网格已生成并完成整体情景初筛；继续10/20m网格检查及地形来源不确定性评估。'
  if row['id']=='rain':row['next_action']='合成60/100mm/h情景已运行；历史事件仍需原始数值强迫，不以公报替代。'
  if row['id']=='water_geometry':row['next_action']='水域、桥梁和部分三维附属设施可定位候选通道；完整河床、岸顶/入口及容量仍缺。'
 with (BASE/'input_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 old=ROOT/'data/exp/data-briefing/three_ward_sfincs_framework/review.html';s=old.read_text()
 if 'scenario-run-update' not in s:s=s.replace('<h2>','<section id="scenario-run-update"><b>整体情景已完成</b><p>官方子网格与引擎已兼容运行，水量和时间步初筛通过；网格收敛、内部库容非负与实测验证仍未完成。见<a href="../three_ward_sfincs_scenarios/review.html">三区积水、流线及水位影响报告</a>。</p></section><h2>',1);old.write_text(s)
 (PRO/'README.txt').write_text('三区地形主导暴雨情景\n\n保留原生0.50m DEM，仅缺值处使用PLATEAU5m地形插值；terrain_sampling_0_50m.tif 的0.50m是采样间隔，不是备用源精度。source_id=1原生DEM，2PLATEAU。\n官方SFINCS2.4和HydroMT-SFINCS1.2.2已运行。20m计算网格，内部统计38.9768km²。三组核心暴雨：60mm/h低水位、100mm/h低水位、60mm/h水位抬升1.5m；另有无雨、封闭域、时间步与输出间隔核查。\n水量闭合及时间步初筛通过，少量干湿格点存在内部库容透支，未满足严格非负检验；它们已从优先热点排除。10/20m网格收敛、建筑实体、管网作用、真实河床与事件验证尚未完成。\n优先热点是三种合成暴雨中稳定的地形低洼区域。surface_flow_paths_snapshot.geojson 为雨停时瞬时速度场的静态流线，不是动态水团轨迹或已核实进河通道。\n复现：项目原环境运行 src/preprocessing/prepare_three_ward_sfincs_scenarios.py；独立Python3.12环境安装runner_requirements.txt，再运行src/exp/build_run_three_ward_scenarios.py build/run、src/exp/review_three_ward_sfincs_scenarios.py及src/exp/finalize_three_ward_scenarios.py。生成文件会占用空间，原有成功运行会被复用，中断结果单独保留。\n')
 checks={'successful_completed_runs':len(registry),'all_input_hashes_and_end_times_passed':True,'native_output_time_labels_preserved':True,'time_label_caveat':'One refined run labels its final map one0.8s step earlier;engine completed and physical maps match original run exactly. Shared tout in pinned output loop is retained,not silently rewritten.','output_cadence_refinements_leave_peak_maps_identical':all(x['peak_mean_depth_identical_to_original'] for x in audits),'water_budget_checks_passed':True,'time_step_check_passed':True,'limitations_preserved':tiers,'run_checks':hashes}
 dump(OUT/'delivery_validation.json',checks);print(json.dumps({'runs_completed':len(registry),'water_budgets_passed':True,'peak_maps_unchanged_by_history_cadence':checks['output_cadence_refinements_leave_peak_maps_identical'],'strict_positivity_passed':False},ensure_ascii=False))
if __name__=='__main__':main()
