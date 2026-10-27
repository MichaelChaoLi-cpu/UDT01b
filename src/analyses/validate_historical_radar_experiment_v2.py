"""Independent on-disk delivery validation, unchanged budgets and interval bounds."""
from pathlib import Path
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import json,csv,hashlib,re,sys
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from audit_three_ward_negative_volumes import structured
P=ROOT/'data/processed/three_ward_historical_radar_experiment_v2';R=ROOT/'data/exp/estimation/three_ward_historical_radar_experiment_v2'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def rr(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
 checks=[]
 def check(name,ok,details=None):checks.append(dict(check=name,passed=bool(ok),details=details))
 m=json.loads((P/'run_manifest.json').read_text());check('exactly_two_declared_cases',len(m['runs'])==2 and {r['case_id'] for r in m['runs']}=={'radar_low','radar_raised'})
 check('one_historical_parent_outside_formal_split',len({r['identity']['parent_id'] for r in m['runs']})==1 and all(r['split']=='external_historical_candidate' and not r['formal_library_member'] for r in m['runs']))
 old=ROOT/'data/processed/three_ward_historical_radar_experiment_v1/runs/radar_low';ex=json.loads((old/'execution.json').read_text());check('aborted_v1_preserved',ex['returncode']!=0 and not ex['completion_marker'] and 'requires piecewise linear' in (old/'solver_stdout.txt').read_text())
 same_files=['sfincs.dep','sfincs.msk','sfincs.ind','sfincs_subgrid.nc','sfincs.crs','sfincs.bnd','building_closures.thd','historical_rain.nc','sfincs.inp']
 for name in same_files:check('paired_input_'+name,len({r['input_sha256'][name] for r in m['runs']})==1)
 support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];area=support['Ground Area'];road=support['Road Ground Area'];ward=support['Ward Code']
 a=np.load(P/'paired_exposure_bounds.npz');good=a['common_admitted']
 labels=[np.load(P/'execution'/(case+'_candidate_reference.npz')) for case in ['radar_low','radar_raised']]
 check('common_admitted_support_exact',np.array_equal(good,valid&labels[0]['Admitted Mask']&labels[1]['Admitted Mask']&~labels[0]['Quarantine Mask']&~labels[1]['Quarantine Mask']))
 rows=rr(R/'wet_extent.csv');scopes={'Whole domain':None,'Three wards':-1,'Chiyoda':1,'Chuo':2,'Minato':3,'Cross-ward buffer':0};worst=0
 for row in rows:
  code=scopes[row['scope']];region=valid if code is None else valid&(ward>0) if code==-1 else valid&(ward==code)
  isroad=row['support']=='Classified ground roads';w=road if isroad else area;use=region&good&(w>0);th=float(row['threshold_m'])
  for arm,key in [('low',0),('raised',1)]:
   for name,direction in [('certain','lower'),('possible','upper')]:
    v=a[('road_' if isroad else 'ground_')+direction][key];exact=float(w[use&(v>th)].sum()/1e6);worst=max(worst,abs(exact-float(row[arm+'_'+name+'_km2'])))
 check('independent_all_extent_recalculation',worst<1e-12,dict(max_abs_km2=worst))
 qual=json.loads((R/'solver_qualification.json').read_text());check('production_fixtures_all_pass',qual['all_passed'])
 checks_by_name={x['check']:x['passed'] for x in qual['checks']}
 check('actual_budget_and_rejection_profiles_tested',all(checks_by_name.get(k) for k in ['radar_budget_profile_fixture_v2_1','radar_budget_profile_fixture_v2_4','budget_reject_unsupported','budget_reject_infiltration','budget_reject_clock_mismatch']))
 entrance_rows=rr(R/'entrance_exposure.csv');screen=json.loads((R/'summary.json').read_text())['entrance_screen']
 check('entrance_screen_counts_match_saved_rows',len(entrance_rows)==screen['inventory_points'] and sum(x['neighborhood_quality']=='True' for x in entrance_rows)==screen['quality_passed'] and sum(x['low_screen_trigger']=='True' for x in entrance_rows)==screen['low_trigger_points'] and sum(x['raised_screen_trigger']=='True' for x in entrance_rows)==screen['raised_trigger_points'])
 budgets=rr(R/'water_budgets.csv')
 for b in budgets:
  source=float(b['rain_m3'])+float(b['inflow_m3']);scale=max(source,float(b['outflow_m3']),1)
  check(b['case_id']+'_unchanged_endpoint_budget',abs(float(b['final_budget_residual_m3']))<=max(1.,.001*scale))
  check(b['case_id']+'_nonnegative_volume',float(b['minimum_native_volume_m3'])>=0 and int(b['negative_cells'])==0)
 groups=rr(R/'water_group_exchange.csv')
 check('seven_water_groups_preserve_source_codes',{int(x['water_group_code']) for x in groups}=={17,51,71,79,106,110,111})
 for b in budgets:
  case=b['case_id']
  check(case+'_group_gross_exchange_matches_domain',abs(sum(float(g[case+'_out_m3']) for g in groups)-float(b['outflow_m3']))<1e-5 and abs(sum(float(g[case+'_in_m3']) for g in groups)-float(b['inflow_m3']))<1e-5)
 for run in m['runs']:
  case=run['case_id'];folder=ROOT/run['folder'];ex=json.loads((folder/'execution.json').read_text());audit=json.loads((R/'execution'/case/'audit.json').read_text())
  check(case+'_all_numerical_gates',audit['all_checks_passed'] and audit['admission']['technical_reference'])
  check(case+'_completion',ex['returncode']==0 and ex['completion_marker'])
  clock=structured(folder/'udt_output_clock.csv');maps=clock[clock['write_map']=='T'];actual=maps['actual_state_time']
  with Dataset(folder/'sfincs_map.nc') as raw:
   check(case+'_actual_clock_original_tolerances',len(actual)==109 and np.max(abs(actual-np.arange(109)*300))<=.001 and np.max(np.diff(actual))<=300.001 and np.array_equal(np.asarray(raw['time'][:]),maps['nominal_output_time'].astype('f4').astype('f8')))

  check(case+'_all_inputs_frozen',all(sha(folder/n)==h for n,h in run['input_sha256'].items()))
  with Dataset(folder/'historical_rain.nc') as nc:
   times=np.asarray(nc['time'][:])*60;inside=np.flipud(valid);rates=[]
   for i in range(109):rates.append(float(nc['Precipitation'][i,1:-1,1:-1].data[inside].astype('f8').sum()*100/3.6e6))
  volume=float(np.dot(np.diff(times),rates[:-1]));expected=json.loads((ROOT/'data/processed/tokyo_historical_radar_20240821_v1/spatial_forcing_qa.json').read_text())['total_reconstructed_source_rain_volume_m3']
  check(case+'_interval_not_trapezoid_volume',abs(volume-expected)<=max(1.,expected)*1e-6,dict(encoded_m3=volume,source_m3=expected))
 old_labels=np.load(P/'execution_initial_clock_audit/radar_low_candidate_reference.npz');new_labels=np.load(P/'execution/radar_low_candidate_reference.npz')
 check('clock_adapter_did_not_change_low_label_arrays',old_labels.files==new_labels.files and all(np.array_equal(old_labels[k],new_labels[k],equal_nan=True) for k in old_labels.files))
 manifest=json.loads((R/'artifact_manifest.json').read_text());cache={};bad=[]
 for rel,h in manifest['files_sha256'].items():
  p=ROOT/rel;st=p.stat();key=(st.st_dev,st.st_ino)
  if key not in cache:cache[key]=sha(p)
  if cache[key]!=h:bad.append(rel)
 check('all_artifact_hashes_match',not bad,bad)
 check('all_protected_inputs_and_formal_split_unchanged',all(sha(ROOT/rel)==h for rel,h in json.loads((P/'protected_sources.json').read_text()).items()))
 text=(R/'review.html').read_text();badlinks=[x for x in re.findall(r'(?:src|href)="([^"]+)"',text) if not (R/x).resolve().is_file()];check('report_links_images_exist',not badlinks,badlinks)
 s=json.loads((R/'summary.json').read_text());check('no_training_test_population_or_accuracy_claim',not any(s[k] for k in ['training','sealed_test','people_counted','formal_reference_release','real_world_flood_accuracy_established','river_stage_observed']))
 result=dict(all_passed=all(x['passed'] for x in checks),checks=checks);(R/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2));assert result['all_passed']

if __name__=='__main__':main()
