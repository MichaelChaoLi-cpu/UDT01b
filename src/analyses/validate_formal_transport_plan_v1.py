"""Pre-launch qualification of frozen matrix, exact forcing and native adapters."""
from pathlib import Path
import sys,json,itertools
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_formal_transport_experiments_v1 import P,R,D,H,HR,RAIN,PAIR,IMAGE,FACTORS,load,read,dump,sha
from audit_formal_transport_run_v1 import receiver_geometry
from audit_urban_geometry_terrain_control_v1 import receiver_geometry as terrain_receiver
from reference_label_policy import CONTRACT

def main():
 m=json.loads((P/'run_manifest.json').read_text());p=json.loads((D/'protocol.json').read_text());checks=[]
 def check(name,ok):checks.append(dict(check=name,passed=bool(ok)));assert ok,name
 check('protocol_frozen_before_execution',sha(D/'protocol.json')==m['protocol_sha256'])
 check('complete_3x2x2_hydraulic_factorial',len(m['runs'])==12 and {(r['rainfall'],r['stage'],r['arm']) for r in m['runs']}==set(itertools.product(FACTORS,['L','H'],['T','U'])))
 new=[r for r in m['runs'] if not r['reused']];reuse=[r for r in m['runs'] if r['reused']];check('exactly_two_reuses_ten_new',len(reuse)==2 and len(new)==10)
 check('finite_serial_queue_and_first_batch',len(set(m['new_execution_order']))==10 and set(m['new_execution_order'])=={r['case_id'] for r in new} and m['new_execution_order'][:2]==['R1__L__T','R1__H__T'])
 check('resource_limits_registered',m['budget']['concurrent_hydraulic_runs']==1 and m['budget']['maximum_new_runs']==10 and m['budget']['maximum_run_active_wall_s']==7200 and m['budget']['maximum_campaign_active_wall_s']==50400 and m['budget']['automatic_retries']==0)
 check('strict_supplement_unknown_branches_retained',p['transport_postprocessing']['road_projection']==['strict_classified_road','ground_path_supplement'] and len(p['transport_postprocessing']['uncertainty'])==2 and len(p['transport_postprocessing']['station_identity'])==2)
 check('network_factorial_count_and_synchronization',p['transport_postprocessing']['maximum_network_evaluations']==12*13*4*2*2*2*2 and p['grid']['synchronized_times_s']==sorted(set(range(0,32401,3600))|{12600,16200}))
 check('scope_excludes_people_learning_and_sealed_tests',not any(p[k] for k in ['people_counted','training','sealed_tests_opened']))
 check('unchanged_numerical_policy',p['reference_policy_sha256']==sha(ROOT/'src/preprocessing/reference_label_policy.py') and CONTRACT['time_tolerance_s']==.001 and CONTRACT['budget']['rain_volume_relative_tolerance']==1e-6)
 check('protected_inputs_unchanged',all(sha(ROOT/k)==h for k,h in json.loads((P/'protected_sources.json').read_text()).items()))
 for r in reuse:
  a=json.loads((ROOT/r['original_audit']).read_text());check('reuse_'+r['case_id']+'_qualified_matching_image_clock',a['all_checks_passed'] and a['summary']['map_records']==109 and r['image_id']==IMAGE)
 geometry=load(PAIR/'paired_support_and_native_mapping.npz');expected_volume=json.loads((RAIN/'spatial_forcing_qa.json').read_text())['total_reconstructed_source_rain_volume_m3'];allfields=True;allvolumes=True;allidentities=True;allstages=True;notopologychange=True;adapters={}
 for r in new:
  folder=ROOT/r['folder'];allidentities &= all(sha(folder/n)==h for n,h in r['input_sha256'].items()) and r['identity']['solver_digest']==IMAGE
  inside=geometry['Terrain Active Mask']==1 if r['arm']=='T' else geometry['Urban Active Mask']==1;rates=[]
  with Dataset(folder/'historical_rain.nc') as d,Dataset(RAIN/'forcing_intervals_10m.nc') as src:
   times=np.asarray(d['time'][:])*60;allfields &= np.max(abs(times-np.arange(109)*300))<5e-7 and d['time'].units=='minutes since 2024-08-21 06:00:00' and d['Precipitation'].unit=='mm.hr-1' and np.all(d['Precipitation'][-1]==0)
   for k in range(108):
    field=np.flipud(np.asarray(d['Precipitation'][k]))[1:-1,1:-1];expected=(np.ma.asarray(src['terrain_effective_rainfall' if r['arm']=='T' else 'urban_effective_rainfall'][k]).filled(np.nan).astype('f8')*r['rainfall_scale']).astype('f4');allfields &= np.array_equal(field,expected);rates.append(field[inside].sum(dtype='f8')*100/3.6e6)
  encoded=float(np.array(rates)@np.diff(times));allvolumes &= abs(encoded-expected_volume*r['rainfall_scale'])<=max(1.,expected_volume*r['rainfall_scale'])*1e-6
  stage=H/'runs'/('radar_raised' if r['stage']=='H' else 'radar_low')/'sfincs.bzs';allstages &= sha(stage)==sha(folder/'sfincs.bzs')
  if r['arm'] not in adapters:
   ok,closed,groups,names,mid=terrain_receiver(folder) if r['arm']=='T' else receiver_geometry(folder);adapters[r['arm']]=bool(ok and (not closed.any() if r['arm']=='T' else closed.any()))
  notopologychange &= ('building_closures.thd' in r['input_sha256'])==(r['arm']=='U')
 check('all_new_input_identities_and_bytes_bound',allidentities);check('all_new_109_time_fields_match_exact_factor_and_source_allocation',allfields);check('all_ten_case_rain_volumes_pass_original_1e6_relative_gate',allvolumes);check('T_U_and_rain_cases_use_exact_same_stage_bytes_at_fixed_stage',allstages);check('native_T_and_U_receiver_adapters_are_independently_valid',all(adapters.values()));check('geometry_barriers_present_only_in_urban_arm',notopologychange)
 check('no_hydraulic_attempt_before_preflight',not list((P/'runs').glob('*/launch.json')))
 dump(R/'plan_validation.json',dict(all_passed=True,checks=checks,hydraulic_runs_started=0));print('PLAN QUALIFIED',len(checks),'checks',flush=True)
if __name__=='__main__':main()
