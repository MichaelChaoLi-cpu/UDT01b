"""Match every positivity decision to the actual accounted integration step."""
from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_conservative_positivity import OUT,REPORT,CASES
from prepare_sfincs_cpu_build import dump

def main(only=None):
 out=ROOT/OUT;report=ROOT/REPORT;assert not(out/'freeze.json').exists();manifest=json.loads((out/'run_manifest.json').read_text());data={};checks=[]
 for run in manifest['runs']:
  case=run['case_id']
  if only and case!=only:continue
  folder=ROOT/run['folder'];a=np.genfromtxt(folder/'udt_positivity_steps.csv',delimiter=',',names=True);b=np.genfromtxt(folder/'udt_step_budget.csv',delimiter=',',names=True)
  def check(name,ok):checks.append({'case':case,'check':name,'pass':bool(ok)})
  check('one_limiter_receipt_per_actual_applied_step',len(a)==len(b)-1 and np.array_equal(a['actual_end_time_s'],b['actual_state_time'][1:])and np.array_equal(a['dt_s'],b['actual_step_dt'][1:]))
  check('finite_limiter_values_and_valid_factors',all(np.all(np.isfinite(a[k]))for k in a.dtype.names)and np.all((a['minimum_factor']>=0)&(a['minimum_factor']<=1))and np.all(a['held_requested_transport_m3']>=0)and np.all(a['limited_donors']>=0)and np.all(a['limited_faces']>=0))
  check('limiter_actually_activated',np.any(a['limited_faces']>0)and np.sum(a['held_requested_transport_m3'])>0)
  check('strict_global_storage_nonnegative_all_actual_steps',np.all(b['minimum_native_volume_since_initialization']>=0)and np.all(b['negative_volume_sum']==0)and np.all(b['negative_native_count']==0))
  receipt=(folder/'udt_positivity_receipt.txt').read_text();check('receipt_explicitly_excludes_clipping_and_inflow_prespending','no same-step inflow credit; no storage clipping'in receipt)
  active=a['limited_faces']>0;data[case]={'actual_steps':len(a),'steps_with_limiting':int(active.sum()),'first_limited_actual_endpoint_s':float(a['actual_end_time_s'][active][0])if active.any()else None,'limited_face_step_observations':int(a['limited_faces'].sum()),'limited_donor_step_observations':int(a['limited_donors'].sum()),'held_requested_transport_m3':float(a['held_requested_transport_m3'].sum()),'minimum_factor':float(a['minimum_factor'].min()),'maximum_limited_faces_per_step':int(a['limited_faces'].max()),'minimum_native_volume_m3':float(b['minimum_native_volume_since_initialization'].min()),'held_transport_semantics':'difference from proposed flux before limiting, summed once per face and step; not added water, a source/sink or a measure benefit','strict_positivity_verified':bool(np.all(b['minimum_native_volume_since_initialization']>=0))}
 result={'cases':data,'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'early_v1_attempt_retained_and_not_admitted':True,'completed_v2_hydrodynamic_runs':len(data),'training':0,'batch_58':0};dump(report/(((only+'_')if only else'')+'positivity_diagnostic_audit.json'),result)
 print(json.dumps(result,indent=2));assert result['all_checks_passed']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--only',choices=CASES);main(p.parse_args().only)
