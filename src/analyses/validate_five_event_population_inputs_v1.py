"""Independent preflight for bounded six-case extension; no numerical-gate changes."""
from pathlib import Path
import sys,json,datetime as dt
import numpy as np
from scipy.sparse import load_npz
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_five_event_population_validation_v1 import P,R,NEW,OLD,PRIOR,IMAGE,PAIR,load,dump,sha as uncached_sha
_digest_cache={}
def sha(path):
 st=path.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
 if key not in _digest_cache:_digest_cache[key]=uncached_sha(path)
 return _digest_cache[key]

def main():
 checks={};m=json.loads((P/'run_manifest.json').read_text());p=json.loads((P/'protocol.json').read_text());runs=m['runs'];checks['six_unique_cases_three_parents']=len(runs)==6 and len({r['case_id']for r in runs})==6 and len({r['parent_id']for r in runs})==3;checks['unchanged_solver_image']=all(r['image_id']==r['identity']['solver_digest']==IMAGE for r in runs);checks['no_prior_attempt']=not any((ROOT/r['folder']/'launch.json').exists()for r in runs);checks['protocol_bound']=sha(P/'protocol.json')==m['protocol_sha256'];old=json.loads((PRIOR/'run_manifest.json').read_text())['runs'];template=next(r for r in old if r['case_id']=='E20240830__L__U');src=ROOT/template['folder'];kv=lambda f:{line.partition('=')[0].strip():line.partition('=')[2].strip()for line in f.read_text().splitlines()if'='in line};before=kv(src/'sfincs.inp');mat=load_npz(OLD/'urban_source_area_matrix.npz');pair=load(PAIR/'paired_support_and_native_mapping.npz');active=pair['Urban Active Mask']==1
 for r in runs:
  folder=ROOT/r['folder'];case=r['case_id'];checks[case+'_all_inputs_bound']=all(sha(folder/n)==h for n,h in r['input_sha256'].items());new=kv(folder/'sfincs.inp');checks[case+'_only_utc_epochs_change']=set(before)==set(new)and all(before[k]==new[k]for k in before if k not in ['tref','tstart','tstop']);begin=dt.datetime.fromisoformat(r['window_start_utc']);checks[case+'_utc_epoch']=new['tref']==new['tstart']==begin.strftime('%Y%m%d %H%M%S')and new['tstop']==(begin+dt.timedelta(hours=9)).strftime('%Y%m%d %H%M%S');oldstage=ROOT/next(x['folder']for x in old if x['case_id']=='E20240830__'+r['stage']+'__U')/'sfincs.bzs';checks[case+'_unchanged_stage']=sha(folder/'sfincs.bzs')==sha(oldstage);checks[case+'_no_automatic_training_release']=r['training']is False and r['sealed_test']is False
 for e in p['events']:
  eid=e['event_id'];rad=load(NEW/eid/'radar_intervals.npz');flat=rad['representative_intensity_mm_h'].reshape(108,-1).astype('f8');area=np.asarray(mat.sum(0)).ravel();used=area>0;flat[:,~used]=0;good=np.isfinite(flat[:,used]).all()and not(rad['level'].reshape(108,-1)[:,used]==251).any();checks[eid+'_source_clock']=np.all(rad['interval_end_unix_s']-rad['interval_start_unix_s']==300)and np.array_equal(rad['interval_end_unix_s'][:-1],rad['interval_start_unix_s'][1:]);total=0.
  with Dataset(P/'forcing'/eid/'historical_rain.nc')as nc,Dataset(P/'rainfall'/eid/'forcing_intervals_10m.nc')as ref:
   nc.set_auto_mask(False);checks[eid+'_seconds_sentinel']=np.array_equal(nc['time'][:]*60,np.arange(109)*300)and np.all(nc['Precipitation'][108]==0);checks[eid+'_rain_epoch']=nc['time'].units=='minutes since '+dt.datetime.fromisoformat(e['window_start_utc']).strftime('%Y-%m-%d %H:%M:%S')
   for i in range(108):
    expected=(mat@flat[i]/100).astype('f4').reshape(1012,770);got=np.flipud(nc['Precipitation'][i,1:-1,1:-1]);good &=np.array_equal(expected,got)and np.array_equal(got,np.asarray(ref['urban_effective_rainfall'][i]));total+=float(got[active].sum(dtype='f8')*100/12000)
  nominal=float((flat@area).sum()/12000);checks[eid+'_source_recipient_exact']=bool(good);checks[eid+'_original_rain_volume_tolerance']=abs(total-nominal)<=max(1.,nominal)*1e-6
 protected=json.loads((P/'protected_sources.json').read_text());checks['protected_sources_unchanged']=all(sha(ROOT/n)==h for n,h in protected.items());checks['original_numerical_auditor_bound']=protected['src/analyses/audit_cross_event_protection_run_v1.py']==sha(ROOT/'src/analyses/audit_cross_event_protection_run_v1.py');assert all(checks.values()),checks;checks={k:bool(v)for k,v in checks.items()};dump(R/'plan_validation.json',dict(all_passed=True,checks=checks,count=len(checks),maximum_runs=6,numerical_gate_changes=0));print('ALL',len(checks),'PREFLIGHT CHECKS PASSED',flush=True)
if __name__=='__main__':main()
