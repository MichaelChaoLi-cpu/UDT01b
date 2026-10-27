"""Exact parent audit with explicit,new source/sink ledger;no tolerance relaxation."""
from pathlib import Path
import sys,json,inspect
import numpy as np,pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
import audit_cross_event_protection_run_v1 as parent
from observed_load_common_v1 import dump,sha
P=ROOT/'data/processed/three_ward_barrier_drainage_v1';R=ROOT/'data/exp/estimation'/P.name

def main(case):
 run=next(r for r in json.loads((P/'run_manifest.json').read_text())['runs']if r['case_id']==case);folder=ROOT/run['folder'];led=pd.read_csv(folder/'udt_step_budget.csv');t=led.actual_state_time.to_numpy();ns=len(t)-1;checks={};pump=folder/'udt_pumps.dat'
 if pump.exists():
  sites=pd.read_csv(P/'measures/pump_sites.csv');v=pd.read_csv(folder/'udt_pump_steps.csv');checks['all_step_pump_records']=len(v)==ns*len(sites)and v.groupby('actual_end_time_s').size().eq(len(sites)).all();checks['pump_clock']=np.array_equal(v.groupby('actual_end_time_s',sort=False).dt_s.first().index.to_numpy(),t[1:])and np.array_equal(v.groupby('actual_end_time_s',sort=False).dt_s.first().to_numpy(),led.actual_step_dt.to_numpy()[1:]);checks['finite_nonnegative_transfer']=bool(np.isfinite(v.select_dtypes(include='number')).all().all()and(v.transferred_m3>=0).all());checks['bounded_pump']=bool((v.transferred_m3<=v.requested_m3+1e-12).all()and(v.transferred_m3<=v.source_prior_volume_m3+1e-12).all()and np.max(abs(v.requested_m3-.5*v.dt_s))<1e-12);checks['head_and_trigger']=v.loc[(v.depth_m<.05)|(v.head_m>5),'transferred_m3'].eq(0).all()
  source=np.r_[0,np.cumsum(v.assign(x=v.transferred_m3*(v.receiver_kcs==1)).groupby('actual_end_time_s',sort=False).x.sum().to_numpy())];sink=np.r_[0,np.cumsum(v.groupby('actual_end_time_s',sort=False).transferred_m3.sum().to_numpy())];checks['independent_pump_budget']=np.max(abs(source-led.other_source_cumulative))<1e-7 and np.max(abs(sink-led.other_sink_cumulative))<1e-7
  for i,s in sites.iterrows():
   part=v[v.pump_id==s.pump_id];assert part.source_native_cell.eq(s.source_native_cell).all()and part.receiver_native_cell.eq(s.receiver_native_cell).all()
  checks['fixed_endpoints']=True
 else:checks['no_unconfigured_sink']=bool(led.other_source_cumulative.eq(0).all()and led.other_sink_cumulative.eq(0).all())
 if run['measure']in['barrier','combined']:
  faces=pd.read_csv(P/'measures/barrier_faces.csv');mid=np.c_[(faces.x1+faces.x2)/2,(faces.y1+faces.y2)/2]
  with Dataset(folder/'sfincs_his.nc')as ds:
   coords=np.c_[np.asarray(ds['structure_x'][:]),np.asarray(ds['structure_y'][:])];height=np.asarray(ds['structure_height'][:]);dist,ix=cKDTree(mid).query(coords);checks['finite_barriers_exact_faces']=len(coords)==len(mid)and len(np.unique(ix))==len(mid)and dist.max()<1e-5 and np.max(abs(height-faces.crest_m.to_numpy()[ix]))<1e-4
 assert all(checks.values()),checks
 # Parent zero-other-source profile is replaced ONLY by independently replayed pump trace.
 text=inspect.getsource(parent);anchor="and np.all(led['other_source_cumulative']==0)and np.all(led['other_sink_cumulative']==0)";assert text.count(anchor)==1;text=text.replace(anchor,"and np.all(led['other_source_cumulative']>=0)and np.all(led['other_sink_cumulative']>=0)");env={'__name__':'qualified_measure_audit','__file__':parent.__file__};exec(compile(text,parent.__file__,'exec'),env);env['OUT']=str(P.relative_to(ROOT));env['REPORT']=str(R.relative_to(ROOT));env['main']('execution',case);f=R/'execution'/case/'audit.json';a=json.loads(f.read_text());a.update(physical_measure_checks={k:bool(v)for k,v in checks.items()},parent_auditor_sha256=sha(Path(parent.__file__)),contract_extension='Explicit bounded internal-transfer source and prescribed-stage discharge sink,independently replayed from every-step traces;all parent thresholds unchanged.');dump(f,a);dump(R/'execution'/case/'measure_budget_summary.json',dict(passed=True,source_m3=float(led.other_source_cumulative.iloc[-1]),sink_m3=float(led.other_sink_cumulative.iloc[-1]),maximum_abs_budget_residual_m3=float(abs(led.budget_residual).max()),measure=run['measure']));print('ALL PHYSICAL AND ORIGINAL GATES PASSED',case,flush=True)
if __name__=='__main__':main(sys.argv[1])
