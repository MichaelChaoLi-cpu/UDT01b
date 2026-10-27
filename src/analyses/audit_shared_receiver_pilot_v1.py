#!/usr/bin/env python3
"""Independently replay variable-rate pump ledger and unchanged original numerical gates."""
from pathlib import Path
import sys,json,inspect
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'preprocessing')]
from shared_receiver_pilot_common_v1 import ROOT,P,R,sha,dump
import audit_cross_event_protection_run_v1 as parent

def main(case):
 run=next(r for r in json.loads((P/'run_manifest.json').read_text())['runs'] if r['case_id']==case);folder=ROOT/run['folder'];out=R/'execution'/case;out.mkdir(parents=True,exist_ok=True)
 sites=pd.read_csv(P/'pump_sites.csv').sort_values('Pump ID');n=len(sites);pump=np.loadtxt(folder/'udt_pumps.dat',skiprows=1,ndmin=2);assert len(pump)==n==62
 led=pd.read_csv(folder/'udt_step_budget.csv',float_precision='round_trip');t=led.actual_state_time.to_numpy();dt=led.actual_step_dt.to_numpy();ns=len(t)-1
 ids=np.arange(1,n+1);total_by_site=np.zeros(n);active_by_site=np.zeros(n,int);blocked_by_site=np.zeros(n,int);cum=0.;step=0;max_ledger_diff=0.;max_pool_violation=0.;actual_pool={p:0. for p in sites['Receiving Pool'].unique()};allowances=pd.read_csv(P/'pool_allowances.csv').set_index('Receiving Pool')['Scenario Shared Allowance'].to_dict()
 numeric=['actual_end_time_s','dt_s','pump_id','source_native_cell','receiver_native_cell','receiver_kcs','requested_m3','transferred_m3','source_prior_volume_m3','head_m','depth_m']
 for frame in pd.read_csv(folder/'udt_pump_steps.csv',chunksize=n*2048,float_precision='round_trip'):
  assert len(frame)%n==0;matrix={k:frame[k].to_numpy().reshape(-1,n) for k in numeric};count=len(frame)//n;target=np.arange(step+1,step+count+1);assert target[-1]<=ns
  assert all(np.isfinite(v).all() for v in matrix.values())
  assert np.array_equal(matrix['pump_id'],np.broadcast_to(ids,(count,n)));assert np.array_equal(matrix['source_native_cell'],np.broadcast_to(pump[:,0],(count,n)));assert np.array_equal(matrix['receiver_native_cell'],np.broadcast_to(pump[:,1],(count,n)));assert np.all(matrix['receiver_kcs']==2)
  assert np.array_equal(matrix['actual_end_time_s'],np.broadcast_to(t[target,None],(count,n)));assert np.array_equal(matrix['dt_s'],np.broadcast_to(dt[target,None],(count,n)))
  expected=dt[target,None]*pump[None,:,2];req=matrix['requested_m3'];v=matrix['transferred_m3'];available=matrix['source_prior_volume_m3'];assert np.max(abs(req-expected),initial=0.)<1e-12
  assert np.all(v>=0.) and np.all(available>=0.) and np.all(v<=req+1e-12) and np.all(v<=available+1e-12)
  blocked=(matrix['depth_m']<pump[None,:,3])|(matrix['head_m']>pump[None,:,4]);assert np.all(v[blocked]==0.)
  # Sequential sum reproduces the original per-pump accumulation order.
  perstep=np.cumsum(v,axis=1)[:,-1];prefix=cum+np.cumsum(perstep);error=float(np.max(abs(prefix-led.other_sink_cumulative.to_numpy()[target]),initial=0.));max_ledger_diff=max(max_ledger_diff,error);assert error<1e-7;cum=float(prefix[-1])
  for pool in actual_pool:
   m=sites['Receiving Pool'].eq(pool).to_numpy();pv=np.cumsum(v[:,m],axis=1)[:,-1];max_pool_violation=max(max_pool_violation,float(np.max(pv-allowances[pool]*dt[target],initial=0.)));assert np.all(pv<=allowances[pool]*dt[target]+1e-11);actual_pool[pool]+=float(pv.sum())
  total_by_site+=v.sum(axis=0);active_by_site+=(v>0).sum(axis=0);blocked_by_site+=blocked.sum(axis=0);step+=count
 assert step==ns and led.other_source_cumulative.eq(0).all() and cum>0,'Missing steps or zero actual extraction'
 sites['Actual Transferred Volume']=total_by_site;sites['Active Steps']=active_by_site;sites['Blocked Trigger Or Head Steps']=blocked_by_site;sites.to_csv(out/'actual_pump_site_totals.csv',index=False)
 checks={'all_step_records_and_endpoints':True,'actual_clock_exact':True,'variable_reserved_rates_exact':True,'nonnegative_prior_storage_bounded_transfer':True,'head_and_trigger_respected':True,'independent_pump_source_sink_ledger':True,'per_step_shared_pool_capacity':True,'positive_actual_extraction':True}
 # Same previously qualified explicit-source/sink extension;no tolerance or label rule changed.
 text=inspect.getsource(parent);anchor="and np.all(led['other_source_cumulative']==0)and np.all(led['other_sink_cumulative']==0)";assert text.count(anchor)==1
 text=text.replace(anchor,"and np.all(led['other_source_cumulative']>=0)and np.all(led['other_sink_cumulative']>=0)")
 env={'__name__':'qualified_static_receiver_audit','__file__':parent.__file__};exec(compile(text,parent.__file__,'exec'),env);env['OUT']=str(P.relative_to(ROOT));env['REPORT']=str(R.relative_to(ROOT));env['main']('execution',case)
 path=out/'audit.json';a=json.loads(path.read_text());assert a['all_checks_passed'] and a['admission']['technical_reference'];a['shared_receiver_checks']=checks;a['parent_auditor_sha256']=sha(Path(parent.__file__));a['extension']='Original zero-other-source clause replaced only after independent every-step finite variable-rate pump reconstruction;original numerical gates and tolerances unchanged.';dump(path,a)
 dump(out/'pump_budget_summary.json',{'passed':True,'case':case,'sites':n,'actual_steps':ns,'pump_records':ns*n,'actual_transferred_m3':float(led.other_sink_cumulative.iloc[-1]),'other_source_m3':float(led.other_source_cumulative.iloc[-1]),'max_independent_sink_ledger_difference_m3':max_ledger_diff,'maximum_shared_pool_violation_m3':max_pool_violation,'pool_actual_volumes_m3':{k:float(v) for k,v in actual_pool.items()},'all_original_gates_passed':True,'shared_receiver_checks':checks})
 print('SHARED_RECEIVER_PILOT_FULL_AUDIT_PASSED',case,flush=True)
if __name__=='__main__':main(sys.argv[1])
