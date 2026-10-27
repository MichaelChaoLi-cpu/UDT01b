"""Post-admission intervention transfer: three new storms,48bounded cells+8controls.
Uses original exact optimizer/gates; fixed protection,OD,capacity and candidate rules.
"""
from pathlib import Path
import os,sys,json,time,argparse,shutil
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','2');os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/udt01b-numba-five-event')
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
import numpy as np,pandas as pd
from scipy.sparse import load_npz
import run_combined_transport_measures_v1 as original
from prepare_five_event_population_validation_v1 import P as CAMPAIGN,R as REPORT,load,dump,sha
P=CAMPAIGN/'traffic';R=REPORT/'traffic';D=ROOT/'data/processed';OLD=D/'three_ward_combined_transport_measures_v1';OLDR=ROOT/'data/exp/estimation/three_ward_combined_transport_measures_v1';MEASURE=D/'three_ward_demand_protection_feedback_v1';M=D/'three_ward_motor_flood_access_v1';C=D/'three_ward_normal_vehicle_choice_v1';REP=D/'three_ward_normal_vehicle_repair_v1';CASES=['E20240731__H__U','E20250710__H__U','E20240618__H__U']

def check():
 if(CAMPAIGN/'STOP').exists():raise RuntimeError('User STOP during traffic')
 while(CAMPAIGN/'PAUSE').exists():
  if(CAMPAIGN/'STOP').exists():raise RuntimeError('User STOP while traffic paused')
  time.sleep(10)
 sources=json.loads((CAMPAIGN/'protected_sources.json').read_text());assert all(sha(ROOT/n)==h for n,h in sources.items())
def paths_for(case,arm,kind):return MEASURE/f'normal_{kind}.npz'if case=='normal'or arm=='normal'else P/'paths'/f'{case}__{arm}_{kind}.npz'
def state(case,arm):
 support=load(CAMPAIGN/'motor_common_support.npz');q=support['qualified_roads'];closed=np.zeros(len(q),bool);slow=closed.copy()
 if case!='normal':
  ex=load(CAMPAIGN/'motor_exposures'/(case+'.npz'));assert abs(ex['actual_times'][5]-14400)<.001;depth=ex['lower'][5];closed=q&(depth>=.3);slow=q&(depth>=.1)&~closed
  if arm=='targeted':closed[support['targeted_effective']]=False;slow[support['targeted_effective']]=False
 pr=load(M/'road_projection.npz');t=load(C/'choice_costs.npz')['travel_seconds'].copy();cap=load(MEASURE/'capacity_assumptions.npz')['hourly_capacity'].copy();t[slow[pr['arc_to_road']]]*=2;cap[slow[pr['arc_to_road']]]*=.5
 return closed,slow,t,cap

def configure():
 P.mkdir(exist_ok=True);R.mkdir(exist_ok=True);(P/'paths').mkdir(exist_ok=True);(P/'fits').mkdir(exist_ok=True);original.P=P;original.R=R;original.check=check;original.paths_for=paths_for;original.state=state;original.inherit=lambda *args:None

def selfcheck():
 configure();original.optimizer_test();q=original.original_request(load(REP/'multi_routing_demands.npz'));h=load_npz(MEASURE/'normal_A.npz');assert h.shape[1]==len(q);assert np.isfinite(q).all()and(q>=0).all()
 for r in [0.,.1,.2,.3]:
  for mode in ['UE','SO']:
   name=original.key('normal','normal',r,mode);v=json.loads((OLDR/(name+'_fit_validation.json')).read_text());assert v['passed'];z=load(OLD/'fits'/(name+'.npz'));assert np.array_equal(z['original_requested'],q)and np.allclose(z['retained_requested'],q*(1-r),rtol=0,atol=1e-10)
 dump(R/'interface_selfcheck.json',dict(passed=True,original_optimizer_analytic_selfcheck=True,normal_controls_unchanged=8,OD_count=len(q),maximum_new_fits=48,new_hydraulic_runs=0));print('TRAFFIC INTERFACE SELFCHECK PASSED',flush=True)

def main():
 started=time.monotonic();assert json.loads((REPORT/'population/validation.json').read_text())['passed'];selfcheck();check();protocol=json.loads((CAMPAIGN/'protocol.json').read_text());dump(P/'protocol.json',dict(decision=protocol['decision'],cases=CASES,snapshot_elapsed_s=14400,new_fits=48,normal_controls_reused=8,reductions=[0,.1,.2,.3],guidance=['UE','SO'],protection=['flood','targeted'],budget=protocol['postprocessing_budget'],assumptions=protocol['traffic'],population='Separate resident-stock consequences; no vehicle-person conversion',common_support='Use intersection from all6new admitted runs and prior qualified support; if this changes old road evidence,old traffic result scalars are not treated as matched comparisons.',measures=protocol['measures']))
 for r in [0.,.1,.2,.3]:
  for mode in ['UE','SO']:
   name=original.key('normal','normal',r,mode)
   for src,dest in [(OLD/'fits'/(name+'.npz'),P/'fits'/(name+'.npz')),(OLDR/(name+'_fit_validation.json'),R/(name+'_fit_validation.json'))]:
    if not dest.exists():shutil.copy2(src,dest)
 for case in CASES:
  check();original.CASES=['unused_no_inheritance',case];original.generate_paths();shutil.copy2(R/'path_campaign.json',R/(case+'_path_campaign.json'));original.fit(case);print('FOLLOW-ON48CELL CAMPAIGN',case,'QUALIFIED',flush=True)
 dem=load(REP/'multi_routing_demands.npz');q0=original.original_request(dem);core=load(MEASURE/'protection_masks.npz')['core_endpoint_od']&dem['base'];rows=[];effects=[];fullcore=float(q0[core].sum())
 for case in CASES:
  reference=load(P/'fits'/(original.key(case,'flood',0.,'UE')+'.npz'));cohort=core&reference['reachable']&(q0>0);weights=q0[cohort];baseline=float(weights@reference['mean_od_cost'][cohort]/weights.sum());assert baseline>0
  for arm in ['flood','targeted']:
   for r in [0.,.1,.2,.3]:
    ue=load(P/'fits'/(original.key(case,arm,r,'UE')+'.npz'))
    for mode in ['UE','SO']:
     z=load(P/'fits'/(original.key(case,arm,r,mode)+'.npz'));ret=z['retained_requested'];served=core&z['reachable'];failed=core&~z['reachable'];deferred=float((q0[core]-ret[core]).sum());completed=float(ret[served].sum());unreach=float(ret[failed].sum());assert abs(fullcore-deferred-completed-unreach)<1e-6;assert z['reachable'][cohort].all();cohortcost=float(weights@z['mean_od_cost'][cohort]/weights.sum());rows.append(dict(case_id=case,protection=arm,deferred_fraction=r,guidance=mode,original_core_request_proxy=fullcore,deferred_core_proxy=deferred,retained_served_core_proxy=completed,retained_unreachable_core_proxy=unreach,completion_fraction_original=completed/fullcore,fixed_original_cohort_cost_index=100*cohortcost/baseline,total_network_model_cost=float(z['total_model_cost']),same_retained_guidance_network_cost_index=100*float(z['total_model_cost'])/float(ue['total_model_cost'])))
     if mode=='SO':effects.append(dict(case_id=case,protection=arm,deferred_fraction=r,network_cost_reduction_fraction=1-float(z['total_model_cost'])/float(ue['total_model_cost']),same_retained_demand=True))
 tab=pd.DataFrame(rows);tab.to_csv(P/'combined_outcomes.csv',index=False,encoding='utf-8-sig');pd.DataFrame(effects).to_csv(P/'same_demand_guidance_effects.csv',index=False,encoding='utf-8-sig');vals=[json.loads(f.read_text())for f in R.glob('*_fit_validation.json')];paths=[json.loads(f.read_text())for f in R.glob('*_path_validation.json')];assert len(vals)==56 and len(paths)==12;assert all(v['passed']for v in vals+paths);assert sum(v['seconds']for v in vals if v['case_id']!='normal')<=7200;assert sum(v['seconds']for v in paths)<=2700
 # Independent recomputation of every new optimizer result,including original OD ledgers.
 maxcost=0.;maxledger=0.;maxflow=0.
 for v in vals:
  if v['case_id']=='normal':continue
  case=v['case_id'];arm=v['arm'];r=v['deferred_fraction'];z=load(P/'fits'/(original.key(case,arm,r,v['guidance'])+'.npz'));closed,slow,tref,cap=state(case,arm);expected=tref*(1+.15*(z['arc_flow']/cap)**4);maxcost=max(maxcost,float(abs(expected-z['arc_cost']).max(initial=0)));q=z['retained_requested'];a=load_npz(paths_for(case,arm,'A'))[:,z['active_od_index']];b=load_npz(paths_for(case,arm,'B'))[:,z['active_od_index']];qactive=q[z['active_od_index']];flow=np.asarray(a@(qactive-z['routeB'])+b@z['routeB']).ravel();maxflow=max(maxflow,float(abs(flow-z['arc_flow']).max(initial=0)));ledger=(q0-q)+q*z['reachable']+q*~z['reachable'];maxledger=max(maxledger,float(abs(ledger-q0).max(initial=0)))
 assert maxcost<1e-6 and maxflow<1e-6 and maxledger<1e-6;check();oldq=load(D/'three_ward_cross_event_protection_v1/motor_common_support.npz')['qualified_roads'];newq=load(CAMPAIGN/'motor_common_support.npz')['qualified_roads'];same_support=np.array_equal(oldq,newq);val=dict(passed=True,new_event_fits=48,normal_controls_reused=8,path_sweeps=12,maximum_gap=max(v['gap']for v in vals),maximum_component_balance=max(v['balance_residual']for v in vals),max_independent_cost_error=maxcost,max_independent_route_loading_error=maxflow,max_per_OD_ledger_error=maxledger,new_fit_seconds=sum(v['seconds']for v in vals if v['case_id']!='normal'),path_seconds=sum(v['seconds']for v in paths),old_and_new_road_qualification_identical=same_support,old_two_events_directly_comparable=same_support,seconds=time.monotonic()-started,real_queue_delay_validated=False,people_inferred_from_trips=False,training=False,sealed_test=False);dump(R/'validation.json',val);dump(R/'summary.json',dict(validation=val,outcomes=rows,interpretation='Static two-candidate-path cost comparisons with explicit deferred-demand ledger;not actual queue delays or observed traffic. New events share17h proxy load and fixed modern geometry;no policy efficacy certification.'));print('48 NEW COMBINATION CELLS QUALIFIED',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--selfcheck',action='store_true');args=a.parse_args();selfcheck()if args.selfcheck else main()
