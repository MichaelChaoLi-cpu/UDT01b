"""Physical-state UE costs at frozen baseline impact clocks;no ideal dry overlay."""
import sys,json,time,inspect,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import run_combined_transport_measures_v1 as parent
from temporal_transport_common_v1 import load,dump,sha
P=ROOT/'data/processed/three_ward_barrier_drainage_v1';R=ROOT/'data/exp/estimation'/P.name;D=ROOT/'data/processed';TP=P/'traffic';TR=R/'traffic';TEMPLATE=D/'three_ward_five_event_temporal_analysis_v1'
def main():
 assert json.loads((R/'postprocessing_validation.json').read_text())['passed'];start=time.monotonic()
 for f in[TP,TR,TP/'paths',TP/'fits']:f.mkdir(exist_ok=True)
 runs=json.loads((P/'run_manifest.json').read_text())['runs'];anchors=pd.read_csv(TEMPLATE/'reference_states.csv');anchors=anchors[anchors.selection_reasons.str.contains('H_noaction_closed_length_peak')].set_index('event_id');by={};q=load(P/'motor_common_support.npz')['qualified_roads'];pr=load(D/'three_ward_motor_flood_access_v1/road_projection.npz');tref=load(D/'three_ward_normal_vehicle_choice_v1/choice_costs.npz')['travel_seconds'];cap=load(D/'three_ward_demand_protection_feedback_v1/capacity_assumptions.npz')['hourly_capacity'];sources={str(f.relative_to(ROOT)):sha(f)for f in[D/'three_ward_normal_vehicle_repair_v1/turn_router.npz',D/'three_ward_normal_vehicle_repair_v1/multi_routing_demands.npz',D/'three_ward_normal_vehicle_repair_v1/multi_endpoint_components.npz',D/'three_ward_motor_flood_access_v1/road_projection.npz',D/'three_ward_normal_vehicle_choice_v1/choice_costs.npz',D/'three_ward_demand_protection_feedback_v1/capacity_assumptions.npz']}
 for run in runs:
  frame=int(anchors.loc[run['event_id'],'frame']);by[run['case_id']+'__f'+str(frame).zfill(3)]=dict(run=run,frame=frame)
 def check():
  if(P/'STOP').exists():raise RuntimeError('User STOP')
  while(P/'PAUSE').exists():time.sleep(5)
  assert time.monotonic()-start<14400,'Traffic budget';assert all(sha(ROOT/n)==h for n,h in sources.items())
 def state(case,arm):
  z=load(P/'exposure'/(by[case]['run']['case_id']+'.npz'));code=z['road_state_codes'][by[case]['frame'],0];indices=z['road_indices'];c=np.zeros(len(q),bool);s=c.copy();c[indices]=code==2;s[indices]=code==1;t=tref.copy();capacity=cap.copy();t[s[pr['arc_to_road']]]*=2;capacity[s[pr['arc_to_road']]]*=.5;return c,s,t,capacity
 def paths(case,arm,kind):return TP/'paths'/(case+'__flood_'+kind+'.npz')
 parent.P=TP;parent.R=TR;parent.check=check;parent.state=state;parent.paths_for=paths;parent.inherit=lambda *args:None;parent.REDUCTIONS=[0.]
 raw_minimize=parent.minimize;raw_dijkstra=parent.dijkstra
 def tick():
  if(P/'STOP').exists():raise RuntimeError('User STOP')
  while(P/'PAUSE').exists():time.sleep(5)
  assert time.monotonic()-start<14400,'Traffic stage budget'
 def guarded_minimize(fun,*args,**kw):
  def guarded_fun(x):tick();return fun(x)
  return raw_minimize(guarded_fun,*args,**kw)
 def guarded_dijkstra(*args,**kw):tick();return raw_dijkstra(*args,**kw)
 parent.minimize=guarded_minimize;parent.dijkstra=guarded_dijkstra
 env=parent.__dict__.copy();env.update(P=TP,R=TR,check=check,state=state,paths_for=paths,inherit=lambda *args:None,REDUCTIONS=[0.]);pathsource=inspect.getsource(parent.generate_paths);assert pathsource.count("for arm in ['flood','targeted']:")==1;pathsource=pathsource.replace("for arm in ['flood','targeted']:","for arm in ['flood']:").replace("sweeps=4","sweeps=2");fitsource=inspect.getsource(parent.fit);assert "arms=['normal']if case=='normal'else['flood','targeted']"in fitsource;fitsource=fitsource.replace("arms=['normal']if case=='normal'else['flood','targeted']","arms=['flood']").replace("for mode in ['UE','SO']:","for mode in ['UE']:");exec(compile(pathsource+'\n'+fitsource,parent.__file__,'exec'),env);dump(TP/'protocol.json',dict(question='Physical-treatment passability and unchanged proxy traffic cost',states=len(by),fits=16,path_sweeps_max=32,deferral=[0],allocation='Same original UE only',anchors='Frozen pre-measure H/no-action road-closure peak;identical elapsed time in L and every treatment',cost_cohort='Intersection of baseline-reachable core ODs remaining reachable in ALL four arms for each event-stage;excluded original requests separately counted. No claims for newly restored-trip cost.',static_sources=sources,parent_sha256=sha(Path(parent.__file__)),loop_adaptation_sha256=hashlib.sha256((pathsource+fitsource).encode()).hexdigest(),new_path_seconds_max=5760,new_fit_seconds_max=1920,no_ideal_dry_overlay=True))
 for i,case in enumerate(by,1):
  check();env['CASES']=['unused',case];env['generate_paths']();env['fit'](case);dump(R/'traffic_progress.json',dict(status='running',states_completed=i,states_total=16,current_case=case,elapsed_s=time.monotonic()-start));print('PHYSICAL TRAFFIC QUALIFIED',case,i,'/16',flush=True)
 dem=load(D/'three_ward_normal_vehicle_repair_v1/multi_routing_demands.npz');q0=parent.original_request(dem);core=load(D/'three_ward_demand_protection_feedback_v1/protection_masks.npz')['core_endpoint_od']&dem['base'];rows=[]
 for event in ['E20250911','E20240618']:
  for level in ['H','L']:
   labels=[(k,x)for k,x in by.items()if x['run']['event_id']==event and x['run']['stage']==level];fits={x['run']['measure']:load(TP/'fits'/(parent.key(k,'flood',0,'UE')+'.npz'))for k,x in labels};b=fits['none'];cohort=core&(q0>0)
   for f in fits.values():cohort&=f['reachable']
   assert q0[cohort].sum()>0;basecost=float(q0[cohort]@b['mean_od_cost'][cohort]/q0[cohort].sum());original=float(q0[core].sum())
   for arm,z in fits.items():
    served=float(q0[core&z['reachable']].sum());failed=float(q0[core&~z['reachable']].sum());assert abs(original-served-failed)<1e-6;rows.append(dict(event_id=event,stage=level,measure=arm,frame=int(anchors.loc[event,'frame']),original_request_proxy=original,served_proxy=served,unreachable_proxy=failed,completion_fraction_original=served/original,common_survivor_original_request_proxy=float(q0[cohort].sum()),baseline_reachable_excluded_from_cost_proxy=float(q0[core&b['reachable']&~cohort].sum()),common_survivor_cost_index=100*float(q0[cohort]@z['mean_od_cost'][cohort]/q0[cohort].sum())/basecost,total_network_cost=float(z['total_model_cost'])))
 pd.DataFrame(rows).to_csv(TP/'physical_traffic_outcomes.csv',index=False,encoding='utf-8-sig');vals=[json.loads(f.read_text())for f in TR.glob('*_fit_validation.json')];assert len(vals)==16 and all(v['passed']for v in vals);dump(R/'traffic_validation.json',dict(passed=True,states=16,maximum_gap=max(v['gap']for v in vals),maximum_balance=max(v['balance_residual']for v in vals),no_ideal_dry_overlay=True,no_deferral=True,common_cost_cohort_across_all_arms=True,actual_queue_validated=False,seconds=time.monotonic()-start));print('ALL PHYSICAL TRAFFIC GATES PASSED',flush=True)
if __name__=='__main__':main()
