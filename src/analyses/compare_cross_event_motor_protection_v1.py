"""Fixed protection transfer after8qualified hydraulics, with common admission and native car rules."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('NUMBA_NUM_THREADS','2')
from pathlib import Path
import sys,json,collections,time
import numpy as np
from netCDF4 import Dataset
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_cross_event_protection_v1 import P,R,D,F,MEASURE,MOTOR,load,read,write,dump,sha
from prepare_historical_surface_coverage_v2 import edge_maximum
from run_motor_flood_access_v1 import graph,route
ARMS=['none','corridor','distributed']

def user_gate():
 if (P/'STOP').exists():raise RuntimeError('User STOP during transport postprocessing')
 while (P/'PAUSE').exists():
  dump(R/'motor_execution_state.json',dict(status='paused_by_user',no_new_hydraulics=True))
  if (P/'STOP').exists():raise RuntimeError('User STOP while transport paused')
  time.sleep(5)

def same(x,y):
 return np.array_equal(np.isfinite(x),np.isfinite(y))and np.allclose(x[np.isfinite(x)],y[np.isfinite(y)],atol=1e-6,rtol=0)

def main():
 start=time.monotonic();user_gate();manifest=json.loads((P/'run_manifest.json').read_text());protocol=json.loads((D/'protocol.json').read_text());runs=manifest['runs'];assert len(runs)==8;assert all(json.loads((R/'execution'/r['case_id']/'audit.json').read_text())['all_checks_passed']for r in runs)
 assert all(sha(ROOT/f)==h for f,h in json.loads((P/'protected_sources.json').read_text()).items());assert not(P/'motor_accessibility.npz').exists(),'Do not overwrite completed postprocessing'
 p=load(MOTOR/'road_projection.npz');router=load(MOTOR/'turn_router.npz');probes=read(MOTOR/'probe_inventory.csv');roads=read(MOTOR/'road_inventory.csv');prior_support=load(MOTOR/'evidence_support.npz');fixed=load(MEASURE/'protected_roads.npz')['protected'];common=load(F/'factor_comparison_v1/common_support.npz')['common_admitted'].copy()
 for run in runs:common &=load(P/'execution'/(run['case_id']+'_candidate_reference.npz'))['Admitted Mask']
 qualified=p['qualified'].copy();coarse=p['point_coarse'];refs=p['point_refs'];offset=p['offsets']
 for j in np.flatnonzero(qualified):qualified[j]=common.ravel()[coarse[refs[offset[j]:offset[j+1]]]].all()
 unknown=~qualified;effective=fixed&qualified[None,:];np.savez_compressed(P/'motor_common_support.npz',common_admitted=common,qualified_roads=qualified,original_fixed_protection=fixed,effective_known_protection=effective)
 # Match the normal unknown branch to the same common road support, not to the old support.
 normal=np.stack([route(router,graph(router,np.zeros(len(unknown),bool))),route(router,graph(router,unknown))]);baseok=np.isfinite(normal);eligible_out=baseok[:,0].any(axis=1).all(axis=0)&prior_support['eligible_outward'];eligible_two=baseok.any(axis=2).all(axis=1).all(axis=0)&prior_support['eligible_two_direction']
 z=load(MOTOR/'source_motor_graph.npz');ix={str(n):i for i,n in enumerate(z['node_ids'])};incident=np.zeros(len(z['node_ids']),bool);arcs=qualified[p['arc_to_road']];incident[p['from_index'][arcs]]=True;incident[p['to_index'][arcs]]=True;ip=np.array([incident[ix[pr['car_source_node']]]if pr['candidate_available']=='True'else False for pr in probes]);eligible_out &=ip;eligible_two &=ip
 eligible=[eligible_out,eligible_two];commonpairs=[np.isfinite(normal[:,d]).all(axis=0)&eligible[d][None,:]for d in range(2)]
 out=P/'motor_exposures';out.mkdir(exist_ok=True);frames=[str(t)for t in protocol['grid']['times_s']]+['peak_envelope'];states=[];distance=[];cache={};checks=[]
 def check(name,v):checks.append(dict(check=name,passed=bool(v)));assert v,name
 for run in runs:
  user_gate();case=run['case_id'];clock=load(P/'execution'/(case+'_output_clock.npz'))['Actual Map Time'];targets=np.array(protocol['grid']['times_s']);indices=[int(np.argmin(abs(clock-t)))for t in targets];check(case+'_actual_clock',max(abs(clock[indices]-targets))<=.001);lower=[];upper=[]
  with Dataset(ROOT/run['folder']/'sfincs_map.nc')as ds:
   bed=np.flipud(np.ma.asarray(ds['zb'][:],dtype='f8').filled(np.nan)).ravel()
   for k in indices+[None]:
    eta=np.flipud(np.ma.asarray(ds['zs'][k]if k is not None else ds['zsmax'][0],dtype='f8').filled(np.nan)).ravel();lo,up=edge_maximum(p['point_z'],coarse,refs,offset,qualified,eta,bed);lower.append(lo);upper.append(up)
  lower=np.array(lower);upper=np.array(upper);check(case+'_exposure_bounds',np.all(np.isfinite(lower[:,qualified]))and np.all(lower[:,qualified]<=upper[:,qualified]));np.savez_compressed(out/(case+'.npz'),lower=lower,upper=upper,actual_times=clock[indices])
  for ti,frame in enumerate(frames):
   user_gate()
   for threshold in [.1,.3]:
    for branch in range(2):
     depth=lower[ti]if branch==0 else upper[ti];closed=(qualified&(depth>=threshold))|(unknown if branch else False);result=[]
     for ai,arm in enumerate(ARMS):
      c=closed&~effective[ai];check(f'no_unknown_reopened_{len(states)}_{arm}',np.array_equal(c[unknown],closed[unknown]));sig=np.packbits(c).tobytes()
      if sig not in cache:cache[sig]=route(router,graph(router,c))
      result.append(cache[sig]);before=normal[branch];after=result[-1];check(f'no_shortcut_beyond_normal_{len(states)}_{arm}',not(np.isfinite(after)&~np.isfinite(before)).any()and np.all(after[np.isfinite(before)]>=before[np.isfinite(before)]-1e-6))
     result=np.stack(result)
     for ai in [1,2]:
      base=result[0];ok=np.isfinite(base);check(f'monotonic_{len(states)}_{ai}',np.isfinite(result[ai][ok]).all()and np.all(result[ai][ok]<=base[ok]+1e-6))
     distance.append(result);states.append(dict(state_index=len(states),case_id=case,event_id=run['event_id'],parent_id=run['parent_id'],geometry=run['arm'],stage=run['stage'],frame_id=frame,simultaneous=frame!='peak_envelope',threshold_m=threshold,unknown_branch=['unknown_available','unknown_withheld'][branch],closed_known_wet_roads=int((qualified&(depth>=threshold)).sum())))
  dump(R/'motor_execution_state.json',dict(status='running',hydraulic_cases_processed=len({s['case_id']for s in states}),hydraulic_cases_total=8,network_states_with3arms=len(states)*3,network_total=1248,elapsed_s=time.monotonic()-start));print('CROSS-EVENT CAR',case,len(states)*3,'/1248',flush=True)
 distance=np.stack(distance);np.savez_compressed(P/'motor_accessibility.npz',distance_m=distance,normal_distance_m=normal,eligible_outward=eligible_out,eligible_two_direction=eligible_two,incident_probe=ip);write(P/'motor_scenario_index.csv',states)
 groups=collections.defaultdict(list)
 for i,s in enumerate(states):groups[s['case_id'],s['frame_id']].append(i)
 classes=[];pc=[];burden=[]
 for ai,arm in enumerate(ARMS):
  for(case,frame),ids in groups.items():
   finite=np.isfinite(distance[ids,ai]);outward=finite[:,0].any(axis=1);two=finite.any(axis=2).all(axis=1)
   for d,values in enumerate([outward,two]):
    ok=eligible[d];classes.append(dict(arm=arm,case_id=case,frame_id=frame,simultaneous=frame!='peak_envelope',direction=['outward','two_direction'][d],robust_modeled_loss=int((ok&~values.any(axis=0)).sum()),conditional_sensitivity=int((ok&values.any(axis=0)&~values.all(axis=0)).sum()),retained_access=int((ok&values.all(axis=0)).sum()),insufficient_evidence=int((~ok).sum())))
   for j,pr in enumerate(probes):
    label='insufficient_evidence'if not eligible_out[j]else'robust_modeled_loss'if not outward[:,j].any()else'retained_access'if outward[:,j].all()else'conditional_sensitivity';pc.append(dict(arm=arm,case_id=case,frame_id=frame,probe_id=pr['source_display_cell_id'],classification=label,x_m=pr['x_m'],y_m=pr['y_m']))
  for i,s in enumerate(states):
   for d in range(2):
    cp=commonpairs[d];before=distance[i,0,d];after=distance[i,ai,d];paired=cp&np.isfinite(before);saving=before[paired]-after[paired];burden.append(dict(arm=arm,**s,direction=['outward','return'][d],common_direction_pairs=int(cp.sum()),lost_direction_pairs=int((cp&~np.isfinite(after)).sum()),restored_direction_pairs=int((cp&~np.isfinite(before)&np.isfinite(after)).sum()),baseline_reachable_pairs=int(paired.sum()),paired_distance_saving_median_m=float(np.median(saving))if len(saving)else None,paired_distance_saving_p90_m=float(np.quantile(saving,.9))if len(saving)else None))
 write(R/'motor_classification_summary.csv',classes);write(R/'motor_probe_classification.csv',pc);write(R/'motor_directional_benefits.csv',burden)
 # Sampled independent rerouting, plus source rule-preserving witnesses on actual newly restored paths.
 witness=0;nstate=len(router['state_physical_arc'])
 with(P/'restored_route_witnesses.jsonl').open('w')as output:
  for run in runs:
   case=run['case_id'];i=next(j for j,s in enumerate(states)if s['case_id']==case and s['frame_id']=='14400'and s['threshold_m']==.3 and s['unknown_branch']=='unknown_available');e=load(out/(case+'.npz'));closed=qualified&(e['lower'][5]>=.3)
   for ai in [1,2]:
    c=closed&~effective[ai];g=graph(router,c);check(case+'_'+ARMS[ai]+'_independent_reroute',same(route(router,g),distance[i,ai]));restored=commonpairs[0]&~np.isfinite(distance[i,0,0])&np.isfinite(distance[i,ai,0]);ds,pred=dijkstra(g.T.tocsr(),indices=router['target_destination'],return_predecessors=True,directed=True)
    for target,j in zip(*np.where(restored)):
     node=int(router['probe_source'][j]);stop=int(router['target_destination'][target]);nodes=[node];seen=set()
     while node!=stop:
      assert node not in seen;seen.add(node);node=int(pred[target,node]);assert node>=0;nodes.append(node)
     native=np.array([router['state_physical_arc'][n]for n in nodes if n<nstate],dtype='i4');rr=p['arc_to_road'][native];total=float(p['length_m'][native].sum());check(case+'_'+ARMS[ai]+'_witness_'+str(witness),not c[rr].any()and np.any(closed[rr]&effective[ai,rr])and abs(total-distance[i,ai,0,target,j])<1e-6);output.write(json.dumps(dict(case_id=case,arm=ARMS[ai],frame_id='14400',threshold_m=.3,branch='unknown_available',probe_id=probes[j]['source_display_cell_id'],target=int(target),original_directed_arc_indices=p['physical_source_arc_index'][native].tolist(),road_distance_m=total,protected_previously_wet_roads=np.unique(rr[closed[rr]&effective[ai,rr]]).tolist(),unknown_road_arcs=int(unknown[rr].sum()),not_safe_route_or_actual_traffic=True))+'\n');witness+=1
 check('sources_and_fixed_measures_still_unchanged',all(sha(ROOT/f)==h for f,h in json.loads((P/'protected_sources.json').read_text()).items())and all(sha(ROOT/f)==h for f,h in json.loads((MEASURE/'frozen_selection_binding.json').read_text()).items()));check('no_expansion_of_original_probe_support',eligible_out.sum()<=93 and eligible_two.sum()<=89);check('complete1248network_states',distance.shape==(416,3,2,4,209));check('all209points_partitioned',all(sum(r[k]for k in ['robust_modeled_loss','conditional_sensitivity','retained_access','insufficient_evidence'])==209 for r in classes))
 length=np.array([float(r['length_m'])for r in roads]);summary=dict(status='qualified_complete',additional_historical_parents=2,hydraulic_cases=8,network_evaluations=1248,unique_flood_graphs=len(cache),eligible_outward=int(eligible_out.sum()),eligible_two_direction=int(eligible_two.sum()),original_cohort_outward=93,additional_support_attrition=93-int(eligible_out.sum()),qualified_source_roads=int(qualified.sum()),newly_unknown_source_roads=int((p['qualified']&~qualified).sum()),fixed_length_m={arm:float(length[fixed[ai]].sum())for ai,arm in enumerate(ARMS)},effective_known_length_m={arm:float(length[effective[ai]].sum())for ai,arm in enumerate(ARMS)},fixed_elapsed4h_outward=[r for r in classes if r['frame_id']=='14400'and r['direction']=='outward'],restored_witnesses=witness,elapsed_s=time.monotonic()-start,people_counted=False,training=False,sealed_tests_opened=False,independent_observed_flood_accuracy=False)
 dump(R/'motor_summary.json',summary);dump(R/'motor_validation.json',dict(all_passed=True,checks=checks,count=len(checks),restored_route_witnesses=witness,original_turn_router_sha256=sha(MOTOR/'turn_router.npz')));dump(R/'motor_execution_state.json',dict(status='qualified_complete',network_states_with3arms=1248,elapsed_s=time.monotonic()-start));print('CROSS-EVENT TRANSPORT QUALIFIED',len(checks),flush=True)
if __name__=='__main__':main()
