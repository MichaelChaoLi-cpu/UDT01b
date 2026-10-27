"""Numerical/software checks and a fixed19:00 native road-path witness export."""
from pathlib import Path
import os,sys,json,collections
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_motor_flood_access_v1 import P,R,N,CASES,read,write,dump,load,sha
from run_motor_flood_access_v1 import graph
from normal_transport_rules_v1 import restriction_allows

def main():
 checks=[]
 def check(name,v):
  checks.append(dict(check=name,passed=bool(v)))
  if not v:raise AssertionError(name)
 bind=json.loads((P/'source_bindings.json').read_text());check('existing_inputs_unchanged',all(sha(ROOT/k)==h for k,h in bind.items()));check('normal_router_reuses18_existing_distances',json.loads((R/'normal_route_reuse_validation.json').read_text())['all_passed'])
 p=load(P/'road_projection.npz');router=load(P/'turn_router.npz');a=load(P/'accessibility.npz');scenarios=read(P/'scenario_index.csv');probes=read(P/'probe_inventory.csv');z=load(P/'source_motor_graph.npz');evidence=load(P/'evidence_support.npz');normal=a['normal_road_distance_m'];dist=a['road_distance_m'];contract=json.loads((P/'analysis_contract.json').read_text());frames=contract['frame_ids'];qualified=p['qualified'];unknown=~qualified
 check('complete624_fixed_case_time_threshold_uncertainty_conditions',len(scenarios)==624 and dist.shape==(624,2,4,209)and normal.shape==(2,2,4,209))
 check('original_car_graph_unchanged',all(np.array_equal(z[k],load(N/'drive_conservative_graph.npz')[k])for k in ['node_ids','from_index','to_index','length_m','osm_way_id','inside_core']))
 nstate=len(router['state_physical_arc']);to=router['indices'];entered=to<nstate;expected=np.full(len(to),-1,'i4');expected[entered]=p['arc_to_road'][router['state_physical_arc'][to[entered]]]
 check('every_road_entry_including_origin_has_its_flood_gate',np.array_equal(expected,router['road_index']))
 rowsby=collections.defaultdict(list);masks={};ecache={};dryok=True;never_improve=True
 for i,row in enumerate(scenarios):
  case=row['case_id'];e=ecache.get(case)
  if e is None:ecache.clear();e=load(P/'exposures'/(case+'.npz'));ecache[case]=e
  ti=frames.index(row['frame_id']);branch=0 if row['unknown_branch']=='unknown_available'else 1;bound='lower'if branch==0 else'upper';wet=qualified&(e[bound][ti]>=float(row['threshold_m']));closed=wet|(unknown if branch else False);saved=np.unpackbits(a['closed_roads_packed'][i])[:len(closed)].astype(bool)
  check('closure_'+str(i),np.array_equal(closed,saved))
  base=normal[branch];after=dist[i];finite=np.isfinite(after);never_improve &=not np.any(finite&~np.isfinite(base))and np.all(after[finite]>=base[finite]-1e-6)
  if case in ['R0__L__T','R0__L__U']:dryok &=not wet.any()and np.array_equal(after,base)
  rowsby[case,row['frame_id']].append(i);masks[case,row['frame_id'],float(row['threshold_m']),branch]=closed
 check('closed_roads_do_not_create_shortcuts',never_improve);check('dry_low_controls_equal_matched_normal_network',dryok)
 nested=True
 for case in CASES:
  e=load(P/'exposures'/(case+'.npz'));check(case+'_bounded_depth_and_actual_clock',np.all(e['lower'][:,qualified]<=e['upper'][:,qualified])and max(abs(e['actual_times']-np.array(contract['times_s'])))<=.001)
  for f in frames:
   nested &=not np.any(masks[case,f,.3,0]&~masks[case,f,.1,0])and not np.any(masks[case,f,.3,1]&~masks[case,f,.1,1])and not np.any(masks[case,f,.1,0]&~masks[case,f,.1,1])and not np.any(masks[case,f,.3,0]&~masks[case,f,.3,1])
 check('depth_and_unknown_sensitivity_masks_nested',nested)
 classes=read(R/'probe_classification.csv');classok=True;pid={p['source_display_cell_id']:i for i,p in enumerate(probes)}
 for row in classes:
  ids=rowsby[row['case_id'],row['frame_id']];j=pid[row['probe_id']];available=np.isfinite(dist[ids]);out=available[:,0,:,j].any(axis=1);two=available[:,:,:,j].any(axis=2).all(axis=1)
  for field,values,ok in [('outward',out,evidence['eligible_outward'][j]),('two_direction',two,evidence['eligible_two_direction'][j])]:
   label='insufficient_evidence'if not ok else'robust_modeled_loss'if not values.any()else'retained_access'if values.all()else'conditional_sensitivity';classok &=label==row[field+'_classification']
 check('all_probe_classifications_use_matched_normal_support',classok)
 # One fixed regional witness set, not a pointwise investigation.
 si=next(i for i,row in enumerate(scenarios)if row['case_id']=='R1__H__U'and row['frame_id']=='14400'and float(row['threshold_m'])==.3 and row['unknown_branch']=='unknown_available');closed=masks['R1__H__U','14400',.3,0];g=graph(router,closed);rules=json.loads((N/'supported_motorcar_restrictions.json').read_text());bylast=collections.defaultdict(list);prefixes=set();maxhist=1
 for r in rules:
  ww=[r['prefix'][0]]if r['via_node']is not None else[r['prefix'][-1]]if r['kind']=='no'else r['prefix']
  for w in set(ww):bylast[w].append(r)
  maxhist=max(maxhist,len(r['prefix']))
  for j in range(1,len(r['prefix'])+1):prefixes.add(tuple(r['prefix'][:j]))
 keys=list(map(str,z['node_ids']));records=reachable=0;usage=collections.Counter();routecheck=True
 with(P/'car_route_witnesses_19jst.jsonl').open('w')as output:
  for direction in range(2):
   for target in range(4):
    source=int(router['target_destination'][target]if direction==0 else router['target_source'][target]);d,pred=dijkstra(g.T.tocsr()if direction==0 else g,indices=source,directed=True,return_predecessors=True)
    for j,probe in enumerate(probes):
     endpoint=int(router['probe_source'][j]if direction==0 else router['probe_destination'][j]);nodes=[endpoint];node=endpoint;seen=set();native=[];length=None
     if np.isfinite(d[endpoint]):
      while node!=source:
       assert node not in seen;seen.add(node);node=int(pred[node]);assert node>=0;nodes.append(node)
      if direction==1:nodes=nodes[::-1]
      native=[int(router['state_physical_arc'][q])for q in nodes if q<nstate];length=float(p['length_m'][native].sum());assert abs(length-d[endpoint])<1e-6 and abs(length-dist[si,direction,target,j])<1e-6
      assert not closed[p['arc_to_road'][native]].any();hist=();previous=None
      for arc in native:
       way=int(p['osm_way_id'][arc]);fromkey=keys[int(p['from_index'][arc])];tokey=keys[int(p['to_index'][arc])];nativeid=int(fromkey.split(':')[1]);assert restriction_allows(hist,way,nativeid,previous,tokey,bylast.get(hist[-1],[])if hist else [])
       if not hist or hist[-1]!=way:
        candidate=(hist+(way,))[-maxhist:];hist=(way,)
        for size in range(len(candidate),0,-1):
         if candidate[-size:]in prefixes:hist=candidate[-size:];break
       previous=fromkey
      for road in set(map(int,p['arc_to_road'][native])):usage[road]+=1
      reachable+=1
     else:assert not np.isfinite(dist[si,direction,target,j])
     record=dict(case_id='R1__H__U',frame_id='14400',threshold_m=.3,unknown_branch='unknown_available',direction=['outward','return'][direction],target=['North','East','South','West'][target],probe_id=probe['source_display_cell_id'],reachable=length is not None,road_distance_m=length,original_directed_arc_indices=[int(p['physical_source_arc_index'][arc])for arc in native],unknown_road_arcs=int(unknown[p['arc_to_road'][native]].sum())if native else 0,not_observed_traffic_or_safe_route=True);output.write(json.dumps(record,ensure_ascii=False)+'\n');records+=1
 check('fixed19jst_witnesses_follow_source_directions_turns_and_open_roads',routecheck and records==209*8)
 roadmeta=read(P/'road_inventory.csv');write(R/'shared_car_route_segments_19jst.csv',[dict(**roadmeta[i],selected_directional_paths_using_road=n,not_traffic_flow=True)for i,n in usage.most_common(100)])
 dump(R/'route_witness_summary.json',dict(records=records,reachable=reachable,case_id='R1__H__U',frame_id='14400',threshold_m=.3,branch='unknown_available',all_direction_and_turn_rules_passed=True))
 check('source_bindings_still_unchanged',all(sha(ROOT/k)==h for k,h in bind.items()))
 dump(R/'validation.json',dict(all_passed=True,checks=checks,new_hydraulics=False,people_counted=False));print('CAR ACCESS CHECKS',len(checks),'PASSED; WITNESSES',records,reachable)
if __name__=='__main__':main()
