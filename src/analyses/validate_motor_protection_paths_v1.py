"""Native directed/turn-aware witnesses for restored common source-sector connections."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,collections
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from compare_motor_road_protection_v1 import P,R,SOURCE,N,setup,common_pairs,graph,load,read,dump
from normal_transport_rules_v1 import restriction_allows

def main():
 router,p,roads,probes,old,support,scenarios=setup();a=load(P/'accessibility.npz')['road_distance_m'];mask=load(P/'protected_roads.npz')['protected'];z=load(SOURCE/'source_motor_graph.npz');keys=list(map(str,z['node_ids']));statecount=len(router['state_physical_arc']);rules=json.loads((N/'supported_motorcar_restrictions.json').read_text());bylast=collections.defaultdict(list);prefixes=set();maxhist=1
 for r in rules:
  ww=[r['prefix'][0]]if r['via_node']is not None else[r['prefix'][-1]]if r['kind']=='no'else r['prefix']
  for w in set(ww):bylast[w].append(r)
  maxhist=max(maxhist,len(r['prefix']))
  for j in range(1,len(r['prefix'])+1):prefixes.add(tuple(r['prefix'][:j]))
 records=0;contexts=[]
 with(P/'restored_route_witnesses_19jst.jsonl').open('w')as out:
  for ai in [1,2]:
   for i,s in enumerate(scenarios):
    if s['case_id']not in ['R1__L__U','R1__H__U','R125__H__U']or s['frame_id']!='14400'or float(s['threshold_m'])!=.3:continue
    closed0=np.unpackbits(old['closed_roads_packed'][i])[:len(roads)].astype(bool);closed=closed0&~mask[ai];g=graph(router,closed);contexts.append(dict(arm=ai,scenario_index=i))
    for direction in range(2):
     common=common_pairs(old,support,direction);restored=common&~np.isfinite(a[0,i,direction])&np.isfinite(a[ai,i,direction]);expected=int(restored.sum());count=0
     if not expected:continue
     starts=router['target_destination']if direction==0 else router['target_source'];d,pred=dijkstra(g.T.tocsr()if direction==0 else g,indices=starts,return_predecessors=True,directed=True)
     for target,j in zip(*np.where(restored)):
      endpoint=int(router['probe_source'][j]if direction==0 else router['probe_destination'][j]);stop=int(starts[target]);nodes=[endpoint];node=endpoint;seen=set()
      while node!=stop:
       assert node not in seen;seen.add(node);node=int(pred[target,node]);assert node>=0;nodes.append(node)
      if direction==1:nodes=nodes[::-1]
      native=[int(router['state_physical_arc'][q])for q in nodes if q<statecount];rr=p['arc_to_road'][native];dist=float(p['length_m'][native].sum());assert abs(dist-a[ai,i,direction,target,j])<1e-6;assert not closed[rr].any();protected_wet=np.unique(rr[closed0[rr]&mask[ai,rr]]);assert len(protected_wet)>0
      hist=();previous=None
      for arc in native:
       way=int(p['osm_way_id'][arc]);fk=keys[int(p['from_index'][arc])];tk=keys[int(p['to_index'][arc])];assert restriction_allows(hist,way,int(fk.split(':')[1]),previous,tk,bylast.get(hist[-1],[])if hist else [])
       if not hist or hist[-1]!=way:
        test=(hist+(way,))[-maxhist:];hist=(way,)
        for size in range(len(test),0,-1):
         if test[-size:]in prefixes:hist=test[-size:];break
       previous=fk
      record=dict(arm=['none','corridor','distributed'][ai],scenario_index=i,case_id=s['case_id'],unknown_branch=s['unknown_branch'],frame_id='14400',threshold_m=.3,direction=['outward','return'][direction],target=['North','East','South','West'][target],probe_id=probes[j]['source_display_cell_id'],road_distance_m=dist,original_directed_arc_indices=p['physical_source_arc_index'][native].tolist(),protected_wet_road_indices=protected_wet.tolist(),unknown_road_arcs=int((~p['qualified'][rr]).sum()),assumed_passability_not_safe_route=True);out.write(json.dumps(record,ensure_ascii=False)+'\n');records+=1;count+=1
     assert count==expected
 dump(R/'path_validation.json',dict(all_passed=True,restored_direction_witnesses=records,case_branch_arm_contexts=len(contexts),checked_original_directions_turns_open_roads_and_distance=True,restored_paths_use_protected_previously_wet_roads=True,unknown_roads_not_promoted=True));print('RESTORED PATH WITNESSES',records,flush=True)
if __name__=='__main__':main()
