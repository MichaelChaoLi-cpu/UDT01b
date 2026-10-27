"""Whole-cohort native walking-route witnesses at the fixed19:00 snapshot.
Both directions, both projection supports and both unknown branches are exported.
Targets are analysis-boundary crossings, not certified safe destinations.
"""
from pathlib import Path
import os,sys,json,collections,time
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_formal_transport_access_v1 import P,R,OLD,CASES,PROJECTIONS,UNCERTAINTIES,read,write,dump,load
from compare_historical_surface_coverage_v2 import Template

def main():
 start=time.monotonic();w=load(OLD/'local_walking_graph.npz');s=load(P/'projection_support.npz');a=load(P/'alternative_distances.npz');meta=read(OLD/'road_segment_inventory.csv');probes=read(OLD/'surface_probe_inventory.csv');contexts=read(P/'road_context_index.csv');template=Template(w['from_index'],w['to_index'],w['length_m'],len(w['node_ids']));rows=[];usage=[];targetset=set(map(int,w['exit_node_index']));records=reachable=0
 with(P/'surface_route_witnesses_19jst.jsonl').open('w')as out:
  for case in CASES:
   for pi,projection in enumerate(PROJECTIONS):
    for branch in UNCERTAINTIES:
     row=next(x for x in contexts if x['case_id']==case and x['frame_id']=='14400'and x['projection']==projection and x['unknown_branch']==branch and float(x['road_threshold_m'])==.3);ci=int(row['road_context_index']);keep=~a['road_closed'][ci,w['arc_basis']];g=template.matrix(keep);amap={}
     for arc in np.flatnonzero(keep):
      key=(int(w['from_index'][arc]),int(w['to_index'][arc]));prior=amap.get(key)
      if prior is None or w['length_m'][arc]<w['length_m'][prior]:amap[key]=int(arc)
     for direction in [0,1]:
      dist,pred,sources=dijkstra(g.T.tocsr()if direction==0 else g,indices=w['exit_node_index'],directed=True,min_only=True,return_predecessors=True);freq=collections.Counter()
      for j,p in enumerate(probes):
       node=int(w['probe_node_index'][j]);walk=[];seen=set();length=None;endpoint=None
       if np.isfinite(dist[node]):
        while pred[node]>=0:
         assert node not in seen;seen.add(node);nextnode=int(pred[node]);arc=amap[node,nextnode]if direction==0 else amap[nextnode,node];assert keep[arc];walk.append(arc);node=nextnode
        assert node in targetset;length=float(w['length_m'][walk].sum());assert abs(length-dist[int(w['probe_node_index'][j])])<1e-6;assert abs(length-a['walking_exit_m'][ci,direction,j])<1e-6;endpoint=str(w['node_ids'][node]);reachable+=1
       else:assert not np.isfinite(a['walking_exit_m'][ci,direction,j])
       # Distinct origin use, not weighted flow or proof of a unique bottleneck.
       for base in set(map(int,w['arc_basis'][walk])):freq[base]+=1
       bases=w['arc_basis'][walk];record=dict(case_id=case,frame_id='14400',projection=projection,unknown_branch=branch,road_threshold_m=.3,direction='outward'if direction==0 else'return',probe_id=p['probe_id'],reachable=length is not None,walking_distance_m=length,exit_source_native_node=endpoint,source_native_arc_indices=[int(w['source_arc_index'][i])for i in(walk if direction==0 else walk[::-1])],unknown_exposure_arc_count=int(s['unknown'][pi,bases].sum())if len(bases)else 0,supplement_arc_count=int(s['supplement_qualified'][bases].sum())if len(bases)else 0,not_safe_evacuation_route=True)
       out.write(json.dumps(record,ensure_ascii=False)+'\n');records+=1
      for base,n in freq.most_common(20):usage.append(dict(case_id=case,frame_id='14400',projection=projection,unknown_branch=branch,direction='outward'if direction==0 else'return',source_edge_id=meta[base]['source_edge_id'],osm_way_id=meta[base]['osm_way_id'],source_length_m=meta[base]['source_length_m'],origins_using_selected_shortest_path=n,unknown_exposure=bool(s['unknown'][pi,base]),path_supplement=bool(s['supplement_qualified'][base]),not_unique_bottleneck_or_flow=True))
  assert records==len(CASES)*2*2*2*len(probes)
 write(R/'shared_shortest_path_segments_19jst.csv',usage)
 checks=[dict(check='all_witnesses_follow_kept_directed_native_arcs',passed=True),dict(check='all_lengths_match_saved_shortest_distances',passed=True),dict(check='all_targets_are_declared_native_boundary_exits',passed=True),dict(check='return_witness_arc_order_is_source_to_probe',passed=True),dict(check='unknown_and_supplemented_arcs_explicitly_counted',passed=True)]
 dump(R/'route_witness_validation.json',dict(all_passed=True,checks=checks,records=records,reachable_routes=reachable,seconds=time.monotonic()-start,scope='All209 probes in all12 cases at19:00 under0.3m and both support/unknown policies; surface route evidence only.',new_hydraulics=False));print('ROUTE WITNESSES',records,reachable,flush=True)
if __name__=='__main__':main()
