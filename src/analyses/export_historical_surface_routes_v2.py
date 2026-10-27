"""Batch path witnesses at fixed19:00 under declared0.3m/unknown-open rule."""
from pathlib import Path
import os,sys,json
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_historical_surface_coverage_v2 import P,R,CASES,load,read,dump
from compare_historical_surface_coverage_v2 import Template

def main():
 w=load(P/'local_walking_graph.npz');a=load(P/'alternative_distances.npz');contexts=read(P/'road_context_index.csv');probes=read(P/'surface_probe_inventory.csv');t=Template(w['from_index'],w['to_index'],w['length_m'],len(w['node_ids']));records=[];checks=[];supplement=load(P/'ground_path_supplement.npz')
 for case in CASES:
  row=next(r for r in contexts if r['case_id']==case and r['frame_id']=='14400' and r['unknown_branch']=='optimistic' and float(r['road_threshold_m'])==.3);ci=int(row['road_context_index']);keep=~a['road_closed'][ci,w['arc_basis']];g=t.matrix(keep);dist,pred,sources=dijkstra(g.T.tocsr(),indices=w['exit_node_index'],directed=True,min_only=True,return_predecessors=True)
  arcmap={}
  for i in np.flatnonzero(keep):
   key=(int(w['from_index'][i]),int(w['to_index'][i]));prior=arcmap.get(key)
   if prior is None or w['length_m'][i]<w['length_m'][prior]:arcmap[key]=int(i)
  for j,p in enumerate(probes):
   start=int(w['probe_node_index'][j]);node=start;path=[];seen=set()
   if np.isfinite(dist[node]):
    while pred[node]>=0:
     assert node not in seen;seen.add(node);nextnode=int(pred[node]);arc=arcmap[node,nextnode];assert keep[arc];path.append(arc);node=nextnode
    assert node in set(w['exit_node_index']);length=float(w['length_m'][path].sum());assert abs(length-dist[start])<1e-6
   else:length=None
   assert (length is None)==(not np.isfinite(a['walking_exit_m'][ci,0,j]))
   records.append(dict(case_id=case,frame_id='14400',probe_id=p['probe_id'],ward=p['ward'],road_threshold_m=.3,unknown_branch='optimistic',reachable=length is not None,walking_distance_m=length,exit_native_node=str(w['node_ids'][node]) if length is not None else None,source_native_arc_indices=[int(w['source_arc_index'][k]) for k in path],unknown_exposure_segments_on_path=int(a['road_unknown'][w['arc_basis'][path]].sum()) if path else 0,supplemented_segments_on_path=int(supplement['qualified'][w['arc_basis'][path]].sum()) if path else 0,exit_is_analysis_boundary_not_safe_refuge=True))
 checks=[dict(check='all_reachable_witnesses_follow_original_directed_open_arcs',passed=True),dict(check='all_witness_costs_match_shortest_distances',passed=True),dict(check='all_witnesses_end_at_native_boundary_targets',passed=True),dict(check='unknown_exposure_on_paths_counted_not_hidden',passed=True)]
 with (P/'surface_exit_route_witnesses_19jst.jsonl').open('w') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 dump(R/'route_witness_validation.json',dict(all_passed=True,checks=checks,records=len(records),reachable_paths=sum(r['reachable'] for r in records),role='Batch route evidence for a declared branch only, not safe evacuation routes',new_hydraulics=False))
 print('ROUTE WITNESSES',len(records),sum(r['reachable'] for r in records),flush=True)
if __name__=='__main__':main()
