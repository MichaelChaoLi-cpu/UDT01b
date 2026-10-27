"""Reuse saved turn rules in a finite-state road router; compare existing flood cases."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,time,collections
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse.csgraph import connected_components,dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_motor_flood_access_v1 import P,R,N,CASES,read,write,dump,load,sha
from normal_transport_rules_v1 import restriction_allows,shortest_paths

def build_router():
 start=time.monotonic();z=load(P/'source_motor_graph.npz');p=load(P/'road_projection.npz');keys=list(map(str,z['node_ids']));zero=z['osm_way_id']==0;n=len(keys)
 ng,component=connected_components(coo_matrix((np.ones(int(zero.sum())),(z['from_index'][zero],z['to_index'][zero])),shape=(n,n)).tocsr(),directed=False)
 u=p['from_index'];v=p['to_index'];ways=p['osm_way_id'];length=p['length_m'];cu=component[u];cv=component[v];E=len(u);order=np.argsort(cu,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(cu,minlength=ng))]
 rules=json.loads((N/'supported_motorcar_restrictions.json').read_text());bylast=collections.defaultdict(list);prefixes=set();maxhist=1
 for r in rules:
  indexways=[r['prefix'][0]]if r['via_node']is not None else[r['prefix'][-1]]if r['kind']=='no'else r['prefix']
  for w in set(indexways):bylast[w].append(r)
  maxhist=max(maxhist,len(r['prefix']))
  for j in range(1,len(r['prefix'])+1):prefixes.add(tuple(r['prefix'][:j]))
 needs=set(bylast)|{x[0]for x in prefixes};extra={};extra_arc=[];extra_hist=[];bu=[];bv=[];basis=[];state=0
 while state<E+len(extra_arc):
  arc=state if state<E else extra_arc[state-E];way=int(ways[arc]);hist=(way,)if state<E else extra_hist[state-E];out=order[offset[cv[arc]]:offset[cv[arc]+1]]
  if len(hist)==1 and way not in needs:
   bu.extend([state]*len(out));bv.extend(map(int,out));basis.extend(map(int,out))
  else:
   candidates=bylast.get(hist[-1],[]);native=int(keys[int(v[arc])].split(':')[1]);previous=keys[int(u[arc])]
   for q in out:
    q=int(q);newway=int(ways[q])
    if candidates and not restriction_allows(hist,newway,native,previous,keys[int(v[q])],candidates):continue
    nh=hist
    if newway!=hist[-1]:
     test=(hist+(newway,))[-maxhist:];nh=(newway,)
     for size in range(len(test),0,-1):
      if test[-size:]in prefixes:nh=test[-size:];break
    if nh==(newway,):target=q
    else:
     key=(q,nh);target=extra.get(key)
     if target is None:target=E+len(extra_arc);extra[key]=target;extra_arc.append(q);extra_hist.append(nh)
    bu.append(state);bv.append(target);basis.append(q)
  state+=1
  if state%200000==0:print('REUSE ROUTER',state,'states elapsed',round(time.monotonic()-start),flush=True)
 statearc=np.r_[np.arange(E),np.array(extra_arc,dtype='i4')];statecomp=cv[statearc];S=len(statearc);probes=read(P/'probe_inventory.csv');ki={k:i for i,k in enumerate(keys)};probecomp=np.array([component[ki[p['car_source_node']]]if p['candidate_available']=='True'else-1 for p in probes],dtype='i4')
 cross=z['inside_core'][u]!=z['inside_core'][v];outside=np.where(z['inside_core'][u[cross]],v[cross],u[cross]);exitcomp=np.unique(component[outside]);neednodes=set(map(int,outside));coord={}
 with(N/'surface_nodes.csv').open(encoding='utf-8-sig')as f:
  import csv
  for row in csv.DictReader(f):
   if row['node_id']in ki and ki[row['node_id']]in neednodes:coord[ki[row['node_id']]]=(float(row['x_m']),float(row['y_m']))
 exitgroups=collections.defaultdict(set);comp_xy={int(component[i]):coord[int(i)]for i in outside};center=(-7510.,-37380.)
 for c in exitcomp:
  x,y=comp_xy[int(c)];dx=x-center[0];dy=y-center[1];group=(0 if dy>=0 else 2)if abs(dy)>=abs(dx)else(1 if dx>=0 else 3);exitgroups[group].add(int(c))
 starts=np.arange(S,S+len(probes),dtype='i4');ends=starts+len(probes);targetS=np.arange(S+len(probes)*2,S+len(probes)*2+4,dtype='i4');targetD=targetS+4;nn=S+len(probes)*2+8
 dest=collections.defaultdict(list);source=[]
 for j,c in enumerate(probecomp):
  if c>=0:dest[int(c)].append(int(ends[j]));source.append((int(starts[j]),{int(c)}))
 for j in range(4):
  source.append((int(targetS[j]),exitgroups[j]))
  for c in exitgroups[j]:dest[c].append(int(targetD[j]))
 for si,c in enumerate(statecomp):
  for target in dest.get(int(c),[]):bu.append(si);bv.append(target);basis.append(-1)
 for origin,comps in source:
  for c in comps:
   arcs=order[offset[c]:offset[c+1]];bu.extend([origin]*len(arcs));bv.extend(map(int,arcs));basis.extend(map(int,arcs))
   for target in dest.get(c,[]):bu.append(origin);bv.append(target);basis.append(-1)
 bu=np.asarray(bu,'i4');bv=np.asarray(bv,'i4');basis=np.asarray(basis,'i4');sort=np.lexsort((bv,bu));bu=bu[sort];bv=bv[sort];basis=basis[sort];duplicate=(bu[1:]==bu[:-1])&(bv[1:]==bv[:-1]);assert np.all(basis[1:][duplicate]==basis[:-1][duplicate]);unique=np.r_[True,~duplicate];print('DEDUP IDENTICAL ROUTER LINKS',int(duplicate.sum()),flush=True);bu=bu[unique];bv=bv[unique];basis=basis[unique];indptr=np.r_[0,np.cumsum(np.bincount(bu,minlength=nn))].astype('i4');cost=np.zeros(len(basis));nonzero=basis>=0;cost[nonzero]=length[basis[nonzero]];road=np.full(len(basis),-1,'i4');road[nonzero]=p['arc_to_road'][basis[nonzero]]
 np.savez_compressed(P/'turn_router.npz',indices=bv,indptr=indptr,cost=cost,road_index=road,probe_source=starts,probe_destination=ends,target_source=targetS,target_destination=targetD,node_component=component,state_physical_arc=statearc,physical_original_arc=p['physical_source_arc_index'],state_arc_end_node=v[statearc],shape=np.array([nn,nn]),target_group_counts=np.array([len(exitgroups[i])for i in range(4)]))
 dump(R/'router_summary.json',dict(source_arcs=E,turn_memory_states=S,extra_via_way_states=len(extra_arc),supported_turn_rules=len(rules),graph_edges=len(cost),targets=['North','East','South','West'],native_exit_components_by_direction=[len(exitgroups[i])for i in range(4)],build_seconds=time.monotonic()-start,ordinary_node_adjacency_not_substituted=True));print('ROUTER READY',S,len(cost),flush=True)

def graph(router,closed):
 values=router['cost'].copy();ri=router['road_index'];sel=ri>=0;values[sel&closed[np.maximum(ri,0)]]=np.inf
 return csr_matrix((values,router['indices'],router['indptr']),shape=tuple(router['shape']))

def route(router,g):
 outward=dijkstra(g.T.tocsr(),indices=router['target_destination'],directed=True)[:,router['probe_source']];back=dijkstra(g,indices=router['target_source'],directed=True)[:,router['probe_destination']]
 return np.stack([outward,back])

def check_existing_routes(router):
 from scipy.sparse import vstack,hstack
 z=load(P/'source_motor_graph.npz');p=load(P/'road_projection.npz');keys=list(map(str,z['node_ids']));ix={k:i for i,k in enumerate(keys)};comp=router['node_component'];source_comp=comp[p['from_index']];state_end_comp=comp[router['state_arc_end_node']]
 rows=[x for x in read(ROOT/'data/exp/data-preprocessing/three_ward_normal_transport_network_v1/routing_probe_checks.csv')if x['mode']=='drive'and x['road_halo_km']=='10'];origins=sorted(set(x['origin_node']for x in rows));n=int(router['shape'][0]);base=graph(router,np.zeros(len(p['qualified']),bool));rr=[];cc=[];cost=[]
 for j,source in enumerate(origins):
  arcs=np.flatnonzero(source_comp==comp[ix[source]]);rr.extend([j]*len(arcs));cc.extend(map(int,arcs));cost.extend(p['length_m'][arcs])
 bottom=csr_matrix((cost,(rr,cc)),shape=(len(origins),n));full=hstack([vstack([base,bottom],format='csr'),csr_matrix((n+len(origins),len(origins)))],format='csr');dist=dijkstra(full,indices=np.arange(n,n+len(origins)),directed=True);comparison=[]
 for row in rows:
  j=origins.index(row['origin_node']);target=comp[ix[row['target_node']]];states=np.flatnonzero(state_end_comp==target);value=float(np.min(dist[j,states]));expected=float(row['distance_m']);comparison.append(dict(origin=row['origin_node'],target=row['target_node'],expected_m=expected,reused_router_m=value,difference_m=value-expected))
 assert len(comparison)==18 and all(abs(x['difference_m'])<1e-6 for x in comparison),'Existing motorcar routing changed'
 write(R/'normal_route_reuse_check.csv',comparison);dump(R/'normal_route_reuse_validation.json',dict(all_passed=True,existing_routes_reproduced=18,no_new_pointwise_investigation=True))

def main():
 start=time.monotonic();assert json.loads((R/'preparation_validation.json').read_text())['all_passed']
 if not(P/'turn_router.npz').exists():build_router()
 router=load(P/'turn_router.npz');check_existing_routes(router);p=load(P/'road_projection.npz');contract=json.loads((P/'analysis_contract.json').read_text());frames=contract['frame_ids'];unknown=~p['qualified'];baseline=[]
 for branch in range(2):baseline.append(route(router,graph(router,unknown if branch else np.zeros(len(unknown),bool))))
 baseline=np.stack(baseline);distances=[];scenarios=[];closemasks=[];cache={};summary=[]
 for case in CASES:
  e=load(P/'exposures'/(case+'.npz'))
  for ti,frame in enumerate(frames):
   for threshold in contract['depth_thresholds_m']:
    for branch in range(2):
     bound='lower'if branch==0 else'upper';wet=p['qualified']&(e[bound][ti]>=threshold);closed=wet|(unknown if branch else False);sig=np.packbits(closed).tobytes()
     if sig not in cache:cache[sig]=route(router,graph(router,closed))
     i=len(scenarios);distances.append(cache[sig]);closemasks.append(np.packbits(closed));scenarios.append(dict(scenario_index=i,case_id=case,frame_id=frame,simultaneous=frame!='peak_envelope',threshold_m=threshold,unknown_branch=['unknown_available','unknown_withheld'][branch],closed_known_wet_roads=int(wet.sum()),withheld_unknown_roads=int((closed&unknown).sum())))
   dump(R/'execution_state.json',dict(status='running',case_id=case,frame_id=frame,network_cases_done=len(scenarios),network_cases_total=624,elapsed_s=time.monotonic()-start,new_hydraulics=False))
  print('CAR ACCESS',case,len(scenarios),'of624 elapsed',round(time.monotonic()-start),flush=True)
 distances=np.stack(distances);np.savez_compressed(P/'accessibility.npz',road_distance_m=distances,normal_road_distance_m=baseline,closed_roads_packed=np.stack(closemasks));write(P/'scenario_index.csv',scenarios)
 probes=read(P/'probe_inventory.csv');classes=[];aggregates=[];odrows=[];groups=collections.defaultdict(list)
 for i,r in enumerate(scenarios):groups[r['case_id'],r['frame_id']].append(i)
 base_direction=np.isfinite(baseline);normal_any=base_direction.any(axis=2);normal_two=(base_direction[:,:,0:4,:].any(axis=2)).all(axis=1);eligible_out=normal_any[:,0].all(axis=0);eligible_two=normal_two.all(axis=0)
 # Qualification also requires a known incident modeled road, not merely a road retained as unknown.
 z=load(P/'source_motor_graph.npz');physical=p['physical_source_arc_index'];incident=np.zeros(len(z['node_ids']),bool);knownarc=p['qualified'][p['arc_to_road']];incident[p['from_index'][knownarc]]=True;incident[p['to_index'][knownarc]]=True;ki={str(k):i for i,k in enumerate(z['node_ids'])};incident_probe=np.array([incident[ki[x['car_source_node']]]if x['candidate_available']=='True'else False for x in probes]);eligible_out &=incident_probe;eligible_two &=incident_probe
 for(case,frame),ids in groups.items():
  assert len(ids)==4;d=distances[ids];available=np.isfinite(d);out=available[:,0].any(axis=1);two=available.any(axis=2).all(axis=1)
  for j,probe in enumerate(probes):
   def label(v,ok):return'insufficient_evidence'if not ok else'robust_modeled_loss'if not v.any()else'retained_access'if v.all()else'conditional_sensitivity'
   classes.append(dict(case_id=case,frame_id=frame,simultaneous=frame!='peak_envelope',probe_id=probe['source_display_cell_id'],source_cell_context_ward=probe['source_cell_context_ward'],x_m=probe['x_m'],y_m=probe['y_m'],candidate_available=probe['candidate_available'],outward_classification=label(out[:,j],eligible_out[j]),two_direction_classification=label(two[:,j],eligible_two[j]),outward_branches=int(out[:,j].sum()),two_direction_branches=int(two[:,j].sum()),branches=4))
  rows=classes[-len(probes):]
  for direction in ['outward','two_direction']:
   counts=collections.Counter(x[direction+'_classification']for x in rows);aggregates.append(dict(case_id=case,frame_id=frame,direction=direction,simultaneous=frame!='peak_envelope',all_source_cells=len(probes),**{k:counts[k]for k in ['robust_modeled_loss','conditional_sensitivity','insufficient_evidence','retained_access']}))
 for i,r in enumerate(scenarios):
  branch=0 if r['unknown_branch']=='unknown_available'else 1;before=baseline[branch];after=distances[i]
  for direction in range(2):
   for target in range(4):
    norm=np.isfinite(before[direction,target]);afterok=np.isfinite(after[direction,target]);paired=norm&afterok;extra=after[direction,target,paired]-before[direction,target,paired];positive=paired&(before[direction,target]>0);rat=after[direction,target,positive]/before[direction,target,positive]
    odrows.append(dict(**r,direction=['outward','return'][direction],target=['North','East','South','West'][target],normal_reachable_probes=int(norm.sum()),storm_reachable_probes=int((norm&afterok).sum()),new_lost_probes=int((norm&~afterok).sum()),median_extra_distance_m=float(np.median(extra))if len(extra)else None,p90_extra_distance_m=float(np.quantile(extra,.9))if len(extra)else None,median_detour_ratio=float(np.median(rat))if len(rat)else None))
 write(R/'probe_classification.csv',classes);write(R/'classification_summary.csv',aggregates);write(R/'directional_access_and_detours.csv',odrows);np.savez_compressed(P/'evidence_support.npz',eligible_outward=eligible_out,eligible_two_direction=eligible_two,incident_ground_support=incident_probe)
 dump(R/'analysis_summary.json',dict(network_evaluations=len(scenarios),unique_flood_graphs=len(cache),physical_cases=12,synchronous_times=12,separate_peak=True,source_cells=209,native_car_candidates=sum(x['candidate_available']=='True'for x in probes),evidence_outward=int(eligible_out.sum()),evidence_two_direction=int(eligible_two.sum()),seconds=time.monotonic()-start,new_hydraulics=False,people_counted=False,traffic_flow_or_queue_simulated=False));dump(R/'execution_state.json',dict(status='network_evaluations_complete',network_cases_done=len(scenarios),network_cases_total=624,elapsed_s=time.monotonic()-start,new_hydraulics=False));print('CAR NETWORK COMPLETE',len(scenarios),flush=True)
if __name__=='__main__':main()
