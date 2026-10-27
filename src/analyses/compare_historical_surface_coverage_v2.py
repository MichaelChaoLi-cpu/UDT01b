"""Synchronized walking alternatives and conditional rail service graphs."""
from pathlib import Path
import os,sys,json,collections,hashlib,time
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_historical_surface_coverage_v2 import P,R,N,T,H,CASES,TIMES,read,write,dump,load,sha
from compare_transport_shutdown_scenarios_v1 import closure_mask
POLICIES=['original_identity','source_normalized_identity'];KINDS=['entrance','station','section'];THRESHOLDS=[.1,.3];UNCERTAINTIES=['optimistic','pessimistic']
class Template:
 def __init__(self,u,v,weight,n):
  self.n=n;self.order=np.lexsort((v,u));u=u[self.order];v=v[self.order];self.weight=weight[self.order];self.starts=np.r_[0,np.flatnonzero((u[1:]!=u[:-1])|(v[1:]!=v[:-1]))+1];self.u=u[self.starts];self.v=v[self.starts]
 def matrix(self,keep):
  weight=np.minimum.reduceat(np.where(keep[self.order],self.weight,np.inf),self.starts);ok=np.isfinite(weight)
  return csr_matrix((weight[ok],(self.u[ok],self.v[ok])),shape=(self.n,self.n))
def graph_parts():
 w=load(P/'local_walking_graph.npz');old=load(T/'hypothetical_network.npz');edges=json.loads((T/'hypothetical_network_edges.json').read_text());keys=list(map(str,old['node_ids']));wi={k:i for i,k in enumerate(w['node_ids'])};ix={k:i for i,k in enumerate(keys)};offset=len(wi)
 # Old bounded synthetic walk legs are replaced by the actual street graph.
 railkeep=np.array([e['edge_kind'] not in ['initial_walk','final_walk'] for e in edges]);ri=np.flatnonzero(railkeep);ru=old['from_index'][ri]+offset;rv=old['to_index'][ri]+offset
 cu=[];cv=[];cp=[]
 for p in read(T/'portal_inventory.csv'):
  for k in json.loads(p['walk_node_ids']):
   if k not in wi:continue
   cu.extend([wi[k],offset+ix['P:'+p['portal_id']]]);cv.extend([offset+ix['P:'+p['portal_id']],wi[k]]);cp.extend([p['portal_id']]*2)
 u=np.r_[w['from_index'],ru,cu].astype('i4');v=np.r_[w['to_index'],rv,cv].astype('i4');weight=np.r_[w['length_m'],np.zeros(len(ru)+len(cu))]
 anchors=list(map(str,load(T/'paired_accessibility_matrices.npz')['anchor_station_ids']));ao=np.array([offset+ix['O:station:'+s] for s in anchors]);ad=np.array([offset+ix['D:station:'+s] for s in anchors])
 return w,edges,ri,u,v,weight,ao,ad,cp

def main():
 started=time.monotonic();assert json.loads((R/'preparation_validation.json').read_text())['all_passed'];e=load(P/'surface_exposure.npz');w,edges,ri,u,v,weight,ao,ad,cp=graph_parts();metadata=read(P/'road_segment_inventory.csv');supplement=load(P/'ground_path_supplement.npz');probes=read(P/'surface_probe_inventory.csv');nw=len(w['node_ids']);ne=len(w['from_index']);nr=len(ri);template=Template(u,v,weight,nw+len(load(T/'hypothetical_network.npz')['node_ids']));walktemplate=Template(w['from_index'],w['to_index'],w['length_m'],nw)
 # All normal branches share the original declared public-through street graph.
 normalwalk=walktemplate.matrix(np.ones(ne,bool));normalmixed=template.matrix(np.ones(len(u),bool));pnode=w['probe_node_index'];exits=w['exit_node_index']
 def distances(g,targets):return np.stack([dijkstra(g.T.tocsr(),indices=targets,directed=True,min_only=True),dijkstra(g,indices=targets,directed=True,min_only=True)])[:,pnode]
 # Rail outward uses anchor D sinks, return uses anchor O sources.
 def raildist(g):
  assert np.asarray(ad).shape==(6,) and np.asarray(ao).shape==(6,)
  return np.stack([dijkstra(g.T.tocsr(),indices=ad,directed=True,min_only=True),dijkstra(g,indices=ao,directed=True,min_only=True)])[:,pnode]
 
 links={p:collections.defaultdict(set) for p in POLICIES}
 for r in read(ROOT/'data/processed/three_ward_station_surface_risk_v1/station_entrance_identity_links.csv'):
  for p in POLICIES:links[p][r['osm_node_id']].add(r['station_id'])
 for r in read(T/'identity_attribution_sensitivity_links.csv'):links[POLICIES[1]][r['osm_node_id']].add(r['station_id'])
 portalids={p['portal_id'] for p in read(T/'portal_inventory.csv')};off=e['entrance_offsets'];ind=e['entrance_cell_index'];ga=e['ground_area'];gknown=e['ground_known'];quality=[];admitted=[]
 for i in range(len(off)-1):
  cells=ind[off[i]:off[i+1]];total=float(ga[cells].sum());admitted_area=float(ga[cells][gknown[cells]].sum());quality.append(total>0 and admitted_area>=100 and admitted_area/total>=.8);admitted.append(admitted_area)
 quality=np.array(quality);starts=w['edge_sample_offsets'];inds=w['edge_sample_road_index'];known=e['road_known'];grade=np.array([r['known_surface_grade']=='True' and r['ambiguous_parallel_geometry']=='False' for r in metadata]);unknown=np.array([not grade[i] or r['has_out_of_grid_sample']=='True' or starts[i+1]==starts[i] or not known[inds[starts[i]:starts[i+1]]].all() for i,r in enumerate(metadata)])
 unknown &= ~supplement['qualified']
 incident_known=np.zeros(nw,bool)
 for a,b,base in zip(w['from_index'],w['to_index'],w['arc_basis']):
  if not unknown[base]:incident_known[a]=True;incident_known[b]=True
 local_support=incident_known[pnode]
 # Missing-exposure links must be treated identically in each paired normal baseline.
 base_exit=[];base_rail=[]
 for uncertainty in UNCERTAINTIES:
  roadkeep=~unknown[w['arc_basis']] if uncertainty=='pessimistic' else np.ones(ne,bool)
  base_exit.append(distances(walktemplate.matrix(roadkeep),exits));base_rail.append(raildist(template.matrix(np.r_[roadkeep,np.ones(nr+len(cp),bool)])))
 base_exit=np.stack(base_exit);base_rail=np.stack(base_rail)
 baseline_possible=np.isfinite(base_exit[:,0])|np.isfinite(base_rail[:,0]);evidence_ok=baseline_possible.all(axis=0)&local_support
 write(R/'normal_baseline_and_support.csv',[dict(probe_id=p['probe_id'],ward=p['ward'],x_m=p['x_m'],y_m=p['y_m'],unknown_branch=uncertainty,normal_surface_exit_out_m=float(base_exit[b,0,i]) if np.isfinite(base_exit[b,0,i]) else None,normal_surface_exit_return_m=float(base_exit[b,1,i]) if np.isfinite(base_exit[b,1,i]) else None,normal_rail_min_walking_out_m=float(base_rail[b,0,i]) if np.isfinite(base_rail[b,0,i]) else None,normal_rail_min_walking_return_m=float(base_rail[b,1,i]) if np.isfinite(base_rail[b,1,i]) else None,baseline_has_modeled_alternative=bool(baseline_possible[b,i]),has_incident_fully_supported_ground_segment=bool(local_support[i]),classification_evidence_eligible=bool(evidence_ok[i]),population_weighting=False) for b,uncertainty in enumerate(UNCERTAINTIES) for i,p in enumerate(probes)])
 contexts=[];context_masks=[];scenario=[];outdist=[];walkdist=[];selections=[];cache={};wcache={};unknown_source=unknown.copy();partial_known=grade.copy();positive=np.zeros(len(metadata),bool)
 for c,case in enumerate(CASES):
  for t,frame in enumerate(e['frame_ids']):
   actual=None if str(frame)=='peak_envelope' else float(e['actual_times'][c,t]);envelope=str(frame)=='peak_envelope'
   for th in THRESHOLDS:
    for uncertainty in UNCERTAINTIES:
     bound='lower' if uncertainty=='optimistic' else 'upper';cellval=e['road_'+bound][c,t];wet=np.zeros(len(metadata),bool)
     for j in range(len(metadata)):
      ids=inds[starts[j]:starts[j+1]]
      wet[j]=bool(grade[j] and np.any(known[ids]&(cellval[ids]>=th)))
     wet |= supplement['qualified']&(supplement[bound][c,t]>=th)
     closed=wet|(unknown if uncertainty=='pessimistic' else False);roadkeep=~closed[w['arc_basis']];signature=hashlib.sha256(roadkeep.tobytes()).hexdigest();ci=len(contexts)
     if signature not in wcache:wcache[signature]=distances(walktemplate.matrix(roadkeep),exits)
     wd=wcache[signature];walkdist.append(wd);context_masks.append(closed)
     contexts.append(dict(road_context_index=ci,case_id=case,frame_id=str(frame),actual_time_s=actual,simultaneous=not envelope,road_threshold_m=th,unknown_branch=uncertainty,closed_segments=int(closed.sum()),closed_known_wet_segments=int(wet.sum()),withheld_unknown_segments=int((unknown&closed).sum()),closed_native_arcs=int((~roadkeep).sum())))
     for identity in POLICIES:
      portals=set();targets=set();unassigned=0;trigger_points=0
      gd=e['ground_'+bound][c,t]
      for j,osm in enumerate(e['entrance_ids']):
       cells=ind[off[j]:off[j+1]];area=float(ga[cells][gknown[cells]&(gd[cells]>=.3)].sum())
       if not quality[j] or area<100:continue
       trigger_points+=1;ids=links[identity][str(osm)];ep='osm:'+str(osm)
       if ids and ep in portalids:portals.add(ep);targets.update(ids)
       else:unassigned+=1
      selections.append(dict(case_id=case,frame_id=str(frame),unknown_branch=uncertainty,identity_policy=identity,trigger_points=trigger_points,selected_portals=len(portals),target_station_units=len(targets),unassigned_trigger_points=unassigned,selected_portal_ids=';'.join(sorted(portals)),target_station_ids=';'.join(sorted(targets)),shutdown_is_assumed=True)) if th==THRESHOLDS[0] else None
      for kind in KINDS:
       railclosed=closure_mask(edges,kind,portals,targets)[ri];keep=np.r_[roadkeep,~railclosed,np.ones(len(cp),bool)];sig=hashlib.sha256(keep.tobytes()).hexdigest()
       if sig not in cache:cache[sig]=raildist(template.matrix(keep))
       rd=cache[sig];idx=len(scenario);outdist.append(rd)
       scenario.append(dict(scenario_index=idx,road_context_index=ci,case_id=case,frame_id=str(frame),actual_time_s=actual,simultaneous=not envelope,road_threshold_m=th,unknown_branch=uncertainty,identity_policy=identity,shutdown_type=kind,selected_portals=len(portals),target_station_units=len(targets),removed_rail_arcs=int(railclosed.sum())))
   print('FRAME',case,str(frame),'evaluations',len(scenario),'unique mixed',len(cache),flush=True)
 write(P/'road_context_index.csv',contexts);write(P/'scenario_index.csv',scenario);write(P/'station_closure_selection.csv',selections)
 wd=np.stack(walkdist);rd=np.stack(outdist);np.savez_compressed(P/'alternative_distances.npz',walking_exit_m=wd,mixed_min_walking_m=rd,baseline_walking_exit_m=base_exit,baseline_mixed_min_walking_m=base_rail,baseline_possible=baseline_possible,local_projection_supported=local_support,evidence_eligible=evidence_ok,road_unknown=unknown_source,road_closed=np.stack(context_masks),probe_ids=np.array([p['probe_id'] for p in probes]))
 # Classification is per simultaneous state and shutdown kind across all8 branches.
 classrows=[];agg=[]
 for case in CASES:
  for frame in e['frame_ids']:
   for kind in KINDS:
    ix=np.array([i for i,r in enumerate(scenario) if r['case_id']==case and r['frame_id']==str(frame) and r['shutdown_type']==kind]);assert len(ix)==8
    ci=np.array([scenario[i]['road_context_index'] for i in ix]);reachable=np.isfinite(wd[ci,0])|np.isfinite(rd[ix,0]);within=(wd[ci,0]<=2000)|(rd[ix,0]<=2000)
    for j,p in enumerate(probes):
     label='insufficient_evidence' if not evidence_ok[j] else 'robust_modeled_disconnection' if not reachable[:,j].any() else 'stable_modeled_alternative' if reachable[:,j].all() else 'conditional_sensitivity'
     classrows.append(dict(case_id=case,frame_id=str(frame),shutdown_type=kind,probe_id=p['probe_id'],ward=p['ward'],x_m=p['x_m'],y_m=p['y_m'],classification=label,branches_with_outward_alternative=int(reachable[:,j].sum()),branches_with_alternative_within2km=int(within[:,j].sum()),evaluated_branches=8,simultaneous=str(frame)!='peak_envelope',not_population_or_whole_town=True))
    for ward in ['all','13101','13102','13103','boundary_or_unassigned']:
     rr=[r for r in classrows if r['case_id']==case and r['frame_id']==str(frame) and r['shutdown_type']==kind and (ward=='all' or r['ward']==ward)];cnt=collections.Counter(r['classification'] for r in rr)
     agg.append(dict(case_id=case,frame_id=str(frame),shutdown_type=kind,ward=ward,origin_probes=len(rr),**{k:cnt[k] for k in ['robust_modeled_disconnection','conditional_sensitivity','insufficient_evidence','stable_modeled_alternative']},simultaneous=str(frame)!='peak_envelope'))
 write(R/'probe_classification.csv',classrows);write(R/'classification_summary.csv',agg)
 # Branch-resolved path and distance summaries. Walking distance is not evacuation time.
 stats=[]
 for i,r in enumerate(scenario):
  ci=r['road_context_index'];b=UNCERTAINTIES.index(r['unknown_branch']);alternative=np.minimum(wd[ci,0],rd[i,0]);normal=np.minimum(base_exit[b,0],base_rail[b,0]);paired=baseline_possible[b]&np.isfinite(alternative);detour=alternative[paired]-normal[paired];finite=alternative[np.isfinite(alternative)]
  stats.append(dict(**r,probes=len(probes),baseline_connected=int(baseline_possible[b].sum()),surface_exit_outward_reachable=int(np.isfinite(wd[ci,0]).sum()),surface_exit_return_reachable=int(np.isfinite(wd[ci,1]).sum()),mixed_anchor_outward_reachable=int(np.isfinite(rd[i,0]).sum()),mixed_anchor_return_reachable=int(np.isfinite(rd[i,1]).sum()),has_either_outward_alternative=int(np.isfinite(alternative).sum()),new_no_outward_alternative=int((baseline_possible[b]&~np.isfinite(alternative)).sum()),alternative_within2km=int((alternative<=2000).sum()),median_min_walking_m=float(np.median(finite)) if len(finite) else None,p90_min_walking_m=float(np.quantile(finite,.9)) if len(finite) else None,paired_median_added_walking_m=float(np.median(detour)) if len(detour) else None,paired_p90_added_walking_m=float(np.quantile(detour,.9)) if len(detour) else None))
 write(R/'scenario_accessibility_summary.csv',stats)
 # Compare peak rail-only candidates with actual modeled surface alternatives, retaining original cohort.
 old=read(ROOT/'data/exp/estimation/three_ward_historical_transport_islands_v1/probe_accessibility.csv');pi={p['probe_id']:j for j,p in enumerate(probes)};comparison=[]
 for case in CASES:
  for identity in POLICIES:
   for kind in KINDS:
    ids=[pi[r['probe_id']] for r in old if r['case_id']==case and r['identity_policy']==identity and r['shutdown_type']==kind and r['depth_bound']=='lower' and r['network_branch']=='provider_transfers_assumed_bidir' and r['cohort']=='surface_grid' and r['new_all_roundtrip_loss']=='True']
    r=next(s for s in scenario if s['case_id']==case and s['frame_id']=='peak_envelope' and s['identity_policy']==identity and s['shutdown_type']==kind and s['road_threshold_m']==.3 and s['unknown_branch']=='optimistic');ci=r['road_context_index'];i=r['scenario_index'];x=wd[ci,0,ids]
    comparison.append(dict(case_id=case,identity_policy=identity,shutdown_type=kind,old_peak_rail_only_candidate_points=len(ids),surface_exit_reachable=int(np.isfinite(x).sum()),surface_exit_within2km=int((x<=2000).sum()),mixed_or_surface_reachable=int((np.isfinite(x)|np.isfinite(rd[i,0,ids])).sum()),surface_exit_median_m=float(np.median(x[np.isfinite(x)])) if np.isfinite(x).any() else None,peak_envelope_not_simultaneous=True))
 write(R/'prior_rail_candidates_surface_alternatives.csv',comparison)
 dump(R/'analysis_summary.json',dict(evaluations=len(scenario),road_contexts=len(contexts),unique_mixed_graphs=len(cache),unique_walking_graphs=len(wcache),normal_outward_optimistic=int(baseline_possible[0].sum()),normal_outward_pessimistic=int(baseline_possible[1].sum()),normal_outward_both=int(baseline_possible.all(axis=0).sum()),local_ground_projection_supported=int(local_support.sum()),classification_evidence_eligible=int(evidence_ok.sum()),baseline_walking_exit_outward=[int(np.isfinite(x[0]).sum()) for x in base_exit],baseline_mixed_outward=[int(np.isfinite(x[0]).sum()) for x in base_rail],seconds=time.monotonic()-started,new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False))
 print('DONE',json.dumps(json.loads((R/'analysis_summary.json').read_text()),indent=2),flush=True)
if __name__=='__main__':main()
