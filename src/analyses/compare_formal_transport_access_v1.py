"""Finite9984-condition transport evaluation on unchanged directed native graphs."""
from pathlib import Path
import os,sys,json,time,collections,hashlib
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_formal_transport_access_v1 import P,R,OLD,STRICT,T,F,D,CASES,PROJECTIONS,UNCERTAINTIES,KINDS,IDENTITIES,THRESHOLDS,read,write,dump,load,sha,segment_max
from compare_historical_surface_coverage_v2 import graph_parts,Template
from compare_transport_shutdown_scenarios_v1 import closure_mask

def main():
 started=time.monotonic();assert json.loads((R/'preparation_validation.json').read_text())['all_passed'];policy=json.loads((P/'analysis_contract.json').read_text());s=load(P/'projection_support.npz');w,edges,ri,u,v,weight,ao,ad,cp=graph_parts();probes=read(OLD/'surface_probe_inventory.csv');nw=len(w['node_ids']);ne=len(w['from_index']);nr=len(ri);pn=w['probe_node_index'];exits=w['exit_node_index']
 graph=Template(u,v,weight,nw+len(load(T/'hypothetical_network.npz')['node_ids']));walking=Template(w['from_index'],w['to_index'],w['length_m'],nw);idx=w['arc_basis'];off=w['edge_sample_offsets'];ind=w['edge_sample_road_index']
 def walking_distance(g):return np.stack([dijkstra(g.T.tocsr(),indices=exits,directed=True,min_only=True),dijkstra(g,indices=exits,directed=True,min_only=True)])[:,pn]
 def rail_distance(g):
  assert ao.shape==ad.shape==(6,)
  return np.stack([dijkstra(g.T.tocsr(),indices=ad,directed=True,min_only=True),dijkstra(g,indices=ao,directed=True,min_only=True)])[:,pn]
 unknown=s['unknown'];basew=np.zeros((2,2,2,len(pn)));baser=basew.copy();incident=[]
 for pi,projection in enumerate(PROJECTIONS):
  ik=np.zeros(nw,bool);goodarc=~unknown[pi,idx];ik[w['from_index'][goodarc]]=True;ik[w['to_index'][goodarc]]=True;incident.append(ik[pn])
  for bi,branch in enumerate(UNCERTAINTIES):
   keep=~unknown[pi,idx]if bi else np.ones(ne,bool);basew[pi,bi]=walking_distance(walking.matrix(keep));baser[pi,bi]=rail_distance(graph.matrix(np.r_[keep,np.ones(nr+len(cp),bool)]))
 basepossible=np.isfinite(basew)|np.isfinite(baser);incident=np.array(incident);eligible_out=basepossible[:,:,0].all(axis=1)&incident;eligible_bi=basepossible.all(axis=(1,2))&incident
 baserows=[]
 for pi,projection in enumerate(PROJECTIONS):
  for bi,branch in enumerate(UNCERTAINTIES):
   for j,p in enumerate(probes):baserows.append(dict(projection=projection,unknown_branch=branch,probe_id=p['probe_id'],ward=p['ward'],normal_outward=bool(basepossible[pi,bi,0,j]),normal_return=bool(basepossible[pi,bi,1,j]),evidence_eligible_outward=bool(eligible_out[pi,j]),evidence_eligible_two_direction=bool(eligible_bi[pi,j]),cohort='original115'if s['original_supported'][j]else'added29'if s['added_supported'][j]else'previously_insufficient',local_projection_supported=bool(incident[pi,j])))
 write(R/'normal_baseline_and_support.csv',baserows)
 links={p:collections.defaultdict(set)for p in IDENTITIES}
 for row in read(ROOT/'data/processed/three_ward_station_surface_risk_v1/station_entrance_identity_links.csv'):
  for p in IDENTITIES:links[p][row['osm_node_id']].add(row['station_id'])
 for row in read(T/'identity_attribution_sensitivity_links.csv'):links[IDENTITIES[1]][row['osm_node_id']].add(row['station_id'])
 portalids={p['portal_id']for p in read(T/'portal_inventory.csv')};frames=policy['frame_ids'];eo=s['entrance_offsets'];ei=s['entrance_cell_index'];ga=s['ground_area'];gknown=s['ground_known'];quality=s['entrance_quality']
 contexts=[];closed_masks=[];wd=[];rd=[];scenarios=[];selections=[];packed_rail=[];wcache={};rcache={};frames_done=0
 total=len(CASES)*len(frames)*64
 def progress(status,current=None):dump(R/'execution_state.json',dict(status=status,current_case=current,frames_done=frames_done,total_case_frames=156,network_evaluations_completed=len(scenarios),network_evaluations_total=total,unique_walk_graphs=len(wcache),unique_mixed_graphs=len(rcache),elapsed_seconds=time.monotonic()-started,new_hydraulics=False,people_counted=False))
 progress('running')
 for case in CASES:
  e=load(P/'exposures'/(case+'.npz'))
  for ti,frame in enumerate(frames):
   selected={}
   for bi,branch in enumerate(UNCERTAINTIES):
    bound='lower'if bi==0 else'upper';gdepth=e['ground_'+bound][ti]
    triggered=[]
    for j,osm in enumerate(s['entrance_ids']):
     cells=ei[eo[j]:eo[j+1]]
     if quality[j]and ga[cells][gknown[cells]&(gdepth[cells]>=.3)].sum()>=100:triggered.append(str(osm))
    for identity in IDENTITIES:
     portals=set();targets=set();unassigned=0
     for osm in triggered:
      if links[identity][osm]and'osm:'+osm in portalids:portals.add('osm:'+osm);targets.update(links[identity][osm])
      else:unassigned+=1
     rail={kind:(np.zeros(nr,bool)if kind==KINDS[0]else closure_mask(edges,kind,portals,targets)[ri])for kind in KINDS};selected[bi,identity]=(portals,targets,rail)
     selections.append(dict(case_id=case,frame_id=frame,unknown_branch=branch,identity_policy=identity,trigger_points=len(triggered),selected_portals=len(portals),target_station_units=len(targets),unassigned_trigger_points=unassigned,selected_portal_ids=';'.join(sorted(portals)),target_station_ids=';'.join(sorted(targets)),shutdown_is_assumed=True))
   for pi,projection in enumerate(PROJECTIONS):
    for threshold in THRESHOLDS:
     for bi,branch in enumerate(UNCERTAINTIES):
      bound='lower'if bi==0 else'upper';samples=np.where(s['road_known'][ind],e['road_'+bound][ti,ind],-np.inf);mx=segment_max(samples,off);wet=s['grade']&(mx>=threshold)
      if pi:wet |=s['supplement_qualified']&(e['supplement_'+bound][ti]>=threshold)
      closed=wet|(unknown[pi]if bi else False);keep=~closed[idx];sig=hashlib.sha256(keep.tobytes()).digest()
      if sig not in wcache:wcache[sig]=walking_distance(walking.matrix(keep))
      ci=len(contexts);wd.append(wcache[sig]);closed_masks.append(closed)
      actual=None if frame=='peak_envelope'else float(e['actual_times'][ti])
      contexts.append(dict(road_context_index=ci,case_id=case,frame_id=frame,actual_time_s=actual,simultaneous=frame!='peak_envelope',projection=projection,road_threshold_m=threshold,unknown_branch=branch,closed_known_wet_segments=int(wet.sum()),withheld_unknown_segments=int((closed&unknown[pi]).sum()),closed_segments=int(closed.sum())))
      for identity in IDENTITIES:
       portals,targets,rail=selected[bi,identity]
       for kind in KINDS:
        cut=rail[kind];mixed_keep=np.r_[keep,~cut,np.ones(len(cp),bool)];rsig=hashlib.sha256(mixed_keep.tobytes()).digest()
        if rsig not in rcache:rcache[rsig]=rail_distance(graph.matrix(mixed_keep))
        si=len(scenarios);rd.append(rcache[rsig]);packed_rail.append(np.packbits(cut))
        scenarios.append(dict(scenario_index=si,road_context_index=ci,case_id=case,frame_id=frame,actual_time_s=actual,simultaneous=frame!='peak_envelope',projection=projection,road_threshold_m=threshold,unknown_branch=branch,identity_policy=identity,shutdown_type=kind,selected_portals=len(portals)if kind!=KINDS[0]else 0,target_station_units=len(targets)if kind!=KINDS[0]else 0,removed_rail_arcs=int(cut.sum())))
   frames_done+=1;progress('running',case)
  print('NETWORK CASE',case,'evals',len(scenarios),'of',total,'elapsed',round(time.monotonic()-started),flush=True)
 assert len(scenarios)==9984 and len(contexts)==1248
 wd=np.stack(wd);rd=np.stack(rd)
 np.savez_compressed(P/'alternative_distances.npz',walking_exit_m=wd,mixed_min_walking_m=rd,baseline_walking_exit_m=basew,baseline_mixed_min_walking_m=baser,baseline_possible=basepossible,evidence_outward=eligible_out,evidence_two_direction=eligible_bi,local_projection_supported=incident,road_unknown=unknown,road_closed=np.stack(closed_masks),rail_closed_packed=np.stack(packed_rail),rail_arc_indices=ri,probe_ids=np.array([x['probe_id']for x in probes]))
 write(P/'road_context_index.csv',contexts);write(P/'scenario_index.csv',scenarios);write(P/'station_closure_selection.csv',selections)
 classes=[];aggregates=[];stats=[];grouper=collections.defaultdict(list)
 for i,row in enumerate(scenarios):grouper[row['case_id'],row['frame_id'],row['projection'],row['shutdown_type']].append(i)
 for(case,frame,projection,kind),ids in grouper.items():
  assert len(ids)==8;pi=PROJECTIONS.index(projection);ctx=np.array([scenarios[i]['road_context_index']for i in ids]);reach=np.isfinite(wd[ctx])|np.isfinite(rd[ids]);two=reach.all(axis=1)
  for j,p in enumerate(probes):
   def classify(values,eligible):return'insufficient_evidence'if not eligible else'robust_modeled_disconnection'if not values.any()else'stable_modeled_alternative'if values.all()else'conditional_sensitivity'
   classes.append(dict(case_id=case,frame_id=frame,projection=projection,shutdown_type=kind,probe_id=p['probe_id'],ward=p['ward'],x_m=p['x_m'],y_m=p['y_m'],outward_classification=classify(reach[:,0,j],eligible_out[pi,j]),two_direction_classification=classify(two[:,j],eligible_bi[pi,j]),branches_outward=int(reach[:,0,j].sum()),branches_return=int(reach[:,1,j].sum()),branches_two_direction=int(two[:,j].sum()),evaluated_branches=8,cohort='original115'if s['original_supported'][j]else'added29'if s['added_supported'][j]else'previously_insufficient',simultaneous=frame!='peak_envelope',not_population_or_whole_town=True))
  chunk=classes[-len(probes):]
  for ward in ['all','13101','13102','13103','boundary_or_unassigned']:
   rows=[x for x in chunk if ward=='all'or x['ward']==ward]
   for direction in ['outward','two_direction']:
    counts=collections.Counter(x[direction+'_classification']for x in rows);aggregates.append(dict(case_id=case,frame_id=frame,projection=projection,shutdown_type=kind,ward=ward,direction=direction,origin_probes=len(rows),**{k:counts[k]for k in ['robust_modeled_disconnection','conditional_sensitivity','insufficient_evidence','stable_modeled_alternative']},simultaneous=frame!='peak_envelope'))
 for i,row in enumerate(scenarios):
  ci=row['road_context_index'];pi=PROJECTIONS.index(row['projection']);bi=UNCERTAINTIES.index(row['unknown_branch']);alternative=np.minimum(wd[ci],rd[i]);normal=np.minimum(basew[pi,bi],baser[pi,bi]);finite=np.isfinite(alternative);paired=basepossible[pi,bi]&finite;outcost=alternative[0,finite[0]];diff=alternative[0,paired[0]]-normal[0,paired[0]]
  stats.append(dict(**row,has_outward_alternative=int(finite[0].sum()),has_return_alternative=int(finite[1].sum()),has_two_direction_alternative=int(finite.all(axis=0).sum()),new_outward_loss=int((basepossible[pi,bi,0]&~finite[0]).sum()),new_two_direction_loss=int((basepossible[pi,bi].all(axis=0)&~finite.all(axis=0)).sum()),surface_outward_reachable=int(np.isfinite(wd[ci,0]).sum()),surface_return_reachable=int(np.isfinite(wd[ci,1]).sum()),mixed_outward_reachable=int(np.isfinite(rd[i,0]).sum()),mixed_return_reachable=int(np.isfinite(rd[i,1]).sum()),median_outward_min_walking_m=float(np.median(outcost))if len(outcost)else None,paired_median_added_walking_m=float(np.median(diff))if len(diff)else None,paired_p90_added_walking_m=float(np.quantile(diff,.9))if len(diff)else None))
 write(R/'probe_classification.csv',classes);write(R/'classification_summary.csv',aggregates);write(R/'scenario_accessibility_summary.csv',stats)
 # Matched classifications compare identical probes and evidence rules; no population weighting.
 ix={(r['case_id'],r['frame_id'],r['projection'],r['shutdown_type'],r['probe_id']):r for r in classes};changes=[]
 for rain in ['R0','R1','R125']:
  for stage in ['L','H']:
   for frame in frames:
    for projection in PROJECTIONS:
     for kind in KINDS:
      for direction in ['outward','two_direction']:
       for cohort in ['all','original115','added29']:
        pairs=[(ix[f'{rain}__{stage}__T',frame,projection,kind,p['probe_id']],ix[f'{rain}__{stage}__U',frame,projection,kind,p['probe_id']])for p in probes]
        if cohort!='all':pairs=[(a,b)for a,b in pairs if a['cohort']==cohort]
        key=direction+'_classification';good=[(a,b)for a,b in pairs if a[key]!='insufficient_evidence'and b[key]!='insufficient_evidence'];robust='robust_modeled_disconnection'
        changes.append(dict(rainfall=rain,stage=stage,frame_id=frame,projection=projection,shutdown_type=kind,direction=direction,cohort=cohort,paired_eligible=len(good),T_robust=sum(a[key]==robust for a,b in good),U_robust=sum(b[key]==robust for a,b in good),new_robust_in_U=sum(b[key]==robust and a[key]!=robust for a,b in good),removed_robust_in_U=sum(a[key]==robust and b[key]!=robust for a,b in good),classification_changes=sum(a[key]!=b[key]for a,b in good),simultaneous=frame!='peak_envelope'))
 write(R/'geometry_classification_changes.csv',changes)
 dump(R/'analysis_summary.json',dict(network_evaluations=len(scenarios),road_contexts=len(contexts),physical_cases=len(CASES),synchronous_times=12,separate_peak_envelope=True,unique_walking_graphs=len(wcache),unique_mixed_graphs=len(rcache),eligible_outward_by_projection=eligible_out.sum(axis=1).tolist(),eligible_two_direction_by_projection=eligible_bi.sum(axis=1).tolist(),original_supported_cohort=int(s['original_supported'].sum()),added_supported_cohort=int(s['added_supported'].sum()),elapsed_seconds=time.monotonic()-started,new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False))
 progress('network_evaluations_complete');print('NETWORK COMPLETE',len(scenarios),flush=True)
if __name__=='__main__':main()
