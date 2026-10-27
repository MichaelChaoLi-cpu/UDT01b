"""Paired accessibility comparisons under declared, static shutdown hypotheses."""
from pathlib import Path
import sys,json,csv,collections,hashlib,time
import numpy as np
from shapely.geometry import shape,Point
from shapely.ops import unary_union
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_transport_shutdown_scenarios_v1';R=ROOT/'data/exp/estimation/three_ward_transport_shutdown_scenarios_v1';S=ROOT/'data/processed/three_ward_station_surface_risk_v1'
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from screen_station_surface_risk_v1 import public_operator
from build_normal_surface_network_v1 import dump,csvwrite
from build_transport_shutdown_scenarios_v1 import rr,sha,canon

def graph_arrays(edges,keys):
 ix={k:i for i,k in enumerate(keys)};return np.array([ix[e['from_node']] for e in edges]),np.array([ix[e['to_node']] for e in edges])
def closure_mask(edges,kind,portals,targets):
 closed=np.zeros(len(edges),bool)
 for i,e in enumerate(edges):
  if e['edge_kind'] in ['station_entry','station_exit'] and e['portal_id'] in portals:closed[i]=True
  if kind in ['station','section']:
   if e['station_id'] in targets and e['edge_kind'] in ['station_entry','station_exit','board','alight']:closed[i]=True
   if e['edge_kind']=='transfer' and targets&set(e.get('transfer_units',[])):closed[i]=True
  if kind=='section' and e['edge_kind']=='rail_service' and targets&set(e['section_exposure_station_ids']):closed[i]=True
 return closed

def semantic_checks():
 # A closed entrance retains a second entry; station non-stop keeps rail
 # through movement; section suspension breaks it. Origins/sinks cannot relay.
 E=[]
 def add(a,b,k,s='',p='',cov=()):E.append({'from_node':a,'to_node':b,'edge_kind':k,'station_id':s,'portal_id':p,'section_exposure_station_ids':list(cov)})
 add('O','p1','initial_walk');add('O','p2','initial_walk');add('p1','h','station_entry','X','p1');add('p2','h','station_entry','X','p2');add('h','r1','board','X');add('ra','r1','rail_service',cov=['X']);add('r1','rb','rail_service',cov=['X']);add('rb','D','final_walk')
 keys=['O','p1','p2','h','r1','ra','rb','D'];ix={k:i for i,k in enumerate(keys)};u,v=graph_arrays(E,keys)
 def reaches(kind,portals,targets,start,end):
  keep=~closure_mask(E,kind,portals,targets);g=coo_matrix((np.ones(keep.sum(),np.uint8),(u[keep],v[keep])),shape=(len(keys),len(keys))).tocsr();return ix[end] in breadth_first_order(g,ix[start],directed=True,return_predecessors=False)
 checks=[{'check':'entrance_closure_retains_second_entrance','pass':reaches('entrance',{'p1'},{'X'},'O','D')},{'check':'station_closure_blocks_local_boarding','pass':not reaches('station',{'p1'},{'X'},'O','D')},{'check':'station_closure_retains_through_service','pass':reaches('station',set(),{'X'},'ra','D')},{'check':'section_suspension_breaks_through_service','pass':not reaches('section',set(),{'X'},'ra','D')}]
 assert all(x['pass'] for x in checks);return checks

def main():
 checks=semantic_checks();policy=json.loads((P/'scenario_policy.json').read_text());policy_sha=sha(P/'scenario_policy.json');z=np.load(P/'hypothetical_network.npz');keys=list(map(str,z['node_ids']));ix={k:i for i,k in enumerate(keys)};u=z['from_index'];v=z['to_index'];edges=json.loads((P/'hypothetical_network_edges.json').read_text());probes=rr(P/'access_probes.csv');portalmeta={r['portal_id']:r for r in rr(P/'portal_inventory.csv')};coremeta={x['station_id']:x for x in rr(S/'station_screen_summary.csv')};origins=[p for p in probes if p['cohort'] in ['surface_grid','core_station']];anchorids=[x['operator']+':'+canon(x['name']) for x in policy['anchors']];originindices=np.array([ix['O:'+p['probe_id']] for p in origins]);destindices=np.array([ix['D:'+p['probe_id']] for p in origins]);anchorO=[ix['O:station:'+sid] for sid in anchorids];anchorD=[ix['D:station:'+sid] for sid in anchorids]
 def od(keep):
  g=coo_matrix((np.ones(int(keep.sum()),np.int32),(u[keep],v[keep])),shape=(len(keys),len(keys))).tocsr();gt=g.T.tocsr();out=np.zeros((len(origins),6),bool);back=out.copy()
  for j in range(6):
   reachable=np.zeros(len(keys),bool);reachable[breadth_first_order(gt,anchorD[j],directed=True,return_predecessors=False)]=True;out[:,j]=reachable[originindices]
   reachable[:]=False;reachable[breadth_first_order(g,anchorO[j],directed=True,return_predecessors=False)]=True;back[:,j]=reachable[destindices]
  return out,back
 eligiblebranches={'intra_operator_hubs_only':~z['provider_transfer'],'provider_transfers_assumed_bidir':np.ones(len(edges),bool)};baseline={};baseline_rows=[];od_baseline=[]
 for branch,keep in eligiblebranches.items():
  out,back=od(keep);baseline[branch]=(out,back);od_baseline.append(np.stack([out,back]));mutual=out&back
  for cohort in ['surface_grid','core_station']:
   for ward in ['all','13101','13102','13103','boundary_or_unassigned']:
    use=np.array([p['cohort']==cohort and (ward=='all' or p['ward']==ward) for p in origins]);n=int(use.sum())
    if not n:continue
    baseline_rows.append({'network_branch':branch,'cohort':cohort,'ward':ward,'origin_probes':n,'declared_OD_pairs':n*6,'reachable_outward_pairs':int(out[use].sum()),'reachable_return_pairs':int(back[use].sum()),'mutual_pairs':int(mutual[use].sum()),'origins_with_any_mutual_anchor':int(mutual[use].any(axis=1).sum()),'origins_without_any_baseline_mutual_anchor':int((~mutual[use].any(axis=1)).sum()),'not_a_population_fraction':True})
 csvwrite(R/'normal_baseline_coverage.csv',baseline_rows)
 exposure=rr(S/'entrance_scenario_surface_screen.csv');exposure=[r for r in exposure if int(r['radius_m'])==20];vicinity=rr(S/'station_vicinity_scenario_screen.csv');vicinity=[r for r in vicinity if int(r['radius_m'])==50];links=collections.defaultdict(set)
 for r in rr(S/'station_entrance_identity_links.csv'):links[r['osm_node_id']].add(r['station_id'])
 events=rr(S/'scenario_bindings.csv');assert len(events)==16 and all(e['split'] in ['train','validation'] for e in events);byevent=collections.defaultdict(list);vbyevent=collections.defaultdict(list)
 for r in exposure:byevent[r['case_id']].append(r)
 for r in vicinity:vbyevent[r['case_id']].append(r)
 supplement_links=collections.defaultdict(set);coregeo=collections.defaultdict(list)
 for f in json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/stations_regional_6677.geojson').read_text())['features']:
  pp=f['properties'];key=public_operator(pp['N02_004'])+':'+canon(pp['N02_005'])
  if pp['intersects_core']:coregeo[key].append(shape(f['geometry']))
 coregeo={k:unary_union(v) for k,v in coregeo.items()};identity_supplement=[]
 for ep,p in portalmeta.items():
  if p['synthetic_unknown']=='True':continue
  for sid in json.loads(p['station_ids']):
   if sid not in coregeo:continue
   dist=coregeo[sid].distance(Point(float(p['x_m']),float(p['y_m'])))
   if dist<=500:
    supplement_links[p['osm_node_id']].add(sid);identity_supplement.append({'osm_node_id':p['osm_node_id'],'station_id':sid,'distance_to_same_identity_official_geometry_m':dist,'prior_risk_link':sid in links[p['osm_node_id']],'basis':'source_area_name_operator_with_explicit_line_annotation_normalization_not_nearest_station'})
 csvwrite(P/'identity_attribution_sensitivity_links.csv',identity_supplement)
 configs=[('primary_0p3_area100','0p3',100,False),('sensitivity_0p1_area100','0p1',100,False),('sensitivity_0p5_area100','0p5',100,False),('sensitivity_0p3_anyarea','0p3',1e-7,False),('sensitivity_vicinity_fallback','0p3',100,True),('sensitivity_normalized_identity','0p3',100,False)]
 selections=[];ledger=[];summary=[];wards=[];probe_details=[];outarray=[];backarray=[];scenario_index=[];monotone=True;no_new_paths=True;walk_preserved=True;station_service_preserved=True;cache={};start=time.monotonic()
 for event in events:
  eid=event['case_id']
  for config,depth,area_min,fallback in configs:
   closed_portals=set();targets=set();trigger_unassigned=0;unknown_rows=0;point_records=[]
   for r in byevent[eid]:
    quality=r['admitted_fraction_of_eligible']!='' and float(r['admitted_fraction_of_eligible'])>=.8 and float(r['admitted_open_ground_m2'])>=100
    if not quality:unknown_rows+=1;continue
    area=float(r['ground_area_lower_ge_'+depth+'_m2']);upper=float(r['ground_area_upper_ge_'+depth+'_m2']);trigger=area+1e-9>=area_min
    if not trigger:continue
    id=r['osm_node_id'];associations=(links[id]|supplement_links[id]) if config=='sensitivity_normalized_identity' else links[id];ep='osm:'+id;qualified=bool(associations) and ep in portalmeta
    if qualified:closed_portals.add(ep);targets|=associations
    else:trigger_unassigned+=1
    point_records.append({'case_id':eid,'exposure_policy':config,'osm_node_id':id,'mapped_core_units':';'.join(sorted(associations)),'exceedance_footprint_lower_m2':area,'exceedance_footprint_upper_m2':upper,'trigger_depth_m':float(depth.replace('p','.')),'trigger_area_m2':area_min,'admitted_fraction':r['admitted_fraction_of_eligible'],'selected_portal':qualified,'basis':'simulated_peak_footprint_plus_declared_shutdown_rule','not_verified_inlet_or_shutdown':True})
   primarytargets=set(targets);fallbacktargets=set()
   if fallback:
    for r in vbyevent[eid]:
     sid=r['station_id'];admitted=float(r['admitted_ground_m2']);eligible=float(r['eligible_ground_m2']);wet=float(r['peak_exceedance_ground_area_lower_ge_0p3_m2'])
     if int(coremeta[sid]['associated_mapped_entrances'])==0 and admitted>=100 and eligible>0 and admitted/eligible>=.8 and wet>=100 and wet/admitted>=.05:fallbacktargets.add(sid)
    targets|=fallbacktargets
   ledger.extend(point_records);selections.append({'case_id':eid,'parent_id':event['parent_id'],'split':event['split'],'exposure_policy':config,'closed_mapped_portals':len(closed_portals),'entrance_evidence_target_units':len(primarytargets),'vicinity_assumed_target_units':len(fallbacktargets),'target_units_total':len(targets),'triggered_but_unassigned_source_points':trigger_unassigned,'unknown_quality_source_points':unknown_rows,'entrance_evidence_station_ids':';'.join(sorted(primarytargets)),'vicinity_assumed_station_ids':';'.join(sorted(fallbacktargets)),'selected_portal_ids':';'.join(sorted(closed_portals))})
   masks={kind:closure_mask(edges,kind,closed_portals,targets) for kind in ['entrance','station','section']};monotone &= not np.any(masks['entrance']&~masks['station']) and not np.any(masks['station']&~masks['section'])
   for kind,removed in masks.items():
    walk_preserved &= all(not removed[i] for i,e in enumerate(edges) if e['edge_kind'] in ['initial_walk','final_walk','station_probe_access','station_probe_egress'])
    if kind=='station':station_service_preserved &= all(not removed[i] for i,e in enumerate(edges) if e['edge_kind']=='rail_service')
    for branch,active in eligiblebranches.items():
     keep=active&~removed;signature=hashlib.sha256(keep.tobytes()).hexdigest()
     if signature not in cache:cache[signature]=od(keep)
     out,back=cache[signature];bo,bb=baseline[branch];bm=bo&bb;mutual=out&back;no_new_paths &= not np.any(out&~bo) and not np.any(back&~bb);lost=bo&~out;lostback=bb&~back;lostmutual=bm&~mutual
     sid=f'{eid}__{config}__{kind}__{branch}';scenario_index.append({'scenario_id':sid,'case_id':eid,'exposure_policy':config,'shutdown_type':kind,'network_branch':branch});outarray.append(out);backarray.append(back)
     row={'scenario_id':sid,'case_id':eid,'parent_id':event['parent_id'],'split':event['split'],'exposure_policy':config,'shutdown_type':kind,'network_branch':branch,'closed_mapped_portals':len(closed_portals),'entrance_evidence_target_units':len(primarytargets),'vicinity_assumed_target_units':len(fallbacktargets),'target_units_total':len(targets),'removed_arcs':int((removed&active).sum()),'removed_service_arcs':sum(bool(removed[i]&active[i]) for i,e in enumerate(edges) if e['edge_kind']=='rail_service'),'baseline_mutual_pairs':int(bm.sum()),'lost_mutual_pairs':int(lostmutual.sum()),'lost_mutual_fraction':float(lostmutual.sum()/bm.sum()) if bm.sum() else None}
     for cohort in ['surface_grid','core_station']:
      use=np.array([p['cohort']==cohort for p in origins]);before=bm[use].any(axis=1);after=mutual[use].any(axis=1);base=int(bm[use].sum());row[cohort+'_baseline_mutual_pairs']=base;row[cohort+'_lost_outward_pairs']=int(lost[use].sum());row[cohort+'_lost_return_pairs']=int(lostback[use].sum());row[cohort+'_lost_mutual_pairs']=int(lostmutual[use].sum());row[cohort+'_lost_mutual_fraction']=float(lostmutual[use].sum()/base) if base else None;row[cohort+'_baseline_origins_with_any_anchor']=int(before.sum());row[cohort+'_lost_all_declared_mutual_anchors']=int((before&~after).sum());row[cohort+'_origins_lost_some_pairs']=int(lostmutual[use].any(axis=1).sum())
      for ward in ['13101','13102','13103','boundary_or_unassigned']:
       ww=np.array([p['cohort']==cohort and p['ward']==ward for p in origins]);npairs=int(bm[ww].sum())
       if not ww.any():continue
       wards.append({'scenario_id':sid,'case_id':eid,'exposure_policy':config,'shutdown_type':kind,'network_branch':branch,'cohort':cohort,'ward':ward,'origin_probes':int(ww.sum()),'baseline_mutual_pairs':npairs,'lost_mutual_pairs':int(lostmutual[ww].sum()),'lost_mutual_fraction':float(lostmutual[ww].sum()/npairs) if npairs else None,'lost_all_declared_mutual_anchors':int((bm[ww].any(axis=1)&~mutual[ww].any(axis=1)).sum()),'population_weighting':False})
     summary.append(row)
     if config in ['primary_0p3_area100','sensitivity_vicinity_fallback','sensitivity_normalized_identity'] and branch=='provider_transfers_assumed_bidir':
      for i,p in enumerate(origins):probe_details.append({'scenario_id':sid,'case_id':eid,'exposure_policy':config,'shutdown_type':kind,'probe_id':p['probe_id'],'cohort':p['cohort'],'ward':p['ward'],'x_m':p['x_m'],'y_m':p['y_m'],'baseline_mutual_anchor_count':int(bm[i].sum()),'remaining_mutual_anchor_count':int(mutual[i].sum()),'lost_mutual_anchor_count':int(lostmutual[i].sum()),'lost_all_declared_mutual_anchors':bool(bm[i].any() and not mutual[i].any()),'baseline_uncovered':not bool(bm[i].any()),'not_population_or_proven_isolation':True})
  print('CASE',eid,'evaluations',len(summary),'cached_graphs',len(cache),flush=True)
 csvwrite(P/'closure_selection_by_parent.csv',selections);csvwrite(P/'exposure_to_hypothesis_ledger.csv',ledger);csvwrite(R/'scenario_accessibility_summary.csv',summary);csvwrite(R/'ward_accessibility_summary.csv',wards);csvwrite(R/'primary_probe_accessibility.csv',probe_details);csvwrite(P/'scenario_index.csv',scenario_index)
 np.savez_compressed(P/'paired_accessibility_matrices.npz',outward=np.stack(outarray),return_trip=np.stack(backarray),baseline=np.stack(od_baseline),origin_probe_ids=np.array([p['probe_id'] for p in origins]),anchor_station_ids=np.array(anchorids),baseline_branch_names=np.array(list(eligiblebranches)))
 checks.extend([{'check':'closure_arc_sets_nested_A_B_C','pass':monotone},{'check':'deleting_arcs_never_creates_new_OD_paths','pass':no_new_paths},{'check':'all_surface_walk_legs_unchanged','pass':walk_preserved},{'check':'station_nonstop_retains_service_arcs','pass':station_service_preserved},{'check':'policy_unchanged_during_evaluation','pass':sha(P/'scenario_policy.json')==policy_sha},{'check':'origins_have_no_incoming_and_destinations_no_outgoing','pass':not any(e['to_node'].startswith('O:') or e['from_node'].startswith('D:') for e in edges)},{'check':'all_walk_legs_are_native_source_paths_le600m','pass':all(float(x['distance_m'])<=600+1e-8 and x['source_path']=='True' for x in rr(P/'bounded_walk_legs.csv'))}])
 bindings=json.loads((P/'source_bindings.json').read_text());checks.append({'check':'frozen_network_risk_population_bindings_unchanged','pass':all(sha(ROOT/x['file'])==x['sha256'] for x in bindings)})
 assert all(x['pass'] for x in checks),checks
 dump(R/'validation.json',{'checks':checks,'all_passed':True,'hypotheses_not_observed_shutdowns':True,'new_hydraulics':False,'people_counted':False,'sealed_tests_opened':False});dump(R/'evaluation_receipt.json',{'human_decision':'MILI-D-20261016-024','policy_sha256':policy_sha,'evaluations':len(summary),'unique_evaluated_graphs':len(cache),'parent_events':len(events),'seconds':time.monotonic()-start,'date':'2026-10-16','network_branches':list(eligiblebranches),'inputs_unchanged':True,'population_weighted':False});print('DONE',len(summary),'evaluations',len(cache),'unique graphs',flush=True)
if __name__=='__main__':main()
