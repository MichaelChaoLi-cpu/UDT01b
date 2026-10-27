"""Independent reconstruction of historical shutdown edges, reachability and totals.
Uses saved outputs and frozen local inputs; never executes a hydraulic solver.
"""
from pathlib import Path
import os,csv,json,hashlib,re,collections,itertools
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2]
N=ROOT/'data/processed/three_ward_transport_shutdown_scenarios_v1';H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2';S=ROOT/'data/processed/three_ward_station_surface_risk_v1';P=ROOT/'data/processed/three_ward_historical_transport_islands_v1';R=ROOT/'data/exp/estimation/three_ward_historical_transport_islands_v1'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_npz(p):
 with np.load(p) as f:return {k:f[k] for k in f.files}
def main():
 checks=[]
 def check(name,ok):
  checks.append(dict(check=name,passed=bool(ok)));assert ok,name
 bindings=json.loads((P/'source_bindings.json').read_text());check('all_bound_inputs_unchanged',all(sha(ROOT/p)==v for p,v in bindings.items()))
 check('prior_normal_network_sources_unchanged',all(sha(ROOT/x['file'])==x['sha256'] for x in json.loads((N/'source_bindings.json').read_text())))
 check('no_sealed_test_content_in_bound_inputs',not any('sealed_test-' in p for p in bindings))
 # Check the already declared protected metadata only; do not open sealed inputs.
 protected=json.loads((H/'protected_sources.json').read_text());protected_meta=[p for p in protected if p in bindings]
 check('original_metadata_and_masks_still_match_pre_hydraulic_hashes',all(bindings[p]==protected[p] for p in protected_meta))
 policy=json.loads((P/'comparison_contract.json').read_text());original=json.loads((N/'scenario_policy.json').read_text())
 check('original_exposure_and_shutdown_policies_retained',policy['primary_exposure_rule']==original['primary_exposure_rule'] and policy['shutdown_types']==original['shutdown_types'])
 check('one_external_parent_not_12_plus_4',policy['parent_id']=='HIST-RADAR-TOKYO-20240821' and 'external_historical_candidate' in policy['partition'])
 check('excluded_tasks_remain_excluded',all(policy[k] is False for k in ['new_hydraulics','training','sealed_tests_opened','people_counted']))
 n=load_npz(N/'hypothetical_network.npz');z=load_npz(P/'paired_accessibility.npz');old=load_npz(N/'paired_accessibility_matrices.npz');edges=json.loads((N/'hypothetical_network_edges.json').read_text());keys=list(map(str,n['node_ids']));ix={k:i for i,k in enumerate(keys)}
 probes=[p for p in read(N/'access_probes.csv') if p['cohort'] in ['surface_grid','core_station']];check('same_origin_and_anchor_order',np.array_equal(z['origin_probe_ids'],old['origin_probe_ids']) and np.array_equal(z['anchor_station_ids'],old['anchor_station_ids']) and list(z['origin_probe_ids'])==[p['probe_id'] for p in probes])
 check('normal_baseline_exactly_preserved',np.array_equal(z['baseline_od'],old['baseline']))
 check('network_edge_indices_consistent',all(keys[int(n['from_index'][i])]==e['from_node'] and keys[int(n['to_index'][i])]==e['to_node'] for i,e in enumerate(edges)))
 check('origins_and_destinations_cannot_relay',all(not e['to_node'].startswith('O:') and not e['from_node'].startswith('D:') for e in edges))
 check('frozen_walk_legs_are_source_paths_under_600m',all(r['source_path']=='True' and float(r['distance_m'])<=600+1e-8 for r in read(N/'bounded_walk_legs.csv')))
 # Recompute20m screen on frozen affine grid, including quality and interval bounds.
 b=load_npz(H/'paired_exposure_bounds.npz');s=load_npz(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');aff=s['Affine Transform'];a=b['ground_area'];valid=s['Eligible Ground Mask'];good=b['common_admitted']
 check('grid_affine_and_crs_supported',np.allclose(aff,[10,0,-11360,0,-10,-32320]) and int(s['EPSG'])==6677)
 check('common_quality_mask_exactly_matches_labels',np.array_equal(good,valid&load_npz(H/'execution/radar_low_candidate_reference.npz')['Admitted Mask']&~load_npz(H/'execution/radar_low_candidate_reference.npz')['Quarantine Mask']&load_npz(H/'execution/radar_raised_candidate_reference.npz')['Admitted Mask']&~load_npz(H/'execution/radar_raised_candidate_reference.npz')['Quarantine Mask']))
 ep={(r['case_id'],r['osm_node_id']):r for r in read(P/'entrance_exposure.csv')};exposure_ok=True
 for ent in read(S/'entrance_screen_summary.csv'):
  x,y=float(ent['x_m']),float(ent['y_m']);c=math_floor((x-aff[2])/aff[0]);r=math_floor((y-aff[5])/aff[4]);rr,cc=np.mgrid[max(0,r-3):min(valid.shape[0],r+4),max(0,c-3):min(valid.shape[1],c+4)]
  if rr.size:
   mask=(aff[2]+(cc+.5)*aff[0]-x)**2+(aff[5]+(rr+.5)*aff[4]-y)**2<=400;mask &= valid[rr,cc];rr,cc=rr[mask],cc[mask]
  else:rr=cc=np.array([],int)
  eligible=float(a[rr,cc].sum());keep=good[rr,cc];rr,cc=rr[keep],cc[keep];admitted=float(a[rr,cc].sum());quality=eligible>0 and admitted>=100 and admitted/eligible>=.8
  for j,case in enumerate(['radar_low','radar_raised']):
   e=ep[case,ent['osm_node_id']];exposure_ok &= abs(eligible-float(e['eligible_ground_m2']))<1e-8 and abs(admitted-float(e['admitted_ground_m2']))<1e-8 and (e['quality_pass']=='True')==quality
   for bound,key in [('lower','ground_lower'),('upper','ground_upper')]:
    wet=float(a[rr,cc][b[key][j,rr,cc]>=.3].sum());exposure_ok &= abs(wet-float(e[bound+'_exceedance_m2']))<1e-8 and (e[bound+'_trigger']=='True')==(quality and wet>=100)
 check('independent_entrance_area_quality_and_bounds_reconstruction',exposure_ok)
 links={pol:collections.defaultdict(set) for pol in ['original_identity','source_normalized_identity']}
 for row in read(S/'station_entrance_identity_links.csv'):
  for pol in links:links[pol][row['osm_node_id']].add(row['station_id'])
 for row in read(N/'identity_attribution_sensitivity_links.csv'):links['source_normalized_identity'][row['osm_node_id']].add(row['station_id'])
 portalids={r['portal_id'] for r in read(N/'portal_inventory.csv')};select=read(P/'closure_selection.csv');selection={};selectok=True
 for row in select:
  key=(row['case_id'],row['identity_policy'],row['depth_bound']);ports=set();targets=set();unassigned=0
  for (case,osm),e in ep.items():
   if case!=row['case_id'] or e[row['depth_bound']+'_trigger']!='True':continue
   ids=links[row['identity_policy']][osm];portal='osm:'+osm
   if ids and portal in portalids:ports.add(portal);targets.update(ids)
   else:unassigned+=1
  selection[key]=(ports,targets)
  selectok &= ports==set(filter(None,row['selected_portal_ids'].split(';'))) and targets==set(filter(None,row['target_station_ids'].split(';'))) and unassigned==int(row['unassigned_trigger_points']) and len(ports)==int(row['selected_portals']) and len(targets)==int(row['target_station_units'])
 check('all_closure_selections_reconstructed_from_screen_and_source_identity',selectok)
 check('numeric_lower_selection_subset_of_upper',all(selection[(c,p,'lower')][0]<=selection[(c,p,'upper')][0] and selection[(c,p,'lower')][1]<=selection[(c,p,'upper')][1] for c in ['radar_low','radar_raised'] for p in links))
 check('original_identity_selection_subset_of_normalized',all(selection[(c,'original_identity',bnd)][0]<=selection[(c,'source_normalized_identity',bnd)][0] and selection[(c,'original_identity',bnd)][1]<=selection[(c,'source_normalized_identity',bnd)][1] for c in ['radar_low','radar_raised'] for bnd in ['lower','upper']))
 index=read(P/'scenario_index.csv');branches=list(map(str,z['baseline_branch_names']));oi=[ix['O:'+p['probe_id']] for p in probes];di=[ix['D:'+p['probe_id']] for p in probes];ao=[ix['O:station:'+s] for s in z['anchor_station_ids']];ad=[ix['D:station:'+s] for s in z['anchor_station_ids']]
 check('complete_48_comparison_factorial',len(index)==48 and len({(r['case_id'],r['identity_policy'],r['depth_bound'],r['shutdown_type'],r['network_branch']) for r in index})==48)
 # Independent semantic reconstruction; do NOT import the production closure function.
 allmasks=[];walksafe=True;passsafe=True;odok=True;maskok=True
 for row in index:
  i=int(row['scenario_index']);ports,targets=selection[row['case_id'],row['identity_policy'],row['depth_bound']];kind=row['shutdown_type'];remove=[]
  for e in edges:
   cut=e['edge_kind'] in ['station_entry','station_exit'] and e['portal_id'] in ports
   if kind!='entrance':
    cut |= e['edge_kind'] in ['station_entry','station_exit','board','alight'] and e['station_id'] in targets
    cut |= e['edge_kind']=='transfer' and bool(targets.intersection(e.get('transfer_units',[])))
   if kind=='section':cut |= e['edge_kind']=='rail_service' and bool(targets.intersection(e['section_exposure_station_ids']))
   remove.append(cut)
  remove=np.array(remove,bool);active=~n['provider_transfer'] if row['network_branch']==branches[0] else np.ones(len(edges),bool);remove &= active;allmasks.append(remove);maskok &= np.array_equal(remove,z['removed_arcs'][i])
  walksafe &= not any(remove[j] for j,e in enumerate(edges) if e['edge_kind'] in ['initial_walk','final_walk','station_probe_access','station_probe_egress'])
  if kind=='station':passsafe &= not any(remove[j] for j,e in enumerate(edges) if e['edge_kind']=='rail_service')
  keep=active&~remove;g=coo_matrix((np.ones(int(keep.sum()),np.int32),(n['from_index'][keep],n['to_index'][keep])),shape=(len(keys),len(keys))).tocsr()
  # Unweighted shortest paths provide an independent reachability calculation.
  out=np.isfinite(dijkstra(g.T.tocsr(),indices=ad,directed=True,unweighted=True)[:,oi]).T;back=np.isfinite(dijkstra(g,indices=ao,directed=True,unweighted=True)[:,di]).T
  odok &= np.array_equal(np.stack([out,back]),z['scenario_od'][i])
 check('independent_removed_arc_sets_match_all48',maskok);check('independent_shortest_path_reachability_matches_all48',odok);check('all_pedestrian_access_legs_unchanged',walksafe);check('whole_station_closure_retains_through_running',passsafe)
 base=np.stack([z['baseline_od'][branches.index(r['network_branch'])] for r in index]);check('closures_never_create_paths',not np.any(z['scenario_od']&~base))
 lookup={(r['case_id'],r['identity_policy'],r['depth_bound'],r['shutdown_type'],r['network_branch']):int(r['scenario_index']) for r in index};nested=True;reachnested=True
 for c,p,bnd,br in itertools.product(['radar_low','radar_raised'],links,['lower','upper'],branches):
  ids=[lookup[c,p,bnd,k,br] for k in ['entrance','station','section']]
  for ia,ib in zip(ids,ids[1:]):nested &= not np.any(allmasks[ia]&~allmasks[ib]);reachnested &= not np.any(z['scenario_od'][ib]&~z['scenario_od'][ia])
 check('entrance_station_section_arc_sets_nested',nested);check('nested_closures_have_nonincreasing_reachability',reachnested)
 # Recompute every saved cohort/ward row and per-probe classification.
 rows=read(R/'accessibility_summary.csv');summaryok=True
 for row in rows:
  i=int(row['scenario_index']);before=base[i];after=z['scenario_od'][i];bm=before[0]&before[1];mut=after[0]&after[1];use=np.array([p['cohort']==row['cohort'] and (row['ward']=='all' or p['ward']==row['ward']) for p in probes]);n0=int(bm[use].sum());lost=bm&~mut;new=bm.any(axis=1)&~mut.any(axis=1);outlost=before[0].any(axis=1)&~after[0].any(axis=1);backlost=before[1].any(axis=1)&~after[1].any(axis=1)
  vals=dict(origin_probes=int(use.sum()),baseline_covered_probes=int(bm[use].any(axis=1).sum()),baseline_unknown_probes=int((~bm[use].any(axis=1)).sum()),baseline_mutual_od=n0,lost_outward_od=int((before[0][use]&~after[0][use]).sum()),lost_return_od=int((before[1][use]&~after[1][use]).sum()),lost_mutual_od=int(lost[use].sum()),new_all_roundtrip_loss_probes=int(new[use].sum()),new_no_outward_anchor_probes=int(outlost[use].sum()),new_no_return_anchor_probes=int(backlost[use].sum()),new_both_directions_lost_probes=int((outlost[use]&backlost[use]).sum()))
  summaryok &= all(int(row[k])==v for k,v in vals.items()) and (not n0 or abs(float(row['lost_mutual_fraction'])-vals['lost_mutual_od']/n0)<1e-12)
 check('every_cohort_ward_direction_and_denominator_recomputed',summaryok)
 details=read(R/'probe_accessibility.csv');pindex={p['probe_id']:j for j,p in enumerate(probes)};detailok=True
 for r in details:
  i=int(r['scenario_index']);j=pindex[r['probe_id']];bm=base[i,0,j]&base[i,1,j];mut=z['scenario_od'][i,0,j]&z['scenario_od'][i,1,j];p=probes[j]
  detailok &= int(r['remaining_roundtrip_anchors'])==int(mut.sum()) and int(r['baseline_roundtrip_anchors'])==int(bm.sum()) and int(r['lost_roundtrip_anchors'])==int((bm&~mut).sum()) and (r['new_all_roundtrip_loss']=='True')==bool(bm.any() and not mut.any()) and (r['baseline_unknown']=='True')==bool(not bm.any()) and float(r['x_m'])==float(p['x_m']) and float(r['y_m'])==float(p['y_m']) and r['ward']==p['ward']
 check('every_probe_classification_and_location_recomputed',detailok)
 candidates=read(R/'rail_access_island_candidates.csv');expected=[r for r in details if r['cohort']=='surface_grid' and r['new_all_roundtrip_loss']=='True' and r['depth_bound']=='lower' and r['network_branch']==branches[1]]
 check('candidate_inventory_exact_without_baseline_gaps',candidates==expected and all(r['baseline_unknown']=='False' for r in candidates))
 geo=json.loads((P/'probe_accessibility_6677.geojson').read_text());check('geospatial_output_contains_points_not_island_polygons',all(f['geometry']['type']=='Point' for f in geo['features']))
 check('no_population_values_in_spatial_output',all(not any(k in f['properties'] for k in ['official_2020_residents','point_mean','envelope_lower','envelope_upper','affected_people','trapped_people']) and f['properties']['population_weighting'] is False for f in geo['features']))
 page=(R/'review.html').read_text();links=re.findall(r'(?:src|href)="([^"]+)"',page);broken=[l for l in links if not (R/l).resolve().is_file() and not l.endswith('independent_validation.json')]
 check('all_existing_report_links_resolve',not broken)
 check('required_interpretation_limits_present',all(t in page for t in ['不一定同时','68/209','不是人数','未计算人数','公交替代','原归属','补充归属','不是孤岛边界']))
 dump(R/'independent_validation.json',dict(all_passed=True,checks=checks,method='Independent edge-semantic reconstruction and unweighted shortest paths for all48 directed cases; all tables reaggregated from OD arrays.',new_hydraulics=False,people_counted=False,sealed_tests_opened=False))
 check('final_report_links_resolve',all((R/l).resolve().is_file() for l in links))
 dump(R/'independent_validation.json',dict(all_passed=True,checks=checks,method='Independent edge-semantic reconstruction and unweighted shortest paths for all48 directed cases; all tables reaggregated from OD arrays.',new_hydraulics=False,people_counted=False,sealed_tests_opened=False))
 files=[f for folder in [P,R] for f in folder.rglob('*') if f.is_file() and f.name!='delivery_manifest.json']+[ROOT/'src/analyses'/name for name in ['compare_historical_transport_islands_v1.py','report_historical_transport_islands_v1.py','validate_historical_transport_islands_v1.py','compare_transport_shutdown_scenarios_v1.py']]
 dump(R/'delivery_manifest.json',dict(files_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},input_bindings_sha256=sha(P/'source_bindings.json'),human_decision='MILI-D-20261017-001',checks_passed=len(checks),new_hydraulics=False,people_counted=False))
 print(json.dumps(dict(all_passed=True,checks=len(checks),candidate_records=len(candidates),report=str(R/'review.html')),ensure_ascii=False,indent=2))
def math_floor(x):return int(np.floor(x))
if __name__=='__main__':main()
