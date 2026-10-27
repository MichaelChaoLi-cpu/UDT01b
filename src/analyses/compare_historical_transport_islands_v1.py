"""Historical flood envelopes -> declared rail closures -> paired regional access.
No hydraulic run, learning, sealed-test access or population weighting.
"""
from pathlib import Path
import os, sys, json, csv, hashlib, math, collections, time
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']: os.environ.pop(k,None)
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order
from shapely import STRtree
from shapely.geometry import Point, shape, box, mapping
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses'))
from compare_transport_shutdown_scenarios_v1 import closure_mask, semantic_checks
N=ROOT/'data/processed/three_ward_transport_shutdown_scenarios_v1'
H=ROOT/'data/processed/three_ward_historical_radar_experiment_v2'
HR=ROOT/'data/exp/estimation/three_ward_historical_radar_experiment_v2'
S=ROOT/'data/processed/three_ward_station_surface_risk_v1'
P=ROOT/'data/processed/three_ward_historical_transport_islands_v1'
R=ROOT/'data/exp/estimation/three_ward_historical_transport_islands_v1'
CASES=['radar_low','radar_raised'];TYPES=['entrance','station','section']
BRANCHES=['intra_operator_hubs_only','provider_transfers_assumed_bidir']
POLICIES=['original_identity','source_normalized_identity']
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def source_paths():
 files=[N/n for n in ['scenario_policy.json','hypothetical_network.npz','hypothetical_network_edges.json','access_probes.csv','portal_inventory.csv','bounded_walk_legs.csv','station_units.csv','stop_station_memberships.json','identity_attribution_sensitivity_links.csv','source_bindings.json','paired_accessibility_matrices.npz','transfer_evidence_and_assumptions.csv','section_span_audit.csv']]
 files += [S/n for n in ['entrance_screen_summary.csv','station_entrance_identity_links.csv','station_screen_summary.csv']]
 files += [H/'paired_exposure_bounds.npz',H/'experiment_contract.json',H/'solver_binding.json',H/'run_manifest.json',HR/'entrance_exposure.csv',HR/'independent_validation.json',HR/'runtime_qualification.json']
 files += [H/'execution'/(c+'_candidate_reference.npz') for c in CASES]
 files += [ROOT/'data/processed/three_ward_minimum_scenarios_v1'/n for n in ['catalog.json','partition_manifest.json']]
 files += [ROOT/'data/processed/three_ward_reference_expansion_12_4_v1/selection.json']
 files += [ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz', ROOT/'data/processed/three_ward_population_block_stability_v1/block_population_stability_6677.geojson',ROOT/'data/exp/data-preprocessing/three_ward_population_block_stability_v1/delivery_manifest.json']
 return files

def main():
 start=time.monotonic();P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 assert json.loads((HR/'independent_validation.json').read_text())['all_passed']
 sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths()}
 original=json.loads((N/'scenario_policy.json').read_text())
 policy={'human_decision':'MILI-D-20261017-001','normal_network':'Frozen existing directed layered graph, source-native <=600m initial/final pedestrian legs, rail operating arcs and source-supported station identities. No new edges or road closures.',
 'parent_id':'HIST-RADAR-TOKYO-20240821','partition':'external_historical_candidate; outside original 12+4 and sealed tests',
 'forcing':'2024-08-21 15:00-24:00 JST archived radar-derived representative intensity; approximate1km/5min, not observed interval totals or observed inundation.',
 'river_stage':'radar_low uses assumed low stages; radar_raised imposes+1.5m ramp/plateau/recession in seven frozen groups. Neither is an observed hydrograph.',
 'primary_exposure_rule':original['primary_exposure_rule'],'shutdown_types':original['shutdown_types'],
 'temporal_assumption':'Per-case static joint closure selected from non-simultaneous cell event peaks. Hypothetical stress envelope only; no real closure timing, duration or simultaneous-isolation claim.',
 'quality':'Same common admitted open-ground support in both cases; at least100m2 and80% of eligible20m cell-centre disk. Quarantine, buildings, water and missing data are not dry.',
 'identity_policies':{'original_identity':'Frozen original surface-screen entrance identity links; no nearest-station assignment.','source_normalized_identity':'Union with frozen source stop-area name/operator normalization, distance<=500m to same-identity official geometry. Identity sensitivity, not verified internal circulation.'},
 'interval_policy':'Select using lower and upper depth bounds separately under the identical0.3m/100m2 rule. Differences measure numerical-interval sensitivity only; not full uncertainty bounds.',
 'network_branches':BRANCHES,'anchors':original['anchors'],'baseline':'Reproduce the existing normal baseline in each branch. Report only newly lost OD; never relabel baseline gaps as storm loss.',
 'island_candidate':'A fixed500m-cell ground probe previously able to make a round trip to at least one declared external anchor now has no such anchor. Also report outward and return direction separately. This is a rail-access candidate, not total street or city isolation.',
 'area_context':'Names describe official towns intersecting the fixed display cell; no claim the whole town or500m cell is isolated. No island area is estimated.',
 'other_modes':original['other_modes'],'population':'Frozen official night block baseline and allocation uncertainty remain referenced and unmodified; no numerical population values joined, no people counted. Daytime population not invented.',
 'epochs':'City geometry retained; official station lines2025 and OSM edit snapshot2026-10-15 do not reconstruct full2024 network.',
 'new_hydraulics':False,'training':False,'sealed_tests_opened':False,'people_counted':False}
 dump(P/'comparison_contract.json',policy);dump(P/'source_bindings.json',sources)
 z=np.load(N/'hypothetical_network.npz');keys=list(map(str,z['node_ids']));ix={k:i for i,k in enumerate(keys)};u=z['from_index'];v=z['to_index']
 edges=json.loads((N/'hypothetical_network_edges.json').read_text());probes=[p for p in read(N/'access_probes.csv') if p['cohort'] in ['surface_grid','core_station']]
 old=np.load(N/'paired_accessibility_matrices.npz');assert [p['probe_id'] for p in probes]==list(old['origin_probe_ids']);anchors=list(map(str,old['anchor_station_ids']))
 oi=np.array([ix['O:'+p['probe_id']] for p in probes]);di=np.array([ix['D:'+p['probe_id']] for p in probes]);ao=[ix['O:station:'+s] for s in anchors];ad=[ix['D:station:'+s] for s in anchors]
 active={BRANCHES[0]:~z['provider_transfer'],BRANCHES[1]:np.ones(len(edges),bool)}
 def od(keep):
  g=coo_matrix((np.ones(int(keep.sum()),np.int32),(u[keep],v[keep])),shape=(len(keys),len(keys))).tocsr();gt=g.T.tocsr();out=np.zeros((len(probes),len(anchors)),bool);back=out.copy()
  for j in range(len(anchors)):
   a=np.zeros(len(keys),bool);a[breadth_first_order(gt,ad[j],directed=True,return_predecessors=False)]=True;out[:,j]=a[oi]
   a[:]=False;a[breadth_first_order(g,ao[j],directed=True,return_predecessors=False)]=True;back[:,j]=a[di]
  return np.stack([out,back])
 baselines=np.stack([od(active[b]) for b in BRANCHES]);assert np.array_equal(baselines,old['baseline'])
 links={p:collections.defaultdict(set) for p in POLICIES}
 for row in read(S/'station_entrance_identity_links.csv'):
  for p in POLICIES:links[p][row['osm_node_id']].add(row['station_id'])
 for row in read(N/'identity_attribution_sensitivity_links.csv'):links[POLICIES[1]][row['osm_node_id']].add(row['station_id'])
 portals={p['portal_id']:p for p in read(N/'portal_inventory.csv')}
 bounds=np.load(H/'paired_exposure_bounds.npz');support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')
 eligible=support['Eligible Ground Mask'];good=bounds['common_admitted'];area=bounds['ground_area'];lower=bounds['ground_lower'];upper=bounds['ground_upper']
 assert np.all(lower[:,good]<=upper[:,good]) and np.isfinite(upper[:,good]).all()
 exposure=[];bycase=collections.defaultdict(list)
 for e in read(S/'entrance_screen_summary.csv'):
  x,y=float(e['x_m']),float(e['y_m']);row=math.floor((-32320-y)/10);col=math.floor((x+11360)/10)
  r0,r1=max(0,row-3),min(1012,row+4);c0,c1=max(0,col-3),min(770,col+4)
  ri=ci=np.array([],int)
  if r1>r0 and c1>c0:
   rr,cc=np.mgrid[r0:r1,c0:c1];use=((-11360+(cc+.5)*10-x)**2+(-32320-(rr+.5)*10-y)**2<=400)&eligible[rr,cc];ri,ci=rr[use],cc[use]
  total=float(area[ri,ci].sum());ok=good[ri,ci];ri,ci=ri[ok],ci[ok];admitted=float(area[ri,ci].sum());quality=total>0 and admitted>=100 and admitted/total>=.8
  for j,case in enumerate(CASES):
   lo=float(area[ri,ci][lower[j,ri,ci]>=.3].sum());up=float(area[ri,ci][upper[j,ri,ci]>=.3].sum())
   rec=dict(case_id=case,osm_node_id=e['osm_node_id'],x_m=x,y_m=y,eligible_ground_m2=total,admitted_ground_m2=admitted,quality_pass=quality,lower_exceedance_m2=lo,upper_exceedance_m2=up,lower_trigger=quality and lo>=100,upper_trigger=quality and up>=100,observed_ingress=False,observed_shutdown=False)
   exposure.append(rec);bycase[case].append(rec)
 write(P/'entrance_exposure.csv',exposure)
 previous={e['osm_node_id']:e for e in read(HR/'entrance_exposure.csv')}
 for j,case in enumerate(CASES):
  name=['low','raised'][j]
  assert all(e['quality_pass']==(previous[e['osm_node_id']]['neighborhood_quality']=='True') and e['lower_trigger']==(previous[e['osm_node_id']][name+'_screen_trigger']=='True') and abs(e['lower_exceedance_m2']-float(previous[e['osm_node_id']][name+'_certain_0p3_footprint_m2']))<1e-8 for e in bycase[case])
 # Use only town names and geometries for context; do not join population values.
 blocks=json.loads((ROOT/'data/processed/three_ward_population_block_stability_v1/block_population_stability_6677.geojson').read_text())['features']
 geoms=[shape(f['geometry']) for f in blocks];tree=STRtree(geoms);context=[]
 for p in probes:
  x,y=float(p['x_m']),float(p['y_m']);cell=box(math.floor(x/500)*500,math.floor(y/500)*500,(math.floor(x/500)+1)*500,(math.floor(y/500)+1)*500)
  names=sorted({blocks[int(i)]['properties']['name'] for i in tree.query(cell,predicate='intersects')})
  context.append(';'.join(names))
 selections=[];ledger=[];index=[];summaries=[];details=[];removed_all=[];ods=[];cache={};checks=semantic_checks()
 for case in CASES:
  for ident in POLICIES:
   for bound in ['lower','upper']:
    closed=set();targets=set();unassigned=0
    for e in bycase[case]:
     ids=links[ident][e['osm_node_id']];ep='osm:'+e['osm_node_id'];trigger=e[bound+'_trigger'];qualifies=bool(ids) and ep in portals
     if trigger and qualifies:closed.add(ep);targets|=ids
     if trigger and not qualifies:unassigned+=1
     ledger.append(dict(case_id=case,identity_policy=ident,depth_bound=bound,osm_node_id=e['osm_node_id'],quality_pass=e['quality_pass'],exceedance_footprint_m2=e[bound+'_exceedance_m2'],exposure_trigger=trigger,associated_station_ids=';'.join(sorted(ids)),selected_portal=trigger and qualifies,unassigned_trigger=trigger and not qualifies,closure_is_hypothesis=True))
    sel=dict(case_id=case,parent_id=policy['parent_id'],partition='external_historical_candidate',identity_policy=ident,depth_bound=bound,quality_passed=sum(e['quality_pass'] for e in bycase[case]),exposure_trigger_points=sum(e[bound+'_trigger'] for e in bycase[case]),selected_portals=len(closed),target_station_units=len(targets),unassigned_trigger_points=unassigned,selected_portal_ids=';'.join(sorted(closed)),target_station_ids=';'.join(sorted(targets)))
    selections.append(sel)
    masks={kind:closure_mask(edges,kind,closed,targets) for kind in TYPES}
    assert not np.any(masks['entrance']&~masks['station']) and not np.any(masks['station']&~masks['section'])
    for kind,removed in masks.items():
     for ib,branch in enumerate(BRANCHES):
      keep=active[branch]&~removed;sig=hashlib.sha256(keep.tobytes()).hexdigest()
      if sig not in cache:cache[sig]=od(keep)
      current=cache[sig];base=baselines[ib];assert not np.any(current&~base)
      bm=base[0]&base[1];mut=current[0]&current[1];lost=bm&~mut;alllost=bm.any(axis=1)&~mut.any(axis=1)
      sid='__'.join([case,ident,bound,kind,branch]);idx=len(index)
      index.append(dict(scenario_index=idx,scenario_id=sid,case_id=case,identity_policy=ident,depth_bound=bound,shutdown_type=kind,network_branch=branch));ods.append(current);removed_all.append(removed&active[branch])
      for cohort in ['surface_grid','core_station']:
       for ward in ['all','13101','13102','13103','boundary_or_unassigned']:
        use=np.array([p['cohort']==cohort and (ward=='all' or p['ward']==ward) for p in probes]);n=int(use.sum())
        if not n:continue
        num=int(bm[use].sum());outlost=base[0].any(axis=1)&~current[0].any(axis=1);backlost=base[1].any(axis=1)&~current[1].any(axis=1)
        summaries.append(dict(**index[-1],cohort=cohort,ward=ward,origin_probes=n,baseline_covered_probes=int(bm[use].any(axis=1).sum()),baseline_unknown_probes=int((~bm[use].any(axis=1)).sum()),baseline_mutual_od=num,lost_outward_od=int((base[0][use]&~current[0][use]).sum()),lost_return_od=int((base[1][use]&~current[1][use]).sum()),lost_mutual_od=int(lost[use].sum()),lost_mutual_fraction=float(lost[use].sum()/num) if num else None,new_all_roundtrip_loss_probes=int(alllost[use].sum()),new_no_outward_anchor_probes=int(outlost[use].sum()),new_no_return_anchor_probes=int(backlost[use].sum()),new_both_directions_lost_probes=int((outlost[use]&backlost[use]).sum()),removed_arcs=int((removed&active[branch]).sum()),removed_service_arcs=sum(bool(removed[i]&active[branch][i]) for i,e in enumerate(edges) if e['edge_kind']=='rail_service'),people_counted=False))
      for i,p in enumerate(probes):
       details.append(dict(**index[-1],probe_id=p['probe_id'],cohort=p['cohort'],ward=p['ward'],x_m=float(p['x_m']),y_m=float(p['y_m']),baseline_roundtrip_anchors=int(bm[i].sum()),remaining_roundtrip_anchors=int(mut[i].sum()),remaining_outward_anchors=int(current[0,i].sum()),remaining_return_anchors=int(current[1,i].sum()),lost_roundtrip_anchors=int(lost[i].sum()),new_all_roundtrip_loss=bool(alllost[i]),baseline_unknown=not bool(bm[i].any()),towns_intersecting_display_cell=context[i],context_not_whole_town_isolation=True,population_weighting=False))
 print('COMPARISONS',len(index),'unique_graphs',len(cache),flush=True)
 write(P/'closure_selection.csv',selections);write(P/'evidence_to_hypothesis_ledger.csv',ledger);write(P/'scenario_index.csv',index);write(R/'accessibility_summary.csv',summaries);write(R/'probe_accessibility.csv',details)
 np.savez_compressed(P/'paired_accessibility.npz',scenario_od=np.stack(ods),baseline_od=baselines,removed_arcs=np.stack(removed_all),origin_probe_ids=np.array([p['probe_id'] for p in probes]),anchor_station_ids=np.array(anchors),baseline_branch_names=np.array(BRANCHES))
 # Candidate inventory and locations, keeping point samples distinct from area boundaries.
 candidates=[r for r in details if r['cohort']=='surface_grid' and r['new_all_roundtrip_loss'] and r['depth_bound']=='lower' and r['network_branch']==BRANCHES[1]]
 write(R/'rail_access_island_candidates.csv',candidates)
 dump(P/'probe_accessibility_6677.geojson',dict(type='FeatureCollection',name='Historical hypothetical rail accessibility probes; not people or island boundaries',crs=dict(type='name',properties=dict(name='EPSG:6677')),features=[dict(type='Feature',geometry=mapping(Point(r['x_m'],r['y_m'])),properties={k:v for k,v in r.items() if k not in ['x_m','y_m']}) for r in details if r['cohort']=='surface_grid' and r['depth_bound']=='lower' and r['network_branch']==BRANCHES[1]]))
 checks += [dict(check='normal_baseline_identical_to_existing_paired_graph',pass_=True),dict(check='historical_entrance_screen_reproduced',pass_=True),dict(check='nested_closures_and_no_created_OD_paths',pass_=True),dict(check='all_source_bindings_unchanged',pass_=all(sha(ROOT/p)==h for p,h in sources.items())),dict(check='original_frozen_sources_unchanged',pass_=all(sha(ROOT/x['file'])==x['sha256'] for x in json.loads((N/'source_bindings.json').read_text())))]
 checks=[{'check':x['check'],'passed':x.get('pass',x.get('pass_',False))} for x in checks];assert all(c['passed'] for c in checks)
 dump(R/'comparison_validation.json',dict(all_passed=True,checks=checks));dump(R/'summary.json',dict(cases=CASES,comparisons=len(index),unique_graphs=len(cache),surface_grid_probes=sum(p['cohort']=='surface_grid' for p in probes),normal_surface_covered=int((baselines[1,0]&baselines[1,1])[np.array([p['cohort']=='surface_grid' for p in probes])].any(axis=1).sum()),selection=selections,primary_results=[x for x in summaries if x['cohort']=='surface_grid' and x['ward']=='all' and x['depth_bound']=='lower' and x['network_branch']==BRANCHES[1]],elapsed_s=time.monotonic()-start,new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False))
 print(json.dumps(json.loads((R/'summary.json').read_text()),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
