"""Whole-region normal-network validation and bounded routing probes.
Does not read flood responses or redistribute population. Run from repository.
"""
from pathlib import Path
import sys,json,csv,gzip,hashlib,datetime,collections,time
import numpy as np
from shapely.geometry import shape,Point
from shapely.ops import unary_union
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components,breadth_first_order
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_normal_transport_network_v1';R=ROOT/'data/exp/data-preprocessing/three_ward_normal_transport_network_v1'
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from normal_transport_rules_v1 import selfcheck,shortest_paths
from build_normal_rail_access_v1 import boarding_permissions
from build_normal_surface_network_v1 import dump,csvwrite

def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def digest(p,algorithm='sha256'):
 h=hashlib.new(algorithm)
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def partitions(u,v,n):
 sp=coo_matrix((np.ones(len(u),np.uint8),(u,v)),shape=(n,n)).tocsr()
 return {kind:connected_components(sp,directed=True,connection=kind)[1] for kind in ['weak','strong']}
def merge_count(small,big,core):
 # Number of distinct core small-extent groups absorbed by larger groups.
 pairs=set(zip(map(int,small[core]),map(int,big[core])));cc=collections.defaultdict(set)
 for s,b in pairs:cc[b].add(s)
 return sum(len(x)-1 for x in cc.values()),sum(len(x)>1 for x in cc.values())
def main():
 check=selfcheck()
 check += [{'check':'platform_entry_only_blocks_alighting','passed':boarding_permissions('stop','platform_entry_only')==(True,False)},{'check':'platform_exit_only_blocks_boarding','passed':boarding_permissions('stop','platform_exit_only')==(False,True)},{'check':'conflicting_stop_platform_roles_never_open_both','passed':boarding_permissions('stop_exit_only','platform_entry_only')==(False,False)}]
 assert all(x['passed'] for x in check)
 dump(R/'rule_selfchecks.json',check)
 sources=P/'sources';archive=sources/'kanto-261015.osm.pbf';expected=(sources/'kanto-261015.osm.pbf.md5').read_text().split()[0];md5=digest(archive,'md5');assert md5==expected
 receipt=json.loads((P/'osm_extraction_receipt.json').read_text());ts=receipt['header']['replication_timestamp']
 dump(P/'source_acquisition_manifest.json',{'archive':archive.name,'url':'https://download.geofabrik.de/asia/japan/kanto-261015.osm.pbf','bytes':archive.stat().st_size,'sha256':digest(archive),'provider_md5':expected,'computed_md5':md5,'provider_md5_verified':True,'replication_utc':datetime.datetime.fromtimestamp(ts,datetime.timezone.utc).isoformat(),'retrieved_utc_date':'2026-10-16','license':'ODbL 1.0, OpenStreetMap contributors; regional extract by Geofabrik','snapshot_is_not_survey_or_current_legal_access_date':True,'primary_schema':'https://github.com/openstreetmap/OSM-binary/blob/master/osmpbf/osmformat.proto','no_contributor_personal_metadata_stored':True})
 prior=[]
 for folder in ['data/exp/feasibility-check/three_ward_transport_framework_v1','data/exp/data-preprocessing/three_ward_population_block_stability_v1']:
  mf=ROOT/folder/'delivery_manifest.json';m=json.loads(mf.read_text());items=m['products']+m.get('scripts',[]);fail=[]
  for it in items:
   p=ROOT/it['file']
   if not p.exists() or digest(p)!=it['sha256']:fail.append(it['file'])
  prior.append({'manifest':str(mf.relative_to(ROOT)),'sha256':digest(mf),'checked_files':len(items),'failed':fail})
 assert all(not x['failed'] for x in prior)
 surface=readcsv(P/'surface_nodes.csv');coord={r['node_id']:(float(r['x_m']),float(r['y_m'])) for r in surface};smeta={r['node_id']:r for r in surface}
 blocks=json.loads((ROOT/'data/processed/three_ward_population_block_stability_v1/block_population_stability_6677.geojson').read_text())['features'];ward={w:unary_union([shape(f['geometry']) for f in blocks if f['properties']['ward']==w]) for w in ['13101','13102','13103']};wardnames={'13101':'Chiyoda','13102':'Chuo','13103':'Minato'}
 official=json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/stations_regional_6677.geojson').read_text())['features'];dest={}
 for name in ['新宿','渋谷','上野','錦糸町','大井町','豊洲']:
  ff=[f for f in official if f['properties']['N02_005']==name];assert ff,name
  dest[name]=unary_union([shape(f['geometry']) for f in ff]).centroid
 extent=[];probes=[];anchors=[];graphchecks=[]
 for mode in ['walk','drive']:
  z=np.load(P/(mode+'_conservative_graph.npz'));keys=z['node_ids'];u=z['from_index'];v=z['to_index'];length=z['length_m'];ways=z['osm_way_id'];n=len(keys);mask=z['inside_5km'];core=z['inside_core'];xy=np.array([coord[str(k)] for k in keys]);parts5=partitions(u[mask],v[mask],n)
  active5=np.zeros(n,bool);active5[u[mask]]=True;active5[v[mask]]=True
  for kind,basekey in [('weak','weak_component'),('strong','strong_component')]:
   small,big=parts5[kind],z[basekey];merges,bigmerged=merge_count(small,big,core&active5);extent.append({'mode':mode,'partition_kind':kind,'core_nodes':int(core.sum()),'core_nodes_missing_in_5km':int(np.sum(core&~active5)),'core_groups_5km':len(set(small[core&active5])),'core_groups_10km':len(set(big[core&active5])),'core_groups_merged_by_outer_5_to_10km':merges,'larger_groups_receiving_merges':bigmerged,'turn_memory_in_partition':False,'population_measure':False})
  # Isolated graph vertices and zero-length portals are allowed; negative or
  # non-finite lengths and cross-native portals are not.
  iz=ways==0;portal_same=all(str(keys[a]).split(':')[1]==str(keys[b]).split(':')[1] for a,b in zip(u[iz],v[iz]));held=set(json.loads((P/'held_turn_way_ids.json').read_text()))
  graphchecks.append({'graph':mode+'_conservative','indices_valid':bool(np.all(u>=0) and np.all(v>=0) and np.all(u<n) and np.all(v<n)),'lengths_finite_nonnegative':bool(np.isfinite(length).all() and (length>=0).all()),'nonportal_lengths_positive':bool((length[~iz]>0).all()),'portals_same_native_id':portal_same,'held_turn_source_ways_absent':bool(not mode=='drive' or not held&set(map(int,ways)))})
  maincomp=int(np.bincount(z['weak_component'][core]).argmax());candidate=np.where(active5&(z['weak_component']==maincomp)&np.array([smeta[str(k)]['grade_key']=='0|0|surface' for k in keys]))[0];tree=cKDTree(xy[candidate]);origin=[];targets=[]
  for w,g in ward.items():
   p=g.representative_point();di,j=tree.query([p.x,p.y]);ix=int(candidate[j]);origin.append((wardnames[w],ix));anchors.append({'mode':mode,'kind':'synthetic_QA_origin_not_population','name':wardnames[w],'node_id':str(keys[ix]),'distance_to_reference_m':float(di),'x_m':float(xy[ix,0]),'y_m':float(xy[ix,1]),'no_station_connection_added':True})
  for name,p in dest.items():
   di,j=tree.query([p.x,p.y]);ix=int(candidate[j]);targets.append((name,ix));anchors.append({'mode':mode,'kind':'external_station_vicinity_QA_not_station_entrance','name':name,'node_id':str(keys[ix]),'distance_to_reference_m':float(di),'x_m':float(xy[ix,0]),'y_m':float(xy[ix,1]),'no_station_connection_added':True})
  restrictions=json.loads((P/'supported_motorcar_restrictions.json').read_text()) if mode=='drive' else []
  # No departure chosen: normal base restrictions only. Conditional rules are
  # exported separately; an explicit stress branch is not a legal calendar.
  for halo in [5,10]:
   select=mask if halo==5 else np.ones(len(u),bool);adj=collections.defaultdict(list)
   for a,b,l,w in zip(u[select],v[select],length[select],ways[select]):adj[str(keys[a])].append((str(keys[b]),float(l),int(w)))
   for name,ix in origin:
    start=time.monotonic();out,meta=shortest_paths(adj,str(keys[ix]),[str(keys[j]) for _,j in targets],restrictions,max_states=600000);elapsed=time.monotonic()-start
    for dn,j in targets:
     found=str(keys[j]) in out;probes.append({'mode':mode,'road_halo_km':halo,'origin_ward':name,'destination_station_vicinity':dn,'origin_node':str(keys[ix]),'target_node':str(keys[j]),'distance_m':out.get(str(keys[j]),''),'status':'found' if found else 'unknown_state_budget' if meta['state_limit_hit'] else 'unreachable_in_candidate_graph','turn_rules_applied':len(restrictions),'conditional_turn_calendar_resolved':False,'visited_states':meta['states'],'state_budget':600000,'search_seconds':elapsed,'not_population_or_travel_time':True})
    print('PROBE',mode,halo,name,len(out),'of',len(targets),meta,'seconds',round(elapsed,2),flush=True)
   del adj
  del z,keys,u,v,length,ways
 csvwrite(R/'road_extent_comparison.csv',extent);csvwrite(R/'routing_probe_checks.csv',probes);csvwrite(R/'routing_probe_anchors.csv',anchors)
 # Rail extent check retains only regional stop occurrences within 10km,
 # versus the 20km inventory. Access/walk graph stays fixed at 10km.
 coregeom=unary_union(list(ward.values()));rail=readcsv(P/'rail_service_candidate_nodes.csv');railcoord={r['node_id']:(float(r['x_m']),float(r['y_m'])) for r in rail};railcore={r['node_id'] for r in rail if r['inside_core']=='True'};railext=[];multicheck=[];railreach=[]
 for branch in ['source_path','conditional_internal']:
  z=np.load(P/(branch+'_multimodal_graph.npz'));keys=z['node_ids'];u=z['from_index'];v=z['to_index'];n=len(keys);railflag=np.array([str(k).startswith('rail:') for k in keys]);inside10=np.array([not str(k).startswith('rail:') or coregeom.distance(Point(railcoord[str(k)]))<=10000 for k in keys]);select=inside10[u]&inside10[v];small=partitions(u[select],v[select],n);big=partitions(u,v,n);core=np.array([smeta[str(k)]['inside_core']=='True' if str(k) in smeta else str(k) in railcore for k in keys])
  for kind in ['weak','strong']:
   merges,merged=merge_count(small[kind],big[kind],core);railext.append({'branch':branch,'partition_kind':kind,'core_nodes_and_stop_occurrences':int(core.sum()),'core_groups_with_10km_rail':len(set(small[kind][core])),'core_groups_with_20km_rail':len(set(big[kind][core])),'core_groups_merged_by_outer_rail':merges,'station_circulation_is_aggregate_assumption':branch=='conditional_internal','service_calendar_verified':False})
  sp=coo_matrix((np.ones(len(u),np.uint8),(u,v)),shape=(n,n)).tocsr();sprev=sp.T.tocsr();ix={str(k):i for i,k in enumerate(keys)}
  for origin in [x for x in anchors if x['mode']=='walk' and x['kind']=='synthetic_QA_origin_not_population']:
   out=np.zeros(n,bool);back=np.zeros(n,bool);out[breadth_first_order(sp,ix[origin['node_id']],directed=True,return_predecessors=False)]=True;back[breadth_first_order(sprev,ix[origin['node_id']],directed=True,return_predecessors=False)]=True
   for op in ['jr','metro','toei','other']:
    stops=[r for r in rail if r['inside_core']=='True' and r['operator_class']==op];ri=[ix[r['node_id']] for r in stops if r['node_id'] in ix];missing=sum(r['node_id'] not in ix for r in stops)
    railreach.append({'branch':branch,'origin_ward':origin['name'],'operator_class':op,'core_stop_occurrences':len(stops),'stop_occurrences_absent_from_active_graph':missing,'reachable_from_road_origin':int(out[ri].sum()),'can_return_to_road_origin':int(back[ri].sum()),'reachable_both_directions':int((out[ri]&back[ri]).sum()),'potential_service_calendar_unverified':True,'not_actual_station_closures':True})
  multicheck.append({'graph':branch,'indices_valid':bool((u>=0).all() and (v>=0).all() and (u<n).all() and (v<n).all()),'nodes':n,'arcs':len(u),'rail_stop_occurrences_with_any_arc':int(railflag.sum())})
 csvwrite(R/'rail_extent_comparison.csv',railext);csvwrite(R/'multimodal_directional_reachability.csv',railreach)
 # Direct rail edges must be consecutive occurrences in a declared source
 # route. No missing middle stop is skipped and no reverse is fabricated.
 edges=readcsv(P/'rail_service_candidate_edges.csv');railorder=all(int(e['to_node'].split(':')[2])==int(e['from_node'].split(':')[2])+1 and e['to_node'].split(':')[1]==e['from_node'].split(':')[1] for e in edges)
 floors=readcsv(P/'held_station_access_floor_conflicts.csv');near=readcsv(P/'held_entrance_station_match_candidates.csv');cross=readcsv(P/'core_planar_crossing_audit.csv');endpoints=readcsv(P/'core_near_endpoint_candidates.csv');support=readcsv(P/'block_network_support_audit.csv')
 identity=readcsv(P/'official_station_identity_audit.csv');identitycounts=collections.Counter((r['operator'],r['match_state']) for r in identity if r['inside_core']=='True');csvwrite(R/'official_identity_by_operator.csv',[{'operator':op,'match_state':st,'station_line_records':count} for (op,st),count in sorted(identitycounts.items())])
 # Check independent API coordinates only as a parser spot check, not survey
 # validation or proof that all OSM coordinates are accurate.
 api=json.loads((sources/'independent_osm_node_coordinate_checks.json').read_text());apiok=all(x.get('coordinate_agrees_to_1e_7_degree') for x in api)
 validation={'human_decision':'MILI-D-20261016-021','source_md5_verified':md5==expected,'rule_selfchecks_passed':all(x['passed'] for x in check),'rule_selfcheck_count':len(check),'independent_coordinate_spotcheck_count':len(api),'independent_coordinate_spotchecks_passed':apiok,'all_selected_way_nodes_available':receipt['all_selected_way_nodes_present'],'unresolved_raw_relation_node_members_not_leapfrogged':receipt['unresolved_member_node_ids'],'surface_graph_checks':graphchecks,'multimodal_graph_checks':multicheck,'rail_edges_all_source_consecutive_forward':railorder,'no_nearest_station_edges_activated':all(r['graph_edge_added']=='False' for r in near),'no_planar_or_near_endpoint_edges_activated':all(r['graph_edge_added']=='False' for r in cross+endpoints),'floor_conflicts_held_count':len(floors),'floor_conflicts_not_activated':all(r['graph_edge_added']=='False' for r in floors),'block_count_preserved':len(support)==len(blocks)==4746,'population_to_components_not_assigned':all(r['population_to_components_assigned']=='False' for r in support),'population_and_framework_manifests':prior,'original_block_baseline_total':sum(f['properties']['official_2020_residents'] for f in blocks),'new_hydraulic_runs':False,'training':False,'sealed_responses_opened':False,'stranded_people_computed':False,'calendar_or_exact_legal_access_qualification':False}
 validation['technical_checks_passed']=all([validation['source_md5_verified'],validation['rule_selfchecks_passed'],apiok,railorder,validation['no_nearest_station_edges_activated'],validation['no_planar_or_near_endpoint_edges_activated'],validation['floor_conflicts_not_activated'],validation['block_count_preserved'],validation['population_to_components_not_assigned']]+[all(v for k,v in c.items() if k!='graph') for c in graphchecks]+[c['indices_valid'] for c in multicheck]);assert validation['technical_checks_passed']
 dump(R/'validation.json',validation);print('VALIDATION',json.dumps(validation,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
