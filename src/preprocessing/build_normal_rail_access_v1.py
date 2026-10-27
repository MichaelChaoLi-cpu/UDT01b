"""Build ordered rail-service candidates and explicit station access layers.
Exact source-shared pedestrian/platform links and station-area aggregate
assumptions are separate outputs. No nearest-station link is activated.
"""
from pathlib import Path
import sys,json,csv,gzip,collections,itertools,unicodedata,re
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point,shape,LineString
from shapely import STRtree
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/three_ward_normal_transport_network_v1';R=ROOT/'data/exp/data-preprocessing/three_ward_normal_transport_network_v1'
sys.path.insert(0,str(ROOT/'src/preprocessing'));from build_normal_surface_network_v1 import rows,dump,csvwrite,gzwrite
from normal_transport_rules_v1 import grade

def norm(s):return re.sub(r'[\s駅（）()・\-]','',unicodedata.normalize('NFKC',s or ''))
def opclass(t):
 x=' '.join(t.get(k,'') for k in ['operator','operator:ja','operator:en','network','name'])
 if any(k in x for k in ['東京メトロ','東京地下鉄','Tokyo Metro']):return 'metro'
 if any(k in x for k in ['都営','東京都交通局','Toei','Tokyo Metropolitan']):return 'toei'
 if any(k in x for k in ['JR','ＪＲ','東日本旅客','東海旅客','East Japan Railway','Central Japan Railway']):return 'jr'
 return 'other'
def operator_identity(t):
 c=opclass({k:t.get(k,'') for k in ['operator','operator:ja','operator:en','network']})
 if ';' in t.get('operator',''):return None
 if c in ['metro','toei']:return c
 x=' '.join(t.get(k,'') for k in ['operator','operator:ja','operator:en'])
 if any(k in x for k in ['JR-East','JR East','東日本','East Japan Railway']):return 'JR-East'
 if any(k in x for k in ['JR-Central','JR Central','東海','Central Japan Railway']):return 'JR-Central'
 return None
def boarding_permissions(stop_role,platform_role='platform'):
 return stop_role!='stop_exit_only' and platform_role!='platform_exit_only',stop_role!='stop_entry_only' and platform_role!='platform_entry_only'
def main():
 raw=rows(OUT/'osm_nodes.jsonl');xy=Transformer.from_crs(4326,6677,always_xy=True);xx,yy=xy.transform([r['lon'] for r in raw],[r['lat'] for r in raw]);nd={r['id']:{'x':float(x),'y':float(y),'tags':r['tags']} for r,x,y in zip(raw,xx,yy)};del raw
 ways={w['id']:w for w in rows(OUT/'osm_ways.jsonl')};rels=rows(OUT/'osm_relations.jsonl');core=shape(json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/inventory_domains_6677.geojson').read_text())['features'][0]['geometry']);region=core.buffer(20000)
 with(OUT/'surface_nodes.csv').open() as f:surface={r['node_id']:r for r in csv.DictReader(f)}
 walk=np.load(OUT/'walk_conservative_graph.npz');walknode=set(map(str,walk['node_ids']));nativewalk=collections.defaultdict(list)
 for id in walknode:nativewalk[int(id.split(':')[1])].append(id)
 held_levels=[]
 def compatible_walk(id,context=None,role='entrance'):
  result=[];context=context or {}
  for wk in nativewalk.get(id,[]):
   floor=wk.split(':',2)[2].split('|')[0];reason=None
   for tags in [nd.get(id,{}).get('tags',{}),context]:
    level=tags.get('level','').strip()
    if level and ';' not in level and floor!='?':
     try:conflict=float(level)!=float(floor)
     except ValueError:conflict=level!=floor
     if conflict:reason='explicit_node_or_platform_floor_conflicts_with_walk_way_floor'
   if reason:held_levels.append({'osm_node_id':id,'walk_node':wk,'role':role,'source_node_level':nd.get(id,{}).get('tags',{}).get('level',''),'platform_level':context.get('level',''),'issue':reason,'graph_edge_added':False})
   else:result.append(wk)
  return result
 areas={};nodeareas=collections.defaultdict(set);wayareas=collections.defaultdict(set)
 for r in rels:
  if r['tags'].get('public_transport')!='stop_area':continue
  # Filter bus-only groups: at least one railway station/stop/entrance/platform
  # member or an explicit rail mode. A stop_area does not itself imply transfers.
  railmembers=[m for m in r['members'] if m['type']=='node' and m['id'] in nd and nd[m['id']]['tags'].get('railway') in ['station','halt','stop','subway_entrance','train_station_entrance'] or m['type']=='way' and m['id'] in ways and ways[m['id']]['tags'].get('railway') in ['platform','station']]
  if not railmembers:continue
  areas[r['id']]=r
  for m in r['members']:
   if m['type']=='node':nodeareas[m['id']].add(r['id'])
   elif m['type']=='way':wayareas[m['id']].add(r['id'])
 entrances=[];entrancearea=collections.defaultdict(list)
 for id,n in nd.items():
  t=n['tags']
  if t.get('railway') not in ['subway_entrance','train_station_entrance']:continue
  p=Point(n['x'],n['y'])
  if not p.intersects(region):continue
  walkids=compatible_walk(id);aa=sorted(nodeareas[id]);e={'osm_node_id':id,'x_m':n['x'],'y_m':n['y'],'inside_core':core.covers(p),'source_type':t['railway'],'name':t.get('name',''),'exit_ref':t.get('ref',''),'operator_class':opclass(t),'relative_level':t.get('level',''),'walk_native_nodes':json.dumps(walkids),'source_station_area_ids':json.dumps(aa),'has_conservative_walk_path':bool(walkids),'has_declared_station_area':bool(aa),'current_opening_time_verified':False,'source_tags':json.dumps(t,ensure_ascii=False)};entrances.append(e)
  for a in aa:entrancearea[a].append(id)
 csvwrite(OUT/'osm_station_entrance_audit.csv',entrances)
 area_route_operators=collections.defaultdict(set)
 for rr in rels:
  if rr['tags'].get('type')!='route':continue
  identity=operator_identity(rr['tags'])
  if identity is None:continue
  for m in rr['members']:
   if m['type']=='node' and m['role'].startswith('stop'):
    for aid in nodeareas[m['id']]:area_route_operators[aid].add(identity)
 railnodes={};railedges=[];routeaudit=[];boarding=[];aggregate=[];unresolved=[];route_platform_pairs=[]
 for r in rels:
  t=r['tags']
  if t.get('type')!='route' or t.get('route') not in ['subway','train','monorail','light_rail','tram']:continue
  stops=[m for m in r['members'] if m['role'] in ['stop','stop_entry_only','stop_exit_only'] and m['type']=='node'];plats=[m for m in r['members'] if m['role'] in ['platform','platform_entry_only','platform_exit_only']]
  if not stops:routeaudit.append({'route_id':r['id'],'name':t.get('name',''),'operator_class':opclass(t),'ordered_stops':0,'stops_in_region':0,'core_stops':0,'status':'no_explicit_ordered_stops_no_coordinate_sorting'});continue
  valid=[]
  for i,m in enumerate(stops):
   n=nd.get(m['id']);inside=n is not None and region.covers(Point(n['x'],n['y']));valid.append(inside)
   if inside:
    key=f"rail:{r['id']}:{i}";railnodes[key]={'node_id':key,'route_id':r['id'],'stop_index':i,'source_stop_node_id':m['id'],'name':n['tags'].get('name',''),'x_m':n['x'],'y_m':n['y'],'operator_class':opclass(t),'route_ref':t.get('ref',''),'route_name':t.get('name',''),'stop_role':m['role'],'inside_core':core.covers(Point(n['x'],n['y'])),'source_station_area_ids':json.dumps(sorted(nodeareas[m['id']])),'calendar_or_timetable_verified':False}
  nreg=sum(valid);ncore=sum(railnodes[f"rail:{r['id']}:{i}"]['inside_core'] for i in range(len(stops)) if valid[i])
  if not nreg:continue
  routeaudit.append({'route_id':r['id'],'name':t.get('name',''),'operator_class':opclass(t),'ordered_stops':len(stops),'stops_in_region':nreg,'core_stops':ncore,'status':'ordered_source_route_candidate_no_calendar'})
  # Member order is directional evidence. Do not add reverse edges merely
  # because a route has only one source relation or skip unlocated middle stops.
  for i in range(len(stops)-1):
   if not(valid[i] and valid[i+1]):continue
   u,v=f"rail:{r['id']}:{i}",f"rail:{r['id']}:{i+1}";a,b=railnodes[u],railnodes[v];length=float(np.hypot(a['x_m']-b['x_m'],a['y_m']-b['y_m']))
   railedges.append({'edge_id':f"r{r['id']}:{i}",'from_node':u,'to_node':v,'route_id':r['id'],'length_chord_m_not_travel_time':length,'operator_class':opclass(t),'route_name':t.get('name',''),'service_state':'normal_static_potential_assumption','actual_operation_verified':False,'operator_raw':t.get('operator',''),'multiple_operator_source_tag':';' in t.get('operator',''),'source_order_retained':True})
  for i,m in enumerate(stops):
   if not valid[i]:continue
   key=f"rail:{r['id']}:{i}";sa=nodeareas[m['id']];candidates=[]
   # Equal stop/platform counts alone do not authorize pairing: their declared
   # station-area memberships must also agree before a source-ID path is used.
   if len(stops)==len(plats):
    pl=plats[i];pa=nodeareas[pl['id']] if pl['type']=='node' else wayareas[pl['id']] if pl['type']=='way' else set()
    if sa&pa:candidates=[pl]
   else:
    for pl in plats:
     pa=nodeareas[pl['id']] if pl['type']=='node' else wayareas[pl['id']] if pl['type']=='way' else set()
     if sa&pa:candidates.append(pl)
   boardset={}
   for pl in candidates:
    platformnodes=[pl['id']] if pl['type']=='node' else ways.get(pl['id'],{}).get('refs',[])
    for pn in platformnodes:
     for wk in compatible_walk(pn,nd.get(pl['id'],{}).get('tags',{}) if pl['type']=='node' else ways.get(pl['id'],{}).get('tags',{}),'platform'):
      ab,aa=boarding_permissions(m['role'],pl['role'])
      if ab or aa:
       old=boardset.get(wk,(False,False));boardset[wk]=(old[0] or ab,old[1] or aa)
   for wk in sorted(boardset):
    boarding.append({'from_node':wk,'to_node':key,'source_basis':'explicit_route_platform_and_stop_area_plus_shared_native_walk_node','kind':'board_alight_link','allow_board':boardset[wk][0],'allow_alight':boardset[wk][1],'source_route_id':r['id'],'physical_internal_path_invented':False,'station_internal_aggregate_assumption':False})
   if not boardset:unresolved.append({'route_id':r['id'],'rail_node':key,'source_stop_node_id':m['id'],'stop_name':railnodes[key]['name'],'inside_core':railnodes[key]['inside_core'],'issue':'no_explicit_platform_to_walk_connection','nearest_station_snapping_used':False})
   # A second, conditional graph can approximate normal internal circulation
   # from an entrance in the SAME declared station area. This is NOT a verified
   # walkway, fare-gate route or cross-operator transfer. Keep all such links out
   # of the source-path graph, even when it reduces apparent completeness.
   allowed_board,allowed_alight=boarding_permissions(m['role'])
   if candidates:
    pp=[boarding_permissions(m['role'],pl['role']) for pl in candidates];allowed_board=any(x[0] for x in pp);allowed_alight=any(x[1] for x in pp)
   if not (allowed_board or allowed_alight):continue
   for area in sa:
    ar=areas.get(area)
    if not ar:continue
    areaop=operator_identity(ar['tags'])
    if areaop is None and len(area_route_operators[area])==1:areaop=next(iter(area_route_operators[area]))
    for en in entrancearea.get(area,[]):
     ent=next(e for e in entrances if e['osm_node_id']==en)
     # Require declared operator compatibility; unknown operator cannot certify
     # a transfer. It remains a candidate needing provider data/diagram checks.
     if areaop is None or areaop!=operator_identity(t):continue
     for wk in json.loads(ent['walk_native_nodes']):aggregate.append({'from_node':wk,'to_node':key,'source_station_area_id':area,'source_route_id':r['id'],'kind':'conditional_station_internal_access','source_basis':'same_source_station_area_and_consistent_exact_operator_identity','allow_board':allowed_board,'allow_alight':allowed_alight,'physical_internal_path_invented':False,'station_internal_aggregate_assumption':True,'not_a_verified_transfer':True})
 csvwrite(OUT/'rail_service_candidate_nodes.csv',list(railnodes.values()));csvwrite(OUT/'rail_service_candidate_edges.csv',railedges);csvwrite(OUT/'rail_route_audit.csv',routeaudit);csvwrite(OUT/'source_platform_boarding_links.csv',boarding,fields=['from_node','to_node','source_basis','kind','allow_board','allow_alight','source_route_id','physical_internal_path_invented','station_internal_aggregate_assumption']);csvwrite(OUT/'conditional_station_access_links.csv',aggregate,fields=['from_node','to_node','source_station_area_id','source_route_id','kind','source_basis','allow_board','allow_alight','physical_internal_path_invented','station_internal_aggregate_assumption','not_a_verified_transfer']);csvwrite(OUT/'held_rail_access_gaps.csv',unresolved)
 # Nearest plausible entrance/station matches are an audit only, never edges.
 area_points=[];areaids=[]
 for id,a in areas.items():
  members=[nd[m['id']] for m in a['members'] if m['type']=='node' and m['id'] in nd]
  if members:areaids.append(id);area_points.append(Point(float(np.mean([n['x'] for n in members])),float(np.mean([n['y'] for n in members]))))
 atree=STRtree(area_points);matches=[]
 for e in entrances:
  if e['has_declared_station_area']:continue
  p=Point(e['x_m'],e['y_m']);j=int(atree.nearest(p));dist=p.distance(area_points[j]);a=areas[areaids[j]]
  if dist<=300:matches.append({'entrance_osm_node_id':e['osm_node_id'],'inside_core':e['inside_core'],'candidate_station_area':a['id'],'candidate_station_name':a['tags'].get('name',''),'distance_m':dist,'candidate_only':True,'graph_edge_added':False})
 csvwrite(OUT/'held_entrance_station_match_candidates.csv',matches)
 held_levels=list({json.dumps(r,sort_keys=True):r for r in held_levels}.values());csvwrite(OUT/'held_station_access_floor_conflicts.csv',held_levels,fields=['osm_node_id','walk_node','role','source_node_level','platform_level','issue','graph_edge_added'])
 # Source-named station areas are matched to official station-line geometry only
 # when name and operator agree within a bounded distance. Still no transfer.
 n02=json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/stations_regional_6677.geojson').read_text())['features'];official=[]
 for f in n02:
  p=f['properties'];g=shape(f['geometry']);name=norm(p['N02_005']);cands=[]
  for aid,ap in zip(areaids,area_points):
   ar=areas[aid];names=[norm(ar['tags'].get(k,'')) for k in ['name','name:ja','alt_name','short_name']]
   if name not in names:continue
   dd=g.distance(ap)
   expected={'metro':'metro','toei':'toei'}.get(p['operator_class'],'JR-East' if p['N02_004']=='東日本旅客鉄道' else 'JR-Central' if p['N02_004']=='東海旅客鉄道' else None)
   supported=area_route_operators[aid]|({operator_identity(ar['tags'])} if operator_identity(ar['tags']) is not None else set())
   if dd<=300 and expected is not None and expected in supported:cands.append((dd,aid))
  cands.sort();official.append({'n02_station_code':p['N02_005c'],'n02_group_code':p['N02_005g'],'name':p['N02_005'],'operator':p['N02_004'],'legal_route':p['N02_003'],'inside_core':p['intersects_core'],'matched_station_area':cands[0][1] if len(cands)==1 else '', 'candidates':len(cands),'match_distance_m':cands[0][0] if cands else '', 'match_state':'name_operator_geometry_unique' if len(cands)==1 else 'ambiguous' if cands else 'unmatched','not_a_transfer_or_entrance_connection':True})
 csvwrite(OUT/'official_station_identity_audit.csv',official)
 # Assemble source-path and conditional-access multimodal adjacency. Service
 # segments are unweighted reachability links; no chord length is a timetable.
 for branch,links in [('source_path',boarding),('conditional_internal',boarding+aggregate)]:
  keys=list(map(str,walk['node_ids']));pairs=[(keys[u],keys[v]) for u,v in zip(walk['from_index'],walk['to_index'])];pairs += [(e['from_node'],e['to_node']) for e in railedges]
  for e in links:
   if e['allow_board']:pairs.append((e['from_node'],e['to_node']))
   if e['allow_alight']:pairs.append((e['to_node'],e['from_node']))
  keys=sorted({u for p in pairs for u in p});ix={k:i for i,k in enumerate(keys)};np.savez_compressed(OUT/(branch+'_multimodal_graph.npz'),node_ids=np.array(keys),from_index=np.array([ix[u] for u,v in pairs],np.int32),to_index=np.array([ix[v] for u,v in pairs],np.int32),normal_service_assumed=True,station_internal_aggregate_assumed=branch=='conditional_internal')
 summary={'held_station_access_floor_conflicts':len(held_levels),'osm_rail_station_areas':len(areas),'entrances_in_region':len(entrances),'entrances_inside_core':sum(e['inside_core'] for e in entrances),'core_entrances_with_source_walk_path':sum(e['inside_core'] and e['has_conservative_walk_path'] for e in entrances),'core_entrances_with_declared_station_area':sum(e['inside_core'] and e['has_declared_station_area'] for e in entrances),'route_status':dict(collections.Counter(r['status'] for r in routeaudit)),'rail_service_candidate_nodes':len(railnodes),'rail_service_candidate_edges':len(railedges),'source_platform_boarding_links':len(boarding),'conditional_station_internal_links':len(aggregate),'core_rail_stop_occurrences':sum(r['inside_core'] for r in railnodes.values()),'core_rail_stop_occurrences_with_source_boarding':len({e['to_node'] for e in boarding if railnodes[e['to_node']]['inside_core']}),'core_rail_stop_occurrences_with_conditional_access':len({e['to_node'] for e in boarding+aggregate if railnodes[e['to_node']]['inside_core']}),'held_nearest_station_match_candidates':len(matches),'official_core_identity_match_status':dict(collections.Counter(r['match_state'] for r in official if r['inside_core'])),'no_proximity_transfer_edges':True,'railway_timetables_verified':False,'unmatched_access_does_not_establish_station_closure':True,'stranded_or_flood_consequences_computed':False}
 dump(R/'rail_access_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
