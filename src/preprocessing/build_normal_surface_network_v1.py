"""Build native-ID, grade-separated normal walk/motorcar candidate graphs.
No flood states, exposure aggregation or affected-person calculations.
"""
from pathlib import Path
import sys,json,csv,gzip,collections,itertools,hashlib,time
import numpy as np
from shapely.geometry import shape,LineString,Point,mapping
from shapely import STRtree,contains_xy
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/three_ward_normal_transport_network_v1';R=ROOT/'data/exp/data-preprocessing/three_ward_normal_transport_network_v1';R.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'src/preprocessing'));from normal_transport_rules_v1 import access,directions,grade_key,grade,transitions_allowed,node_access,selfcheck

def rows(p):
 with p.open() as f:return [json.loads(x) for x in f]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
def csvwrite(p,rr,fields=None):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
def gzwrite(p,rr):
 with gzip.open(p,'wt') as f:
  for r in rr:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def main():
 protocol={'human_decision':'MILI-D-20261016-021','scope':'Normal static public-through connectivity candidates, core plus 10 km road halo; 5 km extent comparison. No departures, speeds, flood closures or people redistribution.','source':'Native IDs from fixed Geofabrik Kanto 2026-10-15 OSM snapshot; ODbL attribution retained. OSM tags are not a legal-access certification.','access':'Explicit mode > general access; private, destination, customers and delivery access are not public through routes. Mode/class defaults are explicit assumptions. Unknown conditional access or opening hours remains withheld in the conservative graph; an uncertainty branch includes unknowns but never explicit exclusions.','directions':'Motorcar one-way and roundabout defaults separate from oneway:foot/conveying. Conditional directions unresolved and held.','levels':'No planar noding. Graph nodes are native ID + floor/layer/structure. Same native bridge/tunnel endpoints may connect to source-supported ground approaches; interior conflicts withheld. Explicit stairs/lifts may connect levels. Relative level/layer is not absolute height.','barriers':'Mode-specific node barriers and gate permissions applied, unresolved gate opening retained as unknown.','turns':'Preserve all restriction relations and supported via-node/via-way state memory. Conditional/incomplete/unsupported relations remain explicit audit gaps; do not claim full vehicle qualification if unresolved.','missing_links':'Geometric crossings or nearby endpoints alone never create an edge. Such candidates are exported for batch verification. Historical official walking lines are coverage cross-checks, not automatically added to the current graph.','population':'Prior official block totals and building uncertainty remain unchanged and are not attached to component centroids.'}
 dump(OUT/'surface_network_policy.json',protocol);dump(R/'rule_selfchecks.json',selfcheck())
 raw=rows(OUT/'osm_nodes.jsonl');ids=np.array([r['id'] for r in raw],np.int64);xy=Transformer.from_crs(4326,6677,always_xy=True);xx,yy=xy.transform([r['lon'] for r in raw],[r['lat'] for r in raw]);node={r['id']:(float(x),float(y),r['tags']) for r,x,y in zip(raw,xx,yy)};del raw
 domain=json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/inventory_domains_6677.geojson').read_text())['features'];core=shape(domain[0]['geometry']);halo5=core.buffer(5000);halo10=core.buffer(10000)
 rels=rows(OUT/'osm_relations.jsonl');needed_turn_ways={m['id'] for r in rels if r['tags'].get('type')=='restriction' for m in r['members'] if m['type']=='way'}
 sourceways=rows(OUT/'osm_ways.jsonl');source_turn_refs={w['id']:set(w['refs']) for w in sourceways if w['id'] in needed_turn_ways};ways={};geoms={};wayaudit=[]
 for w in sourceways:
  t=w['tags'];refs=w['refs']
  if 'highway' not in t or len(refs)<2:continue
  g=LineString([(node[id][0],node[id][1]) for id in refs])
  if g.length<1e-6 or not g.intersects(halo10):continue
  a={};d={}
  for mode in ['walk','drive']:
   status,reason=access(t,mode);f,b,dr=directions(t,mode)
   if status=='allowed' and t.get('opening_hours') not in [None,'','24/7']:status,reason='unknown','unparsed_opening_calendar'
   if status=='allowed' and ';' in t.get('level','') and t.get('highway') not in ['steps','elevator']:status,reason='unknown','multiple_levels_not_resolved'
   if dr.startswith('unknown') and status=='allowed':status,reason='unknown',dr
   a[mode]=(status,reason);d[mode]=(f,b,dr)
  if all(x[0]=='excluded' for x in a.values()):
   wayaudit.append({'osm_way_id':w['id'],'highway':t['highway'],'walk_state':a['walk'][0],'walk_reason':a['walk'][1],'drive_state':a['drive'][0],'drive_reason':a['drive'][1],'inside_core':g.intersects(core)});continue
  ways[w['id']]={**w,'access':a,'directions':d};geoms[w['id']]=g
  wayaudit.append({'osm_way_id':w['id'],'highway':t['highway'],'walk_state':a['walk'][0],'walk_reason':a['walk'][1],'drive_state':a['drive'][0],'drive_reason':a['drive'][1],'inside_core':g.intersects(core)})
 del sourceways
 csvwrite(OUT/'way_permission_audit.csv',wayaudit);print('SURFACE WAYS',len(ways),flush=True)
 occurrences=collections.Counter(id for w in ways.values() for id in set(w['refs']));special={id for id,(_,_,t) in node.items() if any(k in t for k in ['barrier','entrance','railway','public_transport','highway','level','access','foot','motor_vehicle'])};junctions={id for id,n in occurrences.items() if n>1}|special|{id for w in ways.values() for id in [w['refs'][0],w['refs'][-1]]}
 graphnodes={};physical=[];membership=collections.defaultdict(dict);nativemembers=collections.defaultdict(set)
 def gn(id,t):
  k='osm:'+str(id)+':'+grade_key(t)
  if k not in graphnodes:
   x,y,nt=node[id];graphnodes[k]={'node_id':k,'osm_node_id':id,'x_m':x,'y_m':y,'grade_key':grade_key(t),'inside_core':core.covers(Point(x,y)),'inside_5km':halo5.covers(Point(x,y)),'inside_10km':halo10.covers(Point(x,y)),'source_node_tags':json.dumps(nt,ensure_ascii=False)}
  nativemembers[id].add(k);return k
 for wi,(wid,w) in enumerate(ways.items()):
  refs=w['refs'];t=w['tags'];cuts=[j for j,id in enumerate(refs) if j==0 or j==len(refs)-1 or id in junctions]
  for j in cuts:membership[refs[j]][wid]=(j in [0,len(refs)-1],t)
  for seg,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
   if refs[a]==refs[b] and b-a<=1:continue
   coordinates=[(node[id][0],node[id][1]) for id in refs[a:b+1]];g=LineString(coordinates)
   if g.length<1e-6:continue
   u,v=gn(refs[a],t),gn(refs[b],t);e={'edge_id':f'w{wid}:{seg}','osm_way_id':wid,'from_node':u,'to_node':v,'length_m':g.length,'grade_key':grade_key(t),'structure':grade(t)[2],'highway':t['highway'],'inside_core':g.intersects(core),'inside_5km':halo5.covers(g),'inside_10km':halo10.covers(g),'geometry':mapping(g),'tags':t}
   for mode in ['walk','drive']:
    state,reason=w['access'][mode];nb=[node_access(node[id][2],mode) for id in refs[a:b+1]]
    if any(s=='excluded' for s,r in nb):state='excluded';reason='source_node_barrier_or_permission'
    elif state=='allowed' and any(s=='unknown' for s,r in nb):state='unknown';reason='unresolved_source_node_access'
    fwd,back,direction_state=w['directions'][mode];e[mode+'_direction_state']=direction_state
    for direction,allowed in [('forward',fwd),('reverse',back)]:
     # Direction-specific access overrides the same base keys for that direction.
     tt=dict(t)
     for key in ['access','foot','vehicle','motor_vehicle','motorcar']:
      suffix='forward' if direction=='forward' else 'backward'
      if key+':'+suffix in t:tt[key]=t[key+':'+suffix]
     ds,_=access(tt,mode)
     e[mode+'_'+direction]=bool(allowed and ds!='excluded');e[mode+'_possible_'+direction]=bool(ds!='excluded')
    e[mode+'_state']=state;e[mode+'_reason']=reason
   physical.append(e)
  if wi%50000==0:print('EDGES',wi,len(physical),flush=True)
 portals=[];gradeconflicts=[]
 for id,mm in membership.items():
  if len(mm)<2:continue
  for (wa,(ea,ta)),(wb,(eb,tb)) in itertools.combinations(mm.items(),2):
   ka,kb='osm:'+str(id)+':'+grade_key(ta),'osm:'+str(id)+':'+grade_key(tb)
   if ka==kb or ka not in graphnodes or kb not in graphnodes:continue
   nt=node[id][2];vertical=ta.get('highway') in ['steps','elevator'] or tb.get('highway') in ['steps','elevator'] or nt.get('highway')=='elevator';ok,why=transitions_allowed(ta,tb,ea,eb,vertical)
   row={'osm_node_id':id,'way_a':wa,'way_b':wb,'from_node':ka,'to_node':kb,'reason':why,'inside_core':graphnodes[ka]['inside_core']}
   (portals if ok else gradeconflicts).append(row)
 # Deduplicate grade connectors without removing their underlying source evidence.
 portal_pairs={tuple(sorted([r['from_node'],r['to_node']])) for r in portals}
 csvwrite(OUT/'native_level_transition_evidence.csv',portals,fields=['osm_node_id','way_a','way_b','from_node','to_node','reason','inside_core']);csvwrite(OUT/'held_native_grade_conflicts.csv',gradeconflicts,fields=['osm_node_id','way_a','way_b','from_node','to_node','reason','inside_core'])
 csvwrite(OUT/'surface_nodes.csv',list(graphnodes.values()));gzwrite(OUT/'surface_physical_edges.jsonl.gz',physical)
 # Restrictions remain separate state rules, not ordinary graph edges.
 restr=[];conditional_restr=[];held_turn_ways=set();ra=[]
 for r in rels:
  t=r['tags']
  if t.get('type')!='restriction':continue
  mm=r['members'];ff=[m['id'] for m in mm if m['type']=='way' and m['role']=='from'];to=[m['id'] for m in mm if m['type']=='way' and m['role']=='to'];vn=[m['id'] for m in mm if m['type']=='node' and m['role']=='via'];vw=[m['id'] for m in mm if m['type']=='way' and m['role']=='via'];name=t.get('restriction:motorcar',t.get('restriction:motor_vehicle',t.get('restriction','')));exceptions=t.get('except','').split(';');conditional=any('conditional' in k for k in t);relevant=any(w in ways for w in ff)
  if not relevant:continue
  applicable=['restriction:motorcar','restriction:motor_vehicle','restriction:vehicle','restriction'];explicit=next((t[k] for k in applicable if k in t),'');cond=next((t[k+':conditional'] for k in applicable if k+':conditional' in t),'');conditional=bool(cond)
  if not explicit and not cond:status='other_vehicle_modes_not_motorcar';name=''
  else:
   name=explicit or cond.split('@')[0].strip().split(';')[0].strip();status='unresolved'
   if not vn and not vw and len(ff)==len(to)==1 and ff[0] in ways and to[0] in ways:
    common=set(ways[ff[0]]['refs'])&set(ways[to[0]]['refs'])
    if len(common)==1:vn=[next(iter(common))];status='recovered_unique_native_via'
   parsed=len(ff)>=1 and len(to)>=1 and (len(vn)==1 and not vw or not vn and len(vw)>=1) and name.startswith(('no_','only_')) and (len(to)==1 or name.startswith('no_'))
   if parsed and vw:
    for f in ff:
     for tw in to:
      chain=[f]+vw+[tw]
      if not all(source_turn_refs.get(a,set())&source_turn_refs.get(b,set()) for a,b in zip(chain,chain[1:])):parsed=False
   if parsed:status='conditional_supported' if conditional else status if status=='recovered_unique_native_via' else 'supported'
  if any(x in exceptions for x in ['motorcar','motor_vehicle','vehicle']):status='motorcar_exempt'
  ra.append({'relation_id':r['id'],'restriction':name,'status':status,'from_ways':json.dumps(ff),'to_ways':json.dumps(to),'via_nodes':json.dumps(vn),'via_ways':json.dumps(vw),'conditional':conditional,'tags':json.dumps(t,ensure_ascii=False)})
  if status in ['supported','conditional_supported','recovered_unique_native_via']:
   dest=conditional_restr if status=='conditional_supported' else restr
   for f in ff:
    for tw in to:dest.append({'relation_id':r['id'],'prefix':[f]+vw,'to_way':tw,'via_node':vn[0] if vn else None,'kind':'no' if name.startswith('no_') else 'only','restriction':name,'transition_nodes':[sorted(source_turn_refs.get(a,set())&source_turn_refs.get(b,set())) for a,b in zip([f]+vw,[*vw,tw])] if vw else None})
  elif status=='unresolved':held_turn_ways.update(ff)
 dump(OUT/'supported_motorcar_restrictions.json',restr);dump(OUT/'conditional_motorcar_restrictions.json',conditional_restr);dump(OUT/'held_turn_way_ids.json',sorted(held_turn_ways));csvwrite(OUT/'motorcar_restriction_audit.csv',ra)
 # Build ordinary adjacency per mode; the vehicle route adapter ALSO applies the
 # saved turn-state rules. Topological SCC counts alone ignore those turn rules.
 adjacency={};stats=[];component_maps={};graph_arrays={}
 for mode in ['walk','drive']:
  for branch in ['conservative','unknown_open']:
   adj=collections.defaultdict(list);arcs=[]
   for e in physical:
    if not e['inside_10km'] or e[mode+'_state']=='excluded' or branch=='conservative' and e[mode+'_state']!='allowed':continue
    if mode=='drive' and branch=='conservative' and e['osm_way_id'] in held_turn_ways:continue
    optimistic_direction=branch=='unknown_open' and e[mode+'_direction_state'].startswith('unknown')
    if e[mode+'_possible_forward'] if optimistic_direction else e[mode+'_forward']:arcs.append((e['from_node'],e['to_node'],e['length_m'],e['osm_way_id'],e['inside_5km']))
    if e[mode+'_possible_reverse'] if optimistic_direction else e[mode+'_reverse']:arcs.append((e['to_node'],e['from_node'],e['length_m'],e['osm_way_id'],e['inside_5km']))
   active={a for u,v,l,w,h in arcs for a in [u,v]}
   for a,b in portal_pairs:
    if a in active and b in active:arcs.extend([(a,b,0.,0,graphnodes[a]['inside_5km']),(b,a,0.,0,graphnodes[b]['inside_5km'])])
   keys=sorted({v for a in arcs for v in a[:2]});ix={k:i for i,k in enumerate(keys)};u=np.array([ix[a[0]] for a in arcs],np.int32);v=np.array([ix[a[1]] for a in arcs],np.int32);sp=coo_matrix((np.ones(len(arcs),np.uint8),(u,v)),shape=(len(keys),len(keys))).tocsr();nw,cw=connected_components(sp,directed=True,connection='weak');ns,cs=connected_components(sp,directed=True,connection='strong');in_core=np.array([graphnodes[k]['inside_core'] for k in keys]);counts=np.bincount(cw);big=int(counts.argmax()) if len(counts) else -1
   name=mode+'_'+branch
   np.savez_compressed(OUT/(name+'_graph.npz'),node_ids=np.array(keys),from_index=u,to_index=v,length_m=np.array([a[2] for a in arcs]),osm_way_id=np.array([a[3] for a in arcs],np.int64),inside_5km=np.array([a[4] for a in arcs]),weak_component=cw,strong_component=cs,inside_core=in_core)
   component_maps[name]={k:int(cw[i]) for i,k in enumerate(keys)};graph_arrays[name]=(keys,arcs)
   stats.append({'mode':mode,'branch':branch,'nodes':len(keys),'directed_arcs':len(arcs),'weak_components':int(nw),'strong_components_ignoring_turn_memory':int(ns),'core_nodes':int(in_core.sum()),'core_nodes_in_largest_weak_component':int(np.sum(in_core&(cw==big))),'core_node_largest_weak_fraction':float(np.sum(in_core&(cw==big))/max(1,in_core.sum())),'not_a_population_or_flood_isolation_measure':True})
   print('GRAPH',name,len(keys),len(arcs),nw,flush=True)
 csvwrite(R/'surface_graph_summary.csv',stats)
 # Batch intersect core source ways. No geometric crossing is turned into a link.
 coreids=[id for id,g in geoms.items() if g.intersects(core)];gg=[geoms[id] for id in coreids];tree=STRtree(gg);crossings=[]
 for i,g in enumerate(gg):
  wa=ways[coreids[i]]
  for j in tree.query(g,predicate='intersects'):
   if j<=i:continue
   wb=ways[coreids[j]]
   if set(wa['refs'])&set(wb['refs']):continue
   q=g.intersection(gg[j])
   if q.is_empty:continue
   p=q.representative_point()
   if not core.covers(p):continue
   if q.geom_type not in ['Point','MultiPoint']:kind='overlapping_geometry_without_shared_native_id'
   elif grade_key(wa['tags'])!=grade_key(wb['tags']):kind='different_grade_crossing_kept_separate'
   else:kind='same_grade_crossing_candidate_no_auto_join'
   crossings.append({'way_a':wa['id'],'way_b':wb['id'],'kind':kind,'x_m':p.x,'y_m':p.y,'grade_a':grade_key(wa['tags']),'grade_b':grade_key(wb['tags']),'graph_edge_added':False})
 csvwrite(OUT/'core_planar_crossing_audit.csv',crossings,fields=['way_a','way_b','kind','x_m','y_m','grade_a','grade_b','graph_edge_added'])
 # Near misses are candidates, not repaired by arbitrary proximity.
 endpointids={id for w in ways.values() for id in [w['refs'][0],w['refs'][-1]] if core.covers(Point(node[id][:2]))};pids=sorted(endpointids);pts=[Point(node[id][:2]) for id in pids];ptree=STRtree(pts);near=[]
 for i,p in enumerate(pts):
  for j in ptree.query(p.buffer(2.)):
   if j<=i:continue
   a,b=pids[i],pids[j];dist=p.distance(pts[j])
   if dist<=2 and not set(membership.get(a,{}))&set(membership.get(b,{})):
    near.append({'node_a':a,'node_b':b,'distance_m':dist,'x_m':p.x,'y_m':p.y,'graph_edge_added':False,'role':'near_endpoint_candidate_not_proven_missing_connection'})
 csvwrite(OUT/'core_near_endpoint_candidates.csv',near,fields=['node_a','node_b','distance_m','x_m','y_m','graph_edge_added','role'])
 # Population support audit uses street-block geometry, with no allocation of
 # residents to graph components or a centroid, and no affected-person count.
 blocks=json.loads((ROOT/'data/processed/three_ward_population_block_stability_v1/block_population_stability_6677.geojson').read_text())['features'];btree=STRtree([shape(f['geometry']) for f in blocks]);bcom=collections.defaultdict(set);bn=collections.Counter()
 for k,comp in component_maps['walk_conservative'].items():
  n=graphnodes[k]
  if not n['inside_core']:continue
  pp=Point(n['x_m'],n['y_m'])
  for j in btree.query(pp,predicate='intersects'):bn[int(j)]+=1;bcom[int(j)].add(comp)
 block_edge_support=collections.Counter()
 for e in physical:
  if not e['inside_core'] or e['walk_state']!='allowed':continue
  g=shape(e['geometry'])
  for j in btree.query(g,predicate='intersects'):block_edge_support[int(j)]+=1
 support=[{'block_id':f['properties']['unit_id'],'ward':f['properties']['ward'],'walk_candidate_nodes':bn[i],'intersecting_allowed_walk_edges':block_edge_support[i],'normal_walk_weak_components_intersecting':len(bcom[i]),'population_to_components_assigned':False,'building_population_uncertainty_retained':True,'support_role':'spatial network audit, not a resident access or stranded-person classification'} for i,f in enumerate(blocks)]
 csvwrite(OUT/'block_network_support_audit.csv',support)
 # Keep raw nodes in native-ID lookup, without any personal OSM metadata.
 station_nodes=[{'id':id,'x':x,'y':y,'tags':t} for id,(x,y,t) in node.items() if t.get('railway') in ['subway_entrance','train_station_entrance','station','halt','stop'] or t.get('public_transport') in ['station','stop_position','platform'] or t.get('entrance') and any(k in t for k in ['station','railway'])]
 gzwrite(OUT/'station_osm_node_inventory.jsonl.gz',station_nodes)
 summary={'source_snapshot':'kanto-261015.osm.pbf','surface_ways_retained':len(ways),'physical_edges':len(physical),'native_grade_nodes':len(graphnodes),'source_supported_grade_transitions':len(portal_pairs),'held_native_grade_conflict_records':len(gradeconflicts),'core_held_grade_conflict_records':sum(r['inside_core'] for r in gradeconflicts),'planar_crossing_audit':dict(collections.Counter(r['kind'] for r in crossings)),'near_endpoint_candidates':len(near),'restriction_status':dict(collections.Counter(r['status'] for r in ra)),'supported_turn_rules':len(restr),'conditional_turn_rules':len(conditional_restr),'unresolved_turn_source_ways_withheld_conservatively':len(held_turn_ways),'surface_graphs':stats,'blocks_without_conservative_walk_nodes':sum(r['walk_candidate_nodes']==0 for r in support),'blocks_without_nodes_or_allowed_walk_edges':sum(r['walk_candidate_nodes']==0 and r['intersecting_allowed_walk_edges']==0 for r in support),'population_to_components_assigned':False,'flood_or_isolation_computed':False}
 dump(R/'surface_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
