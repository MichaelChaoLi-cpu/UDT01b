"""Construct an explicit hypothetical station/rail access graph and closure policies.
Uses frozen transport and surface-screen products only. No new flood simulation.
"""
from pathlib import Path
import sys,json,csv,re,collections,math,hashlib,time,unicodedata
import numpy as np
from shapely.geometry import shape,Point,box
from shapely.ops import unary_union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_transport_shutdown_scenarios_v1';R=ROOT/'data/exp/estimation/three_ward_transport_shutdown_scenarios_v1';B=ROOT/'data/processed/three_ward_normal_transport_network_v1';C=ROOT/'data/processed/three_ward_jr_toei_evidence_v1/corrected_source';S=ROOT/'data/processed/three_ward_station_surface_risk_v1';E=C.parent
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from build_normal_surface_network_v1 import dump,csvwrite
from build_jr_toei_corrected_source_v1 import operator_identity
from screen_station_surface_risk_v1 import canon,public_operator

def rr(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canon(s):
 s=unicodedata.normalize('NFKC',s).replace('ヶ','ケ').replace('麴','麹').replace('駅','').strip()
 return re.sub(r'\s*\(([^()]*(?:線|Line|line|地下鉄|Subway)[^()]*)\)\s*','',s).strip()

def opid(t):
 op=operator_identity(t)
 if op:return op
 x=' '.join(t.get(k,'') for k in ['operator','operator:ja','operator:en'])
 for tokens,label in [(['ゆりかもめ','Yurikamome'],'ゆりかもめ'),(['京浜急行','京急','Keikyu'],'京浜急行電鉄'),(['東京モノレール','Tokyo Monorail'],'東京モノレール'),(['首都圏新都市','Metropolitan Intercity'],'首都圏新都市鉄道')]:
  if any(a in x for a in tokens):return 'other:'+label
 return 'source-operator:'+t.get('operator',t.get('operator:en','unresolved'))

def main():
 policy={'human_decision':'MILI-D-20261016-024','purpose':'Conditional accessibility stress test of three hypothesized shutdown types, with no observed shutdown or underground ingress claim.','primary_exposure_rule':{'neighborhood_radius_m':20,'depth_lower_bound_m':.3,'minimum_peak_exceedance_ground_area_m2':100,'minimum_admitted_open_ground_m2':100,'minimum_admitted_fraction_of_eligible_ground':.8,'role':'Analyst-declared screening-to-shutdown hypothesis; not a physical or operator threshold.'},'sensitivity_rules':[{'name':'depth0p1','depth_m':.1,'minimum_area_m2':100},{'name':'depth0p5','depth_m':.5,'minimum_area_m2':100},{'name':'any_supported_area0p3','depth_m':.3,'minimum_area_m2':1e-7}], 'vicinity_fallback_sensitivity':{'radius_m':50,'depth_m':.3,'minimum_area_m2':100,'minimum_fraction_of_admitted_ground':.05,'applies_only_to':'station/section cases at units without source-mapped entrance identity','evidence_tier':'Additional assumption from station vicinity; never promoted to verified inlet evidence.'},'temporal_assumption':'For each of 16 existing open parent scenarios, jointly impose closures selected from cellwise event peak footprints. Peaks are not simultaneous; these are static event stress envelopes, not failure timing or duration. No pooled cross-event closure is used.','shutdown_types':{'entrance':'Remove only selected portal-to-station circulation links in both directions; preserve other mapped entries and street access. Unknown inventories remain explicit.','station':'At any mapped-entry-triggered operator/name unit, close local entry/exit, boarding/alighting and interchange hub links. Trains may pass through retained directed service occurrences without stopping.','section':'Apply station closure plus suspend directed service arcs whose endpoints or same-family reference stop-order spans include a target unit. This models an operating-interval interruption, not a verified tunnel cross-section failure. Retain unknown spans in an audit.'},'identity_normalization':'Normalize Japanese glyph aliases and strip explicit parenthetical line annotations; resolve unknown area operator only from a unique declared route operator. Frozen exposure identity links remain the primary closure evidence.', 'station_unit_scope':'Operator/name inventory units; no automatic same-name cross-operator closure propagation.','normal_circulation_assumption':'Source-declared station-area memberships support identities. Circulation between route occurrences of the same operator/name unit and its entries is an explicit normal-operation aggregate assumption, not a verified internal corridor.','network_branches':['intra_operator_hubs_only','provider_transfers_assumed_bidir'],'identity_supplement_sensitivity':'Use source-native portal IDs whose explicit stop-area name has only a line annotation or known glyph alias removed, operator identity is source-supported, and distance to official same-identity geometry is <=500m. This attribution sensitivity uses the unchanged entrance-exposure fields and is reported separately from the original screen attribution.', 'provider_transfer_assumption':'Activate only resolvable provider-reported transfer relations, with hypothetical normal two-way usability. No nearest-station joins or verified-corridor claim.','surface_origins':'Existing conservative at-grade walk vertices selected once in fixed 500m core cells; station-vicinity and external-station probes are separate cohorts. None represents residents.','walking_legs':'Conservative source-native pedestrian paths up to 600 m for each initial/final leg. Separate origin-only and destination-only nodes prevent unlimited inter-station street-walking chains. Street flood closures are held fixed absent in all variants.','anchors':[{'operator':'JR-East','name':x} for x in ['新宿','渋谷','上野','錦糸町','大井町']]+[{'operator':'metro','name':'豊洲'}],'baseline':'Paired hypothetical normal graph within each evidence branch; loss is counted only for OD pairs reachable in that same baseline. Baseline missing links stay unknown.','other_modes':'No timetable, capacity, congestion, road-vehicle disruption or bus replacement model. Rail/street access probes do not establish total city isolation.','population':'Frozen official night baseline and building uncertainty referenced; no allocation, weighting or affected-person counts.','sealed_tests':False,'new_hydraulics':False}
 dump(P/'scenario_policy.json',policy)
 core_inventory=rr(S/'station_screen_summary.csv');coreids={x['station_id'] for x in core_inventory};coremeta={x['station_id']:x for x in core_inventory}
 rels=[json.loads(s) for s in (B/'osm_relations.jsonl').open()];areas={x['id']:x for x in rels if x['tags'].get('public_transport')=='stop_area'};routes={x['id']:x for x in rels if x['tags'].get('type')=='route'};rail=rr(C/'rail_service_candidate_nodes.csv');railedges=rr(C/'rail_service_candidate_edges.csv');routeops=collections.defaultdict(set)
 for n in rail:
  r=routes[int(n['route_id'])];ident=operator_identity(r['tags'])
  if ident:
   for aid in json.loads(n['source_station_area_ids']):routeops[aid].add(ident)
 area_sid={};stationmeta={};identity_normalization=[];code_sid=collections.defaultdict(set)
 for aid,a in areas.items():
  name=canon(a['tags'].get('name:ja',a['tags'].get('name','')))
  if not name:continue
  op=opid(a['tags'])
  if op.startswith('source-operator:') and len(routeops[aid])==1:op=next(iter(routeops[aid]))
  if op=='source-operator:unresolved':continue
  sid=op+':'+name;area_sid[aid]=sid;identity_normalization.append({'source_stop_area':aid,'source_name':a['tags'].get('name:ja',a['tags'].get('name','')),'normalized_station_id':sid,'operator_evidence':'explicit_or_unique_source_route_operator','line_annotation_stripped':name!=a['tags'].get('name:ja',a['tags'].get('name','')),'physical_station_interior_verified':False})
  for code in re.findall(r'[A-Z]{1,2}\d{1,2}',a['tags'].get('ref','').replace('-','')):code_sid[(op,code)].add(sid)
 csvwrite(P/'station_identity_normalization.csv',identity_normalization)
 # Assign actual source stop occurrences to their explicit station areas. Use
 # same operator/name fallback only when no compatible source-area identity is
 # available; station/corridor ambiguity is retained as a gap.
 stop_sids={};gaps=[];station_xy=collections.defaultdict(list)
 for n in rail:
  t=routes[int(n['route_id'])]['tags'];ids={area_sid[a] for a in json.loads(n['source_station_area_ids']) if a in area_sid};op=opid(t);name=canon(n['name']);preferred={s for s in ids if s.startswith(op+':') and s.split(':')[-1]==name}
  if preferred:ids=preferred
  if not ids and name and op!='source-operator:unresolved':ids={op+':'+name}
  if not ids:gaps.append({'source_node':n['node_id'],'issue':'station_identity_unresolved'});stop_sids[n['node_id']]=set();continue
  # If one source stop is explicitly shared between identities, retain both
  # memberships as source evidence and disclose aggregate-circulation limits.
  stop_sids[n['node_id']]=ids
  for sid in ids:station_xy[sid].append((float(n['x_m']),float(n['y_m'])))
 for sid,coords in station_xy.items():
  stationmeta[sid]={'station_id':sid,'name':sid.split(':')[-1],'operator':sid.rsplit(':',1)[0],'x_m':float(np.mean([p[0] for p in coords])),'y_m':float(np.mean([p[1] for p in coords])),'core':sid in coreids,'origin_entrance_geometry_verified':False}
 # Keep all official core identities, including absent source rail coverage.
 for sid,m in coremeta.items():
  if sid not in stationmeta:stationmeta[sid]={'station_id':sid,'name':m['station_name'],'operator':m['operator'],'x_m':float(m['x_m']),'y_m':float(m['y_m']),'core':True,'origin_entrance_geometry_verified':False}
 # Add known mapped portal identities; a no-entry station gets a synthetic
 # unknown-access hypothesis only if source rail occurrences exist.
 entrance=rr(C/'osm_station_entrance_audit.csv');portals={};station_portals=collections.defaultdict(set);risklinks=collections.defaultdict(set)
 for x in rr(S/'station_entrance_identity_links.csv'):risklinks[int(x['osm_node_id'])].add(x['station_id'])
 for e in entrance:
  ids={area_sid[a] for a in json.loads(e['source_station_area_ids']) if a in area_sid and area_sid[a] in stationmeta};ids|=risklinks[int(e['osm_node_id'])];ids={sid for sid in ids if sid in stationmeta}
  if not ids:continue
  eid='osm:'+e['osm_node_id'];portals[eid]={'portal_id':eid,'osm_node_id':int(e['osm_node_id']),'station_ids':sorted(ids),'x_m':float(e['x_m']),'y_m':float(e['y_m']),'walk_node_ids':json.loads(e['walk_native_nodes']),'synthetic_unknown':False}
  for sid in ids:station_portals[sid].add(eid)
 for sid in station_xy:
  if not station_portals[sid]:
   eid='unknown:'+sid;portals[eid]={'portal_id':eid,'osm_node_id':None,'station_ids':[sid],'x_m':stationmeta[sid]['x_m'],'y_m':stationmeta[sid]['y_m'],'walk_node_ids':[],'synthetic_unknown':True};station_portals[sid].add(eid)
 # Match the stored surface-risk entrance IDs by explicit prior links. Do not
 # silently close a different-area identity inferred during normal graph build.
 csvwrite(P/'station_units.csv',list(stationmeta.values()));csvwrite(P/'portal_inventory.csv',[{**v,'station_ids':json.dumps(v['station_ids'],ensure_ascii=False),'walk_node_ids':json.dumps(v['walk_node_ids'])} for v in portals.values()]);dump(P/'stop_station_memberships.json',{k:sorted(v) for k,v in stop_sids.items()})
 # Reconstruct source road distances. Parallel arcs use the minimum weight,
 # not a sum; explicit zero-length grade portals remain present.
 walk=np.load(B/'walk_conservative_graph.npz');wk=list(map(str,walk['node_ids']));wix={k:i for i,k in enumerate(wk)};u=walk['from_index'];v=walk['to_index'];ll=walk['length_m'];order=np.lexsort((v,u));u,v,ll=u[order],v[order],ll[order];starts=np.r_[0,np.flatnonzero((u[1:]!=u[:-1])|(v[1:]!=v[:-1]))+1];wg=csr_matrix((np.minimum.reduceat(ll,starts),(u[starts],v[starts])),shape=(len(wk),len(wk)))
 blocks=json.loads((ROOT/'data/processed/three_ward_population_block_stability_v1/block_population_stability_6677.geojson').read_text())['features'];wards={w:unary_union([shape(f['geometry']) for f in blocks if f['properties']['ward']==w]) for w in ['13101','13102','13103']};core=unary_union(list(wards.values()))
 def wardof(p):
  match=[w for w,g in wards.items() if g.covers(p)];return match[0] if len(match)==1 else 'boundary_or_unassigned'
 # Deterministic fixed 500m cells; pick an actual allowed ground node closest
 # to each cell center. No station or cross-grade edge is created by proximity.
 candidates={}
 with (B/'surface_nodes.csv').open() as f:
  for n in csv.DictReader(f):
   if n['inside_core']!='True' or n['grade_key']!='0|0|surface' or n['node_id'] not in wix:continue
   x,y=float(n['x_m']),float(n['y_m']);gx,gy=math.floor(x/500),math.floor(y/500);score=(x-(gx+.5)*500)**2+(y-(gy+.5)*500)**2;key=(gx,gy)
   if key not in candidates or (score,n['node_id'])<(candidates[key][0],candidates[key][1]):candidates[key]=(score,n['node_id'],x,y)
 probes=[];probe_sources={}
 for (gx,gy),(score,k,x,y) in sorted(candidates.items()):
  pid=f'grid:{gx}:{gy}';probes.append({'probe_id':pid,'cohort':'surface_grid','station_id':'','x_m':x,'y_m':y,'ward':wardof(Point(x,y)),'source_walk_node':k,'represents_population':False});probe_sources[pid]=[wix[k]]
 anchorids=[x['operator']+':'+canon(x['name']) for x in policy['anchors']];stationprobes=sorted(coreids|set(anchorids))
 for sid in stationprobes:
  m=stationmeta.get(sid)
  if not m:gaps.append({'source_node':sid,'issue':'declared_anchor_station_absent'});continue
  pid='station:'+sid;probes.append({'probe_id':pid,'cohort':'core_station' if sid in coreids else 'external_anchor','station_id':sid,'x_m':m['x_m'],'y_m':m['y_m'],'ward':wardof(Point(m['x_m'],m['y_m'])) if sid in coreids else 'external','source_walk_node':'','represents_population':False});probe_sources[pid]=sorted({wix[k] for ep in station_portals[sid] for k in portals[ep]['walk_node_ids'] if k in wix})
 edges=[]
 def add(a,b,kind,station='',portal='',basis='',family='',coverage=None):
  edges.append({'from_node':a,'to_node':b,'edge_kind':kind,'station_id':station,'portal_id':portal,'basis':basis,'service_family':family,'section_exposure_station_ids':sorted(coverage or []),'provider_transfer':False})
 for sid in station_xy:
  for eid in station_portals[sid]:
   add('P:'+eid,'H:'+sid,'station_entry',sid,eid,'assumed_normal_internal_circulation');add('H:'+sid,'P:'+eid,'station_exit',sid,eid,'assumed_normal_internal_circulation')
 for n in rail:
  for sid in stop_sids[n['node_id']]:
   if n['stop_role']!='stop_exit_only':add('H:'+sid,n['node_id'],'board',sid,basis='explicit_source_station_identity_normal_circulation_assumed')
   if n['stop_role']!='stop_entry_only':add(n['node_id'],'H:'+sid,'alight',sid,basis='explicit_source_station_identity_normal_circulation_assumed')
 # Source station-order families propagate a section cut over skip-stop arcs.
 # This is an operating-corridor hypothesis, not shared-track geometry proof.
 routefamily={};sequences=collections.defaultdict(list);route_nodes=collections.defaultdict(list)
 for n in rail:route_nodes[n['route_id']].append(n)
 for rid,ns in route_nodes.items():
  t=routes[int(rid)]['tags'];op=opid(t);refs=t.get('ref','');family=op+':'+refs if refs else ''
  if '浅草線' in t.get('name',''):family='toei:A'
  elif op=='toei' and refs in ['A','I','S','E']:family='toei:'+refs
  elif '上野東京ライン' in t.get('name','') and not refs:family='JR-East:JU'
  routefamily[rid]=family
  if family:
   seq=[sorted(stop_sids[n['node_id']])[0] if len(stop_sids[n['node_id']])==1 else None for n in sorted(ns,key=lambda n:int(n['stop_index']))];sequences[family].append(seq)
 span_audit=[]
 for e in railedges:
  a,b=e['from_node'],e['to_node'];coverage=set(stop_sids[a])|set(stop_sids[b]);family=routefamily.get(e['route_id'],'');spans=0
  for seq in sequences.get(family,[]):
   for ua in stop_sids[a]:
    for vb in stop_sids[b]:
     if ua==vb:continue
     ia=[i for i,x in enumerate(seq) if x==ua];ib=[i for i,x in enumerate(seq) if x==vb]
     if not ia or not ib:continue
     i,j=min(((i,j) for i in ia for j in ib),key=lambda pair:abs(pair[0]-pair[1]));i,j=sorted([i,j]);coverage|={x for x in seq[i:j+1] if x};spans+=1
  add(a,b,'rail_service',basis='source_ordered_operating_arc_with_reference_span_assumption' if spans else 'source_ordered_operating_arc_endpoint_only',family=family,coverage=coverage);span_audit.append({'source_edge_id':e['edge_id'],'route_id':e['route_id'],'family':family,'reference_spans_found':spans,'core_units_in_endpoint_or_span':';'.join(sorted(coverage&coreids)),'interior_span_unknown':not bool(spans),'physical_track_sharing_verified':False})
 csvwrite(P/'section_span_audit.csv',span_audit)
 transfer=[]
 for t in rr(E/'provider_transfer_relations.csv'):
  fr='toei:'+canon(t['from_station']);desc=t['description'];tos=set();why=''
  if fr not in stationmeta:why='from_identity_absent'
  elif 'JR' in desc:
   named=re.findall('[（(]([^（）()]+)[）)]',desc);named=[canon(x) for x in named if not re.fullmatch(r'[A-Z]-?\d+',x)]
   for name in named or [canon(t['from_station'])]:
    key='JR-East:'+name
    if key in stationmeta:tos.add(key)
   if not tos:why='JR_target_name_not_explicit_or_absent_no_nearest_match'
  elif '東京メトロ' in desc:
   for code in re.findall(r'[A-Z]-\d\d',desc):tos|=code_sid.get(('metro',code.replace('-','')),set())
   for name in re.findall('[（(【〖]([^（）()】〗]+)[）)】〗]',desc):
    key='metro:'+canon(name)
    if key in stationmeta:tos.add(key)
   if not tos and 'metro:'+canon(t['from_station']) in stationmeta:tos.add('metro:'+canon(t['from_station']))
   if not tos:why='Metro_target_identity_unresolved'
  elif 'モノレール' in desc:
   key='other:東京モノレール:'+('浜松町' if '浜松町' in desc else canon(t['from_station']));tos={key} if key in stationmeta else set();why='' if tos else 'other_target_absent'
  elif 'ゆりかもめ' in desc:
   key='other:ゆりかもめ:'+canon(t['from_station']);tos={key} if key in stationmeta else set();why='' if tos else 'other_target_absent'
  else:why='operator_or_target_not_qualified'
  for to in sorted(tos):
   for a,b in [(fr,to),(to,fr)]:
    add('H:'+a,'H:'+b,'transfer',basis='provider_reported_relation_two_way_circulation_assumed');edges[-1]['provider_transfer']=True;edges[-1]['transfer_units']=[fr,to]
  transfer.append({'from_station_id':fr,'description':desc,'resolved_to_station_ids':';'.join(sorted(tos)),'state':'assumed_usable_provider_relation' if tos else 'held','reason':why,'metric_corridor_verified':False,'actual_two_way_gates_verified':False})
 csvwrite(P/'transfer_evidence_and_assumptions.csv',transfer)
 # Source-node bounded access legs. End portals are kept separately so closing
 # one entrance cannot silently be bypassed by the same synthetic portal.
 portal_walk={ep:np.array([wix[k] for k in row['walk_node_ids'] if k in wix],dtype=int) for ep,row in portals.items()};accesslegs=[]
 for count,p in enumerate(probes):
  pid=p['probe_id'];source=probe_sources[pid]
  if source:
   forward=dijkstra(wg,directed=True,indices=source,min_only=True,limit=600);backward=dijkstra(wg.T.tocsr(),directed=True,indices=source,min_only=True,limit=600)
   for eid,inds in portal_walk.items():
    if not len(inds):continue
    f=float(forward[inds].min());b=float(backward[inds].min())
    if np.isfinite(f):add('O:'+pid,'P:'+eid,'initial_walk',portal=eid,basis='native_conservative_walk_path_le600m');accesslegs.append({'probe_id':pid,'portal_id':eid,'direction':'initial','distance_m':f,'source_path':True})
    if np.isfinite(b):add('P:'+eid,'D:'+pid,'final_walk',portal=eid,basis='native_conservative_walk_path_le600m');accesslegs.append({'probe_id':pid,'portal_id':eid,'direction':'final','distance_m':b,'source_path':True})
  # Station-area probes use their own declared station identity. Unknown
  # entrances remain explicit assumed-access hypotheses, not invented points.
  if p['station_id']:
   for eid in station_portals[p['station_id']]:
    add('O:'+pid,'P:'+eid,'station_probe_access',portal=eid,basis='station_area_probe_access_assumed');add('P:'+eid,'D:'+pid,'station_probe_egress',portal=eid,basis='station_area_probe_access_assumed')
  if count%50==0:print('ACCESS',count,'of',len(probes),'edges',len(edges),flush=True)
 csvwrite(P/'access_probes.csv',probes);csvwrite(P/'bounded_walk_legs.csv',accesslegs);csvwrite(P/'identity_gaps.csv',gaps,fields=['source_node','issue'])
 keys=sorted({x for e in edges for x in [e['from_node'],e['to_node']]}|{x for p in probes for x in ['O:'+p['probe_id'],'D:'+p['probe_id']]});ix={k:i for i,k in enumerate(keys)}
 for i,e in enumerate(edges):e['edge_id']='hyp:'+str(i)
 dump(P/'hypothetical_network_edges.json',edges);np.savez_compressed(P/'hypothetical_network.npz',node_ids=np.array(keys),from_index=np.array([ix[e['from_node']] for e in edges],np.int32),to_index=np.array([ix[e['to_node']] for e in edges],np.int32),provider_transfer=np.array([e['provider_transfer'] for e in edges]))
 bindings=[]
 for path in [S/'entrance_scenario_surface_screen.csv',S/'station_vicinity_scenario_screen.csv',S/'station_entrance_identity_links.csv',S/'station_screen_summary.csv',C/'rail_service_candidate_nodes.csv',C/'rail_service_candidate_edges.csv',B/'walk_conservative_graph.npz',E/'provider_transfer_relations.csv',ROOT/'data/exp/data-preprocessing/three_ward_station_surface_risk_v1/delivery_manifest.json',ROOT/'data/exp/data-preprocessing/three_ward_population_block_stability_v1/delivery_manifest.json']:
  bindings.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path)})
 dump(P/'source_bindings.json',bindings)
 summary={'core_station_identity_units':len(coreids),'core_units_with_source_stop_occurrences':len(coreids&set(station_xy)),'regional_station_units_with_source_stops':len(station_xy),'source_stop_occurrences':len(rail),'mapped_portal_identity_units':sum(not x['synthetic_unknown'] for x in portals.values()),'synthetic_unknown_access_units':sum(x['synthetic_unknown'] for x in portals.values()),'surface_grid_probes':sum(p['cohort']=='surface_grid' for p in probes),'core_station_probes':sum(p['cohort']=='core_station' for p in probes),'anchor_probes':sum(p['station_id'] in anchorids for p in probes),'network_nodes':len(keys),'network_arcs':len(edges),'bounded_native_walk_legs':len(accesslegs),'provider_relations_used':sum(x['state']!='held' for x in transfer),'provider_relations_held':sum(x['state']=='held' for x in transfer),'identities_not_corridors_verified':True,'policy_frozen_before_accessibility_evaluation':True,'no_real_closures':True,'new_hydraulics':False};dump(R/'network_build_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
