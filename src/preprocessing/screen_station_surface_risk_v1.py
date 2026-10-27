"""Screen existing station-entrance neighborhoods against existing synthetic peaks.
No operational shutdown inference, population allocation or new solver execution.
"""
from pathlib import Path
import sys,json,csv,re,hashlib,collections,math
import numpy as np
from shapely.geometry import shape,Point,mapping
from shapely.ops import unary_union
from shapely import contains_xy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'preprocessing'))
from build_normal_surface_network_v1 import dump,csvwrite
from build_jr_toei_corrected_source_v1 import operator_identity,opclass
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'data/processed/three_ward_normal_transport_network_v1';E=ROOT/'data/processed/three_ward_jr_toei_evidence_v1';P=ROOT/'data/processed/three_ward_station_surface_risk_v1';R=ROOT/'data/exp/data-preprocessing/three_ward_station_surface_risk_v1'
def rr(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canon(s):return s.replace('ヶ','ケ').replace('駅','').strip()
def public_operator(s):return {'東日本旅客鉄道':'JR-East','東海旅客鉄道':'JR-Central','東京地下鉄':'metro','東京都':'toei'}.get(s,'other:'+s)
def parse_sources():
 S=E/'sources';texts=[p.read_text() for p in sorted(S.glob('toei_web_batch_*.txt'))]+[(S/'transfer_extra_web.txt').read_text()];bodies={};sources=[]
 for txt in texts:
  pat=r'^([^\n]+) ｜ 東京都交通局 \((https://www\.kotsu\.metro\.tokyo\.jp/subway/stations/[^)]+)\)\n'
  matches=list(re.finditer(pat,txt,re.M))
  for i,m in enumerate(matches):
   b=txt[m.end():matches[i+1].start() if i+1<len(matches) else len(txt)];bodies.setdefault(m[1],[]).append(b);sources.append({'operator':'toei','station':m[1],'url':m[2],'retrieved':'2026-10-16','evidence_role':'provider_station_description','survey_date_verified':False})
 facts=[];transfers=[]
 for name,chunks in bodies.items():
  txt='\n'.join(chunks);clean=re.sub(r'L\d+: ?', '',txt);section=re.search(r'### 乗り換え案内\n(.*?)(?=### |\Z)',clean,re.S);labels=[]
  if section:labels=list(dict.fromkeys(x.split('†')[0] for x in re.findall(r'cite\d+†([^]+)',section[1])))
  facts.append({'station':name,'provider':'toei','provider_reports_accessible_route':bool(re.search(r'1ルートの確保\s*:\s*済み',clean)),'diagram_link_present':bool(re.search(r'Image: [^\n]*(?:駅構内図|駅立体図)',clean)),'transfer_descriptions_observed':len(labels),'exact_corridor_geometry_verified':False,'specific_ground_threshold_verified':False,'interpretation':'One accessible route is existential evidence, not verification of every exit or transfer.'})
  for label in labels:transfers.append({'from_station':name,'description':label,'evidence':'provider_reported_transfer_relation','specific_entrance_corridor_and_gate_verified':False,'bidirectional_corridor_verified':False,'activated_edge':False})
 rows=json.loads((S/'jr_station_catalog_verified.json').read_text())
 for row in rows:sources.append({'operator':'JR-East','station':row['name'],'url':row['timetable_url'],'retrieved':'2026-10-16','evidence_role':'station_identity_and_service_catalog_only_no_departure_model','survey_date_verified':False})
 csvwrite(E/'provider_station_facts.csv',facts);csvwrite(E/'provider_transfer_relations.csv',transfers);csvwrite(E/'provider_source_inventory.csv',list({(r['operator'],r['station'],r['url']):r for r in sources}.values()))
 # Read only timetable titles and revision labels; no train-by-train model.
 timefacts=[]
 for p in sorted(S.glob('toei_timetable_batch_*.txt')):
  txt=p.read_text();pat=r'^([^\n]+) ｜ 東京都交通局 \((https://www\.kotsu\.metro\.tokyo\.jp/subway/timetable/[^)]+)\)\n';matches=list(re.finditer(pat,txt,re.M))
  for i,m in enumerate(matches):
   body=txt[m.end():matches[i+1].start() if i+1<len(matches) else len(txt)];code=re.search(r'/([AISE]\d\d)(SD|SU)\.html',m[2]);rev=re.search(r'(令和\d+年\d+月\d+日改正)',body)
   if code:timefacts.append({'station_line_code':code[1],'published_direction':code[2],'title':m[1],'url':m[2],'revision_label':rev[1] if rev else 'not_retained_in_excerpt','use':'directional service evidence only; no departures parsed or activated'})
 csvwrite(E/'provider_direction_evidence.csv',timefacts)
 return {'toei_station_pages':len(bodies),'jr_station_catalogs':len(rows),'reported_transfer_relations':len(transfers),'observed_timetable_direction_headers':len(timefacts),'physical_station_diagrams_valid_locally':0,'detailed_timetable_model_built':False}
def main():
 source_summary=parse_sources()
 # Existing data are frozen. All changes are in new output directories.
 checks=[]
 def ck(name,value):checks.append({'check':name,'pass':bool(value)});assert value,name
 ck('toei_english_does_not_match_tokyo_metro',operator_identity({'operator:en':'Tokyo Metropolitan Bureau of Transportation'})=='toei')
 ck('metro_identity_retained',operator_identity({'operator:en':'Tokyo Metro'})=='metro')
 ck('through_service_not_infrastructure_operator',operator_identity({'operator':'京成電鉄','network':'都営地下鉄'}) is None)
 ck('jr_japanese_alias',operator_identity({'operator':'JR東日本'})=='JR-East')
 ck('multi_operator_identity_held',operator_identity({'operator':'JR-East;JR-Central'}) is None)
 support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');aff=support['Affine Transform'];eligible=support['Eligible Ground Mask'];area=support['Ground Area'];nr,nc=eligible.shape
 ck('canonical_grid_not_gdal_identity',np.allclose(aff,[10,0,-11360,0,-10,-32320]))
 n02=json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/stations_regional_6677.geojson').read_text())['features'];groups=collections.defaultdict(list)
 for f in n02:
  p=f['properties']
  if p['intersects_core']:groups[(public_operator(p['N02_004']),canon(p['N02_005']))].append(f)
 stations={}
 for (op,name),features in groups.items():
  geom=unary_union([shape(f['geometry']) for f in features]);point=geom.centroid;sid=op+':'+name
  stations[sid]={'station_id':sid,'operator':op,'station_name':features[0]['properties']['N02_005'],'station_line_records':len(features),'source_station_codes':';'.join(sorted({f['properties']['N02_005c'] for f in features})),'source_group_codes':';'.join(sorted({f['properties']['N02_005g'] for f in features})),'legal_routes':';'.join(sorted({f['properties']['N02_003'] for f in features})),'x_m':point.x,'y_m':point.y,'entrances':set(),'geometry':geom}
 rels=[json.loads(s) for s in (BASE/'osm_relations.jsonl').open()];areas={r['id']:r for r in rels if r['tags'].get('public_transport')=='stop_area'};nodeareas=collections.defaultdict(set);routeops=collections.defaultdict(set)
 for r in areas.values():
  for m in r['members']:
   if m['type']=='node':nodeareas[m['id']].add(r['id'])
 for r in rels:
  op=operator_identity(r['tags'])
  if r['tags'].get('type')=='route' and op:
   for m in r['members']:
    if m['type']=='node' and m['role'].startswith('stop'):
     for a in nodeareas[m['id']]:routeops[a].add(op)
 entrances=rr(BASE/'osm_station_entrance_audit.csv');links=[];held=[];points={}
 for e in entrances:
  # Include a near-border entrance when it has an explicit membership to a
  # core station identity. No nearest-station join or centroid imputation.
  id=int(e['osm_node_id']);x,y=float(e['x_m']),float(e['y_m']);p=Point(x,y);assign=set()
  for aid in nodeareas[id]:
   ar=areas[aid];op=operator_identity(ar['tags']);basis='explicit_station_area_operator'
   if op is None and len(routeops[aid])==1:op=next(iter(routeops[aid]));basis='unique_explicit_route_operator_in_station_area'
   name=canon(ar['tags'].get('name:ja',ar['tags'].get('name','')));sid=str(op)+':'+name
   if sid not in stations:continue
   dist=stations[sid]['geometry'].distance(p)
   if dist>500:held.append({'osm_node_id':id,'station_id':sid,'issue':'explicit_name_operator_link_but_beyond_500m_identity_check','distance_m':dist});continue
   assign.add(sid);links.append({'osm_node_id':id,'station_id':sid,'source_stop_area':aid,'identity_basis':basis,'distance_to_official_station_geometry_m':dist,'corridor_verified':False})
  if e['inside_core']=='True' or assign:
   points[id]={**e,'x_m':x,'y_m':y,'station_ids':sorted(assign),'entry_ground_position_verified':False,'inlet_threshold_elevation_m':None,'barrier_state_verified':False}
   for sid in assign:stations[sid]['entrances'].add(id)
 ck('all_673_core_source_entrances_retained',sum(p['inside_core']=='True' for p in points.values())==673)
 csvwrite(P/'station_entrance_identity_links.csv',links);csvwrite(P/'held_station_identity_links.csv',held,fields=['osm_node_id','station_id','issue','distance_m'])
 pointrows=[{k:v for k,v in p.items() if k not in ['source_tags','walk_native_nodes','source_station_area_ids']} for p in points.values()];csvwrite(P/'entrance_inventory.csv',pointrows)
 # Native mean-depth fields are open-ground cell means, not pointwise entrance
 # depths. Neighborhoods use cell-center disks and preserve eligible areas.
 neighborhoods={};neighborhood_rows=[]
 for id,p in points.items():
  x,y=p['x_m'],p['y_m'];r=int(math.floor((aff[5]-y)/10));c=int(math.floor((x-aff[2])/10));p['point_row']=r;p['point_col']=c;p['point_in_grid']=0<=r<nr and 0<=c<nc;p['point_has_eligible_ground']=bool(eligible[r,c]) if p['point_in_grid'] else False
  for radius in [10,20,30]:
   r0=max(0,r-math.ceil(radius/10)-1);r1=min(nr,r+math.ceil(radius/10)+2);c0=max(0,c-math.ceil(radius/10)-1);c1=min(nc,c+math.ceil(radius/10)+2);rrr,ccc=np.mgrid[r0:r1,c0:c1];xx=aff[2]+(ccc+.5)*10;yy=aff[5]-(rrr+.5)*10;sel=((xx-x)**2+(yy-y)**2<=radius**2)&eligible[rrr,ccc];ri,ci=rrr[sel],ccc[sel];neighborhoods[id,radius]=(ri,ci);neighborhood_rows.append({'osm_node_id':id,'radius_m':radius,'eligible_cells':len(ri),'eligible_open_ground_area_m2':float(area[ri,ci].sum()),'sample_definition':'10m cell-center disk, eligible open ground only; no nearest-cell filling','point_in_grid':p['point_in_grid'],'point_cell_eligible':p['point_has_eligible_ground']})
 csvwrite(P/'entrance_neighborhood_support.csv',neighborhood_rows)
 events=rr(ROOT/'data/exp/data-preprocessing/three_ward_reference_expansion_12_4_v1/response_inventory.csv');ck('only_existing_open_training_validation_events',len(events)==16 and all(e['split'] in ['train','validation'] and e['response_status']=='qualified_candidate' for e in events))
 stationhoods={};vicinityrows=[]
 for sid,station in stations.items():
  for radius in [50,100]:
   geom=station['geometry'].buffer(radius);x0,y0,x1,y1=geom.bounds;c0=max(0,int(math.floor((x0-aff[2])/10)));c1=min(nc,int(math.ceil((x1-aff[2])/10)));r0=max(0,int(math.floor((aff[5]-y1)/10)));r1=min(nr,int(math.ceil((aff[5]-y0)/10)))
   if c1<=c0 or r1<=r0:ri=ci=np.array([],dtype=int)
   else:
    ri,ci=np.mgrid[r0:r1,c0:c1];ok=contains_xy(geom,aff[2]+(ci+.5)*10,aff[5]-(ri+.5)*10)&eligible[ri,ci];ri,ci=ri[ok],ci[ok]
   stationhoods[sid,radius]=(ri,ci)
 eventrows=[];bindings=[];envelopelo=np.full((nr,nc),np.nan);envelopehi=np.full((nr,nc),np.nan)
 for e in events:
  path=ROOT/e['label_path'];z=np.load(path);lo=z['Peak Mean Depth Lower'];hi=z['Peak Mean Depth Upper'];valid=z['Admitted Mask']&~z['Quarantine Mask']&eligible&np.isfinite(lo)&np.isfinite(hi);ck(e['case_id']+'_bounds_ordered',np.all(hi[valid]+1e-9>=lo[valid]));ck(e['case_id']+'_quarantine_excluded',not np.any(valid&z['Quarantine Mask']))
  envelopelo=np.fmax(envelopelo,np.where(valid,lo,np.nan));envelopehi=np.fmax(envelopehi,np.where(valid,hi,np.nan));bindings.append({'case_id':e['case_id'],'parent_id':e['parent_id'],'split':e['split'],'rain_peak_mm_h':e['rain_peak_mm_h'],'rain_total_mm':e['rain_total_mm'],'stage_amplitude_m':e['stage_amplitude_m'],'label_path':e['label_path'],'label_sha256':sha(path)})
  for id,p in points.items():
   for radius in [10,20,30]:
    ri,ci=neighborhoods[id,radius];ok=valid[ri,ci];ai=float(area[ri,ci].sum());av=float(area[ri[ok],ci[ok]].sum());lv=lo[ri[ok],ci[ok]];uv=hi[ri[ok],ci[ok]];aa=area[ri[ok],ci[ok]]
    row={'case_id':e['case_id'],'parent_id':e['parent_id'],'split':e['split'],'osm_node_id':id,'radius_m':radius,'eligible_open_ground_m2':ai,'admitted_open_ground_m2':av,'admitted_fraction_of_eligible':av/ai if ai else None,'neighborhood_peak_mean_depth_lower_max_m':float(lv.max()) if len(lv) else None,'neighborhood_peak_mean_depth_upper_max_m':float(uv.max()) if len(uv) else None,'support_state':'unknown_no_eligible_ground' if not ai else 'unknown_no_admitted_ground' if not av else 'partly_unknown' if av<ai-1e-7 else 'admitted_neighborhood'}
    for th in [.1,.3,.5]:
     key=str(th).replace('.','p');row['ground_area_lower_ge_'+key+'_m2']=float(aa[lv>=th].sum());row['ground_area_upper_ge_'+key+'_m2']=float(aa[uv>=th].sum())
    eventrows.append(row)
  for sid in stations:
   for radius in [50,100]:
    ri,ci=stationhoods[sid,radius];ok=valid[ri,ci];lv=lo[ri[ok],ci[ok]];uv=hi[ri[ok],ci[ok]];aa=area[ri[ok],ci[ok]];ai=float(area[ri,ci].sum());av=float(aa.sum());row={'case_id':e['case_id'],'station_id':sid,'radius_m':radius,'eligible_ground_m2':ai,'admitted_ground_m2':av,'worst_nearby_peak_mean_depth_lower_m':float(lv.max()) if len(lv) else None,'worst_nearby_peak_mean_depth_upper_m':float(uv.max()) if len(uv) else None,'role':'Official station-line geometry vicinity only; not an entrance or train/deck flood depth'}
    for th in [.1,.3,.5]:row['peak_exceedance_ground_area_lower_ge_'+str(th).replace('.','p')+'_m2']=float(aa[lv>=th].sum())
    vicinityrows.append(row)
  print('SCREEN',e['case_id'],len(eventrows),flush=True)
 csvwrite(P/'station_vicinity_scenario_screen.csv',vicinityrows);csvwrite(P/'entrance_scenario_surface_screen.csv',eventrows);csvwrite(P/'scenario_bindings.csv',bindings)
 primary=[r for r in eventrows if r['radius_m']==20];byid=collections.defaultdict(list)
 for r in primary:byid[r['osm_node_id']].append(r)
 summaries=[]
 for id,p in points.items():
  rows=byid[id];l=[r['neighborhood_peak_mean_depth_lower_max_m'] for r in rows if r['neighborhood_peak_mean_depth_lower_max_m'] is not None];u=[r['neighborhood_peak_mean_depth_upper_max_m'] for r in rows if r['neighborhood_peak_mean_depth_upper_max_m'] is not None];low=max(l) if l else None;up=max(u) if u else None
  row={'osm_node_id':id,'x_m':p['x_m'],'y_m':p['y_m'],'inside_core':p['inside_core']=='True','station_ids':';'.join(p['station_ids']),'exit_ref':p['exit_ref'],'source_relative_level':p['relative_level'],'point_in_grid':p['point_in_grid'],'point_has_eligible_ground':p['point_has_eligible_ground'],'scenarios_with_admitted_neighborhood':len(l),'worst_nearby_peak_mean_depth_lower_m':low,'worst_nearby_peak_mean_depth_upper_m':up,'screen_status':'unknown' if low is None else 'nearby_surface_exposure_ge0p3m' if low>=.3 else 'nearby_surface_exposure_ge0p1m' if low>=.1 else 'below0p1m_on_available_support','not_a_probability':True,'in_station_ingress_verified':False,'closure_inferred':False}
  for th in [.1,.3,.5]:row['scenario_count_lower_ge_'+str(th).replace('.','p')]=sum(r['neighborhood_peak_mean_depth_lower_max_m'] is not None and r['neighborhood_peak_mean_depth_lower_max_m']>=th for r in rows)
  row['largest_peak_exceedance_footprint_ge0p3m_m2']=max(r['ground_area_lower_ge_0p3_m2'] for r in rows)
  row['minimum_admitted_fraction_when_eligible']=min([r['admitted_fraction_of_eligible'] for r in rows if r['admitted_fraction_of_eligible'] is not None],default=None)
  summaries.append(row)
 csvwrite(P/'entrance_screen_summary.csv',summaries);sm={r['osm_node_id']:r for r in summaries};stationrows=[]
 for sid,s in stations.items():
  es=[sm[id] for id in s['entrances']];ls=[x['worst_nearby_peak_mean_depth_lower_m'] for x in es if x['worst_nearby_peak_mean_depth_lower_m'] is not None];us=[x['worst_nearby_peak_mean_depth_upper_m'] for x in es if x['worst_nearby_peak_mean_depth_upper_m'] is not None];lower=max(ls) if ls else None
  row={k:v for k,v in s.items() if k not in ['entrances','geometry']};row.update({'associated_mapped_entrances':len(es),'entrances_with_admitted_neighborhood':len(ls),'entrances_without_admitted_support':len(es)-len(ls),'worst_nearby_peak_mean_depth_lower_m':lower,'worst_nearby_peak_mean_depth_upper_m':max(us) if us else None,'screen_state':'unknown_no_source_entrance_identity' if not es else 'unknown_no_admitted_neighborhood' if not ls else 'prioritize_nearby_surface_exposure_ge0p3m' if lower>=.3 else 'nearby_surface_exposure_ge0p1m' if lower>=.1 else 'below0p1m_on_available_support','entrance_inventory_complete':False,'interior_water_ingress':'unknown','station_shutdown':'not_assessed'})
  for th in [.1,.3,.5]:row['mapped_entrances_lower_ge_'+str(th).replace('.','p')]=sum(x['worst_nearby_peak_mean_depth_lower_m'] is not None and x['worst_nearby_peak_mean_depth_lower_m']>=th for x in es)
  vs=[x for x in vicinityrows if x['station_id']==sid and x['radius_m']==50];vv=[x['worst_nearby_peak_mean_depth_lower_m'] for x in vs if x['worst_nearby_peak_mean_depth_lower_m'] is not None];row['vicinity50m_worst_lower_m']=max(vv) if vv else None;row['vicinity50m_largest_peak_exceedance_footprint_ge0p3m_m2']=max(x['peak_exceedance_ground_area_lower_ge_0p3_m2'] for x in vs);row['mapped_entry_largest_peak_exceedance_footprint_ge0p3m_m2']=max([x['largest_peak_exceedance_footprint_ge0p3m_m2'] for x in es],default=None)
  row['vicinity50m_max_event_peak_exceedance_fraction_ge0p3m']=max([x['peak_exceedance_ground_area_lower_ge_0p3_m2']/x['admitted_ground_m2'] for x in vs if x['admitted_ground_m2']>0],default=None)
  row['station_vicinity_status']='unknown_no_admitted_ground' if not vv else 'vicinity_surface_exposure_ge0p3m' if max(vv)>=.3 else 'vicinity_surface_exposure_ge0p1m' if max(vv)>=.1 else 'below0p1m_on_available_vicinity_support'
  stationrows.append(row)
 csvwrite(P/'station_screen_summary.csv',stationrows)
 def geo(path,rows,key='osm_node_id'):
  dump(path,{'type':'FeatureCollection','name':path.stem,'crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':[{'type':'Feature','geometry':mapping(Point(r['x_m'],r['y_m'])),'properties':r} for r in rows]})
 geo(P/'entrance_surface_risk_6677.geojson',summaries);geo(P/'station_surface_risk_6677.geojson',stationrows,'station_id');np.savez_compressed(P/'display_only_scenario_peak_envelope.npz',lower=envelopelo,upper=envelopehi,affine=aff)
 # No scenario frequency is an annual risk probability; no mean-depth bin is a
 # shutdown or human passability threshold. Missing data never become zero.
 policy={'human_decision':'MILI-D-20261016-023','purpose':'Station flood-risk candidate screening first; operational disruption is a later explicit scenario.','station_units':'114 operator/name identity units from 154 official core station-line records; not 114 proven interchange complexes.','source_epochs':'N02 2025-12-31; OSM 2026-10-15 edit snapshot; existing terrain/hydraulic geometry epoch unchanged and not a certified 2026 station-entrance survey.','association':'Explicit OSM stop_area name plus operator or unique route operator, and <=500m to official same-identity station geometry. No nearest-station assignment. Relative floors and curb/inlet heights are unverified.','station_vicinity_fallback':'Separate 50/100m buffers of official station-line geometry screen nearby open ground, without filling missing entrance identities or assigning surface water to tracks/decks.', 'spatial_metric':'10m admitted open-ground mean-depth peak bounds in 10/20/30m cell-center disks around mapped entrance points. Primary radius20m; this is approach-neighborhood exposure, not point entrance depth.','quality':'Retain original exact/interval and quarantine masks. Nonadmitted, building-dominated, water, outside-grid and absent entrances stay unknown; never dry-filled. Coverage denominators refer to eligible open ground, not the entire footprint.','display_bins_m':[.1,.3,.5],'bins_are_not_shutdown_or_safety_thresholds':True,'time':'Existing per-cell event peaks and cross-event maximum are non-simultaneous stress screens; no isolation duration or simultaneous station shutdown is computed.','probability':'16 synthetic scenarios are not a sampled storm probability distribution; exposure counts are not annual probabilities.','hypothetical_later_shutdowns':['Selected exposed entrance links closed, alternative entrances retained','Selected stations closed or non-stop, through running specified separately','Selected running sections suspended, using declared station or flood-zone scenarios'],'shutdown_scenarios_not_executed':True,'interior_ingress_evidence_needed':'Compatible inlet/ground threshold elevations, barriers and actual inlet/corridor connection, or event observations. Surface exposure alone is not proof of interior flooding.','population_uncertainty_preserved':True,'new_hydraulics':False,'affected_people_computed':False,'sealed_test_opened':False,'detailed_timetable_model':False};dump(P/'screening_contract.json',policy)
 frozen=[]
 for mf in [ROOT/'data/exp/data-preprocessing/three_ward_normal_transport_network_v1/delivery_manifest.json',ROOT/'data/exp/data-preprocessing/three_ward_population_block_stability_v1/delivery_manifest.json']:
  m=json.loads(mf.read_text());items=m['products']+m.get('scripts',[]);failed=[r['file'] for r in items if sha(ROOT/r['file'])!=r['sha256']];ck(mf.parent.name+'_frozen_manifest_unchanged',not failed);frozen.append({'manifest':str(mf.relative_to(ROOT)),'sha256':sha(mf),'files_verified':len(items),'failures':failed})
 summary={'human_decision':'MILI-D-20261016-023','station_units':len(stationrows),'station_line_records':sum(s['station_line_records'] for s in stationrows),'retained_core_osm_entrances':673,'retained_core_and_source_associated_border_entrances':len(points),'mapped_entrances_associated_to_core_station_identity':sum(bool(p['station_ids']) for p in points.values()),'station_vicinity_states':dict(collections.Counter(s['station_vicinity_status'] for s in stationrows)),'station_screen_states':dict(collections.Counter(s['screen_state'] for s in stationrows)),'entrance_screen_states':dict(collections.Counter(s['screen_status'] for s in summaries if s['inside_core'])),'existing_synthetic_events':len(events),'forcing_rain_peak_mm_h_range':[min(float(e['rain_peak_mm_h']) for e in events),max(float(e['rain_peak_mm_h']) for e in events)],'forcing_total_rain_mm_range':[min(float(e['rain_total_mm']) for e in events),max(float(e['rain_total_mm']) for e in events)],'provider_evidence_supplement':source_summary,'verified_interior_flooded_stations':None,'shutdown_stations_computed':False,'affected_people_computed':False,'frozen_deliveries':frozen};dump(R/'summary.json',summary);dump(R/'validation.json',{'checks':checks,'all_passed':all(c['pass'] for c in checks),'frozen':frozen,'shutdown_inference':False,'new_hydraulics':False});print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
