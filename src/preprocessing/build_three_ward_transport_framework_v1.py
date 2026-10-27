"""Audit official transport inputs and build interfaces without isolation counts.
Usage: .venv/bin/python src/preprocessing/build_three_ward_transport_framework_v1.py
No transport disruption calculation, population relocation, training or hydraulics.
"""
from pathlib import Path
import csv,json,math,hashlib,zipfile,io,collections,xml.etree.ElementTree as ET,warnings
import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely.geometry import shape,mapping,box,Point,LineString
from shapely.ops import transform,unary_union
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/three_ward_transport_framework_v1';S=OUT/'sources';R=ROOT/'data/exp/feasibility-check/three_ward_transport_framework_v1';R.mkdir(parents=True,exist_ok=True)
xy=Transformer.from_crs(6668,6677,always_xy=True).transform;ll=Transformer.from_crs(6677,6668,always_xy=True).transform

def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def writecsv(p,rows,fields=None):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def geo(path,features):dump(path,{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':features})
def feat(g,p):return {'type':'Feature','geometry':mapping(g),'properties':p}
# Freeze the completed population delivery; official counts have a distinct role.
POP=ROOT/'data/processed/three_ward_population_block_stability_v1';pm=ROOT/'data/exp/data-preprocessing/three_ward_population_block_stability_v1/delivery_manifest.json';frozen=json.loads(pm.read_text())
assert all(sha(ROOT/r['file'])==r['sha256'] for r in frozen['products']+frozen['scripts'])
baseline=readcsv(POP/'official_2020_block_count_baseline.csv');assert sum(float(r['official_2020_residents']) for r in baseline)==496345
writecsv(OUT/'night_population_baseline.csv',baseline)
wardfeatures=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];wards={f['properties']['ward'][:5]:transform(xy,shape(f['geometry'])) for f in wardfeatures};core=unary_union(list(wards.values()));roadhalo=core.buffer(5000);railhalo=core.buffer(20000)
geo(OUT/'inventory_domains_6677.geojson',[feat(core,{'role':'reporting_core','distance_m':0}),feat(roadhalo,{'role':'provisional_road_acquisition_halo_not_validated_routing_extent','distance_m':5000}),feat(railhalo,{'role':'regional_rail_inventory_not_final_service_extent','distance_m':20000})])
# Full national N02 remains on disk; geometries are not truncated at ward edges.
z=zipfile.ZipFile(S/'N02-25_GML.zip');rail=[];stations=[]
for kind,target in [('RailroadSection',rail),('Station',stations)]:
 n=next(n for n in z.namelist() if n.endswith('_'+kind+'.geojson'))
 for i,f in enumerate(json.loads(z.read(n))['features']):
  g=transform(xy,shape(f['geometry']))
  if not g.intersects(railhalo):continue
  p=f['properties'];pp={**p,'source_row':i,'intersects_core':g.intersects(core),'intersects_road_halo':g.intersects(roadhalo),'operator_class':'metro' if p['N02_004']=='東京地下鉄' else 'toei' if p['N02_004']=='東京都' else 'jr' if p['N02_004'] in ['東日本旅客鉄道','東海旅客鉄道'] else 'other','routing_qualified':False,'inventory_geometry_not_service_sequence':True};target.append(feat(g,pp))
geo(OUT/'rail_sections_regional_6677.geojson',rail);geo(OUT/'stations_regional_6677.geojson',stations)
rail_summary=[]
for op,line in sorted({(f['properties']['N02_004'],f['properties']['N02_003']) for f in rail if f['properties']['intersects_core']}):
 rr=[f for f in rail if f['properties']['N02_004']==op and f['properties']['N02_003']==line];ss=[f for f in stations if f['properties']['N02_004']==op and f['properties']['N02_003']==line];rail_summary.append({'operator':op,'legal_route_name':line,'regional_sections':len(rr),'core_sections':sum(f['properties']['intersects_core'] for f in rr),'regional_station_records':len(ss),'core_station_records':sum(f['properties']['intersects_core'] for f in ss),'has_cross_core_geometry':any(shape(f['geometry']).intersects(core.boundary) for f in rr),'service_order_verified':False,'transfer_links_verified':False})
writecsv(OUT/'core_rail_corridor_inventory.csv',rail_summary)
# Road samples demonstrate schema availability; all required tile requests are listed.
roads=[];roadids=set();dup=0
for p in sorted(S.glob('gsi_rdcl_16_*.geojson')):
 for i,f in enumerate(json.loads(p.read_text())['features']):
  g=transform(xy,shape(f['geometry']));pr=f['properties'];rid=str(pr.get('rID',''));key=(rid,g.wkb_hex)
  if key in roadids:dup+=1;continue
  roadids.add(key);roads.append(feat(g,{**pr,'audit_id':p.stem+':'+str(i),'source_tile':p.stem,'direction_and_mode_access_verified':False}))
geo(OUT/'road_schema_sample_6677.geojson',roads)
road_field_audit=[]
for field in ['rID','type','state','lvOrder','rnkWidth','Width','rdCtg','motorway','lfSpanFr','devDate']:
 vals=[f['properties'].get(field) for f in roads];cnt=collections.Counter(str(v) for v in vals if v not in [None,''])
 road_field_audit.append({'field':field,'features':len(vals),'nonempty':sum(v not in [None,''] for v in vals),'top_values':json.dumps(cnt.most_common(10),ensure_ascii=False)})
writecsv(R/'road_sample_fields.csv',road_field_audit)
lo=transform(ll,roadhalo).bounds
xt=lambda x:int((x+180)/360*65536)
yt=lambda y:int((1-math.asinh(math.tan(math.radians(y)))/math.pi)/2*65536)
def tilepoly(x,y):
 def lat(a):return math.degrees(math.atan(math.sinh(math.pi*(1-2*a/65536))))
 return transform(xy,box(x/65536*360-180,lat(y+1),(x+1)/65536*360-180,lat(y)))
tiles=[]
for x in range(xt(lo[0]),xt(lo[2])+1):
 for y in range(yt(lo[3]),yt(lo[1])+1):
  g=tilepoly(x,y)
  if g.intersects(roadhalo):tiles.append({'zoom':16,'x':x,'y':y,'intersects_core':g.intersects(core),'downloaded_schema_sample':(S/f'gsi_rdcl_16_{x}_{y}.geojson').exists(),'url':f'https://cyberjapandata.gsi.go.jp/xyz/experimental_rdcl/16/{x}/{y}.geojson','extent_status':'provisional_5km_halo_requires_routing_extent_test'})
writecsv(OUT/'provisional_road_tile_requests.csv',tiles)
# Keep each walking source namespace separate. Old and new editions overlap and
# must not be blindly concatenated or joined by two-dimensional proximity.
nodes=[];edges=[];walking=[];edgegeoms={};nodegeoms={};walkgeo=[]
for tag in ['tokyo_2018','central_2017','daimon_2018spec']:
 nf=json.loads((S/f'walking_{tag}_nodes.geojson').read_text())['features'];ef=json.loads((S/f'walking_{tag}_links.geojson').read_text())['features'];nd={str(f['properties']['node_id']).strip():f for f in nf};dangling=0;maxdist=0.;floors=collections.Counter();struct=collections.Counter();bounds=0
 for i,f in enumerate(nf):
  p=f['properties'];g=transform(xy,shape(f['geometry']));nid=tag+':'+str(p['node_id']).strip();nodegeoms[nid]=g;fl=p.get('floor','');floors[str(fl)]+=1;bounds+=int(p.get('in_out',99)==2)
  nodes.append({'node_id':nid,'source_node_id':p['node_id'],'source':tag,'x_6677_m':g.x,'y_6677_m':g.y,'floor_index_not_elevation_m':fl,'inside_outside_code':p.get('in_out',''),'node_role':'generic_facility_boundary_candidate' if p.get('in_out')==2 else 'walking_node','station_identity_verified':False,'survey_epoch_verified_current':False})
 for i,f in enumerate(ef):
  p=f['properties'];a=str(p['start_id']).strip();b=str(p['end_id']).strip();g=transform(xy,shape(f['geometry']));eid=tag+':'+str(p['link_id']).strip();direction=int(p.get('direction',99));st=int(p.get('rt_struct',99));struct[st]+=1
  missing=a not in nd or b not in nd;dangling+=int(missing)
  if not missing:
   ag=nodegeoms[tag+':'+a];bg=nodegeoms[tag+':'+b];cs=list(g.coords);maxdist=max(maxdist,Point(cs[0]).distance(ag),Point(cs[-1]).distance(bg))
  fls=[nd.get(k,{}).get('properties',{}).get('floor') for k in [a,b]]
  grade='underground' if st==5 else 'elevated' if st==6 else 'indoor' if st==7 else 'surface_candidate' if st in [1,2,3,4] and all(str(v) in ['0','0.0'] for v in fls) else 'unverified'
  row={'edge_id':eid,'source':tag,'source_link_id':p['link_id'],'from_node':tag+':'+a,'to_node':tag+':'+b,'length_m':g.length,'direction_code':direction,'allow_forward_candidate':direction in [1,2],'allow_reverse_candidate':direction in [1,3],'route_structure_code':st,'route_type_code':p.get('route_type',''),'grade_class_candidate':grade,'endpoints_present':not missing,'routing_admission':'historical_candidate_not_current_verified','direct_surface_flood_rule_allowed':grade=='surface_candidate'}
  edges.append(row);edgegeoms[eid]=g;walkgeo.append(feat(g,row))
 walking.append({'source':tag,'nodes':len(nf),'links':len(ef),'duplicate_node_ids':len(nf)-len(nd),'duplicate_link_ids':len(ef)-len({f['properties']['link_id'] for f in ef}),'dangling_link_endpoints':dangling,'max_endpoint_offset_m':maxdist,'floor_field_present':any('floor' in f['properties'] for f in nf),'floor_values':dict(floors),'facility_boundary_nodes_not_verified_rail_entrances':bounds,'route_structures':dict(struct),'core_links':sum(transform(xy,shape(f['geometry'])).intersects(core) for f in ef),'coverage_is_complete_three_ward_network':False})
writecsv(OUT/'walking_candidate_nodes.csv',nodes);writecsv(OUT/'walking_candidate_edges.csv',edges);geo(OUT/'walking_candidates_6677.geojson',walkgeo)
dump(R/'walking_source_checks.json',walking)
ids17={e['source_link_id'] for e in edges if e['source']=='central_2017'};ids18={e['source_link_id'] for e in edges if e['source']=='tokyo_2018'}
# Decode the historical Minato CSV; station association does not identify a
# present-day street entrance. Ticket gates and generic facility doors are kept.
z=zipfile.ZipFile(S/'walking_minato_2010_2011.zip');zz=zipfile.ZipFile(io.BytesIO(z.read(next(n for n in z.namelist() if 'csv.zip' in n))));tables={n.encode('cp437').decode('cp932'):list(csv.DictReader(io.StringIO(zz.read(n).decode('cp932')))) for n in zz.namelist() if n.upper().endswith('.CSV')}
oldnodes=next(v for k,v in tables.items() if k.endswith('ノード情報.CSV'));oldentrances=next(v for k,v in tables.items() if k.endswith('出入口情報.CSV'));fac=next(v for k,v in tables.items() if k.endswith('公共施設の情報.CSV'));oldmap={r['ノードID']:r for r in oldnodes};facmap={r['施設ID']:r for r in fac};entrances=[];entrancegeo=[]
def dms(s):
 a=s.strip().split('.');return float(a[0])+float(a[1])/60+float('.'.join(a[2:]))/3600
for r in oldentrances:
 facility=facmap.get(r['対応施設ID'],{});name=facility.get('名称','')
 if '駅' not in name:continue
 n=oldmap.get(r['対応ノードID']);g=None
 if n:
  try:g=Point(*xy(dms(n['経度']),dms(n['緯度'])))
  except (ValueError,IndexError):pass
 desc=r['出入口の名称'];p={'candidate_id':'minato_2010_2011:'+r['出入口ID'],'source_entrance_id':r['出入口ID'],'source_node_id':r['対応ノードID'],'source_facility_id':r['対応施設ID'],'facility_name':name,'entrance_name':desc,'name_hint':'ticket_gate' if '改札' in desc else 'exit' if '出口' in desc or '出入口' in desc else 'unspecified','node_coordinate_found':g is not None,'raw_node_height_or_floor_not_datum':n.get('高さ','') if n else '', 'entrance_threshold_elevation_verified':False,'current_ground_entrance_verified':False,'n02_station_join_verified':False}
 entrances.append(p)
 if g is not None:entrancegeo.append(feat(g,p))
writecsv(OUT/'historical_station_access_candidates.csv',entrances);geo(OUT/'historical_station_access_candidates_6677.geojson',entrancegeo)
# Scan every downloaded PLATEAU transportation member, without reinterpreting
# road polygons or bridge solids as a passenger topology.
plateau=[]
for p in sorted((ROOT/'data/raw/plateau/2025').glob('*/*citygml*.zip')):
 zz=zipfile.ZipFile(p);members=[n for n in zz.namelist() if n.endswith('.gml')];tran=[n for n in members if '/tran/' in n];brid=[n for n in members if '/brid/' in n];rwy=[n for n in members if '/rwy/' in n];counts=collections.Counter()
 for n in tran:
  s=zz.read(n);counts['lod0Network']+=s.count(b'lod0Network>')//2;counts['lod1MultiSurface']+=s.count(b'lod1MultiSurface>')//2
 plateau.append({'archive':str(p.relative_to(ROOT)),'tran_members':len(tran),'bridge_members':len(brid),'railway_members':len(rwy),'lod0_network_occurrences':counts['lod0Network'],'lod1_road_surface_occurrences':counts['lod1MultiSurface'],'routing_topology_qualified':False})
writecsv(R/'plateau_transport_inventory.csv',plateau)
# Inspect one qualified existing map and the saved exact clocks. GDAL exposes
# identity raster georeferencing for this NetCDF: use its x/y arrays and verify.
run=ROOT/'data/processed/three_ward_reference_expansion_12_4_v1/runs/train-06__regular_10m';nc=run/'sfincs_map.nc';support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');aff=Affine(*support['Affine Transform']);nr,ncol=support['Eligible Ground Mask'].shape
clock=readcsv(run/'udt_output_clock.csv');maps=[r for r in clock if r['write_map']=='T'];actual=np.array([float(r['actual_state_time']) for r in maps]);warnings.filterwarnings('ignore',category=rasterio.errors.NotGeoreferencedWarning)
def readnc(v):
 with rasterio.open('NETCDF:"'+str(nc)+'":'+v) as d:return d.read(1),d.tags(),d.count,d.units
xs,_,_,_=readnc('x');ys,_,_,_=readnc('y');expected_x=aff.c+(np.arange(ncol)+.5)*aff.a;expected_y=aff.f+(np.arange(nr)+.5)*aff.e;coorderror=max(float(np.abs(xs-expected_x[None,:]).max()),float(np.abs(ys-expected_y[:,None]).max()));assert coorderror<1e-5
hazardvars={}
for v in ['zs','h','u','v','subgrid_volume']:
 with rasterio.open('NETCDF:"'+str(nc)+'":'+v) as d:hazardvars[v]={'count':d.count,'shape':[d.height,d.width],'units':d.units[0],'long_name':d.tags().get(v+'#long_name'),'crs_available_in_gdal':d.crs is not None,'identity_transform_in_gdal':d.transform==Affine.identity()}
assert all(v['count']==len(actual)==49 for v in hazardvars.values())
audit=json.loads((ROOT/'data/exp/data-preprocessing/three_ward_reference_expansion_12_4_v1/expansion/train-06/audit.json').read_text());assert audit['all_checks_passed']
dump(R/'hazard_interface_audit.json',{'example_case':'train-06','role':'Existing qualified synthetic scenario only; no new hydraulics','run':str(run.relative_to(ROOT)),'case_audit_sha256':sha(ROOT/'data/exp/data-preprocessing/three_ward_reference_expansion_12_4_v1/expansion/train-06/audit.json'),'variables':hazardvars,'actual_map_time_s':actual.tolist(),'maximum_actual_output_gap_s':float(np.diff(actual).max()),'maximum_coordinate_error_m':coorderror,'canonical_north_up_affine':list(aff)[:6],'epsg':6677,'gdal_identity_transform_must_not_be_used':True,'native_shallow_censoring_and_reference_masks_must_be_retained':True,'reanalysis_is_not_a_real_observed_storm':True})
# An actual geometric edge-cell join proves the format interface, not passability.
# Use only the two 2018-spec sources, keeping the superseded central subset out.
joins=[];joinchecks=[];grid=box(aff.c,aff.f+nr*aff.e,aff.c+ncol*aff.a,aff.f);eligible=support['Eligible Ground Mask']
for e in edges:
 if e['source']=='central_2017':continue
 g=edgegeoms[e['edge_id']];within=g.intersection(grid);minx,miny,maxx,maxy=g.bounds;c0=max(0,int(math.floor((minx-aff.c)/10)));c1=min(ncol-1,int(math.floor((maxx-aff.c)/10)));r0=max(0,int(math.floor((aff.f-maxy)/10)));r1=min(nr-1,int(math.floor((aff.f-miny)/10)));total=0.
 # Tiny deterministic shift resolves the measure-zero case of a line exactly
 # following a cell edge; half-open ownership is defined by interior samples.
 # Intersections on cell boundaries are deduplicated by each fragment midpoint.
 owned=[]
 for r in range(r0,r1+1):
  for c in range(c0,c1+1):
   cell=box(aff.c+c*10,aff.f-(r+1)*10,aff.c+(c+1)*10,aff.f-r*10);piece=g.intersection(cell)
   if piece.length<=1e-7:continue
   parts=[piece] if piece.geom_type=='LineString' else list(piece.geoms) if hasattr(piece,'geoms') else []
   length=0.
   for part in parts:
    if part.geom_type!='LineString' or part.length<=1e-7:continue
    pt=part.interpolate(.5,normalized=True);owner_c=min(ncol-1,int(math.floor((pt.x-aff.c)/10)));owner_r=min(nr-1,int(math.floor((aff.f-pt.y)/10)))
    if owner_c==c and owner_r==r:length+=part.length
   if length<=1e-7:continue
   total+=length;joins.append({'edge_id':e['edge_id'],'row':r,'col':c,'intersection_length_m':length,'eligible_ground_cell':bool(eligible[r,c]),'grade_class_candidate':e['grade_class_candidate'],'quality_gate':'spatial_join_only_not_passability'})
 joinchecks.append({'edge_id':e['edge_id'],'length_in_grid_m':within.length,'joined_length_m':total,'join_residual_m':total-within.length})
writecsv(OUT/'walking_edge_hazard_cells.csv',joins);writecsv(R/'edge_hazard_join_checks.csv',joinchecks)
assert max(abs(r['join_residual_m']) for r in joinchecks)<1e-4
# Explicit contracts prevent administrative boundaries and incomplete networks
# from generating unqualified islands or artificially precise people counts.
contract={
 'human_decision':'MILI-D-20261016-020','stage':'Data feasibility and executable interfaces only; no real isolation or affected counts',
 'population':{'baseline':'night_population_baseline.csv','census_date':'2020-10-01','units':4746,'total':496345,'fixed_total_per_block':True,'not_daytime_or_trip_demand':True,'building_sidecar':str(POP.relative_to(ROOT)),'split_block_rule':'Do not attach all residents to a centroid or sum block and building populations. If a future network component splits a block, keep the official total fixed but location unresolved; default component support [0,P_block] subject to components summing to P_block until a new constrained within-block allocation is independently qualified. Frozen mesh-based building intervals cannot be silently rescaled to official block totals.','current_population_movement':False},
 'extent':{'core':'Chiyoda, Chuo, Minato','road_initial_inventory_halo_m':5000,'rail_inventory_halo_m':20000,'buffers_are_not_sufficiency_proofs':True,'final_extent_test':'After a routable baseline exists, expand road halo 5 to 10 km and retain rail corridors to verified external interchanges/termini and through-running operators; expand further if core-to-fixed-destination reachability or detours change. Keep full national N02 source for expansion.','outside_hazard_domain':'unknown hazard; optimistic available and pessimistic impaired branches must be explicit. Outside is not silently dry or a safe destination.','external_anchors':'verified destination nodes or regional connectors with stated hazard/service assumptions, not arbitrary clipped endpoints or every ward-boundary crossing'},
 'layers':{'walk':'pedestrian access with explicit directions, levels and opening times','road_vehicle':'separate directed vehicle-access graph; one-way, access restrictions, ramps and deck elevations pending','rail':'operator and service-specific station/platform nodes plus verified consecutive-service links','station_access':'ground entry, gate, concourse and platform are distinct; station group codes are candidate matching aids, not transfers','transfer':'only verified same-level paths and explicit stairs/lifts or interchange links; do not connect every planar crossing or nearby station','other_modes':'keep geometry inventory for other operators; include relevant through-running and substitutes before claiming citywide multimodal isolation. Bus service and capacity are not presently modeled.'},
 'hazard_coupling':{'crs':'EPSG:6677 horizontal, north-up canonical affine verified against NetCDF x/y; native arrays require explicit orientation checks','time':'actual solver output timestamps, 49 states about 300 s apart; never synthesize simultaneous states by combining cellwise peaks','primary_surface_metric':'h_road(t)=max(0,eta_surface(t)-z_travel_surface), only with compatible vertical datum and valid wet-state/terrain interpretation. Dry or censored output is not deep flooding or proof of exact zero.','mean_depth_role':'open-ground 10 m mean-depth intervals support scenario screening only; they are not pointwise curb, deck or entrance water depth.','grade_rules':'surface streets sample valid local water state; elevated decks and indoor/underground links require deck/portal/path information and separate assumptions. Floodwater under a bridge does not close its deck automatically.','entrance_rules':'external approach exposure, ground threshold, barrier status and connecting passage state are separate; generic facility boundary node is not a verified station entrance.','rail_rules':'an entrance closure disables only that access link. A station can have alternative access; station non-stop, a platform closure, running-section suspension and line-wide suspension are separate explicit scenarios. Surface flooding alone does not predict operator decisions or tunnel water.','quality_masks':'retain Exact, Interval, Wide Interval and Quarantine masks. Quarantine, nodata, unknown vertical datum and outside-domain links remain unknown.','threshold_policy':'No physical passability, operational shutdown, travel-speed or human-safety threshold is selected here. Later threshold/duration choices are scenario assumptions unless source-validated.','wet_depth_velocity':'retain eta/depth and u/v units, staggering and definition before any depth-velocity criteria; no universal safety formula is inferred','peak_envelope_role':'event stress screen only, not simultaneous isolation duration or service recovery'},
 'outputs_after_future_qualification':['mode-specific reachability and loss relative to matched normal network','connected but degraded access/detour separated from strict no-path','station access availability separated from rail service availability','block accessibility classes with unresolved split-block population; no stranded count without additional definition and demand'],
 'release_gates':['complete and epoch-labeled road/walk coverage','mode permissions and grade-separated topology checks','verified station-access and transfer mapping','regional network closure and extent sensitivity','hazard datum, orientation, actual time and quality handling','predeclared destinations and scenario service states'],
 'no_current_claims':['No validated routable three-ward multimodal network','No real road or rail closures computed','No operator shutdown prediction','No affected or stranded people','No new simulation or training']}
dump(OUT/'framework_contract.json',contract)
schema={'crs_epsg':6677,'node_fields':{'node_id':'unique stable namespace/source ID','layer':'walk / vehicle / access / platform / rail / external_anchor','x_m':'easting','y_m':'northing','z_m':'null unless compatible absolute travel-surface elevation verified','level_index':'relative floor index, never substitute for metres','operator_service_station_ids':'null unless verified','source_epoch':'survey/release/update separately','quality_state':'candidate/verified/unknown'},'edge_fields':{'edge_id':'unique source-derived ID','from_node':'directed start','to_node':'directed end','edge_type':'walk/vehicle/access/vertical/transfer/rail_service','length_m':'geometry length, not rail travel time','direction':'explicit modes and forward/reverse permission','grade':'surface/elevated/underground/indoor/unknown','valid_time_window':'service/opening calendar; null=unknown','hazard_link':'sparse length-weighted cell intersections for suitable surfaces; portal association otherwise','travel_time_s':'null until defensible speed/timetable/transfer cost','source_epoch':'data provenance','topology_admission':'pending/qualified'},'state_fields':['event_id','actual_time_s','edge_id','physical_state:open/closed/unknown','service_state:operating/suspended/unknown/not_applicable','state_basis:observation/explicit_scenario','reason','threshold_assumption_id','quality_mask','outside_hazard_domain'],'population_link_fields':['block_id','network_access_support_ids','support_status','fixed_official_2020_residents','building_uncertainty_sidecar','component_allocation_pending']}
dump(OUT/'network_schema.json',schema)
scenarios=[{'scenario_id':k,'road_state':road,'metro_service_state':me,'jr_service_state':jr,'purpose':why,'actual_interruption_computed':False} for k,road,me,jr,why in [('B0','normal','normal','normal','matched reference topology, not a dry-hazard prediction'),('H0','hazard-conditioned with unknown branches','normal','normal','road and station-approach impairment channel'),('H_M','same as H0','specified station/section/service cuts','same as H0','additional metro service loss conditional on a declared cut set'),('H_J','same as H0','same as H0','specified station/section/service cuts','additional JR service loss conditional on a declared cut set'),('H_MJ','same as H0','same specified metro cuts','same specified JR cuts','compound consequence; no claim all operators halt together')]]
writecsv(OUT/'scenario_matrix.csv',scenarios)
summary={'human_decision':'MILI-D-20261016-020','baseline_blocks':4746,'official_2020_night_residents':496345,'n02_reference_date':'2025-12-31','n02_source_release':'2026-04','regional_rail_sections':len(rail),'regional_station_line_records':len(stations),'core_station_line_records':sum(f['properties']['intersects_core'] for f in stations),'core_station_group_candidates':len({f['properties']['N02_005g'] for f in stations if f['properties']['intersects_core']}),'core_rail_corridors':len(rail_summary),'core_operator_counts':dict(collections.Counter(f['properties']['operator_class'] for f in stations if f['properties']['intersects_core'])),'road_sample_tiles':len(list(S.glob('gsi_rdcl_16_*.geojson'))),'road_sample_features':len(roads),'planned_provisional_road_tiles':len(tiles),'walking_sources':walking,'old_new_walking_shared_link_ids':len(ids17&ids18),'historical_minato_nodes':len(oldnodes),'historical_minato_links':len(next(v for k,v in tables.items() if k.endswith('リンクの情報.CSV'))),'historical_minato_generic_entrances':len(oldentrances),'historical_station_access_candidates':len(entrances),'historical_station_facilities':len({r['source_facility_id'] for r in entrances}),'historical_station_access_with_coordinates':len(entrancegeo),'ticket_gate_name_hints':sum(r['name_hint']=='ticket_gate' for r in entrances),'surface_edge_hazard_join_rows':len(joins),'edge_hazard_interface_max_length_error_m':max(abs(r['join_residual_m']) for r in joinchecks),'plateau_transport':plateau,'training':False,'new_hydraulics':False,'stranded_person_calculation':False,'actual_isolation_calculated':False}
dump(R/'summary.json',summary)
checks={'population_source_delivery_unchanged':all(sha(ROOT/r['file'])==r['sha256'] for r in frozen['products']+frozen['scripts']),'official_night_total_conserved':sum(float(r['official_2020_residents']) for r in baseline)==496345,'rail_geometries_not_clipped_at_ward_boundary':all('source_row' in f['properties'] for f in rail),'north_up_coordinates_match_actual_netcdf':coorderror<1e-5,'actual_hazard_times_present_and_increasing':len(actual)==49 and bool(np.all(np.diff(actual)>0)),'edge_hazard_join_conserves_in_domain_length':max(abs(r['join_residual_m']) for r in joinchecks)<1e-4,'no_old_new_walking_blind_merge':all(e['edge_id'].startswith(e['source']+':') for e in edges),'relative_floor_not_treated_as_absolute_height':all(not r['entrance_threshold_elevation_verified'] for r in entrances),'station_entrances_remain_unverified':all(not r['current_ground_entrance_verified'] for r in entrances),'no_population_or_isolation_consequence_calculated':True}
assert all(checks.values());dump(R/'validation.json',{'checks':checks,'all_passed':True});print(json.dumps(summary,ensure_ascii=False,indent=2))
