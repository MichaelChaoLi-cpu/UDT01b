"""Source-preserving traffic baseline candidates; no OD fabrication or assignment.
Run with the project's geospatial Python runtime. See preparation_contract.json.
"""
from pathlib import Path
import os,csv,io,json,hashlib,datetime as dt,collections,zipfile,xml.etree.ElementTree as ET
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(key,None)
import geopandas as gpd
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'data/processed/three_ward_traffic_data_search_v1/sources'
P=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1'
S=P/'sources';R=ROOT/'data/exp/feasibility-check/three_ward_traffic_baseline_inputs_v1'
WARD={'13101':'千代田区','13102':'中央区','13103':'港区'}
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,rows,fields=None):
 rows=list(rows)
 with (P/name).open('w',encoding='utf-8-sig',newline='')as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def load(name):return list(csv.reader(io.StringIO((OLD/name).read_bytes().decode('cp932'))))
def number(v):
 v=v.strip()
 if not v:return None
 try:return float(v)
 except ValueError:raise ValueError('Uninterpreted numeric code: '+repr(v))
def xlsx_rows(p,sheet):
 with zipfile.ZipFile(p)as z:
  strings=[''.join((t.text or '')for t in si.findall('m:t',NS)+si.findall('m:r/m:t',NS))for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',NS)]
  out=[]
  for row in ET.fromstring(z.read(f'xl/worksheets/sheet{sheet}.xml')).findall('.//m:row',NS):
   d={}
   for c in row.findall('m:c',NS):
    v=c.find('m:v',NS);v=v.text if v is not None else ''
    if c.get('t')=='s' and v:v=strings[int(v)]
    d[''.join(t for t in c.get('r')if t.isalpha())]=v
   out.append((row.get('r'),d))
 return out

def main():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 sources=[OLD/'mlit_tokyo_2021_sections.csv',OLD/'mlit_tokyo_2021_hourly.csv',OLD/'tokyo_wards_2021_hourly.csv',S/'H30_zonecode.xlsx',S/'H30_gis.zip',S/'pt_data_guide.pdf',S/'road_census_2021_methodology.pdf']
 before={str(p.relative_to(ROOT)):sha(p)for p in sources}
 base=load('mlit_tokyo_2021_sections.csv');hour=load('mlit_tokyo_2021_hourly.csv');core=[(i,r)for i,r in enumerate(base[1:],2)if r[18]in WARD]
 attr=[];cross=[];fieldmap={}
 cols={'section_id':0,'road_type_code':3,'route_number':4,'route_name':5,'municipality_code':18,'motorway_code':19,'length_km':23,'oneway_flag':26,'count_authority_code':30,'count_unit_id':31,'volume_12h_both_directions_veh':58,'volume_24h_both_directions_veh':61,'congestion_degree_12h':65,'road_width_m':112,'carriageway_area_width_m':113,'carriageway_width_m':114,'lane_count':122,'speed_limit_kmh':140}
 numeric=set(cols)-{'section_id','road_type_code','route_number','route_name','municipality_code','motorway_code','oneway_flag','count_authority_code','count_unit_id'}
 for k,j in cols.items():fieldmap[k]={'source':'mlit_tokyo_2021_sections.csv','zero_based_column':j,'native_header':base[0][j],'numeric_blank_policy':'empty stays missing; numeric zero retained'if k in numeric else 'native string preserved'}
 for i,r in core:
  a={k:number(r[j])if k in numeric else r[j]for k,j in cols.items()};a['source_row_1_based']=i;attr.append(a)
  cross.append({'section_id':r[0],'municipality_code':r[18],'count_authority_code':r[30],'count_unit_id':r[31],'source_row_1_based':i,'native_start_connection':r[9],'native_end_connection':r[14],'network_link_id':'','network_match_status':'not_constructed'})
 write('core_road_attributes.csv',attr);write('section_to_count_unit.csv',cross)
 # Directional values are retained separately. Flag 1 = measured, 2 = estimated.
 directional=[]
 for i,r in core:
  for direction,st,speedst in [('1',32,72),('2',44,79)]:
   directional.append({'section_id':r[0],'municipality_code':r[18],'direction_code':direction,'native_observation_section_id':r[st],'native_observation_location':r[st+1],'count_observed_estimated_code':r[st+2].strip(),'individual_observation_reuse_code':r[st+3].strip(),'observation_duration_code':r[st+4].strip(),'observation_date_raw':r[st+5].strip(),'volume_12h_small_veh':number(r[st+6]),'volume_12h_large_veh':number(r[st+7]),'volume_12h_total_veh':number(r[st+8]),'volume_24h_total_veh':number(r[st+11]),'peak_speed_measurement_code':r[speedst].strip(),'peak_speed_total_kmh':number(r[speedst+3]),'offpeak_speed_measurement_code':r[speedst+14].strip(),'offpeak_speed_total_kmh':number(r[speedst+17]),'source_row_1_based':i})
 write('core_directional_observations.csv',directional)
 keys={(r[30],r[31])for _,r in core};matched=[(i,r)for i,r in enumerate(hour[1:],2)if tuple(r[:2])in keys]
 long=[];quality=[]
 for i,r in matched:
  vals=[number(v)for v in r[11:35]];v12=number(r[35]);v24=number(r[36]);n=sum(v is not None for v in vals)
  err12=sum(vals[:12])-v12 if all(v is not None for v in vals[:12])and v12 is not None else None
  err24=sum(vals)-v24 if n==24 and v24 is not None else None
  q={'count_authority_code':r[0],'count_unit_id':r[1],'direction_code':r[9],'vehicle_class_code':r[10],'observed_estimated_code':r[4].strip(),'individual_observation_reuse_code':r[5].strip(),'duration_code':r[6].strip(),'observation_date_raw':r[7].strip(),'weather_code':r[8].strip(),'available_hour_count':n,'native_12h_veh':v12,'native_24h_veh':v24,'sum_12h_minus_native_veh':err12,'sum_24h_minus_native_veh':err24,'source_row_1_based':i}
  quality.append(q)
  for k,v in enumerate(vals):
   hh=(k+7)%24
   long.append({**{key:q[key]for key in ['count_authority_code','count_unit_id','direction_code','vehicle_class_code','observed_estimated_code','duration_code','observation_date_raw','weather_code','source_row_1_based']},'hour_start_jst':hh,'survey_day_offset':int(k>=17),'hourly_volume_veh_per_hour':v,'value_status':'missing'if v is None else ('observed_source'if q['observed_estimated_code']=='1'else 'estimated_source'),'daily_total_usable_for_hour_fraction':bool(n==24 and v24 and err24==0),'fraction_of_24h':v/v24 if n==24 and v24 and err24==0 else None})
 write('core_count_hourly_long.csv',long);write('hourly_profile_quality.csv',quality)
 matchedkeys={tuple(r[:2])for _,r in matched}
 write('count_units_without_hourly_rows.csv',[{'count_authority_code':k[0],'count_unit_id':k[1]}for k in sorted(keys-matchedkeys)])
 # Source zone codes are strings. Strip phonetic annotations rather than joining all XML text.
 zones=[]
 for i,d in xlsx_rows(S/'H30_zonecode.xlsx',1):
  if i=='1':continue
  if not d.get('F'):continue
  zones.append({'prefecture':d.get('A',''),'municipality':d.get('B',''),'large_zone':d.get('C',''),'middle_zone':d.get('D',''),'planning_zone':d.get('E',''),'small_zone':d.get('F',''),'source_sheet':'市区町村別ゾーン一覧','source_excel_row':i})
 write('pt_2018_zone_code_crosswalk.csv',zones)
 cz=[r for r in zones if r['prefecture']=='東京都' and r['municipality']in WARD.values()];write('pt_2018_core_zone_codes.csv',cz)
 corecodes={r['planning_zone']for r in cz};core_small={r['small_zone']for r in cz}
 archive=(S/'H30_gis.zip').resolve()
 kg=gpd.read_file(f'/vsizip/{archive}/H30_gis/H30_kzone.shp',encoding='cp932');kg['planning_zone']=kg.kzone.map(lambda x:f'{int(x):04d}')
 sg=gpd.read_file(f'/vsizip/{archive}/H30_gis/H30_szone.shp',encoding='cp932');sg['small_zone']=sg.szone.map(lambda x:f'{int(x):05d}')
 kc=kg[kg.planning_zone.isin(corecodes)].copy();sc=sg[sg.small_zone.isin(core_small)].copy()
 # Buffer is an acquisition pool only, not a closed OD boundary or a final routing connector scheme.
 halo=kc.geometry.union_all().buffer(10000);pool=kg[kg.intersects(halo)].copy();pool['role']=pool.planning_zone.map(lambda x:'core'if x in corecodes else 'external_candidate_10km_pool')
 kc.to_file(P/'pt_2018_core_planning_zones_6677.geojson',driver='GeoJSON');sc.to_file(P/'pt_2018_core_small_zones_6677.geojson',driver='GeoJSON');pool.to_file(P/'pt_2018_candidate_external_zones_6677.geojson',driver='GeoJSON')
 geometry_invalid={'core_planning':int((~kc.is_valid).sum()),'core_small':int((~sc.is_valid).sum()),'external_pool':int((~pool.is_valid).sum())}
 counts={w:{'small':len({r['small_zone']for r in cz if r['municipality']==w}),'planning':len({r['planning_zone']for r in cz if r['municipality']==w})}for w in WARD.values()}
 summary={'prepared_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'road_observation_candidates_and_pt_zones_prepared; OD_table_access_pending_affiliation','core_census_sections':len(core),'core_census_rows_by_municipality':dict(collections.Counter(r[18]for _,r in core)),'distinct_core_count_units':len(keys),'count_units_with_hourly_rows':len(matchedkeys),'count_units_without_hourly_rows':len(keys-matchedkeys),'hourly_profiles':len(matched),'hourly_long_rows':len(long),'hourly_nonmissing_values':sum(r['hourly_volume_veh_per_hour']is not None for r in long),'hourly_profile_observation_flags':dict(collections.Counter(r['observed_estimated_code']for r in quality)),'complete_24h_profiles':sum(r['available_hour_count']==24 for r in quality),'profiles_with_any_hourly_values':sum(r['available_hour_count']>0 for r in quality),'profiles_12h_nonzero_sum_error':sum(r['sum_12h_minus_native_veh']not in [None,0]for r in quality),'profiles_24h_nonzero_sum_error':sum(r['sum_24h_minus_native_veh']not in [None,0]for r in quality),'pt_core_zone_counts':counts,'pt_core_planning_codes':sorted(corecodes),'pt_external_pool_planning_zones':len(pool),'geometry_invalid':geometry_invalid,'pt_od_downloaded':False,'motor_network_crosswalk_constructed':False,'capacities_assigned':False,'congestion_simulated':False,'people_counted':False,'running_hydraulic_campaign_modified':False}
 dump(R/'summary.json',summary)
 contract={'purpose':'Candidate inputs for a survey-constrained normal motor-traffic baseline; not yet a calibrated baseline.','decision':'MILI-D-20261017-014','years':{'PT':2018,'road_census':2021,'resident_population':2020,'road_network_snapshot':'2026-10-15','rain_events':[2024,2025]},'source_attribution':['東京都市圏交通計画協議会ホームページを加工して作成 (https://www.tokyo-pt.jp/data/01_01)','国土交通省 令和3年度 全国道路・街路交通情勢調査を加工して作成 (https://www.mlit.go.jp/road/census/r3/index.html)'],'field_map':fieldmap,'hourly_rules':{'source':'mlit_tokyo_2021_hourly.csv','exact_join':['count_authority_code','count_unit_id'],'join_cardinality':'many basic sections to one count unit; never sum repeated counts across linked sections','direction_codes':{'1':'up native direction','2':'down native direction'},'vehicle_codes':{'1':'small vehicles','2':'large vehicles'},'observed_estimated_codes':{'1':'observed','2':'estimated'},'duration_codes':{'1':'12 hours','2':'24 hours'},'clock':'7:00 through next-day7:00 JST, 24 one-hour bins; after midnight day_offset1','missing':'whitespace stays empty/None; no interpolation or zero filling','date':'YYYYMM00 month-only identifiers stay native strings; no invented date','observations':'normal survey traffic or census estimates; not storm-hour traffic','temporal_profiles':'fractions only for complete24hour rows with exact published24hour sum; no extra normalization across missing hours'},'zone_rules':{'source_code_table_sheet':'市区町村別ゾーン一覧','code_types':'strings preserve leading zeros','crs':'EPSG:6677 JGD2011 / Japan Plane Rectangular CS IX','geometry':'source boundaries preserved; invalid geometries reported, no silent repairs','external_pool':'planning zones intersecting a10km buffer around core PT zones; acquisition candidate pool only. Keep all metropolitan codes and original geometry; do not delete outside OD or through traffic.','enclaves':'official GIS readme warns municipality labels can include enclaves of other municipalities'},'unit_rules':['PT d-1 mode3 is automobile PERSON-trips, not vehicles; includes passengers and multiple auto classes.','PT c-2/c-3 are departure/arrival trip ENDS, not full OD-by-time; do not add origins and destinations to count trips.','PT b-3 driver/non-driver information can constrain vehicle conversion but is residence-based, not destination-specific OD.','PT b-4 counts survey-derived time-specific staying persons, not trips or a complete census of visitors.','PT mainly resident persons aged5+; commercial vehicle operations, external-resident vehicles and occupants require separate coverage checks.','Road census volumes are vehicle passages, not independent vehicles across all road links or unique people.','Congestion degree is volume/capacity in12h units; no direct hourly capacity column. Do not divide its implied12h capacity by12 and treat as hourly observed capacity.'],'next_required':['Acquire d-1,b-3,c-2/c-3 and e-2 after truthful affiliation form submission; inspect5-3 vehicle convention.','Build auditable regional-zone to motor-road connectors and aggregate external inflow/outflow; preserve through trips.','Map census units to existing road graph in batches using source geometry or route/endpoint evidence, without copying counts to nearby unrelated roads.','Use observed speed/volume constraints to check normal flow; assign explicit minor-road capacity and uncovered-demand scenarios.','Specify low/central/high demand and capacity factors before assignment; no numeric factors have been selected here.'],'not_done':['No affiliation invented or form submitted','No OD table values invented','No census counts summed across repeated units','No hourly capacity invented','No traffic assignment','No people impact calculation','No change to active hydraulic campaign']}
 dump(P/'preparation_contract.json',contract)
 checks={'source_hashes_unchanged':all(sha(ROOT/k)==v for k,v in before.items()),'391_core_sections_preserved':len(core)==391,'section_rows_unique':len({r['section_id']for r in attr})==len(attr),'hourly_profile_keys_unique':len({tuple(r[j]for j in [0,1,9,10])for _,r in matched})==len(matched),'hourly_output24_rows_per_profile':len(long)==24*len(matched),'hourly_values_roundtrip':all(next(x for x in long if x['source_row_1_based']==i and x['hour_start_jst']==(k+7)%24)['hourly_volume_veh_per_hour']==number(r[11+k])for i,r in matched for k in range(24)),'hourly_missing_not_zero_filled':sum(x['hourly_volume_veh_per_hour']is None for x in long)==sum(not x.strip()for _,r in matched for x in r[11:35]),'core_zone_codes_all_in_geometry':set(kc.planning_zone)==corecodes and set(sc.small_zone)==core_small,'three_wards_present':len(counts)==3 and all(v['small']>0 for v in counts.values()),'pt_leading_zeros_preserved':all(len(x)==4 and x.startswith('00')for x in corecodes),'crs6677':kc.crs.to_epsg()==6677 and sc.crs.to_epsg()==6677,'no_fabricated_od_or_capacity':not summary['pt_od_downloaded']and not summary['capacities_assigned']}
 dump(R/'validation.json',{'passed':all(checks.values()),'checks':checks,'warnings':{'geometry_invalid':geometry_invalid,'hourly_native_sum_mismatches':[r for r in quality if r['sum_12h_minus_native_veh']not in [None,0]or r['sum_24h_minus_native_veh']not in [None,0]]}})
 dump(P/'source_bindings.json',{'sha256':before,'prepared_utc':summary['prepared_utc']})
 print(json.dumps(summary,ensure_ascii=False,indent=2));print('validation',all(checks.values()))
 if not all(checks.values()):raise SystemExit('Preparation validation failed')
if __name__=='__main__':main()
