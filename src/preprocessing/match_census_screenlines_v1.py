"""Batch-match exact observation-section geometries to one modeled cross-section.
Geometry-only ormulti-level ambiguity remains held; flow alongwhole sections isnot summed.
"""
from pathlib import Path
import os,json,csv,gzip,re,unicodedata,hashlib,datetime as dt,time
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np,pandas as pd
from shapely.geometry import shape,LineString,Point,mapping
from shapely.ops import transform
from shapely import STRtree
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_normal_vehicle_baseline_v1';R=ROOT/'data/exp/estimation/three_ward_normal_vehicle_baseline_v1';B=ROOT/'data/processed/three_ward_traffic_baseline_inputs_v1';N=ROOT/'data/processed/three_ward_normal_transport_network_v1';M=ROOT/'data/processed/three_ward_motor_flood_access_v1'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item()if isinstance(v,np.generic)else str(v))+'\n')
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def numeric(s):
 try:return float(str(s).replace(',',''))
 except(ValueError,TypeError):return np.nan
def norm(s):
 s=unicodedata.normalize('NFKC',str(s))
 for term in ['一般国道','東京都道','主要地方道','首都','高速','路線','線','号',' ']:s=s.replace(term,'')
 return s

def main():
 started=time.monotonic();policy={'registered_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'method':'one transverse cut in native observation section,not sum over its longitudinal road chain','cut_fraction_of_longest_component':0.47,'cut_half_width_m':'max25,min50,0.6*reportedroadwidth+15','max_source_centerline_distance_m':20,'minimum_local_alignment_cosine':0.8,'minimum_distance_from_osm_segment_endpoint_m':1,'qualified_match':'1or2intersecting physical roads,route-ref/name evidence,same relativegrade,noendpoint intersection. Othersheld.','split':'SHA256(observation_section_id)mod4==0 validation,otherwise calibration. One deterministic unitperobservationsection forfit;sharedobservationunitsheldasduplicates.','fit':'Scalar target-versus-modeled12h factor on calibration units only;diagnosticnotappliedtosourceOD,validationuntouched. No capacity fitting.'};dump(P/'screenline_matching_policy.json',policy)
 ep=load(P/'endpoint_components.npz');xy=ep['node_xy'];p=load(M/'road_projection.npz');roads=pd.read_csv(M/'road_inventory.csv',dtype={'source_edge_id':str});allids={str(r.source_edge_id):int(r.road_index)for r in roads.itertuples()};geomdata=json.loads((P/'official_census_geometries_4326.geojson').read_text());tx=Transformer.from_crs(4326,6677,always_xy=True).transform;geometries={str(f['properties']['R03cen']):transform(tx,shape(f['geometry']))for f in geomdata['features']}
 bounds=np.array([g.bounds for g in geometries.values()]);xmin=bounds[:,0].min()-100;ymin=bounds[:,1].min()-100;xmax=bounds[:,2].max()+100;ymax=bounds[:,3].max()+100;first=np.full(len(roads),-1,int);first[p['arc_to_road']]=np.arange(len(p['arc_to_road']));a=xy[p['from_index'][first]];b=xy[p['to_index'][first]];keep=(np.maximum(a[:,0],b[:,0])>=xmin)&(np.minimum(a[:,0],b[:,0])<=xmax)&(np.maximum(a[:,1],b[:,1])>=ymin)&(np.minimum(a[:,1],b[:,1])<=ymax);wanted=set(roads.loc[keep,'source_edge_id']);road_geoms=[];roadix=[];tags=[]
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as f:
  for line in f:
   e=json.loads(line)
   if e['edge_id']in wanted:road_geoms.append(shape(e['geometry']));roadix.append(allids[e['edge_id']]);tags.append(e['tags'])
 tree=STRtree(road_geoms);obs=pd.read_csv(B/'core_directional_observations.csv',dtype=str,keep_default_na=False);cross=pd.read_csv(B/'section_to_count_unit.csv',dtype=str,keep_default_na=False);cross=cross.merge(obs[['section_id','native_observation_section_id']].drop_duplicates(),on='section_id',how='left');profiles=pd.read_csv(B/'hourly_profile_quality.csv',dtype={'count_authority_code':str,'count_unit_id':str,'direction_code':str,'vehicle_class_code':str,'observed_estimated_code':str});hour=pd.read_csv(B/'core_count_hourly_long.csv',dtype={'count_authority_code':str,'count_unit_id':str,'direction_code':str,'vehicle_class_code':str,'observed_estimated_code':str});raw=list(csv.reader((ROOT/'data/processed/three_ward_traffic_data_search_v1/sources/mlit_tokyo_2021_sections.csv').open(encoding='cp932')));native={r[0]:r for r in raw[1:]};rows=[];features=[];mapping_rows=[];usedobs=set()
 for key,group in profiles.groupby(['count_authority_code','count_unit_id'],sort=True):
  ids=cross.loc[cross.count_authority_code.eq(key[0])&cross.count_unit_id.eq(key[1]),'native_observation_section_id'].dropna();ids=sorted({s for s in ids if s.isdigit()and len(s)==11});record={'count_authority_code':key[0],'count_unit_id':key[1],'observation_section_ids':'|'.join(ids),'profile_rows':len(group),'all_profiles_observed':set(group.observed_estimated_code)=={'1'},'matched_road_indices':'','matching_status':'held_no_unique_observation_section','split':'not_eligible'}
  if len(ids)!=1:rows.append(record);continue
  sid=ids[0];record['observation_section_id']=sid;record['split']='validation'if int(hashlib.sha256(sid.encode()).hexdigest()[:12],16)%4==0 else'calibration';record['duplicate_observation_unit']=sid in usedobs;usedobs.add(sid)
  geo=geometries.get(sid);fattr=P/'sources'/f'{sid}.json';attr=json.loads(fattr.read_text(encoding='utf-8-sig'))if fattr.exists()else{};nat=native.get(sid);record['source_route_name']=attr.get('路線名','');record['source_road_type']=nat[3]if nat else'';record['source_route_number']=nat[4]if nat else'';record['source_peak_mean_kmh']=float(np.nanmean([numeric(v)for k,v in attr.items()if k.startswith('朝夕旅行速度')]))if any(k.startswith('朝夕旅行速度')for k in attr)else np.nan;record['source_offpeak_mean_kmh']=float(np.nanmean([numeric(v)for k,v in attr.items()if k.startswith('昼間旅行速度')]))if any(k.startswith('昼間旅行速度')for k in attr)else np.nan
  if geo is None:record['matching_status']='held_missing_official_geometry';rows.append(record);continue
  components=list(geo.geoms)if geo.geom_type=='MultiLineString'else[geo];line=max(components,key=lambda x:x.length);at=.47*line.length;c=line.interpolate(at);v0=line.interpolate(max(0,at-10));v1=line.interpolate(min(line.length,at+10));vec=np.array([v1.x-v0.x,v1.y-v0.y]);vec=vec/np.linalg.norm(vec);normal=np.array([-vec[1],vec[0]]);width=numeric(nat[112])if nat else 30.;width=width if np.isfinite(width)else 30.;half=max(25.,min(50.,.6*width+15));cut=LineString([np.array([c.x,c.y])-normal*half,np.array([c.x,c.y])+normal*half]);features.append({'type':'Feature','geometry':mapping(cut),'properties':{'count_unit':key[0]+'-'+key[1],'observation_section_id':sid}});candidates=[]
  for j in tree.query(cut,predicate='intersects'):
   g=road_geoms[j];d=g.project(c);s0=g.interpolate(max(0,d-10));s1=g.interpolate(min(g.length,d+10));rv=np.array([s1.x-s0.x,s1.y-s0.y]);den=np.linalg.norm(rv);alignment=abs(float(np.dot(rv,vec)/den))if den else 0.;close=g.distance(line);ridx=roadix[j];tag=tags[j];highway=str(tag.get('highway',''));express=record['source_road_type']in['1','2'];classok=highway.startswith('motorway')if express else not highway.startswith('motorway');route=False
   if express:
    rn=norm(record['source_route_name']);on=norm(tag.get('name',''));route=bool(rn and on and(rn in on or on in rn))
   else:
    number=record['source_route_number'];route=bool(number and re.search(r'(?<!\d)'+re.escape(str(int(number)))+r'(?!\d)',unicodedata.normalize('NFKC',tag.get('ref',''))))if number.isdigit()else False
   nearendpoint=min(d,g.length-d)<1
   valid=classok and alignment>=.8 and close<=20
   if valid:candidates.append({'road_index':ridx,'route_gate':route,'near_endpoint':nearendpoint,'grade':str(roads.iloc[ridx].grade_key),'distance_m':float(close),'alignment':alignment,'name':str(tag.get('name','')),'ref':str(tag.get('ref',''))})
  strong=[q for q in candidates if q['route_gate']];chosen=strong if strong else candidates;record['candidate_road_count']=len(chosen);record['route_evidence_count']=len(strong);record['cut_x_m']=c.x;record['cut_y_m']=c.y
  if not chosen:status='held_no_geometry_class_candidate'
  elif not strong:status='held_geometry_only_no_route_evidence'
  elif len(chosen)>2:status='held_more_than_two_parallel_or_branching_roads'
  elif len({q['grade']for q in chosen})!=1:status='held_multiple_relative_levels'
  elif any(q['near_endpoint']for q in chosen):status='held_cut_at_source_segment_endpoint'
  else:status='qualified_route_gated_screenline'
  record['matching_status']=status;record['matched_road_indices']='|'.join(str(q['road_index'])for q in chosen);record['candidates_json']=json.dumps(chosen,ensure_ascii=False)
  hh=hour[hour.count_authority_code.eq(key[0])&hour.count_unit_id.eq(key[1])]
  for name,hours in [('daily',list(range(24))),('hour8',[8]),('hour10',[10]),('hour17',[17]),('daytime12h',list(range(7,19)))]:
   for cls,label in [(None,'all'),('1','small')]:
    gg=hh[hh.hour_start_jst.isin(hours)]
    expected=len(group)*len(hours)
    if cls is not None:gg=gg[gg.vehicle_class_code.eq(cls)];expected=int(group.vehicle_class_code.eq(cls).sum())*len(hours)
    value=float(gg.hourly_volume_veh_per_hour.sum())if len(gg)==expected and gg.hourly_volume_veh_per_hour.notna().all()and record['all_profiles_observed']else np.nan;record['observed_'+label+'_'+name]=value
  if status=='qualified_route_gated_screenline':
   for q in chosen:mapping_rows.append({'count_authority_code':key[0],'count_unit_id':key[1],'observation_section_id':sid,'road_index':q['road_index'],'normal_grade':q['grade'],'source_route_gate':q['route_gate']})
  rows.append(record)
 pd.DataFrame(rows).to_csv(P/'census_screenline_crosswalk.csv',index=False,encoding='utf-8-sig');pd.DataFrame(mapping_rows).to_csv(P/'qualified_census_to_source_roads.csv',index=False,encoding='utf-8-sig');(P/'census_screenlines_6677.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False)+'\n');summary={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'count_units':len(rows),'statuses':pd.Series([r['matching_status']for r in rows]).value_counts().to_dict(),'qualified_unique_observed_units_12h':sum(r['matching_status']=='qualified_route_gated_screenline'and r['all_profiles_observed']and not r.get('duplicate_observation_unit',False)and np.isfinite(r.get('observed_all_daytime12h',np.nan))for r in rows),'native_road_geometries_in_index':len(road_geoms),'elapsed_s':time.monotonic()-started};dump(R/'screenline_matching_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:v.item()if isinstance(v,np.generic)else str(v)),flush=True)
if __name__=='__main__':main()
