"""Inventory PLATEAU underground-access objects without inferring station thresholds."""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
from pathlib import Path
import zipfile,xml.etree.ElementTree as E,collections,hashlib,json,csv
import numpy as np
from shapely.geometry import Point,shape,mapping
from shapely.ops import transform
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/processed/three_ward_transport_framework_v1';R=ROOT/'data/exp/feasibility-check/three_ward_transport_framework_v1';G='{http://www.opengis.net/gml}';F='{http://www.opengis.net/citygml/cityfurniture/2.0}';xy=Transformer.from_crs(6668,6677,always_xy=True).transform
records=[];byid={};railclass=[]
for p in sorted((moga_raw(ROOT, 'plateau/2025')).glob('*/*citygml*.zip')):
 z=zipfile.ZipFile(p);names=z.namelist();railn=0
 for n in names:
  if '/tran/' in n and n.endswith('.gml'):
   s=z.read(n);railn+=s.count(b'<tran:Railway')
 railclass.append({'archive':moga_source(p, ROOT),'Railway_class_occurrences_in_tran':railn})
 for n in names:
  if '/frn/' not in n or not n.endswith('.gml'):continue
  root=E.fromstring(z.read(n))
  for e in root.findall('.//'+F+'CityFurniture'):
   f=e.find(F+'function')
   if f is None or f.text!='4020':continue
   pts=[]
   for pos in e.findall('.//'+G+'posList')+e.findall('.//'+G+'pos'):
    a=np.fromstring(pos.text,sep=' ')
    if len(a)%3==0:pts.append(a.reshape(-1,3))
   a=np.concatenate(pts) if pts else np.empty((0,3));id=e.attrib[G+'id'];digest=hashlib.sha256(a.tobytes()).hexdigest();key=(id,digest)
   leaves={}
   for v in e.iter():
    tag=v.tag.split('}')[-1]
    if len(v)==0 and v.text and tag not in ['posList','pos']:
     value=v.text.strip()
     if value and len(value)<150:leaves.setdefault(tag,set()).add(value)
   meta={k:sorted(v) for k,v in leaves.items()};srs=next((v.attrib['srsName'] for v in e.iter() if 'srsName' in v.attrib),'inherited from document')
   r={'gml_id':id,'source_archive':moga_source(p, ROOT),'source_member':n,'function_code':'4020','function_description':'underground access','coordinate_points':len(a),'source_geometry_sha256':digest,'declared_srs':srs,'source_leaf_metadata':json.dumps(meta,ensure_ascii=False),'z_min_model_geometry_m':float(a[:,2].min()) if len(a) else None,'z_max_model_geometry_m':float(a[:,2].max()) if len(a) else None,'creation_dates':'|'.join(meta.get('creationDate',[])),'survey_year_values':'|'.join(meta.get('surveyYear',[])),'station_identity_verified':False,'minimum_z_is_an_entrance_threshold':False,'complete_station_entrance_coverage':False}
   records.append(r)
   if key not in byid:
    centre=a[:,:2].mean(axis=0);x,y=xy(centre[1],centre[0]);byid[key]={'record':r,'geometry':Point(x,y),'occurrences':0}
   byid[key]['occurrences']+=1
keys=list(dict.fromkeys(k for r in records for k in r))
with(OUT/'plateau_underground_access_occurrences.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
features=[]
for k,v in byid.items():features.append({'type':'Feature','geometry':mapping(v['geometry']),'properties':{**v['record'],'cross_archive_occurrences':v['occurrences'],'inventory_point_is_geometry_vertex_centroid':True}})
(OUT/'plateau_underground_access_candidates_6677.geojson').write_text(json.dumps({'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':features},ensure_ascii=False)+'\n')
s={'source_function_code':'4020','definition_from_bundled_codelist':'地下出入口','source_occurrences':len(records),'unique_gml_id_geometry_variants':len(byid),'unique_gml_ids':len({r['gml_id'] for r in records}),'cross_archive_repeated_occurrences':len(records)-len(byid),'geometry_variant_ids':sum(len({r['source_geometry_sha256'] for r in records if r['gml_id']==id})>1 for id in {r['gml_id'] for r in records}),'by_archive':dict(collections.Counter(r['source_archive'] for r in records)),'creation_dates':dict(collections.Counter(r['creation_dates'] for r in records)),'survey_year_values':dict(collections.Counter(r['survey_year_values'] for r in records)),'railway_class_checks':railclass,'station_or_public_ground_entrance_verified':False,'threshold_elevation_verified':False,'these_are_not_all_station_entrances':True}
(R/'plateau_access_audit.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');print(json.dumps(s,ensure_ascii=False,indent=2))
