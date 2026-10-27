"""Actual PLATEAU LOD2 geometry, illustrative materials, portable WebGL viewer."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,zipfile,xml.etree.ElementTree as E,collections,hashlib
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon,shape,box
from shapely import constrained_delaunay_triangles,make_valid
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/tokyo_station_3d';OUT=ROOT/'data/results/tokyo_station_3d';XY=Transformer.from_crs(6668,6677,always_xy=True)
STATION='bldg_6af58cef-669e-4aca-bc01-355e870d1ad5';CX=-6090.055;CY=-35368.748;BASE=2.93

def tag(q):return q.tag.rsplit('}',1)[-1]
def polygons(q):
 out=[]
 for po in q.iter():
  if tag(po)!='Polygon':continue
  rings=[]
  for r in po.iter():
   if tag(r)!='LinearRing':continue
   p=next((e for e in r.iter() if tag(e)=='posList' and e.text),None)
   if p is None:continue
   a=np.fromstring(p.text,sep=' ')
   if len(a)%3 or len(a)<12:continue
   a=a.reshape(-1,3);x,y=XY.transform(a[:,1],a[:,0]);v=np.c_[x-CX,a[:,2]-BASE,-(y-CY)]
   if np.linalg.norm(v[0]-v[-1])<1e-5:v=v[:-1]
   if len(v)>=3:rings.append(v)
  if rings:out.append((next((v for k,v in po.attrib.items() if k.endswith('}id')),''),rings))
 return out

def triangulate(rings):
 points=np.concatenate(rings);origin=points.mean(0);_,s,vh=np.linalg.svd(points-origin,full_matrices=False)
 if len(s)<2 or s[1]<1e-7:return []
 basis=vh[:2];proj=[(v-origin)@basis.T for v in rings];g=make_valid(Polygon(proj[0],proj[1:]))
 if g.is_empty:return []
 tris=constrained_delaunay_triangles(g);tree=cKDTree(np.concatenate(proj));out=[]
 for t in getattr(tris,'geoms',[]):
  if t.geom_type!='Polygon' or t.area<1e-8:continue
  a=np.asarray(t.exterior.coords)[:3];dist,ids=tree.query(a);v=points[ids] if max(dist)<1e-5 else origin+a@basis
  if np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[0]))>1e-7:out.append(v)
 return out

def extract():
 a=ROOT/'data/raw/plateau/2025/13101-chiyoda-ku/13101_chiyoda-ku_pref_2025_citygml_1_op.zip';member='udx/bldg/53394611_bldg_6697_op.gml';records=[];inventory=[];seen=set()
 with zipfile.ZipFile(a) as z,z.open(member) as f:
  ctx=E.iterparse(f,events=['start','end']);_,root=next(ctx)
  for event,ob in ctx:
   if event!='end' or tag(ob)!='cityObjectMember':continue
   o=next(iter(ob),None)
   if o is None:root.clear();continue
   ident=next((v for k,v in o.attrib.items() if k.endswith('}id')),'');p=next((q for q in o.iter() if tag(q)=='posList' and q.text),None)
   if p is None:root.clear();continue
   aa=np.fromstring(p.text,sep=' ').reshape(-1,3);xx,yy=XY.transform(aa[:,1],aa[:,0])
   if min(xx)>CX+340 or max(xx)<CX-340 or min(yy)>CY+360 or max(yy)<CY-360:root.clear();continue
   semantic=[q for q in o.iter() if tag(q) in ['RoofSurface','WallSurface','GroundSurface','ClosureSurface','OuterCeilingSurface','OuterFloorSurface']];parts=[(tag(q),po) for q in semantic for po in polygons(q)]
   # Actual LOD2 installations (domes/roof equipment), separate from semantic faces.
   parts.extend(('Installation',po) for q in o.iter() if tag(q)=='lod2Geometry' for po in polygons(q))
   lod=2
   if not parts:parts=[('LOD1',po) for q in o.iter() if tag(q)=='lod1Solid' for po in polygons(q)];lod=1
   cnt=0;bounds=[]
   for surface,(pid,rings) in parts:
    key=pid or hashlib.sha256(np.concatenate(rings).tobytes()).hexdigest()
    if key in seen:continue
    seen.add(key)
    try:tris=triangulate(rings)
    except Exception as err:raise RuntimeError(f'{ident}/{pid}: {err}')
    if not tris:continue
    station=ident==STATION
    if station:color=[.48,.54,.57] if surface=='RoofSurface' else [.64,.31,.24] if surface=='WallSurface' else [.70,.45,.34]
    else:color=[.69,.74,.76] if surface=='RoofSurface' else [.76,.80,.80]
    for tri in tris:records.append({'p':np.round(tri,5).tolist(),'c':color,'group':'station' if station else 'context'})
    cnt+=len(tris);bounds.extend(np.concatenate(rings).tolist())
   if cnt:
    bb=np.asarray(bounds);inventory.append({'id':ident,'name':next((q.text for q in o.iter() if tag(q)=='name' and q.text),''),'lod':lod,'triangles':cnt,'is_station_marunouchi':ident==STATION,'bounds_local':np.array([bb.min(0),bb.max(0)]).tolist()})
   root.clear()
 source={'archive':str(a.relative_to(ROOT)),'member':member,'member_crc32':f'{zipfile.ZipFile(a).getinfo(member).CRC:08x}','package_year':2025,'horizontal_crs':6677,'local_origin_easting_northing':[CX,CY],'local_vertical_origin_m':BASE,'station_id':STATION,'materials':'Illustrative brick/red walls and gray roofs; not photographic textures or measured facade colors.','no_height_exaggeration':True,'station_geometry':'Original LOD2 semantic surfaces and LOD2 installations; no flood-model +10m obstacles used.','ground':'Flat presentation plane; road plan geometry drawn at display-plane height, not surveyed 3D ground.'}
 assert any(q['is_station_marunouchi'] and q['lod']==2 for q in inventory)
 # Roads are planimetric context only; source LOD1 Z=0 is not used as elevation.
 fs=json.loads((ROOT/'data/processed/three_ward_urban_constraints/constraint_vectors_6677.geojson').read_text())['features'];roi=box(CX-450,CY-450,CX+450,CY+450);roads=[]
 for f in fs:
  if f['properties']['class']!='road_ground_candidate':continue
  g=shape(f['geometry'])
  if not g.intersects(roi):continue
  g=g.intersection(roi)
  for poly in getattr(g,'geoms',[g]):
   if poly.geom_type!='Polygon':continue
   rings=[np.array([[x-CX,.03,-(y-CY)] for x,y in r.coords]) for r in [poly.exterior,*poly.interiors]]
   for tri in triangulate(rings):roads.append({'p':np.round(tri,5).tolist(),'c':[.87,.88,.87],'group':'road'})
 records.extend(roads)
 manifest={'source':source,'buildings':inventory,'building_count':len(inventory),'triangle_count':len(records),'station_triangles':sum(q['triangles'] for q in inventory if q['is_station_marunouchi'])};(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(P/'mesh.json').write_text(json.dumps(records,separators=(',',':')))
 return records,manifest

def main():
 P.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 if (P/'mesh.json').exists():mesh=json.loads((P/'mesh.json').read_text());manifest=json.loads((P/'manifest.json').read_text())
 else:mesh,manifest=extract()
 template=(Path(__file__).with_name('tokyo_station_viewer.html')).read_text();html=template.replace('__MESH_DATA__',json.dumps(mesh,separators=(',',':')).replace('<','\\u003c')).replace('__MANIFEST__',json.dumps(manifest,ensure_ascii=False,separators=(',',':')));(OUT/'tokyo_station_3d.html').write_text(html);print(json.dumps({k:manifest[k] for k in ['building_count','triangle_count','station_triangles']},ensure_ascii=False))
if __name__=='__main__':main()
