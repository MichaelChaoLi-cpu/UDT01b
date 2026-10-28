"""Complete PLATEAU tile53394611, colored buildings/roads/native terrain/water.
Original geometric layers remain separate; no textures, material files or flood
obstacles. Road plan Z=0 is draped on the packaged TIN for visualization only.
"""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,zipfile,xml.etree.ElementTree as E,hashlib,collections
import numpy as np
from shapely.geometry import Polygon,Point
from shapely import STRtree,points
import render_tokyo_station as shared
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/tokyo_mesh_53394611';OUT=ROOT/'data/results/tokyo_mesh_53394611'
LON0,LAT0=139.7625,35.675;LON1,LAT1=139.775,35.68333333333333
x,y=shared.XY.transform([LON0,LON1,LON1,LON0],[LAT0,LAT0,LAT1,LAT1]);CX=float(np.mean(x));CY=float(np.mean(y));shared.CX=CX;shared.CY=CY;shared.BASE=0.;RING=np.c_[np.array(x)-CX,-(np.array(y)-CY)]
a=RING[1]-RING[0];b=RING[2]-RING[1]
if a[0]*b[1]-a[1]*b[0]<0:RING=RING[::-1]
ROI=Polygon(RING);BOUNDS=ROI.bounds
A=moga_raw(ROOT, 'plateau/2025/13101-chiyoda-ku/13101_chiyoda-ku_pref_2025_citygml_1_op.zip')

def clip(tri):
 poly=list(np.asarray(tri,dtype=float));xz=np.asarray(tri)[:,[0,2]]
 # Most triangles are entirely inside the cell; avoid polygon operations there.
 ins=True
 for p,q in zip(RING,np.roll(RING,-1,axis=0)):
  d=q-p;v=d[0]*(xz[:,1]-p[1])-d[1]*(xz[:,0]-p[0])
  if np.all(v< -1e-6):return []
  if not np.all(v>=-1e-6):ins=False
 if ins:return [np.asarray(tri)]
 for p,q in zip(RING,np.roll(RING,-1,axis=0)):
  d=q-p;new=[]
  if not poly:return []
  prev=poly[-1];fp=d[0]*(prev[2]-p[1])-d[1]*(prev[0]-p[0]);inp=fp>=-1e-7
  for cur in poly:
   fc=d[0]*(cur[2]-p[1])-d[1]*(cur[0]-p[0]);inc=fc>=-1e-7
   if inc!=inp:new.append(prev+(cur-prev)*(fp/(fp-fc)))
   if inc:new.append(cur)
   prev=cur;fp=fc;inp=inc
  poly=new
 return [np.array([poly[0],poly[i],poly[i+1]]) for i in range(1,len(poly)-1) if np.linalg.norm(np.cross(poly[i]-poly[0],poly[i+1]-poly[0]))>1e-7]

def member_objects(archive,member):
 with archive.open(member) as f:
  ctx=E.iterparse(f,events=['start','end']);_,root=next(ctx)
  for ev,ob in ctx:
   if ev!='end' or shared.tag(ob)!='cityObjectMember':continue
   if len(ob):yield next(iter(ob))
   root.clear()

def geometry_parts(o,lod):return [po for q in o.iter() if shared.tag(q)==lod for po in shared.polygons(q)]
def add_faces(records,parts,group,seen):
 count=0
 for pid,rings in parts:
  key=pid or hashlib.sha256(np.concatenate(rings).tobytes()).hexdigest()
  if key in seen:continue
  seen.add(key)
  for tri in shared.triangulate(rings):
   for t in clip(tri):records.append({'p':np.round(t,5).tolist(),'group':group});count+=1
 return count

def terrain(archive,member):
 result=[];seen=set();scanned=0
 class Parser:
  def __init__(self):self.level=0;self.capture=False;self.text=[];self.batch=[]
  def start(self,tag,attrs):
   if tag.endswith('}Triangle'):self.level+=1
   if self.level and tag.endswith('}posList'):self.capture=True;self.text=[]
  def data(self,s):
   if self.capture:self.text.append(s)
  def end(self,tag):
   nonlocal scanned
   if self.capture and tag.endswith('}posList'):
    self.capture=False;a=np.fromstring(''.join(self.text),sep=' ')
    if len(a)==12:self.batch.append(a.reshape(4,3)[:3]);scanned+=1
    if len(self.batch)>=20000:self.flush()
   if tag.endswith('}Triangle'):self.level-=1
  def close(self):self.flush()
  def flush(self):
   if not self.batch:return
   a=np.array(self.batch);self.batch=[];ok=(a[:,:,0].max(1)>=LAT0)&(a[:,:,0].min(1)<=LAT1)&(a[:,:,1].max(1)>=LON0)&(a[:,:,1].min(1)<=LON1);a=a[ok]
   if not len(a):return
   xx,yy=shared.XY.transform(a[:,:,1],a[:,:,0]);v=np.stack([xx-CX,a[:,:,2],-(yy-CY)],axis=2)
   for tri in v:
    for t in clip(tri):
     order=np.lexsort((t[:,2],t[:,1],t[:,0]));key=np.round(t[order],4).tobytes()
     if key in seen:continue
     seen.add(key);result.append(t)
 target=Parser();parser=E.XMLParser(target=target)
 with archive.open(member) as f:
  while b:=f.read(1024*1024):parser.feed(b)
 parser.close();return np.asarray(result),scanned

def heights(tin,coords):
 tree=STRtree([Polygon(t[:,[0,2]]) for t in tin]);pair=tree.query(points(coords[:,0],coords[:,1]),predicate='covered_by');out=np.full(len(coords),np.nan)
 # Rounding/projected cell edges may place a point a fraction of a micron
 # outside a TIN triangle. Accept only <=1e-6 m boundary roundoff, not gaps.
 missing=np.setdiff1d(np.arange(len(coords)),pair[0])
 if len(missing):
  near=tree.query_nearest(points(coords[missing,0],coords[missing,1]),max_distance=1e-6,all_matches=False)
  if len(near[0]):pair=np.concatenate([pair,np.stack([missing[near[0]],near[1]])],axis=1)
 if len(pair[0]):
  ids,first=np.unique(pair[0],return_index=True);tr=tin[pair[1,first]];p=coords[ids];a=tr[:,0][:,[0,2]];b=tr[:,1][:,[0,2]];c=tr[:,2][:,[0,2]];den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1]);w0=((b[:,1]-c[:,1])*(p[:,0]-c[:,0])+(c[:,0]-b[:,0])*(p[:,1]-c[:,1]))/den;w1=((c[:,1]-a[:,1])*(p[:,0]-c[:,0])+(a[:,0]-c[:,0])*(p[:,1]-c[:,1]))/den;out[ids]=w0*tr[:,0,1]+w1*tr[:,1,1]+(1-w0-w1)*tr[:,2,1]
 return out

def subdivide(t,maxedge=10.):
 edges=[np.linalg.norm(t[(j+1)%3,[0,2]]-t[j,[0,2]]) for j in range(3)];j=int(np.argmax(edges))
 if edges[j]<=maxedge:return [t]
 k=(j+1)%3;l=(j+2)%3;m=(t[j]+t[k])/2
 return subdivide(np.array([t[j],m,t[l]]),maxedge)+subdivide(np.array([m,t[k],t[l]]),maxedge)

def build():
 records=[];source=[];inventory=[];counts=collections.Counter();outlines=[]
 with zipfile.ZipFile(A) as z:
  for theme in ['bldg','tran','wtr','dem','brid']:
   member=f'udx/{theme}/'+('533946' if theme in ['dem','wtr'] else '53394611')+f'_{theme}_6697_op.gml';info=z.getinfo(member);source.append({'archive':moga_source(A, ROOT),'member':member,'crc32':f'{info.CRC:08x}','bytes':info.file_size})
  print('Load original full-mesh terrain TIN',flush=True);tin,scanned=terrain(z,'udx/dem/533946_dem_6697_op.gml');print('TIN',len(tin),'triangles / scanned',scanned,flush=True)
  # Check coverage before any draping; do not invent terrain in missing cells.
  aa=tin[:,1][:,[0,2]]-tin[:,0][:,[0,2]];bb=tin[:,2][:,[0,2]]-tin[:,0][:,[0,2]];area=float(np.sum(abs(aa[:,0]*bb[:,1]-aa[:,1]*bb[:,0]))*.5);assert abs(area/ROI.area-1)<.002,f'Incomplete or overlapping TIN: {area/ROI.area}'
  records.extend({'p':np.round(t,5).tolist(),'group':'terrain'} for t in tin);counts['terrain_triangles']=len(tin)
  seen=set()
  for i,o in enumerate(member_objects(z,'udx/bldg/53394611_bldg_6697_op.gml')):
   ident=next((v for k,v in o.attrib.items() if k.endswith('}id')),'');semantic=[q for q in o.iter() if shared.tag(q) in ['RoofSurface','WallSurface','GroundSurface','ClosureSurface','OuterCeilingSurface','OuterFloorSurface']];parts=[po for q in semantic for po in shared.polygons(q)];parts+=geometry_parts(o,'lod2Geometry');lod=2
   if not parts:parts=geometry_parts(o,'lod1Solid');lod=1
   n=add_faces(records,parts,'buildings',seen)
   if n:counts['buildings']+=1;counts[f'buildings_lod{lod}']+=1;inventory.append({'id':ident,'lod':lod,'triangles':n})
  print('Buildings',dict(counts),flush=True)
  roadtris=[];road_edges=[];seen=set()
  for o in member_objects(z,'udx/tran/53394611_tran_6697_op.gml'):
   parts=geometry_parts(o,'lod1MultiSurface');n=0
   for pid,rings in parts:
    key=pid or hashlib.sha256(np.concatenate(rings).tobytes()).hexdigest()
    if key in seen:continue
    seen.add(key)
    for tri in shared.triangulate(rings):
     for t in clip(tri):roadtris.extend(subdivide(t));n+=1
    for ring in rings:
     for a,b in zip(ring,np.roll(ring,-1,axis=0)):
      length=np.linalg.norm(a[[0,2]]-b[[0,2]]);nn=max(1,int(np.ceil(length/5)));rr=np.linspace(a,b,nn+1)
      for aa,bb in zip(rr[:-1],rr[1:]):
       if ROI.covers(Point((aa[0]+bb[0])/2,(aa[2]+bb[2])/2)):road_edges.append([aa,bb])
   if n:counts['roads']+=1
  verts=np.asarray(roadtris).reshape(-1,3);edges=np.asarray(road_edges).reshape(-1,3);coords=np.concatenate([verts[:,[0,2]],edges[:,[0,2]]]);uv,inverse=np.unique(np.round(coords,7),axis=0,return_inverse=True);hh=heights(tin,uv)[inverse];verts[:,1]=hh[:len(verts)]+.055;edges[:,1]=hh[len(verts):]+.18;roadtris=verts.reshape(-1,3,3);good=np.isfinite(roadtris).all((1,2));counts['road_triangles_without_terrain']=int((~good).sum());records.extend({'p':np.round(t,5).tolist(),'group':'roads'} for t in roadtris[good]);edgearray=edges.reshape(-1,2,3);outlines.extend({'p':np.round(t,5).tolist(),'group':'roads'} for t in edgearray[np.isfinite(edgearray).all((1,2))]);print('Roads',counts['roads'],'faces',int(good.sum()),flush=True)
  seen=set()
  for o in member_objects(z,'udx/wtr/533946_wtr_6697_op.gml'):
   n=add_faces(records,geometry_parts(o,'lod1MultiSurface'),'water',seen)
   if n:counts['water_bodies']+=1
  # Bridges retain actual solid geometry, in the road layer.
  seen=set()
  for o in member_objects(z,'udx/brid/53394611_brid_6697_op.gml'):
   sem=[q for q in o.iter() if shared.tag(q) in ['RoofSurface','WallSurface','GroundSurface','ClosureSurface','OuterCeilingSurface','OuterFloorSurface']];parts=[po for q in sem for po in shared.polygons(q)];parts+=geometry_parts(o,'lod2Geometry')
   if not parts:parts=geometry_parts(o,'lod1Solid')
   if add_faces(records,parts,'roads',seen):counts['bridge_objects']+=1
  for group in ['buildings','terrain','roads','water']:counts[group+'_triangles']=sum(t['group']==group for t in records)
 assert counts['roads']>500, 'Unexpectedly sparse road layer'
 manifest={'build_recipe_version':3,'mesh_id':'53394611','geographic_bounds':[LON0,LAT0,LON1,LAT1],'projected_cell_corners':np.c_[x,y].tolist(),'local_origin_easting_northing':[CX,CY],'local_vertical_origin_m':0,'crs':6677,'tile_area_m2':ROI.area,'tin_clipped_area_m2':area,'tin_area_coverage_ratio':area/ROI.area,'terrain_source':'Original PLATEAU DEM TIN triangles, clipped to complete third-order cell. Not a flat floor or raster replacement.','source':source,'counts':dict(counts),'building_inventory':inventory,'materials':'None; all surfaces white; no textures or per-building material coloring.','height_exaggeration':1,'roads':'LOD1 plan footprints subdivided to max10m edge and draped barycentrically on original TIN (+0.055m drawing offset; <=1e-6m tolerance for projected-boundary roundoff); road outlines +0.18m. These display heights are not independently measured road/bridge elevations. Explicit bridge solids retained. Original road Z=0 not used as ground.','water':'Original explicit LOD1 water surfaces, retaining packaged Z. Z is not validated event water level or channel-bed elevation. Some pools may be below TIN and can be inspected with terrain hidden.','scope':'Full tile53394611, not a station object or a station-centered radius. Source objects deduplicated and triangles clipped to cell boundary. All four independently switchable layers are included.'}
 (P/'mesh.json').write_text(json.dumps(records,separators=(',',':')));(P/'outlines.json').write_text(json.dumps(outlines,separators=(',',':')));(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(counts),indent=2),flush=True);return records,outlines,manifest

def obj_export(mesh):
 # No mtllib/usemtl statements: geometry-only OBJ with one group per layer.
 vmap={};verts=[];groups=collections.defaultdict(list)
 for t in mesh:
  ids=[]
  for a in t['p']:
   key=tuple(a)
   if key not in vmap:vmap[key]=len(verts)+1;verts.append(a)
   ids.append(vmap[key])
  groups[t['group']].append(ids)
 p=OUT/'tokyo_mesh_53394611.obj'
 with p.open('w') as f:
  f.write('# PLATEAU53394611; local X=east,Y=original elevation,Z=-north; no materials\n')
  for a in verts:f.write('v '+' '.join(f'{v:.5f}' for v in a)+'\n')
  for g in ['terrain','roads','water','buildings']:
   f.write('g '+g+'\n')
   for tri in groups[g]:f.write('f '+' '.join(map(str,tri))+'\n')
 return {'vertices':len(verts),'faces':sum(map(len,groups.values())),'groups':{g:len(a) for g,a in groups.items()},'material_library':False}

def main():
 P.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 if (P/'manifest.json').exists() and json.loads((P/'manifest.json').read_text()).get('build_recipe_version')==3:mesh=json.loads((P/'mesh.json').read_text());lines=json.loads((P/'outlines.json').read_text());manifest=json.loads((P/'manifest.json').read_text())
 else:mesh,lines,manifest=build()
 manifest=dict(manifest)
 manifest.update({'palette':{'buildings':'#FFFFFF','roads':'#7D858C','terrain':'#D9CFB7','water':'#ADD8EF'},'annotation_language':'en','annotations':'none','controls':'Hidden by default; press H to show or hide English layer controls.','materials':'Geometry-only OBJ has no materials or textures; viewer uses solid layer colors with geometric lighting.'})
 (OUT/'metadata.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='building_inventory'},ensure_ascii=False,indent=2)+'\n')
 template=Path(__file__).with_name('tokyo_mesh_viewer.html').read_text();safe=lambda x:json.dumps(x,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c');html=template.replace('__MESH_DATA__',safe(mesh)).replace('__LINES__',safe(lines)).replace('__MANIFEST__',safe(manifest));(OUT/'tokyo_mesh_53394611.html').write_text(html);obj=obj_export(mesh);(OUT/'obj_validation.json').write_text(json.dumps(obj,indent=2)+'\n');print('EXPORT',obj,flush=True)
if __name__=='__main__':main()
