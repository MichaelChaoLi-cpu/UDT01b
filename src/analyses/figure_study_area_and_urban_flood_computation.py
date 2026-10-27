#!/usr/bin/env python3
"""Study Area and Urban Flood Computation.
AnaSOP 5: contextual representation, not empirical validation or an estimator.
AnaSOP 6: existing 2D hydrodynamics, road rules, restricted traffic and resident access.
AnaSOP 7-8: four panels ab / ac / ad; no new simulation or fitting.
Read-only source data; all render caches live in data/exp/figure-table-generation.
"""
from __future__ import annotations
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-context-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
for k in ['PROJ_DATA','PROJ_LIB','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,hashlib,zipfile,collections,subprocess,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection,LineCollection
from matplotlib.patches import PathPatch,Patch,Rectangle
from matplotlib.path import Path as MPath
from matplotlib.colors import LightSource,to_rgb
from matplotlib.ticker import FuncFormatter,MaxNLocator
from matplotlib.lines import Line2D
from PIL import Image
from pyproj import Transformer
from shapely.geometry import shape,box,Polygon
from shapely.ops import transform as geometry_transform
from shapely import contains_xy,union_all,transform as shapely_transform
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from scipy.interpolate import RegularGridInterpolator
import seaborn as sns
ROOT=Path(__file__).resolve().parents[2]
TITLE='Study Area and Urban Flood Computation';SLUG='study_area_and_urban_flood_computation'
A=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'/SLUG
OUT=ROOT/'data/results/figures'/f'Figure_{SLUG}.png'
XY=Transformer.from_crs(4326,6677,always_xy=True)
LL=Transformer.from_crs(6677,4326,always_xy=True)
ROI=box(-7240,-36520,-6160,-35520)
CX,CY=ROI.centroid.x,ROI.centroid.y
COL={'buildings':'#FFFFFF','roads':'#7D858C','terrain':'#D9CFB7','water':'#ADD8EF','ward':'#BB6A32','face':'#B24B3D'}
V=ROOT/'data/processed/three_ward_urban_constraints/constraint_vectors_6677.geojson'
DEM=ROOT/'data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif'
WTR=ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson'
WARDS=ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson'
RUN=ROOT/'data/processed/three_ward_s1r_cross_event_validation_v1/batches/E20240830_Q5/runs/E20240830__H__U__s1r_q5'
MAN=RUN.parent.parent/'run_manifest.json'
ARCH=ROOT/'data/raw/plateau/2025/13101-chiyoda-ku/13101_chiyoda-ku_pref_2025_citygml_1_op.zip'
N03=ROOT/'data/raw/ksj/N03-20240101_13_GML.zip'
SOURCE={};STATS={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def bind(p):
 p=Path(p);SOURCE[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(bind(p).read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pieces(g):
 if g.is_empty:return []
 if g.geom_type=='Polygon':return [g]
 return [p for q in g.geoms for p in pieces(q)] if hasattr(g,'geoms') else []
def patch(g,to_geo=False):
 vs=[];cs=[]
 for ring in [g.exterior,*g.interiors]:
  p=np.asarray(ring.coords)
  if to_geo:p=np.column_stack(LL.transform(p[:,0],p[:,1]))
  vs.extend(p[:,:2]);cs.extend([MPath.MOVETO]+[MPath.LINETO]*(len(p)-2)+[MPath.CLOSEPOLY])
 return PathPatch(MPath(np.asarray(vs),cs))
def polycollection(ax,gs,**kw):
 p=[patch(q) for g in gs for q in pieces(g)]
 if p:ax.add_collection(PatchCollection(p,**kw))
def metric(g):return shapely_transform(g,XY.transform,interleaved=False)
def prepare():
 A.mkdir(parents=True,exist_ok=True)
 doc=bind(ROOT/'docs/AnaSOP.md').read_text();assert f'| {TITLE} |' in doc
 wards=[(f['properties']['ward'][:5],metric(shape(f['geometry']))) for f in read(WARDS)['features']]
 waters=[(f,metric(shape(f['geometry']))) for f in read(WTR)['features']]
 native=read(MAN)['runs'][0];assert native['identity']['geometry_sha256']=='a642d0738b553740650599bd6fd5ab9f5e45011cdcd152c2c23794f85091f2fd'
 for n in ['sfincs.dep','sfincs.msk','sfincs.ind','sfincs.inp','building_closures.thd']:
  bind(RUN/n);assert sha(RUN/n)==native['input_sha256'][n]
 dep=np.fromfile(RUN/'sfincs.dep',dtype='<f4');idx=np.fromfile(RUN/'sfincs.ind',dtype='<i4');msk=np.fromfile(RUN/'sfincs.msk',dtype='i1')
 assert idx[0]==len(dep)==len(msk)==368872 and len(idx)==len(dep)+1
 ii=idx[1:]-1;row=ii%1012;col=ii//1012
 assert col.max()<770 and row.max()<1012
 xs=-11360+(col+.5)*10;ys=-42440+(row+.5)*10
 with np.load(bind(ROOT/'data/processed/three_ward_barrier_drainage_v1/measures/spatial_contract.npz')) as z:
  np.testing.assert_allclose(np.column_stack([xs,ys]),z['native_xy'],atol=1e-5,rtol=0)
 grid=np.full((1012,770),np.nan,dtype='f4');grid[1011-row,col]=dep
 active=np.zeros_like(grid,dtype=bool);active[1011-row,col]=msk>0
 # Display terrain is reduced only for the context map. Native hydraulic data stay unchanged.
 dsstat={'size':DEM.stat().st_size,'mtime_ns':DEM.stat().st_mtime_ns}
 with rasterio.open(DEM) as ds:
  overview=ds.read(1,out_shape=(1012,770),resampling=Resampling.average)
  b=ds.bounds;extent=(b.left,b.right,b.bottom,b.top)
  win=from_bounds(*ROI.bounds,ds.transform);rh=200;rw=216
  terrain=ds.read(1,window=win,out_shape=(rh,rw),resampling=Resampling.average)
  tx=rasterio.transform.from_bounds(*ROI.bounds,rw,rh)
 assert np.isfinite(terrain).sum()>terrain.size*.6
 STATS['detail_terrain_missing_fraction']=float((~np.isfinite(terrain)).mean())
 # Small file bindings; the large terrain file is checked by stat, not rehashed repeatedly.
 STATS.update(terrain_source={'path':str(DEM.relative_to(ROOT)),**dsstat},native_cells=len(dep),native_case=native['case_id'],native_geometry_sha256=native['identity']['geometry_sha256'],native_solver_digest=native['identity']['solver_digest'])
 watergs=[g for f,g in waters]
 wm=geometry_mask([(g,1) for g in watergs],transform=tx,out_shape=terrain.shape,invert=True)
 # Same projected rectangle in c and d, not a favorable flood-result selection.
 localwaters=[(f,g.intersection(ROI)) for f,g in waters if g.intersects(ROI)]
 print('Read three-ward geometry',flush=True)
 fs=read(V)['features'];buildings=[];roads=[];localroads=[];counts=collections.Counter()
 roi0=ROI.bounds
 for f in fs:
  k=f['properties'].get('class');counts[k]+=1
  if k not in ('building','road_ground_candidate'):continue
  g=shape(f['geometry'])
  if k=='building':buildings.append(g)
  else:
   roads.append(g)
   b=g.bounds
   if b[0]<roi0[2] and b[2]>roi0[0] and b[1]<roi0[3] and b[3]>roi0[1]:localroads.extend(pieces(g.intersection(ROI)))
 del fs
 ll=read(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/wards_6677.geojson')
 # Native fully closed building faces: exact current thd file, not a guessed raster graph.
 lines=(RUN/'building_closures.thd').read_text().splitlines();faces=[];i=0
 while i<len(lines):
  assert lines[i].startswith('BuildingClosure');n,m=map(int,lines[i+1].split());assert (n,m)==(2,2)
  pp=np.array([list(map(float,lines[i+2].split())),list(map(float,lines[i+3].split()))]);i+=4
  if pp[:,0].max()>=ROI.bounds[0] and pp[:,0].min()<=ROI.bounds[2] and pp[:,1].max()>=ROI.bounds[1] and pp[:,1].min()<=ROI.bounds[3]:faces.append(pp)
 STATS.update(ward_codes=sorted(k for k,g in wards),water_objects=len(waters),vector_counts=dict(counts),local_closed_faces=len(faces),local_water_objects=len(localwaters),local_water_footprint_m2=float(sum(g.area for f,g in localwaters)),detail_bounds_epsg6677=list(ROI.bounds),same_detail_bounds=True)
 assert len(faces)>0 and len(localwaters)>0 and len(localroads)>0
 # Model extent is (left, right, bottom, top), with whole ward boundaries retained.
 native_rect=box(extent[0],extent[2],extent[1],extent[3])
 STATS['ward_fraction_outside_overview']={code:float(g.difference(native_rect).area/g.area) for code,g in wards}
 assert max(STATS['ward_fraction_outside_overview'].values())<1e-8
 return dict(wards=wards,waters=waters,buildings=buildings,roads=roads,localroads=localroads,localwaters=localwaters,overview=overview,grid=grid,active=active,terrain=terrain,watermask=wm,terrain_transform=tx,extent=extent,faces=faces,demstat=dsstat)

def build_scene(z):
 cache=A/'scene_geometry.json';receipt=A/'scene_sources.json'
 recipe={'bounds':list(ROI.bounds),'recipe':1,'terrain_stat':z['demstat'],'vectors_sha256':SOURCE[str(V.relative_to(ROOT))],'water_sha256':SOURCE[str(WTR.relative_to(ROOT))],'archive_stat':{'size':ARCH.stat().st_size,'mtime_ns':ARCH.stat().st_mtime_ns}}
 if cache.exists() and receipt.exists() and json.loads(receipt.read_text()).get('recipe')==recipe:
  print('Reuse frozen local scene cache',flush=True);q=json.loads(receipt.read_text());STATS['scene']=q;return cache
 sys.path.insert(0,str(ROOT/'src/visualization'))
 import render_tokyo_mesh as clipper
 import render_tokyo_station as S
 S.CX=CX;S.CY=CY;S.BASE=0
 # Clip every geometry to the same model-coordinate rectangle.
 corners=np.array([[ROI.bounds[0],ROI.bounds[1]],[ROI.bounds[2],ROI.bounds[1]],[ROI.bounds[2],ROI.bounds[3]],[ROI.bounds[0],ROI.bounds[3]]])
 clipper.RING=np.column_stack([corners[:,0]-CX,-(corners[:,1]-CY)])[::-1]
 rec=[];memberbindings=[];objects=collections.Counter();seen=set()
 class Hashed:
  def __init__(self,f):self.f=f;self.h=hashlib.sha256()
  def read(self,n=-1):b=self.f.read(n);self.h.update(b);return b
 def emit(rings,kind):
  n0=len(rec)
  # Original planar three-dimensional polygons; triangulation preserves rings and holes.
  if len(rings)==1 and len(rings[0])==3:triangles=[rings[0]]
  else:triangles=S.triangulate(rings)
  for tri in triangles:
   for t in clipper.clip(tri):rec.append({'p':np.round(t,5).tolist(),'group':kind})
  return len(rec)-n0
 with zipfile.ZipFile(ARCH) as arc:
  members=[f'udx/bldg/{t}_bldg_6697_op.gml' for t in ['53394600','53394601','53394610','53394611']]+['udx/wtr/533946_wtr_6697_op.gml']
  for member in members:
   print('Extract local',member,flush=True);info=arc.getinfo(member)
   with arc.open(member) as raw:
    f=Hashed(raw);it=ET.iterparse(f,events=['start','end']);_,root=next(it)
    for event,ob in it:
     if event!='end' or S.tag(ob)!='cityObjectMember':continue
     if not len(ob):root.clear();continue
     o=next(iter(ob));pos=[q for q in o.iter() if S.tag(q)=='posList' and q.text]
     if not pos:root.clear();continue
     pp=np.concatenate([np.fromstring(q.text,sep=' ').reshape(-1,3) for q in pos])
     xx,yy=XY.transform(pp[:,1],pp[:,0])
     if xx.max()<ROI.bounds[0] or xx.min()>ROI.bounds[2] or yy.max()<ROI.bounds[1] or yy.min()>ROI.bounds[3]:root.clear();continue
     if '/bldg/' in member:
      kind='buildings';sem=[q for q in o.iter() if S.tag(q) in ['RoofSurface','WallSurface','GroundSurface','ClosureSurface','OuterCeilingSurface','OuterFloorSurface']]
      parts=[p for q in sem for p in S.polygons(q)];parts += [p for q in o.iter() if S.tag(q)=='lod2Geometry' for p in S.polygons(q)]
      if not parts:parts=[p for q in o.iter() if S.tag(q)=='lod1Solid' for p in S.polygons(q)]
     else:
      kind='water';parts=[p for q in o.iter() if S.tag(q)=='lod1MultiSurface' for p in S.polygons(q)]
     n=0
     for ident,rings in parts:
      ident=ident or hashlib.sha256(np.concatenate(rings).tobytes()).hexdigest()
      if ident in seen:continue
      seen.add(ident);n+=emit(rings,kind)
     if n:objects[kind]+=1
     root.clear()
    memberbindings.append({'archive':str(ARCH.relative_to(ROOT)),'member':member,'sha256_uncompressed':f.h.hexdigest(),'crc32':f'{info.CRC:08x}','bytes':info.file_size})
 # Ground surface from frozen bare-earth / PLATEAU fill; no flat floor.
 ter=z['terrain'];tr=z['terrain_transform'];x=tr.c+(np.arange(ter.shape[1])+.5)*tr.a;y=tr.f+(np.arange(ter.shape[0])+.5)*tr.e
 interp=RegularGridInterpolator((y[::-1],x),ter[::-1],bounds_error=False,fill_value=None)
 for j in range(ter.shape[0]-1):
  for k in range(ter.shape[1]-1):
   if z['watermask'][j:j+2,k:k+2].any() or not np.isfinite(ter[j:j+2,k:k+2]).all():continue
   ps=np.array([[x[k]-CX,ter[j,k],-(y[j]-CY)],[x[k+1]-CX,ter[j,k+1],-(y[j]-CY)],[x[k+1]-CX,ter[j+1,k+1],-(y[j+1]-CY)],[x[k]-CX,ter[j+1,k],-(y[j+1]-CY)]])
   for ids in [[0,1,2],[0,2,3]]:rec.append({'p':np.round(ps[ids],5).tolist(),'group':'terrain'})
 for g in z['localroads']:
  rings=[]
  for ring in [g.exterior,*g.interiors]:
   q=np.asarray(ring.coords)[:-1];rings.append(np.column_stack([q[:,0]-CX,interp(q[:,[1,0]])+.08,-(q[:,1]-CY)]))
  if not all(np.isfinite(q).all() for q in rings):
   objects['roads_with_unknown_display_height_omitted']+=1;continue
  for tri in S.triangulate(rings):
   for t in clipper.subdivide(tri,maxedge=5):
    t[:,1]=interp(np.column_stack([CY-t[:,2],CX+t[:,0]]))+.08
    if np.isfinite(t).all():rec.append({'p':np.round(t,5).tolist(),'group':'roads'})
 counts=dict(collections.Counter(t['group'] for t in rec));assert all(counts.get(k,0)>0 for k in COL if k in ['buildings','roads','terrain','water'])
 cache.write_text(json.dumps(rec,separators=(',',':'),allow_nan=False))
 q={'recipe':recipe,'counts':counts,'object_counts':dict(objects),'member_bindings':memberbindings,'terrain_display_spacing_m':5,'height_exaggeration':1,'water_height':'Original explicit LOD1 polygon vertex elevations, unverified as hydraulic stage or bathymetry','terrain_water_cutouts':'Terrain display triangles overlapping the water footprint are omitted solely to expose the source water layer','roads_height':'Display draping on terrain +0.08 m, not a surveyed carriageway or bridge elevation','source_archive_stat':{'size':ARCH.stat().st_size,'mtime_ns':ARCH.stat().st_mtime_ns}}
 write(receipt,q);STATS['scene']=q;print('Scene triangles',counts,flush=True);return cache

def render_scene(cache):
 p=A/'local_3d_scene.png';meta=A/'scene_webgl_validation.json'
 signature=sha(cache)
 template_path=bind(ROOT/'src/visualization/tokyo_mesh_viewer.html')
 render_recipe={'mesh_sha256':signature,'template_sha256':sha(template_path),'palette':COL,'camera':[215,58,1640,0,32,0],'recipe':1}
 if p.exists() and meta.exists() and json.loads(meta.read_text()).get('render_recipe')==render_recipe and sha(p)==json.loads(meta.read_text()).get('image_sha256'):
  STATS['webgl']=json.loads(meta.read_text());return p
 template=template_path.read_text()
 manifest={'palette':{k:COL[k] for k in ['buildings','roads','terrain','water']},'mesh_id':'Hibiya-Marunouchi-context','counts':STATS['scene']['counts']}
 html=template.replace('__MESH_DATA__',cache.read_text()).replace('__LINES__','[]').replace('__MANIFEST__',json.dumps(manifest))
 html=html.replace('width=2200','width=1640').replace('target=[0,65,0]','target=[0,32,0]').replace('az=215*Math.PI/180','az=215*Math.PI/180').replace('el=53*Math.PI/180','el=58*Math.PI/180').replace('gl.clearColor(.91,.92,.935,1)','gl.clearColor(1,1,1,1)')
 (A/'local_3d_scene.html').write_text(html)
 js=r'''const fs=require('fs');const {pathToFileURL}=require('url');const {chromium}=require('/Users/lichao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{let d=process.argv[2];const b=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});const p=await b.newPage({viewport:{width:1800,height:1450},deviceScaleFactor:2});let errors=[];p.on('pageerror',e=>errors.push(e.message));await p.goto(pathToFileURL(d+'/local_3d_scene.html').href+'?export=1',{timeout:120000});await p.waitForFunction(()=>window.sceneReady===true,{timeout:120000});await p.screenshot({path:d+'/local_3d_scene.png'});let state=await p.evaluate(()=>({glError:window.lastGlError,stats:window.sceneStats,layers:window.layerState,annotations:document.body.innerText}));if(errors.length||state.glError||Object.values(state.layers).filter((v,i)=>i<4).some(v=>!v))throw Error(JSON.stringify({errors,state}));fs.writeFileSync(d+'/scene_webgl_validation_raw.json',JSON.stringify({passed:true,errors,state},null,2));await b.close();})().catch(e=>{console.error(e);process.exit(1)});'''
 (A/'export_local_scene.cjs').write_text(js)
 subprocess.run(['/opt/homebrew/bin/node',str(A/'export_local_scene.cjs'),str(A)],check=True)
 q=json.loads((A/'scene_webgl_validation_raw.json').read_text());q['mesh_sha256']=signature;q['render_recipe']=render_recipe;q['image_sha256']=sha(p);write(meta,q);STATS['webgl']=q;return p

def mapstyle(ax,bounds,geo=False):
 x0,y0,x1,y1=bounds
 if geo:
  ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect(1/np.cos(np.deg2rad((y0+y1)/2)))
 else:
  ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect('equal')
  # True plan-coordinate axes with longitude/latitude tick labels.
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{LL.transform(x,(y0+y1)/2)[0]:.2f}°E'))
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{LL.transform((x0+x1)/2,y)[1]:.2f}°N'))
 ax.tick_params(labelsize=9);ax.set_xlabel('Longitude',fontsize=10);ax.set_ylabel('Latitude',fontsize=10)
 for s in ax.spines.values():s.set_visible(True);s.set_color('#414A50');s.set_linewidth(.75)

def graticule(ax,bounds,step=.02):
 x0,y0,x1,y1=bounds;lo,la=LL.transform([x0,x1],[y0,y1]);lons=np.arange(np.ceil(min(lo)/step)*step,max(lo),step);lats=np.arange(np.ceil(min(la)/step)*step,max(la),step)
 # Geographic curves are projected into the metric map; values are never resampled.
 for v in lons:
  xx,yy=XY.transform(np.full(120,v),np.linspace(min(la)-.02,max(la)+.02,120));ax.plot(xx,yy,color='#71828A',lw=.45,alpha=.5,ls=(0,(4,4)),zorder=8)
 for v in lats:
  xx,yy=XY.transform(np.linspace(min(lo)-.02,max(lo)+.02,120),np.full(120,v));ax.plot(xx,yy,color='#71828A',lw=.45,alpha=.5,ls=(0,(4,4)),zorder=8)
 # Put ticks at intersections on midline, sufficient at this geographic extent.
 xx,_=XY.transform(lons,np.full(len(lons),(min(la)+max(la))/2));_,yy=XY.transform(np.full(len(lats),(min(lo)+max(lo))/2),lats)
 ax.set_xticks(xx);ax.set_yticks(yy)
 if step<.01:
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{LL.transform(x,(y0+y1)/2)[0]:.3f}°E'));ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{LL.transform((x0+x1)/2,y)[1]:.3f}°N'))

def scalebar(ax,bounds,length):
 x0,y0,x1,y1=bounds;x=x0+.06*(x1-x0);y=y0+.055*(y1-y0)
 ax.plot([x,x+length],[y,y],color='#25333D',lw=2,zorder=15);ax.plot([x,x],[y-15,y+15],color='#25333D',lw=1,zorder=15);ax.plot([x+length,x+length],[y-15,y+15],color='#25333D',lw=1,zorder=15)
 ax.text(x+length/2,y+.015*(y1-y0),f'{length/1000:g} km' if length>=1000 else f'{length:g} m',ha='center',va='bottom',fontsize=9,zorder=16,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1.2})
def heading(ax,label):
 return ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(0,9),textcoords='offset points',fontsize=13,fontweight='bold',ha='left',va='bottom',annotation_clip=False)
def terrain_rgb(d):
 valid=np.isfinite(d);v=np.where(valid,d,np.nanmedian(d));hs=LightSource(azdeg=315,altdeg=40).hillshade(v,vert_exag=1,dx=10,dy=10)
 col=np.array(to_rgb(COL['terrain']));rgb=col[None,None,:]*(.87+.16*hs[:,:,None]);return np.dstack([np.clip(rgb,0,1),valid.astype(float)])
def render(z,scene):
 sns.set_theme(style='white',font='DejaVu Sans',rc={'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
 # Physical panel dimensions preserve map and scene aspect while aligning b/c/d.
 with zipfile.ZipFile(bind(N03)) as arc:nd=json.loads(arc.read('N03-20240101_13.geojson'))
 locator_geometries=[metric(shape(f['geometry'])) for f in nd['features'] if f['properties'].get('N03_007') and 13101<=int(f['properties']['N03_007'])<=13123]
 locbounds=union_all(locator_geometries).bounds
 im=Image.open(scene).convert('RGB');pixels=np.asarray(im);yy,xx=np.where(np.any(pixels<244,axis=2))
 crop=(max(0,int(xx.min())-35),max(0,int(yy.min())-35),min(im.width,int(xx.max())+36),min(im.height,int(yy.max())+36));im=im.crop(crop)
 width,height=15.7,13.6;right_x=11.05;right_w=3.9;top=12.75;gap=.50
 bh=right_w*(locbounds[3]-locbounds[1])/(locbounds[2]-locbounds[0])
 ch=right_w*im.height/im.width
 dh=right_w*(ROI.bounds[3]-ROI.bounds[1])/(ROI.bounds[2]-ROI.bounds[0])
 total_h=bh+ch+dh+2*gap;bottom=top-total_h
 assert bottom>.65,(bottom,bh,ch,dh)
 bb=z['extent'];left_w=total_h*(bb[1]-bb[0])/(bb[3]-bb[2]);left_x=.95
 assert left_x+left_w<right_x-.75
 fig=plt.figure(figsize=(width,height),dpi=100)
 def position(x,y,w,h):return [x/width,y/height,w/width,h/height]
 axes={'a':fig.add_axes(position(left_x,bottom,left_w,total_h)),
       'b':fig.add_axes(position(right_x,top-bh,right_w,bh)),
       'c':fig.add_axes(position(right_x,top-bh-gap-ch,right_w,ch)),
       'd':fig.add_axes(position(right_x,bottom,right_w,dh))}
 ax=axes['a'];bb=z['extent'];bounds=(bb[0],bb[2],bb[1],bb[3]);ax.set_facecolor('#F6F5F0')
 ax.imshow(terrain_rgb(z['overview']),extent=bb,origin='upper',zorder=1,interpolation='nearest')
 polycollection(ax,z['roads'],facecolor=COL['roads'],edgecolor='none',zorder=2,rasterized=True)
 polycollection(ax,z['buildings'],facecolor='white',edgecolor='#A8A99F',linewidth=.10,zorder=3,rasterized=True)
 polycollection(ax,[g for f,g in z['waters']],facecolor=COL['water'],edgecolor='#79A3B7',linewidth=.2,zorder=4)
 for code,g in z['wards']:
  for p in pieces(g):ax.plot(*p.exterior.xy,color='#A66136',lw=1.15,zorder=9)
 for code,xy,label in [('13101',(-8800,-34100),'Chiyoda'),('13102',(-5300,-36900),'Chuo'),('13103',(-9200,-39650),'Minato')]:
  ax.text(*xy,label,fontsize=13,fontweight='bold',color='#353C3F',zorder=12,bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':3})
 ax.add_patch(Rectangle((ROI.bounds[0],ROI.bounds[1]),ROI.bounds[2]-ROI.bounds[0],ROI.bounds[3]-ROI.bounds[1],fill=False,ec='#C05F24',lw=1.7,zorder=12))
 ax.annotate('Hibiya–Marunouchi\nPanels c and d',xy=(ROI.bounds[2],ROI.bounds[1]+100),xytext=(-5000,-38800),fontsize=10,ha='center',arrowprops={'arrowstyle':'-','color':'#A3572A','lw':.9},zorder=13,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':3})
 mapstyle(ax,bounds);graticule(ax,bounds);scalebar(ax,bounds,2000)
 ax.annotate('N',xy=(.94,.94),xytext=(.94,.84),xycoords='axes fraction',ha='center',va='center',fontsize=11,fontweight='bold',arrowprops={'arrowstyle':'-|>','lw':1,'color':'#26343E'},zorder=15)
 heading(ax,'a: Three-ward urban model')
 # Locator: this project's already downloaded official N03, never sibling-derived data.
 ax=axes['b'];groups=[];target=[];codes=set()
 for f in nd['features']:
  code=f['properties'].get('N03_007')
  if not code or not 13101<=int(code)<=13123:continue
  codes.add(code);g=metric(shape(f['geometry']));groups.append(g)
  if code in ['13101','13102','13103']:target.append(g)
 assert len(codes)==23
 STATS['locator_ward_count']=len(codes);STATS['locator_source']='N03 2024-01-01, local source archive'
 polycollection(ax,groups,facecolor='#DEE4E6',edgecolor='white',linewidth=.55)
 polycollection(ax,target,facecolor='#CB8050',edgecolor='#8E4C23',linewidth=.8)
 allg=union_all(groups);x0,y0,x1,y1=allg.bounds;dx=(x1-x0)*.04;dy=(y1-y0)*.04
 ax.set_xlim(x0-dx,x1+dx);ax.set_ylim(y0-dy,y1+dy);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
 for s in ax.spines.values():s.set_visible(True);s.set_color('#414A50');s.set_linewidth(.75)
 ax.text(.04,.95,'Tokyo, Japan',transform=ax.transAxes,va='top',fontsize=11,fontweight='bold');ax.legend(handles=[Patch(fc='#CB8050',ec='#8E4C23',label='Study wards'),Patch(fc='#DEE4E6',ec='white',label='Other special wards')],loc='lower right',frameon=True,facecolor='white',edgecolor='none',fontsize=8.5)
 heading(ax,'b: Location in Tokyo')
 # c: actual depth-tested WebGL surface rendering, original LOD2 building geometry.
 ax=axes['c'];ax.imshow(im);ax.axis('off');heading(ax,'c: Hibiya–Marunouchi in 3D')
 STATS['scene_crop_px']=list(crop)
 # d: actual native ten-metre cells and blocked exchange faces, overlaid source footprints.
 ax=axes['d'];x0,y0,x1,y1=ROI.bounds;c0=int((x0+11360)//10);c1=int((x1+11360)//10);r0=int((-32320-y1)//10);r1=int((-32320-y0)//10)
 gd=z['grid'][r0:r1,c0:c1];nativefinite=np.isfinite(gd)
 ax.set_facecolor('#F0F1EE');ax.imshow(terrain_rgb(gd),extent=(x0,x1,y0,y1),origin='upper',interpolation='nearest',zorder=1)
 localbs=[g.intersection(ROI) for g in z['buildings'] if g.intersects(ROI)]
 polycollection(ax,z['localroads'],facecolor=COL['roads'],edgecolor='none',zorder=2)
 polycollection(ax,localbs,facecolor='white',edgecolor='#C8CBC6',linewidth=.3,zorder=3)
 polycollection(ax,[g for f,g in z['localwaters']],facecolor=COL['water'],edgecolor='#7AA6B9',linewidth=.45,zorder=4)
 # Thin regular grid represents working-cell size; red faces come only from thd.
 glines=[[(x,y0),(x,y1)] for x in np.arange(x0,x1+1,10)]+[[(x0,y),(x1,y)] for y in np.arange(y0,y1+1,10)]
 ax.add_collection(LineCollection(glines,colors='#848A87',lw=.19,alpha=.42,zorder=5))
 ax.add_collection(LineCollection(z['faces'],colors=COL['face'],lw=.65,zorder=7))
 mapstyle(ax,ROI.bounds);graticule(ax,ROI.bounds,step=.004);scalebar(ax,ROI.bounds,200)
 heading(ax,'d: 10 m computational grid')
 ax.legend(handles=[Line2D([0],[0],color=COL['face'],lw=1.5,label='Closed building face'),Line2D([0],[0],color='#848A87',lw=.7,label='10 m cell')],loc='upper right',frameon=True,facecolor='white',edgecolor='none',fontsize=8.5)
 STATS['detail_native_grid_shape']=list(gd.shape);STATS['detail_native_active_cells']=int(nativefinite.sum())
 # Shared semantic key below all four panels; no workflow panel or small-print narrative.
 handles=[Patch(fc=COL[k],ec='#8D969A',lw=.55,label=label) for k,label in [('buildings','Buildings'),('roads','Roads'),('terrain','Terrain'),('water','Water')]]
 leg=fig.legend(handles=handles,loc='center',bbox_to_anchor=(.50,.015),ncol=4,frameon=False,fontsize=11,columnspacing=2.5,handlelength=1.5,handletextpad=.6)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();positions={k:list(ax.get_position().bounds) for k,ax in axes.items()}
 assert fig._suptitle is None and all(not ax.get_title() for ax in axes.values())
 textcheck=[]
 for k,ax in axes.items():
  tb=ax.get_tightbbox(renderer)
  assert tb.x0>=0 and tb.y0>=0 and tb.x1<=fig.bbox.width+1 and tb.y1<=fig.bbox.height+1,(k,tb.bounds)
  for j,other in axes.items():
   if k<j:assert not tb.overlaps(other.get_tightbbox(renderer)),(k,j,'panel overlap')
 # Exact right-column alignment and the two outer top/bottom alignments.
 pb={k:ax.get_position() for k,ax in axes.items()}
 alignment={'b_c_left':abs(pb['b'].x0-pb['c'].x0),'b_d_left':abs(pb['b'].x0-pb['d'].x0),'b_c_right':abs(pb['b'].x1-pb['c'].x1),'b_d_right':abs(pb['b'].x1-pb['d'].x1),'a_b_top':abs(pb['a'].y1-pb['b'].y1),'a_d_bottom':abs(pb['a'].y0-pb['d'].y0)}
 assert max(alignment.values())<1e-9,alignment
 lb=leg.get_window_extent(renderer);assert lb.x0>=0 and lb.y0>=0 and lb.x1<=fig.bbox.width and lb.y1<=fig.bbox.height
 assert all(not lb.overlaps(ax.get_tightbbox(renderer)) for ax in axes.values()),'legend collision'
 # Exactly English labels; proper hyphenated place names are geographic labels.
 texts=[t.get_text() for t in fig.findobj(matplotlib.text.Text)];assert not any(any('\u3400'<=c<='\u9fff' for c in t) for t in texts)
 OUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT,dpi=450,facecolor='white');fig.savefig(A/'preview.png',dpi=120,facecolor='white');plt.close(fig)
 with Image.open(OUT) as p:dim=list(p.size);dpi=list(p.info['dpi']);assert all(449<=v<=451 for v in dpi)
 STATS.update(layout=['ab','ac','ad'],alignment_error_figure_fraction=alignment,workflow_panel_removed=True,panel_positions=positions,dimensions_px=dim,dpi=dpi,english_only=True,no_global_title=True,panels_nonoverlapping=True,all_four_layers_present=True)
 return STATS

def main():
 existing={str(p.relative_to(ROOT)):sha(p) for folder in ['data/results/figures','data/results/tables'] for p in (ROOT/folder).glob('*') if p.is_file() and p!=OUT}
 z=prepare();mesh=build_scene(z);scene=render_scene(mesh);checks=render(z,scene)
 assert {'size':DEM.stat().st_size,'mtime_ns':DEM.stat().st_mtime_ns}==z['demstat']
 assert all(sha(ROOT/p)==h for p,h in SOURCE.items())
 assert all(sha(ROOT/p)==h for p,h in existing.items())
 write(A/'source_bindings.json',SOURCE)
 caption='''Study area, three-dimensional urban information and computational representation. (a) Chiyoda, Chuo and Minato, with the frozen terrain, building footprints, ground-road candidates and PLATEAU water footprints; adjoining context remains visible. The orange rectangle locates the common Hibiya–Marunouchi detail window. (b) The study wards within Tokyo's 23 special wards, using the project's downloaded official N03 administrative zones dated 1 January 2024. (c) An elevated depth-tested rendering of actual PLATEAU building surfaces, terrain, ground-road candidates and explicit LOD1 water surfaces in the same projected rectangle used in d. Buildings are white, roads gray, terrain beige and water pale blue; no facade textures or height exaggeration are used. Fine terrain is reduced to five-meter display samples, with missing ground kept absent. Ground-road geometry is draped on this terrain with an 0.08-meter drawing offset, not a surveyed carriageway or bridge elevation. Original water-surface elevations are retained; these are not independently validated event stages or channel-bed elevations. Display terrain intersecting water footprints is omitted solely to expose the water layer, without modifying any hydraulic input. (d) The same area on the actual regular ten-meter hydraulic grid. Native bed elevations, active support and fully closed building exchange faces are read from an existing frozen urban-constrained case; red segments are the actual building-closure input, not guessed flow connections. Source footprints remain visual context, not a claim that every subcell building is fully impermeable. Light gray gaps identify unsupported or inactive native bed support. The figure explains the implemented framework and introduces no new hydraulic run, transport fit, machine-learning training, empirical flood validation or superiority claim for three-dimensional information.'''
 (A/'caption.md').write_text(caption+'\n')
 write(A/'validation.json',{'passed':True,'status':'generated_awaiting_human_review','title':TITLE,'panels':4,'checks':checks,'source_sha256':SOURCE,'output':str(OUT.relative_to(ROOT)),'output_sha256':sha(OUT),'script_sha256':sha(Path(__file__)),'other_publication_outputs_unchanged':True,'new_hydraulic_runs':0,'new_transport_fits':0,'training':False,'sealed_test_access':False})
 (A/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+TITLE+'</title><style>body{font:16px Arial;margin:28px auto;max-width:1400px;padding:0 22px;color:#24333c}img{width:100%;height:auto}p{line-height:1.6}a{color:#2b6887}</style><h1>'+TITLE+'</h1><p>Four-panel draft · English · 450 dpi · Existing model inputs only</p><a href="../../../../results/figures/Figure_'+SLUG+'.png"><img src="preview.png" alt="Study location, 3D urban model and computational grid"></a><p>'+caption+'</p></html>')
 print(json.dumps({'passed':True,'output':str(OUT),'preview':str(A/'preview.png'),'dimensions':checks['dimensions_px'],'native_faces':checks['local_closed_faces'],'scene_triangles':checks['scene']['counts']},indent=2))
if __name__ == '__main__':
 main()
 from publication_review_state import refresh_review_page
 refresh_review_page(Path(__file__).resolve().parents[2], 'study_area_and_urban_flood_computation')
