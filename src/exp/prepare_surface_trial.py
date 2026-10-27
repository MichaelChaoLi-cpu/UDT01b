"""Build exploratory 20m surface inputs from original PLATEAU triangles and LOD0.
No source ZIP edits, no depression filling, no waterbody-Z boundary stages.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import json,zipfile,xml.etree.ElementTree as ET,math,csv,hashlib
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter,binary_dilation,distance_transform_edt
from pyproj import Transformer
from shapely.geometry import shape,mapping
from shapely.ops import transform,unary_union
ROOT=Path(__file__).resolve().parents[2] if Path(__file__).parent.name=='exp' else Path('/Volumes/Extension/Research/UDT01b')
PRO=ROOT/'data/processed/three_ward_surface_trial';OUT=ROOT/'data/exp/data-briefing/three_ward_surface_trial'
DX=20.;FINE=5.;TOXY=Transformer.from_crs(6668,6677,always_xy=True)

def write(a,name,tf,nodata=None):
 with rasterio.open(PRO/(name+'.tif'),'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype=a.dtype,crs='EPSG:6677',transform=tf,nodata=nodata,compress='deflate',tiled=True) as d:d.write(a,1)

def main():
 PRO.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 wardf=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];wards=[transform(TOXY.transform,shape(f['geometry'])) for f in wardf];union=unary_union(wards);domain=union.buffer(1000);left,bottom,right,top=domain.bounds;left=math.floor(left/DX)*DX;bottom=math.floor(bottom/DX)*DX;right=math.ceil(right/DX)*DX;top=math.ceil(top/DX)*DX;nr,nc=int((top-bottom)/DX),int((right-left)/DX);tf=from_origin(left,top,DX,DX);ftf=from_origin(left,top,FINE,FINE);fshape=(nr*4,nc*4)
 ward=rasterize([(mapping(g),i+1) for i,g in enumerate(wards)],out_shape=(nr,nc),transform=tf).astype('uint8');dm=rasterize([(mapping(domain),1)],out_shape=(nr,nc),transform=tf).astype(bool);z=np.full((nr,nc),np.nan);src=np.zeros((nr,nc),dtype='uint8');seam=[];audit=[];building=np.zeros(fshape,dtype='uint8');road=np.zeros(fshape,dtype='uint8')
 class Terrain:
  def __init__(self,wi):self.wi=wi;self.capture=False;self.text=[];self.batch=[];self.total=0;self.hits=0;self.large=0
  def start(self,tag,attrs):
   if tag.endswith('}posList'):self.capture=True;self.text=[]
  def data(self,d):
   if self.capture:self.text.append(d)
  def end(self,tag):
   if tag.endswith('}posList'):
    self.capture=False;v=np.fromstring(''.join(self.text),sep=' ')
    if len(v)==12:self.batch.append(v.reshape(4,3)[:3]);self.total+=1
    if len(self.batch)>=20000:self.flush()
  def close(self):self.flush()
  def flush(self):
   if not self.batch:return
   a=np.array(self.batch);self.batch=[];x,y=TOXY.transform(a[:,:,1],a[:,:,0]);v=a[:,:,2]
   # Find cell centers inside each original triangle using barycentric coordinates.
   c0=np.ceil((x.min(1)-left)/DX-.5).astype(int);c1=np.floor((x.max(1)-left)/DX-.5).astype(int);r0=np.ceil((top-y.max(1))/DX-.5).astype(int);r1=np.floor((top-y.min(1))/DX-.5).astype(int)
   keep=(c1>=c0)&(r1>=r0)&(c1>=0)&(c0<nc)&(r1>=0)&(r0<nr)
   ids=np.where(keep)[0];self.large+=int(np.sum((c1[ids]>c0[ids])|(r1[ids]>r0[ids])))
   for dc in range(int(np.max(c1[ids]-c0[ids],initial=0))+1):
    for dr in range(int(np.max(r1[ids]-r0[ids],initial=0))+1):
     cs=c0[ids]+dc;rs=r0[ids]+dr;ok=(cs<=c1[ids])&(rs<=r1[ids])&(cs>=0)&(cs<nc)&(rs>=0)&(rs<nr);ii=ids[ok];cc=cs[ok];rr=rs[ok]
     if not len(ii):continue
     xx=left+(cc+.5)*DX;yy=top-(rr+.5)*DX;x0=x[ii,0];x1=x[ii,1];x2=x[ii,2];y0=y[ii,0];y1=y[ii,1];y2=y[ii,2];den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);valid=np.abs(den)>1e-9;den=np.where(valid,den,1);b0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;b1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;b2=1-b0-b1;valid &= (b0>=-1e-7)&(b1>=-1e-7)&(b2>=-1e-7)&dm[rr,cc];rr=rr[valid];cc=cc[valid];ii=ii[valid];zz=b0[valid]*v[ii,0]+b1[valid]*v[ii,1]+b2[valid]*v[ii,2];self.hits+=len(zz)
     existing=np.isfinite(z[rr,cc]);diff=np.abs(z[rr[existing],cc[existing]]-zz[existing]);
     if len(diff):seam.extend(diff[diff>.01].tolist())
     prefer=~existing|((ward[rr,cc]==self.wi)&(src[rr,cc]!=self.wi));z[rr[prefer],cc[prefer]]=zz[prefer];src[rr[prefer],cc[prefer]]=self.wi
 class Vectors:
  def __init__(self,kind):self.kind=kind;self.active=None;self.capture=False;self.text=[];self.rings=[];self.polys={};self.batch=[];self.members=0;self.used=0;self.foot=0;self.roof=0;self.unresolved=0
  def start(self,tag,attrs):
   name=tag.rsplit('}',1)[-1]
   if name=='cityObjectMember':self.polys={}
   if name in (['lod0FootPrint','lod0RoofEdge'] if self.kind=='bldg' else ['lod1MultiSurface']):self.active=name
   if self.active and name=='Polygon':self.rings=[]
   if self.active and name=='posList':self.capture=True;self.text=[]
   if self.active and any(k.endswith('}href') for k in attrs):self.unresolved+=1
  def data(self,d):
   if self.capture:self.text.append(d)
  def end(self,tag):
   name=tag.rsplit('}',1)[-1]
   if self.active and name=='posList':
    self.capture=False;a=np.fromstring(''.join(self.text),sep=' ')
    if len(a)%3==0 and len(a)>=12:
     a=a.reshape(-1,3);x,y=TOXY.transform(a[:,1],a[:,0]);self.rings.append(np.c_[x,y].tolist())
   if self.active and name=='Polygon' and self.rings:self.polys.setdefault(self.active,[]).append({'type':'Polygon','coordinates':self.rings});self.rings=[]
   if name==self.active:self.active=None
   if name=='cityObjectMember':
    self.members+=1;key='lod0FootPrint' if self.polys.get('lod0FootPrint') else 'lod0RoofEdge' if self.kind=='bldg' else 'lod1MultiSurface';p=self.polys.get(key,[])
    if p:self.used+=1;self.foot+=key=='lod0FootPrint';self.roof+=key=='lod0RoofEdge';self.batch.extend(p)
    if len(self.batch)>=500:self.flush()
  def close(self):self.flush()
  def flush(self):
   if self.batch:
    a=rasterize([(g,1) for g in self.batch],out_shape=fshape,transform=ftf,dtype='uint8');target=building if self.kind=='bldg' else road;np.maximum(target,a,out=target);self.batch=[]
 for wi,p in enumerate(sorted((ROOT/'data/raw/plateau/2025').rglob('*citygml*.zip')),1):
  with zipfile.ZipFile(p) as archive:
   for kind in ['dem','bldg','tran']:
    entries=[i for i in archive.infolist() if i.filename.startswith('udx/'+kind+'/') and i.filename.endswith('.gml')]
    for num,i in enumerate(entries,1):
     target=Terrain(wi) if kind=='dem' else Vectors(kind);parser=ET.XMLParser(target=target)
     with archive.open(i) as f:
      while b:=f.read(1024*1024):parser.feed(b)
     parser.close();rec={'archive':str(p.relative_to(ROOT)),'member':i.filename,'bytes':i.file_size,'crc32':f'{i.CRC:08x}','kind':kind,'ward_id':wi}
     rec.update({'triangles':target.total,'grid_center_hits':target.hits,'large_triangle_candidates':target.large} if kind=='dem' else {'members':target.members,'used':target.used,'footprint':target.foot,'roof_edge_proxy':target.roof,'unresolved_xlinks':target.unresolved});audit.append(rec)
     if kind=='dem' or num%5==0 or num==len(entries):print(p.parent.name,kind,num,'/',len(entries),rec.get('grid_center_hits',rec.get('used')),flush=True)
  # Cache each finished ward to retain progress if interrupted.
  np.savez_compressed(PRO/'input_progress.npz',z=z,src=src,building=building,road=road,ward=ward,dm=dm)
 frac=building.reshape(nr,4,nc,4).mean(axis=(1,3)).astype('float32');roads=road.reshape(nr,4,nc,4).mean(axis=(1,3)).astype('float32')
 wtr=json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features'];water_shapes=[];selected=[];catalog=[]
 names={'神田川','日本橋川','隅田川','亀島川','古川','汐留川','築地川','月島川'}
 for i,f in enumerate(wtr,1):
  g=transform(TOXY.transform,shape(f['geometry']));water_shapes.append((mapping(g),i));eligible=f['properties']['name'] in names or '1000' in f['properties']['class_codes'];catalog.append({'id':i,'water_id':f['properties']['water_id'],'name':f['properties']['name'],'class_codes':f['properties']['class_codes'],'candidate_exchange':eligible,'source_water_Z_used':False});
  if eligible:selected.append(i)
 water=rasterize(water_shapes,out_shape=(nr,nc),transform=tf).astype('uint16');candidate=np.isin(water,selected);land=dm&np.isfinite(z)&(water==0)&(frac<.9);outer=land&binary_dilation((~dm)|(~np.isfinite(z)&(water==0)),iterations=1);edge_distance=distance_transform_edt(~outer,sampling=DX);edge_uncertain=land&(edge_distance<500)
 # All water interiors are excluded. Closed moats/ponds are not assumed drains.
 stages=[]
 for i in selected:
  bank=land&binary_dilation(water==i);heights=z[bank];stages.append({'id':i,'name':catalog[i-1]['name'],'bank_samples':int(len(heights)),'synthetic_stage_low_m':float(np.quantile(heights,.1)-.5) if len(heights) else None,'definition':'Artificial stage = 10th percentile of adjacent 20m land elevations minus 0.5m; high scenario adds 1m. Not observed stage, riverbed or surveyed bank crest.'})
 info={'dx_m':DX,'shape':[nr,nc],'transform':list(tf)[:6],'requested_buffer_m':1000,'requested_buffer_is_complete_catchment':False,'terrain_total_coverage_km2':float(np.sum(dm&np.isfinite(z))*DX*DX/1e6),'land_km2':float(land.sum()*DX*DX/1e6),'terrain_cells_missing_by_ward':{str(i):int(np.sum((ward==i)&~np.isfinite(z))) for i in [1,2,3]},'ward_cells':{str(i):int(np.sum(ward==i)) for i in [1,2,3]},'building_fraction_native_raster_m':FINE,'building_proxy':'LOD0 FootPrint preferred; LOD0 RoofEdge fallback; sampled at5m then averaged to20m. No roof height added to bare terrain.','terrain_method':'Original TIN barycentric elevation at20m cell centers; source ward takes precedence inside each ward. No DEM blend, fill or inferred height for missing cells.','terrain_height_reference':'PLATEAU declared JGD2011 / T.P.; source survey epoch/height-result version unresolved. No 0.50m DEM merged.','seam_comparisons_over_0_01m':len(seam),'seam_abs_difference_p95_m':float(np.quantile(seam,.95)) if seam else None,'unreliable_edge_zone_m':500,'water_interior_excluded':True,'upstream_hydrographs_available':False,'synthetic_water_stages':stages,'water_catalog':catalog,'sources':audit}
 (OUT/'input_audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(PRO/'surface_inputs.npz',z=z,ward=ward,land=land,building_fraction=frac,road_fraction=roads,water=water,candidate_water=candidate,edge_uncertain=edge_uncertain,terrain_source=src,domain=dm)
 write(building,'building_roof_proxy_5m',ftf,0);write(road,'road_mask_5m',ftf,0)
 for a,n,no in [(z.astype('float32'),'terrain_20m',np.nan),(frac,'building_fraction_20m',None),(roads,'road_fraction_20m',None),(ward,'ward_id',0),(water,'water_id',0),(land.astype('uint8'),'active_land',0),(edge_uncertain.astype('uint8'),'edge_uncertain',0)]:write(a,n,tf,no)
 print('INPUTS', {k:info[k] for k in ['shape','land_km2','terrain_cells_missing_by_ward','seam_abs_difference_p95_m']},flush=True)
if __name__=='__main__':main()
