"""Prepare original CityGML DEM TIN as a terrain-gap fallback.

No roof elevations, depression fill, extrapolation or datum offset is applied.
The 5m preview is not the native0.50m DEM or a hydraulic simulation grid.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import csv,json,math,hashlib,zipfile,xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.warp import reproject,Resampling
from shapely.geometry import shape,mapping
from shapely.ops import transform,unary_union
from pyproj import Transformer

ROOT=Path(__file__).resolve().parents[2]
PRO=ROOT/'data/processed/three_ward_terrain_fallback'
OUT=ROOT/'data/exp/data-briefing/remote_sensing_fallback'
XY=Transformer.from_crs(6668,6677,always_xy=True)
DX=5.0
def write(name,a,tf,nodata):
 with rasterio.open(PRO/name,'w',driver='GTiff',width=a.shape[1],height=a.shape[0],count=1,dtype=a.dtype,crs='EPSG:6677',transform=tf,nodata=nodata,compress='deflate',tiled=True) as d:d.write(a,1)
def main():
 PRO.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 domain=unary_union([transform(XY.transform,shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/three_ward_dem_domain/native_model_domain_candidate.geojson').read_text())['features']]);l,b,r,t=domain.bounds;l=math.floor(l/DX)*DX;b=math.floor(b/DX)*DX;r=math.ceil(r/DX)*DX;t=math.ceil(t/DX)*DX;tf=from_origin(l,t,DX,DX);dims=(int((t-b)/DX),int((r-l)/DX))
 dm=rasterize([(mapping(domain),1)],out_shape=dims,transform=tf).astype(bool)
 wf=json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features'];ward=rasterize([(mapping(transform(XY.transform,shape(f['geometry']))),i+1) for i,f in enumerate(wf)],out_shape=dims,transform=tf).astype('uint8')
 waters=[transform(XY.transform,shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']];water=rasterize([(mapping(g),1) for g in waters],out_shape=dims,transform=tf).astype(bool);eligible=dm&~water
 z=np.full(dims,np.nan,dtype='float32');owner=np.zeros(dims,dtype='uint8');records=[]
 class Terrain:
  def __init__(self,wi):self.wi=wi;self.capture=False;self.text=[];self.batch=[];self.triangles=0;self.hits=0
  def start(self,tag,attrs):
   if tag.endswith('}posList'):self.capture=True;self.text=[]
  def data(self,s):
   if self.capture:self.text.append(s)
  def end(self,tag):
   if tag.endswith('}posList'):
    self.capture=False;a=np.fromstring(''.join(self.text),sep=' ')
    if len(a)==12:self.batch.append(a.reshape(4,3)[:3]);self.triangles+=1
    if len(self.batch)>=20000:self.flush()
  def close(self):self.flush()
  def flush(self):
   if not self.batch:return
   p=np.array(self.batch);self.batch=[];x,y=XY.transform(p[:,:,1],p[:,:,0]);h=p[:,:,2];c0=np.ceil((x.min(1)-l)/DX-.5).astype(int);c1=np.floor((x.max(1)-l)/DX-.5).astype(int);r0=np.ceil((t-y.max(1))/DX-.5).astype(int);r1=np.floor((t-y.min(1))/DX-.5).astype(int)
   ids=np.where((c1>=c0)&(r1>=r0)&(c1>=0)&(c0<dims[1])&(r1>=0)&(r0<dims[0]))[0]
   for dc in range(int(np.max(c1[ids]-c0[ids],initial=0))+1):
    for dr in range(int(np.max(r1[ids]-r0[ids],initial=0))+1):
     cc=c0[ids]+dc;rr=r0[ids]+dr;ok=(cc<=c1[ids])&(rr<=r1[ids])&(cc>=0)&(cc<dims[1])&(rr>=0)&(rr<dims[0]);ii=ids[ok];cc=cc[ok];rr=rr[ok];ok=eligible[rr,cc];ii=ii[ok];cc=cc[ok];rr=rr[ok]
     if not len(ii):continue
     xx=l+(cc+.5)*DX;yy=t-(rr+.5)*DX;x0,x1,x2=x[ii].T;y0,y1,y2=y[ii].T;den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);ok=abs(den)>1e-9;den=np.where(ok,den,1);b0=((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/den;b1=((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/den;b2=1-b0-b1;ok&=(b0>=-1e-7)&(b1>=-1e-7)&(b2>=-1e-7);ii=ii[ok];cc=cc[ok];rr=rr[ok];height=b0[ok]*h[ii,0]+b1[ok]*h[ii,1]+b2[ok]*h[ii,2];self.hits+=len(height);prefer=~np.isfinite(z[rr,cc])|((ward[rr,cc]==self.wi)&(owner[rr,cc]!=self.wi));z[rr[prefer],cc[prefer]]=height[prefer];owner[rr[prefer],cc[prefer]]=self.wi
 for wi,p in enumerate(sorted((ROOT/'data/raw/plateau/2025').rglob('*citygml*.zip')),1):
  with zipfile.ZipFile(p) as archive:
   for info in [v for v in archive.infolist() if v.filename.startswith('udx/dem/') and v.filename.endswith('.gml')]:
    target=Terrain(wi);parser=ET.XMLParser(target=target);hh=hashlib.sha256()
    with archive.open(info) as f:
     while chunk:=f.read(1024*1024):hh.update(chunk);parser.feed(chunk)
    parser.close();records.append({'archive':str(p.relative_to(ROOT)),'member':info.filename,'crc32':f'{info.CRC:08x}','sha256':hh.hexdigest(),'triangles':target.triangles,'grid_hits':target.hits,'ward_id':wi});print('PLATEAU ground TIN',p.parent.name,info.filename,target.triangles,flush=True)
 write('plateau_ground_tin_5m.tif',z,tf,np.nan);write('plateau_ground_owner_5m.tif',owner,tf,0)
 primary=np.full(dims,np.nan,dtype='float32')
 with rasterio.open(ROOT/'data/processed/three_ward_dem_domain/native_dem_0_50m.vrt') as d:reproject(rasterio.band(d,1),primary,src_transform=d.transform,src_crs='EPSG:6677',src_nodata=-32767,dst_transform=tf,dst_crs='EPSG:6677',dst_nodata=np.nan,resampling=Resampling.nearest,warp_mem_limit=128)
 primary[~eligible]=np.nan;use_primary=eligible&np.isfinite(primary);use_fallback=eligible&~use_primary&np.isfinite(z);candidate=primary.copy();candidate[use_fallback]=z[use_fallback];source=np.zeros(dims,dtype='uint8');source[use_primary]=1;source[use_fallback]=2
 write('terrain_preview_5m.tif',candidate,tf,np.nan);write('terrain_preview_source_5m.tif',source,tf,0)
 rows=[]
 for wi in [1,2,3]:
  region=(ward==wi)&eligible;overlap=region&np.isfinite(primary)&np.isfinite(z);diff=primary[overlap].astype(float)-z[overlap];n=int(region.sum());rows.append({'ward_id':wi,'eligible_5m_cells':n,'official_0_50m_sample_cells':int((region&use_primary).sum()),'plateau_gap_fallback_cells':int((region&use_fallback).sum()),'remaining_NoData_cells':int((region&~np.isfinite(candidate)).sum()),'preview_coverage_fraction':float(np.isfinite(candidate[region]).mean()),'overlap_samples':len(diff),'overlap_median_native_minus_plateau_m':float(np.median(diff)) if len(diff) else None,'overlap_p95_abs_difference_m':float(np.quantile(abs(diff),.95)) if len(diff) else None,'overlap_is_independent_accuracy_validation':False})
 with (OUT/'terrain_fallback_coverage.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 registry={'role':'candidate multi-source terrain; observation accuracy not certified','source_priority':[{'id':1,'path':'data/processed/three_ward_dem_domain/native_dem_0_50m.vrt','native_resolution_m':.5},{'id':2,'path':'data/processed/three_ward_terrain_fallback/plateau_ground_tin_5m.tif','native_geometry':'original PLATEAU udx/dem TIN, approximately5m sampling','raster_sample_m':5,'use':'only primary NoData land gaps'}],'preview_path':'data/processed/three_ward_terrain_fallback/terrain_preview_5m.tif','preview_is_formal_subgrid':False,'roofs_used_as_ground':False,'offset_applied_m':0,'depression_filling':False,'extrapolation':False,'source_datum':'sources declared JGD2011/T.P.; height-result epochs unresolved','formal_builder_policy':'Use two source layers in priority order; preserve fine DEM where available and track coarser fallback. Do not promote5m fallback to0.50m accuracy.','water_policy':'known PLATEAU wtr excluded from5m land preview; model water boundaries need separate treatment','coverage':rows,'original_TIN_members':records}
 (PRO/'elevation_sources.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
 checks={'finite_primary_values_preserved_exactly':bool(np.array_equal(candidate[use_primary],primary[use_primary])),'fallback_only_primary_missing':bool(not (use_fallback&use_primary).any()),'known_water_excluded_from_preview':bool(not np.isfinite(candidate[water]).any()),'fallback_matches_original_TIN_sample':bool(np.array_equal(candidate[use_fallback],z[use_fallback])),'native_resolution_retained_as_separate_source':True,'preview_sample_m':5,'filled_preview_area_km2':float(use_fallback.sum())*25/1e6,'residual_preview_missing_area_km2':float((eligible&~np.isfinite(candidate)).sum())*25/1e6,'datum_epoch_alignment_certified':False}
 assert all(checks[k] for k in ['finite_primary_values_preserved_exactly','fallback_only_primary_missing','known_water_excluded_from_preview','fallback_matches_original_TIN_sample'])
 (OUT/'terrain_fallback_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps({'coverage':rows,'checks':checks},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
