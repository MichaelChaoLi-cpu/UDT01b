"""Screen surface contributing areas, select official Tokyo0.50m DEM sheets.

Regional GSI display elevations are only a domain-screening aid; native Tokyo
DEM source sheets remain unmodified. Filled heights are never a model DEM.
"""

# Resolve original raw names to fixed Moga members; no local raw copies.
import sys as _moga_sys
from pathlib import Path as _MogaPath
_moga_sys.path.insert(0, str(_MogaPath(__file__).resolve().parents[1]))
from moga_raw_inputs import raw_path as moga_raw, source_reference as moga_source, project_path as moga_project_path
import os
for key in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(key,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import deque
import argparse,csv,hashlib,heapq,io,json,math,re,time,zipfile
from datetime import datetime,timezone
import requests
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject,Resampling
from rasterio.features import rasterize,shapes
from pyproj import Transformer
from shapely.geometry import shape,mapping,box,Polygon
from shapely.ops import transform,unary_union
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
RAW=moga_raw(ROOT, 'tokyo_lidar_dem/2024/three_ward_batch')
PROC=ROOT/'data/processed/three_ward_dem_domain'
OUT=ROOT/'data/exp/data-briefing/three_ward_dem_domain'
BASE='https://gic-tokyo.s3.ap-northeast-1.amazonaws.com/2024/dig/'
TOXY=Transformer.from_crs(6668,6677,always_xy=True)
TOLL=Transformer.from_crs(6677,6668,always_xy=True)
WEBXY=Transformer.from_crs(3857,6677,always_xy=True)
NEIGHBORS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fetch(url,p,missing_ok=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():return p
    for attempt in range(5):
        try:
            r=requests.get(url,timeout=(15,90),stream=True)
            if missing_ok and r.status_code==404:return None
            r.raise_for_status()
            part=p.with_suffix(p.suffix+'.part')
            with part.open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
            length=r.headers.get('Content-Length')
            if length and part.stat().st_size!=int(length):raise ValueError('HTTP length mismatch')
            part.replace(p);return p
        except Exception:
            if attempt==4:raise
            time.sleep(min(2**attempt,8))
def fc(path,records):dump(path,{'type':'FeatureCollection','features':[{'type':'Feature','geometry':mapping(transform(TOLL.transform,g)),'properties':p} for g,p in records]})
def write_tif(name,a,tf,nodata):
    with rasterio.open(PROC/name,'w',driver='GTiff',width=a.shape[1],height=a.shape[0],count=1,dtype=a.dtype,transform=tf,crs='EPSG:6677',nodata=nodata,compress='deflate',tiled=True) as d:d.write(a,1)
def geometry(mask,tf):return unary_union([shape(g) for g,v in shapes(mask.astype('uint8'),mask=mask,transform=tf) if v==1])
def wards():
    return [(transform(TOXY.transform,shape(f['geometry'])),f['properties']) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').read_text())['features']]
def tile(lon,lat,z):
    n=2**z;return int((lon+180)/360*n),int((1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)
def tile_ranges(bounds,z):
    poly=transform(TOLL.transform,box(*bounds));l,b,r,t=poly.bounds
    x0,y0=tile(l,t,z);x1,y1=tile(r,b,z)
    return x0,y0,x1,y1
def decode_png(p):
    rgb=np.asarray(Image.open(p).convert('RGB'),dtype='int32');v=rgb[:,:,0]*65536+rgb[:,:,1]*256+rgb[:,:,2]
    a=np.where(v>2**23,v-2**24,v).astype('float32')*.01;a[v==2**23]=np.nan;return a


def region_dem(bounds):
    z=14;x0,y0,x1,y1=tile_ranges(bounds,z);ny,nx=y1-y0+1,x1-x0+1
    mosaic=np.full((ny*256,nx*256),np.nan,dtype='float32');source=np.zeros(mosaic.shape,dtype='uint8');sources=[]
    def job(x,y):
        a=np.full((256,256),np.nan,dtype='float32');src=np.zeros((256,256),dtype='uint8');entries=[]
        for sid,kind in [(1,'dem5a_png'),(2,'dem_png')]:
            url=f'https://cyberjapandata.gsi.go.jp/xyz/{kind}/{z}/{x}/{y}.png';p=RAW/'regional_gsi'/f'{kind}_{z}_{x}_{y}.png'
            p=fetch(url,p,missing_ok=True)
            if p:
                b=decode_png(p);use=~np.isfinite(a)&np.isfinite(b);a[use]=b[use];src[use]=sid
                entries.append({'url':url,'path':moga_source(p, ROOT),'sha256':digest(p),'bytes':p.stat().st_size})
            if np.isfinite(a).all():break
        return x,y,a,src,entries
    jobs=[(x,y) for y in range(y0,y1+1) for x in range(x0,x1+1)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count,f in enumerate(as_completed([pool.submit(job,*j) for j in jobs]),1):
            x,y,a,s,e=f.result();rr=slice((y-y0)*256,(y-y0+1)*256);cc=slice((x-x0)*256,(x-x0+1)*256);mosaic[rr,cc]=a;source[rr,cc]=s;sources+=e
            if count%25==0 or count==len(jobs):print('regional tiles',count,'/',len(jobs),flush=True)
    half=20037508.342789244;span=2*half/2**z;webtf=from_origin(-half+x0*span,half-y0*span,span/256,span/256)
    l,b,r,t=bounds;tf=from_origin(l,t,20,20);dims=(int((t-b)/20),int((r-l)/20));a=np.full(dims,np.nan,dtype='float32');s=np.zeros(dims,dtype='uint8')
    reproject(mosaic,a,src_transform=webtf,src_crs='EPSG:3857',src_nodata=np.nan,dst_transform=tf,dst_crs='EPSG:6677',dst_nodata=np.nan,resampling=Resampling.bilinear)
    reproject(source,s,src_transform=webtf,src_crs='EPSG:3857',dst_transform=tf,dst_crs='EPSG:6677',resampling=Resampling.nearest)
    dump(OUT/'regional_source_manifest.json',{'resources':sources,'role':'screening_only','note':'GSI PNG elevations are smoothed display data with source epochs unresolved; not merged into native0.50m model DEM.'})
    write_tif('regional_screening_20m.tif',a,tf,np.nan);write_tif('regional_source_id.tif',s,tf,0)
    return a,tf


def river_masks(dims,tf):
    G='{http://www.opengis.net/gml/3.2}';K='{http://nlftp.mlit.go.jp/ksj/schemas/ksj-app}';X='{http://www.w3.org/1999/xlink}'
    with zipfile.ZipFile(moga_raw(ROOT, 'ksj/W05-08_13_GML.zip')) as z:root=ET.fromstring(z.read('W05-08_13-g.xml'))
    ids={e.get(G+'id'):e for e in root if e.get(G+'id')};old=Transformer.from_crs(4612,6677,always_xy=True);lines=[]
    from shapely.geometry import LineString
    for e in root.findall(K+'Stream'):
        c=ids[e.find(K+'location').get(X+'href').lstrip('#')];v=np.fromstring(c.find('.//'+G+'posList').text,sep=' ').reshape(-1,2);x,y=old.transform(v[:,1],v[:,0]);lines.append(LineString(np.c_[x,y]).buffer(10))
    wm=[transform(TOXY.transform,shape(f['geometry'])) for f in json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']]
    return rasterize([(mapping(g),1) for g in lines+wm],out_shape=dims,transform=tf,all_touched=True).astype(bool)


def receivers(z,valid,sinks,fill):
    nr,nc=z.shape;total=z.size;parent=np.full(total,-1,dtype='int32');a=z.ravel().copy();vf=valid.ravel();sf=sinks.ravel();order=np.zeros(total,dtype='int32')
    if fill:
        edge=valid&binary_dilation(~valid);edge[0,:]|=valid[0,:];edge[-1,:]|=valid[-1,:];edge[:,0]|=valid[:,0];edge[:,-1]|=valid[:,-1];seed=edge|sinks
        seen=(~valid).ravel().copy();heap=[]
        for k in np.flatnonzero(seed&valid):seen[k]=True;heap.append((float(a[k]),int(k)))
        heapq.heapify(heap);rank=0
        while heap:
            level,i=heapq.heappop(heap);order[i]=rank;rank+=1;r,c=divmod(i,nc)
            for dr,dc in NEIGHBORS:
                rr=r+dr;cc=c+dc
                if 0<=rr<nr and 0<=cc<nc:
                    j=rr*nc+cc
                    if not seen[j]:
                        seen[j]=True;a[j]=max(a[j],level);parent[j]=i;heapq.heappush(heap,(float(a[j]),j))
        print('priority flood visited',rank,flush=True)
    rec=np.full(z.shape,-1,dtype='int32');best=np.zeros(z.shape,dtype='float32');filled=a.reshape(z.shape);ids=np.arange(total,dtype='int32').reshape(z.shape)
    for dr,dc in NEIGHBORS:
        rr=slice(max(0,-dr),min(nr,nr-dr));cc=slice(max(0,-dc),min(nc,nc-dc));tr=slice(max(0,dr),min(nr,nr+dr));tc=slice(max(0,dc),min(nc,nc+dc))
        slope=(filled[rr,cc]-filled[tr,tc])/math.hypot(dr,dc);use=valid[rr,cc]&valid[tr,tc]&(slope>best[rr,cc]);rec[rr,cc][use]=ids[tr,tc][use];best[rr,cc][use]=slope[use]
    if fill:
        flat=valid.ravel()&(rec.ravel()<0)&(parent>=0);rec.ravel()[flat]=parent[flat]
    rec[sinks|~valid]=-1
    return rec,filled


def upstream(rec,target):
    rf=rec.ravel();kids=np.flatnonzero(rf>=0).astype('int32');graph=csr_matrix((np.ones(len(kids),dtype='uint8'),(rf[kids],kids)),shape=(rf.size,rf.size));reach=target.ravel().copy();q=deque(map(int,np.flatnonzero(reach)))
    while q:
        i=q.popleft()
        for j in graph.indices[graph.indptr[i]:graph.indptr[i+1]]:
            if not reach[j]:reach[j]=True;q.append(int(j))
    return reach.reshape(target.shape)


def screen():
    ws=wards();union=unary_union([g for g,p in ws]);l,b,r,t=union.bounds
    bounds=[math.floor((l-10000)/20)*20,math.floor((b-8000)/20)*20,math.ceil((r+5000)/20)*20,math.ceil((t+8000)/20)*20]
    dem=PROC/'regional_screening_20m.tif'
    if dem.exists():
        with rasterio.open(dem) as d:z=d.read(1);tf=d.transform
    else:z,tf=region_dem(bounds)
    valid=np.isfinite(z);ward=rasterize([(mapping(g),i+1) for i,(g,p) in enumerate(ws)],out_shape=z.shape,transform=tf).astype('uint8');sinks=river_masks(z.shape,tf);target=(ward>0)&valid&~sinks
    rec0,_=receivers(z,valid,sinks,False);native=upstream(rec0,target)
    rec1,filled=receivers(z,valid,sinks,True);potential=upstream(rec1,target)
    outside=potential&(ward==0);uncertain=outside&(~native|(filled-z>.5));guard=binary_dilation(potential|((ward>0)&valid),iterations=5)&valid
    # This guard is an acquisition margin, explicitly not a contributing basin.
    wardgeom=union;contribgeom=geometry(outside,tf);acq=unary_union([wardgeom,geometry(guard,tf)])
    fc(PROC/'acquisition_domain.geojson',[(acq,{'role':'provisional_acquisition_extent','not_final_hydraulic_boundary':True,'guard_m':100})])
    fc(PROC/'surface_contributing_candidates.geojson',[(contribgeom,{'role':'potential_direct_surface_contribution','algorithm':'20m GSI D8 with priority-flood screening and W05/wtr channel stops','not_certified_catchment':True})])
    for name,a in [('ward_id.tif',ward),('direct_native_mask.tif',native.astype('uint8')),('potential_contribution_mask.tif',potential.astype('uint8')),('acquisition_mask.tif',guard.astype('uint8')),('scope_uncertainty.tif',uncertain.astype('uint8')),('channel_stop_mask.tif',sinks.astype('uint8'))]:write_tif(name,a,tf,0)
    write_tif('screening_fill_depth.tif',np.where(valid,filled-z,np.nan).astype('float32'),tf,np.nan)
    touch=int(outside[0,:].sum()+outside[-1,:].sum()+outside[:,0].sum()+outside[:,-1].sum())
    stats={'scope_status':'provisional_acquisition_domain','horizontal_crs':'EPSG:6677','screening_cell_m':20,'search_envelope_m':list(dem_bounds(z,tf)),'search_envelope_is_catchment':False,'ward_polygon_area_km2':union.area/1e6,'native_d8_external_km2':float((native&(ward==0)).sum())*.0004,'filled_potential_external_km2':float(outside.sum())*.0004,'external_uncertain_km2':float(uncertain.sum())*.0004,'acquisition_area_km2':acq.area/1e6,'external_reach_at_search_edge_cells':touch,'channel_stops_are_verified_banks':False,'sources':['GSI DEM5A/DEM10B display tiles','KSJ W05 2008 centerlines','PLATEAU wtr and ward boundaries'],'limits':['D8 is a static terrain screening, not rain-driven inundation.','Priority filling is only used to bound potential contributing areas; no source/model DEM modified.','W05 10m corridor/raster all_touched is an artificial channel stop, not observed bank width or inlet. Culverts, banks, bridges and closed channels need domain refinement.','100m guard is data acquisition margin, not a hydrological basin.','Recheck divides and cross-boundary exchange using native0.50m DEM and solver; high-water multi-direction exchange can exceed D8 catchments.','Regional GSI source height epochs unresolved; never blended into Tokyo DEM.']}
    dump(OUT/'domain_screening.json',stats);print(json.dumps(stats,ensure_ascii=False),flush=True)
    if touch:raise RuntimeError('Candidate reaches search edge; expand search before acquisition')
def dem_bounds(z,tf):return [tf.c,tf.f+tf.e*z.shape[0],tf.c+tf.a*z.shape[1],tf.f]


# Minimal protobuf/MVT decoder following vector-tile-spec2.1, without installing packages.
def varint(b,p):
    n=0;s=0
    while True:
        v=b[p];p+=1;n|=(v&127)<<s
        if v<128:return n,p
        s+=7
        if s>63:raise ValueError('invalid varint')
def fields(b):
    p=0
    while p<len(b):
        tag,p=varint(b,p);n,w=tag>>3,tag&7
        if w==0:v,p=varint(b,p)
        elif w==2:l,p=varint(b,p);v=b[p:p+l];p+=l
        elif w in [1,5]:l=8 if w==1 else 4;v=b[p:p+l];p+=l
        else:raise ValueError('unsupported protobuf wire type')
        yield n,w,v
def packed(b):
    vals=[];p=0
    while p<len(b):v,p=varint(b,p);vals.append(v)
    return vals
def index_features(data,z,tx,ty):
    records=[];span=40075016.68557849/2**z
    for fn,w,layer in fields(data):
        if fn!=3:continue
        fs=[];keys=[];values=[];extent=4096
        for n,w,v in fields(layer):
            if n==2:fs.append(v)
            elif n==3:keys.append(v.decode())
            elif n==4:
                value=next(iter(fields(v)));values.append(value[2].decode() if value[0]==1 else value[2])
            elif n==5:extent=v
        for f in fs:
            attrs={};geom=None;typ=None
            for n,w,v in fields(f):
                if n==2:tags=packed(v);attrs={keys[tags[i]]:values[tags[i+1]] for i in range(0,len(tags),2)}
                elif n==3:typ=v
                elif n==4:geom=packed(v)
            urls=[v for v in attrs.values() if isinstance(v,str) and v.startswith('https://') and v.endswith('.zip')]
            if not urls:continue
            if typ!=3:raise ValueError('Official index geometry is not polygon')
            rings=[];ring=[];px=py=0;i=0
            while i<len(geom):
                command=geom[i];i+=1;cid=command&7;count=command>>3
                if cid in [1,2]:
                    if cid==1 and ring:rings.append(ring);ring=[]
                    for _ in range(count):
                        dx,dy=geom[i:i+2];i+=2;px+=(dx>>1)^-(dx&1);py+=(dy>>1)^-(dy&1)
                        wx=-20037508.342789244+(tx+px/extent)*span;wy=20037508.342789244-(ty+py/extent)*span;ring.append(WEBXY.transform(wx,wy))
                elif cid==7:
                    if ring:ring.append(ring[0]);rings.append(ring);ring=[]
                else:raise ValueError('unsupported MVT geometry command')
            if ring:rings.append(ring)
            polygons=[Polygon(r) for r in rings if len(r)>=4]
            for u in urls:records.append((unary_union(polygons),u))
    return records


def select():
    domain=unary_union([transform(TOXY.transform,shape(f['geometry'])) for f in json.loads((PROC/'acquisition_domain.geojson').read_text())['features']]);z=14;x0,y0,x1,y1=tile_ranges(domain.bounds,z);selected={};entries=[];selected_geometries={}
    def job(tx,ty):
        u=f'{BASE}Vectortile/23ku/grd50/{z}/{tx}/{ty}.pbf';p=fetch(u,RAW/'index'/f'grd50_{z}_{tx}_{ty}.pbf',True)
        return tx,ty,p,u
    jobs=[(x,y) for y in range(y0,y1+1) for x in range(x0,x1+1)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(job,*j) for j in jobs]):
            x,y,p,u=f.result()
            if not p:continue
            features=index_features(p.read_bytes(),z,x,y);entries.append({'url':u,'path':moga_source(p, ROOT),'sha256':digest(p),'feature_count':len(features)})
            for g,url in features:
                # z14 geometry quantization: pad20m for selection, verify actual TIFF bounds later.
                if g.intersects(domain.buffer(20)):
                    sheet=Path(url).stem
                    if sheet in selected and selected[sheet]['url']!=url:raise ValueError('Multiple URLs per sheet')
                    selected[sheet]={'sheet':sheet,'url':url,'index_source':u};selected_geometries[sheet]=unary_union([selected_geometries.get(sheet,Polygon()),g])
    dump(OUT/'index_selection.json',{'zoom':z,'index_resources':entries,'selection_geometry_pad_m':20,'selected':sorted(selected.values(),key=lambda x:x['sheet']),'tile_count':len(selected),'source':'official resource URL from PBF feature; no guessed download paths'})
    fc(PROC/'selected_sheet_index.geojson',[(g,{'sheet':name,'index_geometry_only':True,'not_native_raster_bounds':True}) for name,g in selected_geometries.items()]);print('Selected native DEM sheets',len(selected),flush=True)
    if not selected:raise ValueError('No native DEM sheets selected')


def download():
    selection=json.loads((OUT/'index_selection.json').read_text())['selected'];records=[];failures=[]
    existing={p.stem:p for sub in ['pilot_hibiya','hibiya_water_corridor/grd50'] for p in (moga_raw(ROOT, 'tokyo_lidar_dem/2024')/sub).glob('*.zip')}
    def job(row):
        name=row['sheet'];p=existing.get(name,RAW/'grd50'/f'{name}.zip');reuse=p.exists();fetch(row['url'],p)
        with zipfile.ZipFile(p) as archive:
            members=archive.infolist();tifs=[i for i in members if i.filename.endswith('.tif')]
            if len(tifs)!=1 or Path(tifs[0].filename).name!=tifs[0].filename:raise ValueError('Unexpected ZIP members')
            info=tifs[0];dest=p.parent/info.filename
            with archive.open(info) as f:
                blob=f.read() # validates TIFF CRC
            h=hashlib.sha256(blob).hexdigest()
            if dest.exists():
                if digest(dest)!=h:raise ValueError('Existing source TIFF hash differs')
            else:dest.write_bytes(blob)
            for i in members:
                if i.filename!=info.filename:archive.read(i) # validate auxiliary members too
        with rasterio.open(dest) as d:
            if d.res!=(.5,.5) or d.width>800 or d.height>600:raise ValueError('Unexpected native grid resolution or oversized sheet')
            a=d.read(1,masked=True);valid=int((~np.ma.getmaskarray(a)&np.isfinite(a.data)).sum());bound=list(d.bounds);tf=list(d.transform)[:6];crs=d.crs.to_wkt();dims=list(d.shape);nodata=d.nodata;minv=float(a.min()) if valid else None;maxv=float(a.max()) if valid else None
        return {**row,'zip_path':moga_source(p, ROOT),'tif_path':moga_source(dest, ROOT),'reused_existing':name in existing,'resumed_batch_cache':reuse and name not in existing,'zip_bytes':p.stat().st_size,'zip_sha256':digest(p),'tif_sha256':h,'zip_crc_verified':True,'bounds':bound,'transform':tf,'source_crs_wkt':crs,'resolution_m':.5,'shape':dims,'source_nodata':nodata,'valid_pixels':valid,'nodata_pixels':dims[0]*dims[1]-valid,'min_elevation':minv,'max_elevation':maxv,'ZIP_timestamp_is_survey_date':False}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(job,r):r for r in selection}
        for count,f in enumerate(as_completed(futures),1):
            try:records.append(f.result())
            except Exception as e:failures.append({'sheet':futures[f]['sheet'],'url':futures[f]['url'],'error':str(e)});print('DOWNLOAD FAILURE',futures[f]['sheet'],repr(e),flush=True)
            if count%25==0 or count==len(selection):
                dump(OUT/'acquisition_manifest.json',{'complete':count==len(selection) and not failures,'expected_sheets':len(selection),'verified_sheets':len(records),'failures':failures,'resources':records})
                print('native DEM verified',count,'/',len(selection),'failures',len(failures),flush=True)
    records.sort(key=lambda r:r['sheet']);fc(PROC/'native_dem_sheets.geojson',[(box(*r['bounds']),{'sheet':r['sheet'],'tif_path':r['tif_path'],'nodata_pixels':r['nodata_pixels']}) for r in records])
    l=min(r['bounds'][0] for r in records);b=min(r['bounds'][1] for r in records);right=max(r['bounds'][2] for r in records);t=max(r['bounds'][3] for r in records)
    vrt=ET.Element('VRTDataset',rasterXSize=str(round((right-l)/.5)),rasterYSize=str(round((t-b)/.5)));ET.SubElement(vrt,'SRS').text='EPSG:6677';ET.SubElement(vrt,'GeoTransform').text=f'{l},0.5,0,{t},0,-0.5';band=ET.SubElement(vrt,'VRTRasterBand',dataType='Float32',band='1');ET.SubElement(band,'NoDataValue').text='-32767'
    for row in records:
        src=ET.SubElement(band,'ComplexSource');ET.SubElement(src,'SourceFilename',relativeToVRT='1').text=os.path.relpath(moga_project_path(ROOT, row['tif_path']),PROC);ET.SubElement(src,'SourceBand').text='1';ET.SubElement(src,'SrcRect',xOff='0',yOff='0',xSize=str(row['shape'][1]),ySize=str(row['shape'][0]));ET.SubElement(src,'DstRect',xOff=str(round((row['bounds'][0]-l)/.5)),yOff=str(round((t-row['bounds'][3])/.5)),xSize=str(row['shape'][1]),ySize=str(row['shape'][0]));ET.SubElement(src,'NODATA').text=str(row['source_nodata'])
    ET.ElementTree(vrt).write(PROC/'native_dem_0_50m.vrt',encoding='utf-8',xml_declaration=True)
    with rasterio.open(PROC/'native_dem_0_50m.vrt') as d:assert d.res==(.5,.5);print('VRT',d.shape,d.bounds,flush=True)
    dump(OUT/'acquisition_manifest.json',{'complete':not failures,'expected_sheets':len(selection),'verified_sheets':len(records),'downloaded_new_sheets':sum(not r['reused_existing'] for r in records),'reused_sheets':sum(r['reused_existing'] for r in records),'total_archive_bytes':sum(r['zip_bytes'] for r in records),'failures':failures,'resources':records,'source_product':'Tokyo2024 0.50m grd50, native sources preserved','vertical_and_survey_epoch':'not yet certified across all sources; do not mix GSI heights into model DEM'})


if __name__=='__main__':
    for p in [RAW,PROC,OUT]:p.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','select','download']);args=ap.parse_args();{'screen':screen,'select':select,'download':download}[args.stage]()
