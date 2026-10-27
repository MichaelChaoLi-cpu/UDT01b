"""Native-grid screening of possible Hibiya dry-land routes to water banks.

Minimax terrain paths are screening hypotheses, not simulated runoff or outlets.
Sources and raw heights are preserved. Water-interior DEM values are excluded.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import csv,hashlib,heapq,json,math,re,struct,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date,timedelta
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize
from rasterio.transform import rowcol,xy
from scipy.ndimage import binary_dilation
from pyproj import Transformer
from shapely.geometry import Polygon,shape,mapping,box,LineString,Point
from shapely import make_valid,transform as stransform
from shapely.ops import unary_union,nearest_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/exp/data-briefing/hibiya_water_corridor'
PROCESSED=ROOT/'data/processed/hibiya_water_corridor'
TOXY=Transformer.from_crs(6668,6677,always_xy=True);TOLL=Transformer.from_crs(6677,6668,always_xy=True)
G='http://www.opengis.net/gml';B='http://www.opengis.net/citygml/building/2.0';T='http://www.opengis.net/citygml/transportation/2.0'
NS={'g':G,'b':B,'t':T}

def projected(g):return stransform(g,TOXY.transform,interleaved=False)
def geographic(g):return stransform(g,TOLL.transform,interleaved=False)
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def savegeo(path,features):path.write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False)+'\n')
def feature(g,props):return {'type':'Feature','properties':props,'geometry':mapping(geographic(g))}
def writecsv(path,rows):
    if not rows:return
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write_raster(path,a,tf,nodata=None):
    if a.ndim==2:a=a[None]
    with rasterio.open(path,'w',driver='GTiff',height=a.shape[1],width=a.shape[2],count=a.shape[0],dtype=a.dtype,crs='EPSG:6677',transform=tf,nodata=nodata,compress='deflate',tiled=True) as d:d.write(a)
def get_xyz(e):
    a=np.fromstring(e.text or '',sep=' ').reshape(-1,3)
    x,y=TOXY.transform(a[:,1],a[:,0]);return np.c_[x,y,a[:,2]]
def surface(e):
    polys=[]
    for p in e.iter(f'{{{G}}}Polygon'):
        rings=[]
        for ring in p.iter(f'{{{G}}}LinearRing'):
            pos=ring.find('g:posList',NS)
            if pos is not None:rings.append(get_xyz(pos)[:,:2])
        if rings:
            q=make_valid(Polygon(rings[0],rings[1:]))
            if q.area>0:polys.append(q)
    return unary_union(polys)

def local_vectors(domain):
    p=next((ROOT/'data/raw/plateau/2025/13101-chiyoda-ku').glob('*citygml*.zip'))
    buildings=[];roads=[];sources=[]
    with zipfile.ZipFile(p) as z:
        for kind,tag in [('bldg',f'{{{B}}}Building'),('tran',f'{{{T}}}Road')]:
            for i in z.infolist():
                if not i.filename.startswith('udx/'+kind+'/') or not i.filename.endswith('.gml'):continue
                with z.open(i) as f:prefix=f.read(4096).decode()
                corners=re.findall(r'<gml:(?:lowerCorner|upperCorner)>([^<]+)',prefix)
                if len(corners)!=2:raise ValueError('Source envelope missing')
                ll=[list(map(float,v.split())) for v in corners]
                if not projected(box(ll[0][1],ll[0][0],ll[1][1],ll[1][0])).intersects(domain):continue
                sources.append({'archive':str(p.relative_to(ROOT)),'member':i.filename,'bytes':i.file_size,'crc32':f'{i.CRC:08x}'})
                n=0
                with z.open(i) as f:
                    for event,e in ET.iterparse(f,events=['end']):
                        if e.tag!=tag:continue
                        if kind=='bldg':
                            s=e.find('b:lod0FootPrint',NS)
                            if s is None:s=e.find('b:lod0RoofEdge',NS)
                            if s is None:s=e.find('b:lod1Solid',NS)
                            if s is None:e.clear();continue
                            g=surface(s).intersection(domain)
                            if not g.is_empty and g.area>.1:buildings.append((g,e.get(f'{{{G}}}id')));n+=1
                        else:
                            s=e.find('t:lod1MultiSurface',NS)
                            if s is not None:
                                g=surface(s).intersection(domain)
                                if not g.is_empty and g.area>.1:roads.append((g,e.get(f'{{{G}}}id')));n+=1
                        e.clear()
                print('Vectors',i.filename,n,flush=True)
    return buildings,roads,sources

def las_points(path):
    with path.open('rb') as f:h=f.read(227)
    fmt=h[104];size=struct.unpack_from('<H',h,105)[0];count=struct.unpack_from('<I',h,107)[0];offset=struct.unpack_from('<I',h,96)[0]
    if h[:4]!=b'LASF' or fmt!=3 or size!=34 or path.stat().st_size!=offset+size*count:raise ValueError('Unexpected LAS format/length')
    scales=np.array(struct.unpack_from('<3d',h,131));origin=np.array(struct.unpack_from('<3d',h,155))
    dtype=np.dtype({'names':['x','y','z','class','flags'],'formats':['<i4','<i4','<i4','u1','u1'],'offsets':[0,4,8,15,14],'itemsize':size})
    a=np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=(count,))
    year=struct.unpack_from('<H',h,92)[0];day=struct.unpack_from('<H',h,90)[0]
    meta={'path':str(path.relative_to(ROOT)),'point_count':count,'version':list(h[24:26]),'point_format':fmt,'scales':list(scales),'offsets':list(origin),'file_creation_date':str(date(year,1,1)+timedelta(days=day-1)),'global_encoding':struct.unpack_from('<H',h,6)[0]}
    return a,scales,origin,meta

def minimax_paths(dem,blocked,start_mask,targets,tf):
    nrows,ncols=dem.shape;n=dem.size
    flat=dem.ravel();bad=blocked.ravel();best=np.full(n,np.inf);dist=np.full(n,np.inf);parent=np.full(n,-2,np.int32)
    heap=[]
    for s in np.flatnonzero(start_mask&~blocked):
        s=int(s);best[s]=float(flat[s]);dist[s]=0;parent[s]=-1;heapq.heappush(heap,(best[s],0.,s))
    target_flat={k:m.ravel() for k,m in targets.items()};found={};pops=0
    steps=[(-1,0,.5),(1,0,.5),(0,-1,.5),(0,1,.5),(-1,-1,math.sqrt(.5)),(-1,1,math.sqrt(.5)),(1,-1,math.sqrt(.5)),(1,1,math.sqrt(.5))]
    while heap and len(found)<len(targets):
        peak,length,i=heapq.heappop(heap)
        if peak!=best[i] or length!=dist[i]:continue
        pops+=1
        if pops%200000==0:print('Path search settled',pops,'bank targets',len(found),flush=True)
        for name,m in target_flat.items():
            if name not in found and m[i]:found[name]=(i,peak,length);print('Reached bank',name,peak,length,flush=True)
        r,c=divmod(i,ncols)
        for dr,dc,step in steps:
            rr=r+dr;cc=c+dc
            if rr<0 or rr>=nrows or cc<0 or cc>=ncols:continue
            j=rr*ncols+cc
            if bad[j]:continue
            if dr and dc and (bad[r*ncols+cc] or bad[rr*ncols+c]):continue
            h=max(peak,float(flat[j]));d=length+step
            if h<best[j] or (h==best[j] and d<dist[j]):
                best[j]=h;dist[j]=d;parent[j]=i;heapq.heappush(heap,(h,d,j))
    routes=[]
    for name,(i,h,d) in found.items():
        indices=[i]
        while parent[indices[-1]]>=0:indices.append(int(parent[indices[-1]]))
        indices=np.array(indices[::-1]);rr,cc=np.divmod(indices,ncols);xx,yy=xy(tf,rr,cc)
        xyz=np.c_[xx,yy,flat[indices]]
        routes.append({'target':name,'xyz':xyz,'indices':indices,'peak_source_dem_m':h,'length_m':d,'start_dem_m':float(xyz[0,2]),'screening_peak_above_start_m':float(h-xyz[0,2])})
    return routes,pops

def main():
    OUT.mkdir(parents=True,exist_ok=True);PROCESSED.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((OUT/'acquisition_manifest.json').read_text())
    if not manifest['acquisition_complete']:raise ValueError('Acquisition incomplete')
    ds=[rasterio.open(ROOT/s['dem_path']) for s in manifest['sheets']]
    dem,tf=merge(ds,nodata=-32767);dem=dem[0]
    for d in ds:d.close()
    write_raster(PROCESSED/'official_dem_0_50m_mosaic.tif',dem,tf,-32767)
    ortho_paths=[ROOT/next(p for p in e['extracted'] if p.endswith('.tif')) for e in manifest['resources'] if e.get('kind')=='ortho']
    ds=[rasterio.open(p) for p in ortho_paths];ortho,otf=merge(ds)
    for d in ds:d.close()
    write_raster(PROCESSED/'official_orthophoto_0_20m_mosaic.tif',ortho,otf)
    bounds=rasterio.transform.array_bounds(*dem.shape,tf);domain=box(*bounds)
    # All nearby source water footprints are excluded, including unnamed pools.
    allwater=json.loads((ROOT/'data/processed/plateau_water/central_three_wards_wtr.geojson').read_text())['features']
    waters=[(projected(shape(f['geometry'])),f['properties']) for f in allwater if projected(shape(f['geometry'])).intersects(domain)]
    targets={p['name']:g for g,p in waters if p['name'] in ['雲形池','心字池','日比谷濠']}
    watermask=rasterize([(g,1) for g,p in waters],out_shape=dem.shape,transform=tf,all_touched=True,dtype='uint8').astype(bool)
    buildings,roads,vector_sources=local_vectors(domain)
    savegeo(PROCESSED/'local_building_exclusion.geojson',[feature(g,{'source_id':i,'interpretation':'Conservative 2D roof/footprint exclusion; not surveyed wall breakline'}) for g,i in buildings])
    savegeo(PROCESSED/'local_plateau_roads.geojson',[feature(g,{'source_id':i,'interpretation':'LOD1 road footprint only'}) for g,i in roads])
    bmask=rasterize([(g.buffer(.5),1) for g,i in buildings],out_shape=dem.shape,transform=tf,all_touched=True,dtype='uint8').astype(bool)
    count=np.zeros(dem.size,np.uint32);sums=np.zeros(dem.size,np.float64);low=np.full(dem.size,np.inf,np.float32);high=np.full(dem.size,-np.inf,np.float32)
    las_metadata=[];las_paths=[ROOT/next(p for p in e['extracted'] if p.endswith('.las')) for e in manifest['resources'] if e.get('kind')=='lp']
    for path in las_paths:
        a,scale,offset,meta=las_points(path);las_metadata.append(meta)
        for begin in range(0,len(a),1000000):
            chunk=a[begin:begin+1000000];keep=((chunk['class']&31)==2)&((chunk['class']&128)==0)
            q=chunk[keep];x=q['x']*scale[0]+offset[0];y=q['y']*scale[1]+offset[1];z=q['z']*scale[2]+offset[2]
            rr,cc=rowcol(tf,x,y);rr=np.array(rr);cc=np.array(cc);valid=(rr>=0)&(cc>=0)&(rr<dem.shape[0])&(cc<dem.shape[1]);ids=rr[valid]*dem.shape[1]+cc[valid];z=z[valid]
            np.add.at(count,ids,1);np.add.at(sums,ids,z);np.minimum.at(low,ids,z);np.maximum.at(high,ids,z)
        print('Ground grid',path.name,meta['point_count'],flush=True)
    mean=np.divide(sums,count,out=np.full(dem.size,np.nan),where=count>0).reshape(dem.shape)
    counts=count.reshape(dem.shape);spread=(high-low).reshape(dem.shape);spread[counts==0]=np.nan
    write_raster(PROCESSED/'ground_class2_count_native_grid.tif',counts,tf)
    write_raster(PROCESSED/'ground_class2_mean_native_grid.tif',mean.astype(np.float32),tf,np.nan)
    write_raster(PROCESSED/'ground_class2_range_native_grid.tif',spread,tf,np.nan)
    # Preserve true depressions and entrances; no sink filling or water flattening.
    blocked=watermask|bmask|~np.isfinite(dem)|(dem==-32767)
    # Limit search to the dry-land corridor; no northern moat detour outside context.
    ys=tf.f+(np.arange(dem.shape[0])+.5)*tf.e
    blocked[ys>-35630,:]=True
    start=np.array([-6956.,-36465.]);end=np.array([-6923.,-36403.]);u=(end-start)/np.linalg.norm(end-start);v=np.array([-u[1],u[0]])
    lane=LineString([start-3*v,end-3*v]).buffer(.35)
    startmask=rasterize([(lane,1)],out_shape=dem.shape,transform=tf,dtype='uint8').astype(bool)
    banks={};bank_rows=[];bank_features=[]
    for name,g in targets.items():
        ring=g.buffer(3).difference(g.buffer(.75))
        mask=rasterize([(ring,1)],out_shape=dem.shape,transform=tf,all_touched=False,dtype='uint8').astype(bool)&~blocked
        banks[name]=mask
        ids=np.flatnonzero(mask);supported=ids[(count[ids]>=3)&((high-low)[ids]<=.15)]
        use=supported if len(supported) else ids
        idx=int(use[np.argmin(dem.ravel()[use])]);r,c=divmod(idx,dem.shape[1]);x,y=xy(tf,r,c)
        prop={'target':name,'screening_id':{'雲形池':'B1','心字池':'B2','日比谷濠':'B3'}[name],'easting_m':x,'northing_m':y,'dem_z_m':float(dem[r,c]),'class2_count':int(counts[r,c]),'class2_mean_z_m':float(mean[r,c]),'ground_range_m':float(spread[r,c]),'supported_bank_cells':len(supported),'all_bank_cells':len(ids),'bank_dem_p05_m':float(np.percentile(dem.ravel()[ids],5)),'bank_dem_p50_m':float(np.median(dem.ravel()[ids])),'interpretation':'Lowest supported dry-land cell in approximate 0.75-3 m outer shore band; not verified bank crest or inlet'}
        bank_rows.append(prop);bank_features.append(feature(Point(x,y),prop))
    writecsv(OUT/'bank_screening_candidates.csv',bank_rows)
    routes,pops=minimax_paths(dem,blocked,startmask,banks,tf)
    route_features=[];profiles=[];summary_routes=[];candidates=bank_features[:]
    for i,route in enumerate(routes,1):
        coords=route['xyz'];ids=route['indices'];distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(coords[:,:2],axis=0),axis=1))]
        props={k:v for k,v in route.items() if k not in ['xyz','indices']};props['route_id']=f'P{i}'
        props['supported_cell_fraction']=float(np.mean((count[ids]>=3)&((high-low)[ids]<=.15)))
        props['interpretation']='Minimum maximum native DEM elevation to approximate dry bank; NOT confirmed inlet, downhill flow, or water-stage comparison'
        route_features.append(feature(LineString(coords[:,:2]),props));summary_routes.append(props)
        for s,q,idx in zip(distance,coords,ids):profiles.append({'route_id':f'P{i}','target':route['target'],'distance_m':s,'easting_m':q[0],'northing_m':q[1],'dem_z_m':q[2],'class2_count':int(count[idx]),'class2_mean_z_m':float(mean.ravel()[idx]),'class2_range_m':float(spread.ravel()[idx])})
        crest=coords[np.argmax(coords[:,2])];endpt=coords[-1]
        for suffix,q,note in [('C',crest,'Highest native DEM cell on candidate path; investigate barrier geometry'),('E',endpt,'Candidate path reaches approximate outer bank; inlet NOT confirmed')]:
            candidates.append(feature(Point(*q[:2]),{'candidate_id':f'P{i}{suffix}','target':route['target'],'source_dem_z_m':float(q[2]),'interpretation':note}))
    savegeo(PROCESSED/'screening_paths.geojson',route_features)
    savegeo(PROCESSED/'review_candidates.geojson',candidates)
    writecsv(OUT/'screening_path_profiles.csv',profiles)
    # Geometry screening map, with water footprints overriding untrusted water DEM.
    extent=(bounds[0],bounds[2],bounds[1],bounds[3])
    fig,axes=plt.subplots(1,2,figsize=(14,10),layout='constrained')
    axes[0].imshow(np.moveaxis(ortho,0,-1),extent=extent,origin='upper')
    masked=np.ma.array(dem,mask=watermask|(dem==-32767));im=axes[1].imshow(masked,extent=extent,origin='upper',cmap='terrain',vmin=0,vmax=7)
    for ax in axes:
        for name,g in targets.items():
            for q in (list(g.geoms) if hasattr(g,'geoms') else [g]):ax.plot(*np.array(q.exterior.coords).T,color='#00bfff',lw=1)
        for i,route in enumerate(routes,1):ax.plot(*route['xyz'][:,:2].T,lw=1.6,label=f'P{i} '+{'雲形池':'Kumogata','心字池':'Shinji','日比谷濠':'Hibiya moat'}[route['target']])
        ax.plot(*np.array(lane.exterior.coords).T,color='magenta',lw=2,label='Verified road lane: source area')
        for p in bank_rows:ax.annotate(p['screening_id'],(p['easting_m'],p['northing_m']),color='red',fontsize=9,xytext=(4,4),textcoords='offset points')
        ax.set_xlim(-7200,-6500);ax.set_ylim(-36550,-35630);ax.set_aspect('equal');ax.set_xlabel('Easting (m; JGD2011 IX)');ax.set_ylabel('Northing (m)');ax.grid(alpha=.15)
    axes[0].set_title('Official 0.20 m orthophoto + screening paths');axes[1].set_title('Native 0.50 m DEM (water interior excluded)')
    axes[0].legend(loc='lower left',fontsize=8);fig.colorbar(im,ax=axes[1],label='Source ground elevation (m)')
    fig.suptitle('Hibiya: possible terrain corridors, NOT verified drainage connections',fontsize=14)
    fig.savefig(OUT/'corridor_screening_overview.png',dpi=160,bbox_inches='tight');plt.close(fig)
    summary={'mosaic_bounds':list(bounds),'native_dem_shape':list(dem.shape),'dem_resolution_m':.5,'orthophoto_resolution_m':.2,'dem_nodata_cells':int(np.sum(dem==-32767)),'water_cells_excluded':int(watermask.sum()),'building_objects':len(buildings),'road_objects':len(roads),'vector_sources':vector_sources,'las_metadata':las_metadata,'ground_point_count':int(count.sum()),'bank_screening':bank_rows,'routes':summary_routes,'search_settled_cells':pops,'limits':['Minimax route is a static screening hypothesis, not a runoff simulation.','Shore band is a map-derived 0.75-3 m outer band; neither a surveyed crest nor verified inlet.','Building exclusion is conservative; narrow walls/kerbs may be absent from ground DEM.','Unknown water stage and vertical-datum revision prevent an absolute head comparison.','Native DEM and Class 2 are common-source evidence, not independent accuracy validation.']}
    (OUT/'screening_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'banks':bank_rows,'routes':summary_routes},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
