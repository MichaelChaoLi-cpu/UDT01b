"""Build traceable whole-domain SFINCS scenario terrain from existing sources.

No source DEM is edited. The 0.5m sampling grid retains source resolution IDs;
PLATEAU fallback remains a coarser source. No riverbed is inferred from wtr Z.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,math,csv
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize,shapes
from shapely.geometry import shape,mapping,box
from shapely.ops import transform,unary_union
from shapely import STRtree
from scipy.ndimage import binary_dilation,generate_binary_structure
from scipy.spatial import cKDTree
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[2]
PRO=ROOT/'data/processed/three_ward_sfincs_scenarios'
OUT=ROOT/'data/exp/data-briefing/three_ward_sfincs_scenarios'
XY=Transformer.from_crs(6668,6677,always_xy=True)
RIVERS={'古川','神田川','日本橋川','亀島川','隅田川'}
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def save_raster(p,a,tf):
 with rasterio.open(p,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype=a.dtype,crs='EPSG:6677',transform=tf,nodata=0 if a.dtype=='uint8' else np.nan,compress='deflate') as d:d.write(a,1)
def main():
 PRO.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 core=json.loads((ROOT/'data/processed/three_ward_dem_domain/native_model_domain_candidate.geojson').read_text())
 geom=unary_union([transform(XY.transform,shape(f['geometry'])) for f in core['features']]);dx=20.;l,b,r,t=geom.bounds;l=math.floor(l/dx)*dx;b=math.floor(b/dx)*dx;r=math.ceil(r/dx)*dx;t=math.ceil(t/dx)*dx;nr,nc=int((t-b)/dx),int((r-l)/dx);tf=from_origin(l,t,dx,dx);ftf=from_origin(l,t,.5,.5);fshape=(nr*40,nc*40)
 dm=rasterize([(mapping(geom),1)],out_shape=(nr,nc),transform=tf).astype(bool)
 waters=json.loads((ROOT/'data/processed/central_three_water_network/water_inventory.geojson').read_text())['features']
 wg=[transform(XY.transform,shape(f['geometry'])) for f in waters];tree=STRtree(wg)
 selected=[];waternames={}
 for i,f in enumerate(waters,1):
  p=f['properties'];name=p['name'];sea='1000' in p['source_class_codes'];waternames[i]=name
  if name in RIVERS or sea:selected.append((mapping(wg[i-1]),i))
 water_group=rasterize(selected,out_shape=(nr,nc),transform=tf,all_touched=True,dtype='int16')
 finepath=PRO/'terrain_sampling_0_50m.tif';srcpath=PRO/'terrain_source_id_0_50m.tif'
 profile=dict(driver='GTiff',height=fshape[0],width=fshape[1],count=1,crs='EPSG:6677',transform=ftf,tiled=True,compress='deflate',BIGTIFF='IF_SAFER')
 active=np.zeros((nr,nc),bool);watercell=np.zeros_like(active);missingcell=np.zeros_like(active);zmean=np.full((nr,nc),np.nan,np.float32);zmin=zmean.copy();zmax=zmean.copy();fallbackfraction=zmean.copy();total_primary=total_fallback=0
 with rasterio.open(ROOT/'data/processed/three_ward_dem_domain/native_dem_0_50m.vrt') as native,rasterio.open(ROOT/'data/processed/three_ward_terrain_fallback/plateau_ground_tin_5m.tif') as tin,WarpedVRT(native,crs='EPSG:6677',transform=ftf,width=fshape[1],height=fshape[0],resampling=Resampling.nearest,nodata=np.nan) as nv,WarpedVRT(tin,crs='EPSG:6677',transform=ftf,width=fshape[1],height=fshape[0],resampling=Resampling.bilinear,nodata=np.nan) as tv,rasterio.open(finepath,'w',dtype='float32',nodata=np.nan,**profile) as dst,rasterio.open(srcpath,'w',dtype='uint8',nodata=0,**profile) as sid:
  for cr in range(0,nr,40):
   for cc in range(0,nc,50):
    hh=min(40,nr-cr);ww=min(50,nc-cc);w=Window(cc*40,cr*40,ww*40,hh*40);a=nv.read(1,window=w);a[(a==-32767)|(~np.isfinite(a))]=np.nan;fallback=tv.read(1,window=w);use=(~np.isfinite(a))&np.isfinite(fallback);ids=np.where(np.isfinite(a),1,0).astype('uint8');a[use]=fallback[use];ids[use]=2
    qtf=rasterio.windows.transform(w,ftf);tilebox=box(*rasterio.windows.bounds(w,ftf));geoms=[(mapping(wg[j]),1) for j in tree.query(tilebox,predicate='intersects')]
    water=rasterize(geoms,out_shape=a.shape,transform=qtf).astype(bool) if geoms else np.zeros(a.shape,bool)
    # Only complete land cells are active, so the builder cannot interpolate missing active terrain.
    am=a.reshape(hh,40,ww,40);wm=water.reshape(hh,40,ww,40);valid=np.isfinite(am).all(axis=(1,3));wet=wm.any(axis=(1,3));act=dm[cr:cr+hh,cc:cc+ww]&valid&~wet
    active[cr:cr+hh,cc:cc+ww]=act;watercell[cr:cr+hh,cc:cc+ww]=wet;missingcell[cr:cr+hh,cc:cc+ww]=~valid
    zmean[cr:cr+hh,cc:cc+ww]=np.nanmean(am,axis=(1,3));zmin[cr:cr+hh,cc:cc+ww]=np.nanmin(am,axis=(1,3));zmax[cr:cr+hh,cc:cc+ww]=np.nanmax(am,axis=(1,3));fallbackfraction[cr:cr+hh,cc:cc+ww]=(ids.reshape(hh,40,ww,40)==2).mean(axis=(1,3))
    full=np.repeat(np.repeat(act,40,axis=0),40,axis=1);total_primary+=int(((ids==1)&full).sum());total_fallback+=int(((ids==2)&full).sum());dst.write(a,1,window=w);sid.write(ids,1,window=w)
   print(f'fine terrain strip {cr+hh}/{nr}',flush=True)
 # Boundary row is land immediately adjoining named river/sea exclusions. No pond/moat sink.
 receiving=watercell&(water_group>0);near=binary_dilation(receiving,structure=generate_binary_structure(2,1));boundary=active&near
 by,bx=np.where(boundary);wy,wx=np.where(receiving);xy=np.c_[l+(bx+.5)*dx,t-(by+.5)*dx];wxy=np.c_[l+(wx+.5)*dx,t-(wy+.5)*dx];_,ind=cKDTree(wxy).query(xy);gid=water_group[wy[ind],wx[ind]]
 # Local lower-water envelope: below minimum land elevation over neighbouring bank row.
 low=np.full(len(by),np.nan);bmin=zmin[by,bx]
 for group in np.unique(gid):
  take=np.where(gid==group)[0];kt=cKDTree(xy[take]);neighbors=kt.query_ball_point(xy[take],250)
  envelope=np.array([np.quantile(bmin[take[js]],.1) for js in neighbors])-1.
  low[take]=np.minimum(envelope,bmin[take]-.5)
  if any('1000' in waters[int(group)-1]['properties']['source_class_codes'] for _ in [0]):low[take]=np.minimum(0.,bmin[take]-.5)
 msk=active.astype('uint8');msk[boundary]=2
 save_raster(PRO/'active_mask_20m.tif',msk,tf);save_raster(PRO/'ground_mean_20m.tif',zmean,tf);save_raster(PRO/'ground_min_20m.tif',zmin,tf);save_raster(PRO/'ground_max_20m.tif',zmax,tf);save_raster(PRO/'fallback_fraction_20m.tif',fallbackfraction,tf)
 rows=[{'point_id':i+1,'row_north':int(y),'col_east':int(x),'x':float(p[0]),'y':float(p[1]),'water_group':int(g),'water_name':waternames[int(g)],'land_min_m':float(zmin[y,x]),'land_max_m':float(zmax[y,x]),'low_stage_m':float(h),'raised_stage_increment_m':1.5,'basis':'DEM-relative hypothetical stage envelope;not surveyed river stage or bathymetry'} for i,(y,x,p,g,h) in enumerate(zip(by,bx,xy,gid,low))]
 with (PRO/'boundary_points.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(PRO/'boundary_points.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':[{'type':'Feature','properties':row,'geometry':{'type':'Point','coordinates':[row['x'],row['y']]}} for row in rows]})
 missing_features=[{'type':'Feature','properties':{'reason':'remaining source NoData;closed edge'},'geometry':g} for g,v in shapes((dm&~watercell&missingcell).astype('uint8'),transform=tf) if v==1]
 dump(PRO/'excluded_terrain_gaps.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':'EPSG:6677'}},'features':missing_features})
 contract={'domain_candidate_km2':geom.area/1e6,'dx_m':dx,'fine_sampling_m':.5,'nrows':nr,'ncols':nc,'bounds':[l,b,r,t],'active_land_cell_area_km2':active.sum()*400/1e6,'interior_land_area_km2':(msk==1).sum()*400/1e6,'boundary_row_area_km2':boundary.sum()*400/1e6,'excluded_water_intersecting_cells_km2':(dm&watercell).sum()*400/1e6,'excluded_incomplete_land_cells_km2':(dm&~watercell&missingcell).sum()*400/1e6,'fine_active_primary_area_km2':total_primary*.25/1e6,'fine_active_plateau_fallback_area_km2':total_fallback*.25/1e6,'boundary_points':len(rows),'native_primary_resolution_m':.5,'plateau_native_fallback_m':5,'fine_sampling_is_accuracy_claim':False,'active_fine_terrain_is_complete_before_official_builder':True,'source_dem_modified':False,'water_lod1_z_used':False,'river_bathymetry_available':False,'river_stage_is_hypothetical':True,'unknown_moat_pond_exchanges':'closed for this screening','outer_domain':'closed except named receiving waters;no external river discharge imposed','bank_limits':'subgrid terrain retained;boundary land row and neighbouring40m zone are uncertain,not surveyed crest/inlet','building_obstacles':'not imposed from RoofEdge proxy;this is terrain-led surface screening','rain':'uniform synthetic60/100mm per hour for1h,1h dry boundary spinup,2h recession','drainage':'no pipe/pump removal','infiltration_mm_hr':0,'roughness_n':.04,'interpretation':'conditional surface accumulation,not validated street flooding or dry-weather river discharge model'}
 dump(PRO/'terrain_contract.json',contract)
 river={'water_objects':len(waters),'lod1_all_zero_objects':sum(f['properties']['source_lod1_z_stats']['min_m']==f['properties']['source_lod1_z_stats']['max_m']==0 for f in waters),'actual_source_property_names':['name','creationDate','class','lod0MultiCurve','lod1MultiSurface','wtrDataQualityAttribute'],'riverbed_surface_found':False,'lod2_or_lod3_water_solids_found':False,'bank_crest_from_dem':'candidate ground elevations only,not a surveyed cross section','small_fair_weather_flow_user_observation':'used only as qualitative low-stage hypothesis;no measured Q(t) assigned','tidal_and_storm_response':'low baseflow does not fix water elevation or receiving capacity'}
 dump(OUT/'river_3d_availability.json',river);print(json.dumps(contract,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
