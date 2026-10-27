"""Same fixed0.50m roof donors aggregated to5m, conservatively forced on10/5m.
Galibier bilinear sampling at10m centres averages exactly four nested5m cells.
Every-cell precipitation totals must pass the official-engine pilot.
"""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import json,argparse
import numpy as np
import rasterio,pandas as pd,xarray as xr,geopandas as gpd
from scipy.ndimage import label,distance_transform_edt
from shapely.geometry import Point
from hydromt_sfincs import SfincsModel
from prepare_three_ward_adaptive_connectivity import ROOT,P,OLD,FIX,CON,NR,NC,LEFT,TOP,maps,dump,face_coordinates

def allocation5():
 if (P/'allocation5_complete.json').exists():return
 _,cl,_=maps()
 with rasterio.open(OLD/'active_mask_20m.tif') as d:m=d.read(1)
 inside=np.repeat(np.repeat(m==1,40,0),40,1);open_=inside&(cl!=1);labels=np.empty(cl.shape,np.int32);count=label(open_,output=labels);road_ids=np.unique(labels[(cl==2)&open_]);road_ids=road_ids[road_ids>0];allowed=np.zeros(count+1,bool);allowed[road_ids]=True;eligible=allowed[labels]&open_;del labels,allowed
 contributing=open_.reshape(NR//10,10,NC//10,10).sum((1,3),dtype=np.int64)*.25;roof=np.zeros_like(contributing);block=1024;halo=512;donors=0
 for r0 in range(0,NR,block):
  for c0 in range(0,NC,block):
   r1=min(NR,r0+block);c1=min(NC,c0+block);donor=inside[r0:r1,c0:c1]&(cl[r0:r1,c0:c1]==1)
   if not donor.any():continue
   rs=max(0,r0-halo);re=min(NR,r1+halo);cs=max(0,c0-halo);ce=min(NC,c1+halo);e=eligible[rs:re,cs:ce];assert e.any();dist,idx=distance_transform_edt(~e,return_indices=True);dr,dc=np.where(donor);rr=dr+r0-rs;cc=dc+c0-cs;assert np.all(dist[rr,cc]<halo)
   target_r=idx[0,rr,cc]+rs;target_c=idx[1,rr,cc]+cs;assert np.all(eligible[target_r,target_c]);np.add.at(roof,(target_r//10,target_c//10),.25);donors+=len(dr)
  print('Fixed5m roof allocation',r0,'/',NR,flush=True)
 a=contributing+roof
 with rasterio.open(FIX/'contributing_area_10m.tif') as d:a10=d.read(1)
 assert np.array_equal(a.reshape(NR//20,2,NC//20,2).sum((1,3)),a10),'5m donor assignment differs from matched10m'
 assert float(a.sum())==float((m==1).sum()*400)
 with rasterio.open(P/'contributing_area_5m.tif','w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype='float64',crs=6677,transform=rasterio.transform.from_origin(LEFT,TOP,5,5),compress='deflate',tiled=True) as d:d.write(a,1)
 dump(P/'allocation5_complete.json',{'all_roof_area_m2':donors*.25,'total_contributing_area_m2':float(a.sum()),'exactly_nested_to_previous10m':True,'fine_source_sampling_m':.5,'original_fixed_roof_rule':json.loads((FIX/'contract.json').read_text())})

def forcing(folder,pilot):
 from run_three_ward_adaptive_connectivity import config
 allocation5()
 with np.load(P/'adaptive_cell_geometry.npz') as q:g={k:q[k] for k in q.files}
 with np.load(P/'adaptive_face_geometry.npz') as q:face={k:q[k] for k in q.files}
 with rasterio.open(P/'contributing_area_5m.tif') as d:a5=d.read(1)
 rf=g['row_fine']//10;cf=g['col_fine']//10;area=a5[rf,cf].copy();coarse=g['size_m']==10
 area[coarse]=a5[rf[coarse],cf[coarse]]+a5[rf[coarse]+1,cf[coarse]]+a5[rf[coarse],cf[coarse]+1]+a5[rf[coarse]+1,cf[coarse]+1];assert np.all(area[g['mask']!=1]==0)
 assert float(area.sum())==float(a5.sum());g['contributing_area_m2']=area;np.savez_compressed(P/'adaptive_cell_geometry.npz',**g)
 # Keep the physical stage of each original20m boundary cell.
 boundary=pd.read_csv(OLD/'boundary_points.csv');stage_lookup={(int(q.row_north),int(q.col_east)):(float(q.low_stage_m),q.water_name) for q in boundary.itertuples()};sel=np.flatnonzero(g['mask']==2)
 stages=np.array([stage_lookup[(int(g['row_fine'][j]//40),int(g['col_fine'][j]//40))][0] for j in sel]);loc=gpd.GeoDataFrame(index=np.arange(1,len(sel)+1),geometry=[Point(g['x'][j],g['y'][j]) for j in sel],crs=6677)
 sf=SfincsModel(root=folder,mode='r+',write_gis=False,data_libs=[]);sf.read_config();sf.read_grid()
 tstage=pd.Timestamp('2020-01-01')+pd.to_timedelta([0,60] if pilot else [0,14400],unit='s');ts=pd.DataFrame(np.repeat(stages[None,:],len(tstage),0),index=tstage,columns=loc.index);sf.setup_waterlevel_forcing(timeseries=ts,locations=loc,buffer=0,merge=False)
 seconds=np.array([0,60] if pilot else [0,3600,3601,7200,7201,14400]);rates=np.array([60,60] if pilot else [0,0,60,60,0,0],float);weights=np.flipud(np.pad(a5/25,1));x=LEFT-5+np.arange(weights.shape[1])*5;y=TOP-NR*.5-5+np.arange(weights.shape[0])*5
 da=xr.DataArray((rates[:,None,None]*weights).astype('float32'),dims=['time','y','x'],coords={'time':pd.Timestamp('2020-01-01')+pd.to_timedelta(seconds,unit='s'),'y':y,'x':x},name='precip_2d');da.raster.set_crs(6677);da.raster.set_nodata(0.);da.attrs.update(unit='mm.hr-1',standard_name='precipitation');sf.set_forcing(da,name='precip_2d');sf.setup_config(netamprfile='roof_surface_precip.nc',dtwnd=1.,ampr_block=0)
 for key in ['precipfile','srcfile','disfile']:sf.config.pop(key,None)
 sf.setup_config(tstop='20200101 000100' if pilot else '20200101 040000',dtout=60. if pilot else 300.,dtmaxout=60. if pilot else 14400.);sf.write_forcing(fmt='%.9f');sf.write_config()
 a=face['cell_a'];b=face['cell_b'];m=g['mask'];di=face['direction'];sel=np.flatnonzero(((m[a]==1)&(m[b]==2))|((m[a]==2)&(m[b]==1)));x,y,length=face_coordinates(face['row_fine'],face['col_fine'],di,face['length_pixels']);lines=[];sections=[]
 for j in sel:
  sign=1 if m[a[j]]==1 else -1;half=length[j]*.49
  xx=[x[j],x[j]] if di[j]==0 else [x[j]-sign*half,x[j]+sign*half];yy=[y[j]+sign*half,y[j]-sign*half] if di[j]==0 else [y[j],y[j]];name=f'B{len(sections)+1:05d}';lines.append({'name':name,'x':xx,'y':yy});bj=b[j] if m[b[j]]==2 else a[j];water=stage_lookup[(int(g['row_fine'][bj]//40),int(g['col_fine'][bj]//40))][1];sections.append({'name':name,'face':int(j),'water':water,'length_m':float(length[j])})
 from hydromt_sfincs.utils import write_geoms
 write_geoms(folder/'sfincs.crs',lines,stype='crs',fmt='%.6f');dump(folder/'flux_sections.json',sections)
 land=np.flatnonzero(m==1);j=land[len(land)//2];(folder/'sfincs.obs').write_text(f'{g["x"][j]:.3f} {g["y"][j]:.3f} "diagnostic_land"\n');config(folder/'sfincs.inp',{'obsfile':'sfincs.obs','crsfile':'sfincs.crs'})
 return {'case':'adaptive_10_5m','dx':10,'minimum_grid_m':5,'synthetic':True,'active_measures':False,'rain_mm_h':60,'stage_rise_m':0,'pilot':pilot,'name':'pilot' if pilot else 'rain60_low','bounds':[LEFT,TOP-NR*.5,LEFT+NC*.5,TOP],'original_rain_interior_volume_m3':float(area.sum()*np.trapezoid(rates,seconds)/3600/1000),'effective_rain_depth_mm':float(np.trapezoid(rates,seconds)/3600),'forcing_method':'Fixed0.50m roof rule, exact nested5m area weights; pinned Galibier bilinear sampling conservatively averages four5m samples for each10m cell and directly samples5m cells. Every-cell pilot validation required.','roof_source_volume_m3':0,'nlevels':253,'history_interval':.4,'same_raw_terrain':True,'rain_and_roof_allocation_matched_to_baseline':True,'river_stage_rule_matched_to_baseline':True}

if __name__=='__main__':allocation5()
