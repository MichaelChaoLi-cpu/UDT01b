"""Matched whole-domain SFINCS rerun with a fixed-fine roof runoff scenario."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['MPLCONFIGDIR']='/private/tmp/udt01b-mpl'
from pathlib import Path
import argparse,json,hashlib,shutil,time,subprocess
import numpy as np
import pandas as pd
import xarray as xr
import rasterio,geopandas as gpd
from shapely.geometry import Point
from hydromt_sfincs import SfincsModel
from run_three_ward_urban_grid import boundary,contract
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_fixed_runoff';BASE=ROOT/'data/processed/three_ward_sfincs_framework';PRIOR=BASE/'urban_grid_comparison';RUN=BASE/'fixed_runoff_comparison'

def run(dx,pilot=False,levels=64):
 import fcntl
 RUN.mkdir(exist_ok=True)
 if not pilot:assert json.loads((RUN/f'pilot_{dx}m/forcing_validation.json').read_text())['passed'], 'Full runs require successful input pilot'
 folder=RUN/(('pilot' if pilot else 'rain60_low')+('' if levels==64 else f'_storage{levels}')+f'_{dx}m');folder.mkdir(exist_ok=True)
 with (folder/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (folder/'execution.json').exists():
   ex=json.loads((folder/'execution.json').read_text());assert ex['returncode']==0 and ex['completion_marker'];
   if pilot:assert json.loads((folder/'forcing_validation.json').read_text())['passed']
   print('Reuse',folder.name,flush=True);return
  if (folder/'sfincs_map.nc').exists():raise RuntimeError('Preserve interrupted outputs before rerunning')
  template=PRIOR/f'urban_{dx}m_template' if levels==64 else RUN/f'urban_storage{levels}_{dx}m_template'
  if levels!=64:assert json.loads((template/'storage_refinement_complete.json').read_text())['face_piecewise_linear_functions_preserved']
  for p in template.iterdir():
   if p.suffix in ['.ind','.msk','.nc','.obs','.inp','.crs','.dep'] and p.name not in ['sfincs_map.nc','sfincs_his.nc']:shutil.copy2(p,folder/p.name)
  shutil.copy2(template/'flux_sections.json',folder/'flux_sections.json')
  sf=SfincsModel(root=folder,mode='r+',write_gis=False,data_libs=[]);sf.read_config();sf.read_grid()
  pts=boundary(dx);loc=gpd.GeoDataFrame(index=pts.index,geometry=[Point(x,y) for x,y in zip(pts.x,pts.y)],crs=6677)
  tstage=pd.Timestamp('2020-01-01')+pd.to_timedelta([0,60] if pilot else [0,14400],unit='s');df=pd.DataFrame(np.repeat(pts.low_stage_m.to_numpy()[None,:],len(tstage),0),index=tstage,columns=pts.index);sf.setup_waterlevel_forcing(timeseries=df,locations=loc,buffer=0,merge=False)
  with rasterio.open(P/f'contributing_area_{dx}m.tif') as d:a=d.read(1);tf=d.transform
  weights=np.pad(a/dx**2,1);seconds=np.array([0,60] if pilot else [0,3600,3601,7200,7201,14400]);rate=np.array([60,60] if pilot else [0,0,60,60,0,0],dtype='float64')
  # Galibier ncinput reads px(1),py(1) as lower-left CORNERS, then
  # meteo adds half a cell. Encode ascending corner coordinates, not CF centers.
  x=tf.c-dx+np.arange(weights.shape[1])*dx;y=tf.f-a.shape[0]*dx-dx+np.arange(weights.shape[0])*dx
  weights=np.flipud(weights)
  da=xr.DataArray((rate[:,None,None]*weights[None,:,:]).astype('float32'),dims=['time','y','x'],coords={'time':pd.Timestamp('2020-01-01')+pd.to_timedelta(seconds,unit='s'),'y':y,'x':x},name='precip_2d');da.raster.set_crs(6677);da.raster.set_nodata(0.);da.attrs.update(unit='mm.hr-1',standard_name='precipitation')
  sf.set_forcing(da,name='precip_2d');sf.setup_config(netamprfile='roof_surface_precip.nc',dtwnd=1.,ampr_block=0)
  for key in ['precipfile','srcfile','disfile']:sf.config.pop(key,None)
  if pilot:sf.setup_config(tstop='20200101 000100',dtout=60.,dtmaxout=60.)
  sf.write_forcing(fmt='%.9f');sf.write_config()
  c=contract(dx);c.pop('roof_donors');c['legacy_removed_building_roof_area_m2']=c['roof_source_area_m2'];c['roof_source_area_m2']=0.;c['roof_surface_area_m2']=json.loads((P/'contract.json').read_text())['all_roof_area_m2'];c['roof_max_donor_distance_m']=json.loads((P/'contract.json').read_text())['roof_nearest_distance_max_m'];rain_depth=float(np.trapezoid(rate,seconds)/3600);c.update({'name':'pilot' if pilot else 'rain60_low','urban':True,'rain_mm_h':60,'stage_rise_m':0,'original_rain_interior_volume_m3':float(a.sum()*rain_depth/1000),'effective_rain_depth_mm':rain_depth,'roof_source_volume_m3':0,'forcing_method':'Spatial effective rain = (open ground + fixed fine roof donors) area / hydraulic cell area * physical rain; all weights nonnegative; original 1-second rain ramps retained.','roof_allocation_contract':json.loads((P/'contract.json').read_text()),'nlevels':levels,'storage_refined_only':levels!=64,'synthetic':True,'active_measures':False,'hydraulics_identical_to_previous_run':levels==64,'spatial_precip_coordinate_convention':'Galibier-specific ascending lower-left corner coordinates; official reader adds half cell; NOT generic CF pixel-center raster','meteo_update_interval_seconds':1.,'history_interval':.4,'pilot':pilot})
  (folder/'scenario_contract.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
  inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.suffix in ['.inp','.ind','.msk','.nc','.obs','.crs','.dep','.bnd','.bzs']}
  spec=json.loads((BASE/'model_spec.json').read_text());threads=os.environ.get('UDT_SFINCS_THREADS','4');command=['/usr/local/bin/docker','run','--rm','--platform','linux/amd64','--network','none','--cpus',threads,'--memory','6g' if levels!=64 else '4g','-e','OMP_NUM_THREADS='+threads,'--mount',f'type=bind,src={folder},dst=/data',spec['solver']['docker_image']]
  print('RUN',folder.name,flush=True);start=time.monotonic()
  with (folder/'solver_stdout.txt').open('w') as log:res=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=10800)
  log=(folder/'solver_stdout.txt').read_text(errors='replace');finished='Simulation finished' in log or 'Simulation is finished' in log
  ex={'returncode':res.returncode,'completion_marker':finished,'runtime_seconds':time.monotonic()-start,'command':command,'input_sha256':inputs,'official_solver':spec['solver']};(folder/'execution.json').write_text(json.dumps(ex,ensure_ascii=False,indent=2)+'\n')
  assert res.returncode==0 and finished,log[-500:];print('DONE',folder.name,round(ex['runtime_seconds'],1),flush=True)
  if pilot:
   import netCDF4
   with netCDF4.Dataset(folder/'sfincs_map.nc') as d:observed=np.flipud(np.ma.asarray(d['cumprcp'][-1]).filled(0)).astype('float64');m=np.flipud(np.asarray(d['msk'][:]));vol=np.flipud(np.ma.asarray(d['subgrid_volume'][-1]).filled(0)).astype('float64')
   expected=a/dx**2*.001;sel=m==1;err=float(np.max(abs(observed[sel]-expected[sel])));rel=abs(float(observed[sel].sum()*dx**2)-float(a.sum()*.001))/float(a.sum()*.001)
   audit={'max_cell_precip_depth_error_m':err,'rain_total_relative_error':rel,'all_recipients_active_interior':bool(np.all(a[m!=1]==0)),'passed':err<1e-6 and rel<1e-5,'engine_precip_volume_m3':float(observed[sel].sum()*dx**2),'expected_precip_volume_m3':float(a.sum()*.001),'signed_final_storage_m3':float(vol[sel].sum())}
   (folder/'forcing_validation.json').write_text(json.dumps(audit,indent=2)+'\n');assert audit['passed'],audit;print('PILOT',audit,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dx',type=int,choices=[20,10],required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--levels',type=int,choices=[64,253],default=64);args=ap.parse_args();run(args.dx,args.pilot,args.levels)
