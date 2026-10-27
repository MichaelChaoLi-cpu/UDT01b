"""Use pinned official HydroMT-SFINCS and official solver for synthetic Tokyo scenarios."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ['MPLCONFIGDIR']='/private/tmp/udt01b-mpl'
from pathlib import Path
import argparse,json,logging,hashlib,shutil,subprocess,time,importlib.metadata
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from hydromt_sfincs import SfincsModel
ROOT=Path(__file__).resolve().parents[2]
PRO=ROOT/'data/processed/three_ward_sfincs_scenarios'
BASE=ROOT/'data/processed/three_ward_sfincs_framework'
OUT=ROOT/'data/exp/data-briefing/three_ward_sfincs_scenarios'
CASES=[('dry_low',0,0),('rain60_low',60,0),('rain100_low',100,0),('dry_raised',0,1.5),('rain60_raised',60,1.5),('rain60_closed',60,0),('rain60_low_dt1',60,0),('dry_raised_his2',0,1.5),('rain60_raised_his2',60,1.5),('dry_raised_hisstep',0,1.5),('rain60_raised_hisstep',60,1.5)]
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def build():
 c=json.loads((PRO/'terrain_contract.json').read_text());folder=BASE/'terrain_scenario_template_20m';folder.mkdir(exist_ok=True)
 logger=logging.getLogger('udt01b_scenario_builder');logger.setLevel(logging.INFO);logger.addHandler(logging.FileHandler(folder/'builder.log'));logger.addHandler(logging.StreamHandler())
 sf=SfincsModel(root=folder,mode='w',write_gis=False,data_libs=[],logger=logger)
 l,b,r,t=c['bounds'];sf.setup_grid(x0=l,y0=b,dx=20.,dy=20.,nmax=c['nrows'],mmax=c['ncols'],rotation=0.,epsg=6677)
 with rasterio.open(PRO/'active_mask_20m.tif') as d:mask=d.read(1)
 template=sf.reggrid.empty_mask
 template.data=np.flipud(mask).copy();sf.set_grid(template,name='msk')
 with rasterio.open(PRO/'ground_mean_20m.tif') as d:z=d.read(1)
 dep=template.astype('float32').copy();dep.data=np.flipud(z).copy();dep.raster.set_nodata(np.nan);sf.set_grid(dep,name='dep')
 sf.setup_config(indexfile='sfincs.ind',mskfile='sfincs.msk',inputformat='bin',tref='20200101 000000',tstart='20200101 000000',tstop='20200101 040000',zsini=-100.,qinf=0.,dtout=300.,dthisout=300.,dtmaxout=14400.,dtmax=2.,alpha=.4,huthresh=.01,hmin_cfl=.1,advection=1,viscosity=1,coriolis=0,baro=0,storehsubgrid=1,storehmean=1,storezvolume=1,storevel=1,storevelmax=1,storecumprcp=1,timestep_analysis=1,outputformat='net')
 sf.write_grid();sf.write_config()
 print('Building official0.5m-sampled subgrid tables',flush=True)
 sf.setup_subgrid(datasets_dep=[{'elevtn':str(PRO/'terrain_sampling_0_50m.tif'),'reproj_method':'nearest'}],nr_subgrid_pixels=40,nrmax=1600,nlevels=16,max_gradient=99999.,manning_land=.04,manning_sea=.04,huthresh=.01,write_dep_tif=False)
 # Save before subsequent output steps; avoid losing computed official tables.
 sf.write_subgrid();sf.write()
 by,bx=np.where(mask==1);ii=len(by)//2
 (folder/'sfincs.obs').write_text(f'{l+(bx[ii]+.5)*20:.1f} {t-(by[ii]+.5)*20:.1f} "diagnostic_land"\n')
 sf.setup_config(obsfile='sfincs.obs');sf.write_config()
 dump(folder/'builder_environment.json',{p:importlib.metadata.version(p) for p in ['hydromt-sfincs','hydromt','numpy','pandas','xarray','rasterio','numba','netCDF4']})
 code=Path(__import__('hydromt_sfincs.subgrid',fromlist=['x']).__file__)
 dump(folder/'builder_provenance.json',{'module':str(code),'sha256':hashlib.sha256(code.read_bytes()).hexdigest(),'official_builder_version':'1.2.2','active_terrain_complete_before_builder':c['active_fine_terrain_is_complete_before_official_builder'],'formal_subgrid_created':True,'not_calibrated_event_model':True})
 print('Official subgrid finished',flush=True)

def prepare_flux_sections():
 c=json.loads((PRO/'terrain_contract.json').read_text());l,b,r,t=c['bounds']
 with rasterio.open(PRO/'active_mask_20m.tif') as d:mask=d.read(1)
 sections=[];lines=[]
 for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
  shifted=np.roll(mask,(-dr,-dc),(0,1));select=(mask==1)&(shifted==2)
  if dr==1:select[-1]=False
  if dr==-1:select[0]=False
  if dc==1:select[:,-1]=False
  if dc==-1:select[:,0]=False
  for y,x in zip(*np.where(select)):
   xx=l+(x+.5+.5*dc)*20;yy=t-(y+.5+.5*dr)*20
   # Positive discharge to left of directed section: land -> receiving row.
   tx,ty=-dr,-dc;start=(xx-tx*9.8,yy-ty*9.8);end=(xx+tx*9.8,yy+ty*9.8)
   name=f'B{len(sections)+1:05d}';lines += [name,'2 2',f'{start[0]:.3f} {start[1]:.3f}',f'{end[0]:.3f} {end[1]:.3f}']
   sections.append({'name':name,'row':int(y),'col':int(x),'boundary_row':int(y+dr),'boundary_col':int(x+dc),'normal_x':dc,'normal_y':-dr,'positive':'interior_land_to_receiving_water_boundary','coordinates':[start,end]})
 template=BASE/'terrain_scenario_template_20m';(template/'sfincs.crs').write_text('\n'.join(lines)+'\n')
 sf=SfincsModel(root=template,mode='r+',write_gis=False,data_libs=[]);sf.read_config();sf.setup_config(crsfile='sfincs.crs',dthisout=10.);sf.write_config();dump(PRO/'flux_section_index.json',sections)

def cases(dtmax=None):
 template=BASE/'terrain_scenario_template_20m'
 prepare_flux_sections()
 if not (template/'sfincs_subgrid.nc').exists():raise RuntimeError('Build template first')
 points=pd.read_csv(PRO/'boundary_points.csv');ids=points.point_id.to_numpy();locs=gpd.GeoDataFrame(index=ids,geometry=[Point(x,y) for x,y in zip(points.x,points.y)],crs=6677)
 spec=json.loads((BASE/'model_spec.json').read_text());registry=[]
 for name,rain,rise in CASES:
  folder=BASE/('terrain_'+name+'_20m');folder.mkdir(exist_ok=True)
  if (folder/'execution.json').exists():
   previous=json.loads((folder/'execution.json').read_text())
   if previous.get('returncode')==0 and previous.get('completion_marker') and (folder/'sfincs_map.nc').is_file():
    registry.append({**json.loads((folder/'scenario_contract.json').read_text()),'path':str(folder.relative_to(ROOT))});continue
  partial=[folder/n for n in ['sfincs_map.nc','sfincs_his.nc','sfincs.log','solver_stdout.txt','execution.json'] if (folder/n).exists()]
  if partial:
   archive=folder/'interrupted_20261014';archive.mkdir(exist_ok=True)
   hashes={}
   for old in partial:
    target=archive/old.name
    if target.exists():raise RuntimeError('Interrupted output archive already exists;preserve before retry')
    hashes[old.name]=hashlib.sha256(old.read_bytes()).hexdigest();shutil.move(str(old),str(target))
   dump(archive/'preserved_output_sha256.json',hashes)
  for p in template.iterdir():
   if p.suffix in ['.ind','.msk','.nc','.obs','.inp','.crs','.dep'] and p.name not in ['sfincs_map.nc','sfincs_his.nc']:shutil.copy2(p,folder/p.name)
  sf=SfincsModel(root=folder,mode='r+',write_gis=False,data_libs=[]);sf.read_config();sf.read_grid()
  times=pd.Timestamp('2020-01-01')+pd.to_timedelta([0,3600,5400,7200,10800,14400],unit='s');pulse=np.array([0,0,1,1,0,0])*rise
  df=pd.DataFrame(points.low_stage_m.to_numpy()[None,:]+pulse[:,None],index=times,columns=ids)
  if name=='rain60_closed':
   ms=sf.mask.copy();ms.data=np.where(ms.data==2,1,ms.data).astype('uint8');sf.set_grid(ms,name='msk');sf.write_grid();sf.setup_config(bndfile='none',bzsfile='none')
  else:sf.setup_waterlevel_forcing(timeseries=df,locations=locs,buffer=0,merge=False)
  pt=pd.Timestamp('2020-01-01')+pd.to_timedelta([0,3600,3601,7200,7201,14400],unit='s');pr=pd.DataFrame({'rain':[0,0,rain,rain,0,0]},index=pt)
  sf.setup_precip_forcing(timeseries=pr)
  if dtmax or name.endswith('_dt1'):sf.setup_config(dtmax=dtmax or 1.)
  if name.endswith('_his2'):sf.setup_config(dthisout=2.)
  if name.endswith('_hisstep'):sf.setup_config(dthisout=.8)
  sf.write_forcing(fmt='%.5f');sf.write_config()
  inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.suffix in ['.ind','.msk','.nc','.obs','.inp','.crs','.dep','.bnd','.bzs','.precip']}
  command=['/usr/local/bin/docker','run','--rm','--platform','linux/amd64','--network','none','--cpus','4','--memory','4g','-e','OMP_NUM_THREADS=4','--mount',f'type=bind,src={folder},dst=/data',spec['solver']['docker_image']]
  contract={'scenario':name,'synthetic_rain':True,'historical_event_replay':False,'rain_rate_mm_h':rain,'rain_duration_seconds':3600,'river_stage_rise_m':rise,'rain_integral_mm':float(np.trapezoid(pr.rain.to_numpy(),(pt-pt[0]).total_seconds()))/3600,'river_stage_reference':'DEM-relative lower-water envelope,not measured T.P. river profile','drainage_mm_h':0,'infiltration_mm_h':0,'boundary_row_excluded_from_hazard_stats':True,'roughness':.04,'active_measures':False,'boundary_closed_control':name=='rain60_closed','dtmax_seconds':1. if name.endswith('_dt1') else 2.,'history_output_seconds':.8 if name.endswith('_hisstep') else (2. if name.endswith('_his2') else 10.)}
  dump(folder/'scenario_contract.json',contract);start=time.monotonic();print('Running',name,flush=True)
  with (folder/'solver_stdout.txt').open('w') as log:res=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
  log=(folder/'solver_stdout.txt').read_text(errors='replace');finished='Simulation finished' in log or 'Simulation is finished' in log
  execution={'scenario':name,'returncode':res.returncode,'completion_marker':finished,'runtime_seconds':time.monotonic()-start,'command':command,'input_sha256':inputs,'synthetic':True,'official_engine':spec['solver'],'map_created':(folder/'sfincs_map.nc').exists()}
  dump(folder/'execution.json',execution)
  if res.returncode or not finished or not execution['map_created']:raise RuntimeError(f'{name} engine failed;inspect{folder}/solver_stdout.txt')
  print('Completed',name,round(execution['runtime_seconds'],2),flush=True);registry.append({**contract,'path':str(folder.relative_to(ROOT))})
 dump(OUT/'scenario_registry.json',registry)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','run']);a=ap.parse_args();build() if a.action=='build' else cases()
