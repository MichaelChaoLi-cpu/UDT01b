"""Pinned official SFINCS nested-grid experiment; no bespoke flow equations."""
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
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_urban_constraints';OLD=ROOT/'data/processed/three_ward_sfincs_scenarios';BASE=ROOT/'data/processed/three_ward_sfincs_framework';RUN=BASE/'urban_grid_comparison';RUN.mkdir(exist_ok=True)
CASES=[('rain60_low',60,0),('rain100_low',100,0),('dry_raised',0,1.5),('rain60_raised',60,1.5)]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def paths(dx,urban):return RUN/(('urban' if urban else 'terrain64')+f'_{dx}m_template')
def mask(dx,urban):
 with rasterio.open((P/f'active_mask_{dx}m.tif') if urban else (OLD/'active_mask_20m.tif')) as d:return d.read(1)
def contract(dx):
 if (P/'model_contracts.json').exists():return next(q for q in json.loads((P/'model_contracts.json').read_text()) if q['dx']==dx)
 old=json.loads((OLD/'terrain_contract.json').read_text());return {'dx':dx,'bounds':old['bounds'],'original_interior_area_m2':old['interior_land_area_km2']*1e6,'roof_donors':[],'roof_source_area_m2':0}
def boundary(dx):
 old=pd.read_csv(OLD/'boundary_points.csv');data=[];c=json.loads((OLD/'terrain_contract.json').read_text());l,b,r,t=c['bounds'];fac=20//dx
 for q in old.itertuples():
  for dr in range(fac):
   for dc in range(fac):
    rr=q.row_north*fac+dr;cc=q.col_east*fac+dc;data.append({'row':rr,'col':cc,'x':l+(cc+.5)*dx,'y':t-(rr+.5)*dx,'low_stage_m':q.low_stage_m,'water_name':q.water_name})
 df=pd.DataFrame(data);df.index=np.arange(1,len(df)+1);return df

def sections(m,dx,folder):
 l,b,r,t=contract(dx)['bounds'];rows=[];lines=[];water={(q.row,q.col):q.water_name for q in boundary(dx).itertuples()}
 for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
  sel=(m==1)&(np.roll(m,(-dr,-dc),(0,1))==2)
  if dr==1:sel[-1]=False
  if dr==-1:sel[0]=False
  if dc==1:sel[:,-1]=False
  if dc==-1:sel[:,0]=False
  for rr,cc in zip(*np.where(sel)):
   x=l+(cc+.5+.5*dc)*dx;y=t-(rr+.5+.5*dr)*dx;tx,ty=-dr,-dc;half=dx*.49;start=(x-tx*half,y-ty*half);end=(x+tx*half,y+ty*half);name=f'B{len(rows)+1:05d}'
   lines += [name,'2 2',f'{start[0]:.4f} {start[1]:.4f}',f'{end[0]:.4f} {end[1]:.4f}'];rows.append({'name':name,'row':int(rr),'col':int(cc),'water':water[(rr+dr,cc+dc)]})
 (folder/'sfincs.crs').write_text('\n'.join(lines)+'\n');dump(folder/'flux_sections.json',rows)

def template_source_hashes(dx,urban):
 sources=[P/f'active_mask_{dx}m.tif',P/'hydraulic_obstacle_0_50m.tif',P/'manning_0_50m.tif'] if urban else [OLD/'active_mask_20m.tif',OLD/'terrain_sampling_0_50m.tif']
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

def _build(dx,urban=True):
 folder=paths(dx,urban);folder.mkdir(exist_ok=True)
 if (folder/'builder_complete.json').exists():
  saved=json.loads((folder/'builder_complete.json').read_text())
  if saved.get('source_sha256')!=template_source_hashes(dx,urban):raise RuntimeError('Template source changed; use a new experiment folder and preserve existing runs')
  print('Reuse template',folder.name,flush=True);return
 m=mask(dx,urban);c=contract(dx);l,b,r,t=c['bounds'];logger=logging.getLogger(folder.name);logger.setLevel(logging.INFO);logger.addHandler(logging.FileHandler(folder/'builder.log'));logger.addHandler(logging.StreamHandler())
 sf=SfincsModel(root=folder,mode='w',write_gis=False,data_libs=[],logger=logger);sf.setup_grid(x0=l,y0=b,dx=dx,dy=dx,nmax=m.shape[0],mmax=m.shape[1],rotation=0,epsg=6677)
 ar=sf.reggrid.empty_mask;ar.data=np.flipud(m).copy();sf.set_grid(ar,name='msk')
 with rasterio.open(OLD/'ground_mean_20m.tif') as d:ground=d.read(1)
 z=np.repeat(np.repeat(ground,20//dx,axis=0),20//dx,axis=1);dep=ar.astype('float32').copy();dep.data=np.flipud(z).copy();dep.raster.set_nodata(np.nan);sf.set_grid(dep,name='dep')
 sf.setup_config(indexfile='sfincs.ind',mskfile='sfincs.msk',inputformat='bin',tref='20200101 000000',tstart='20200101 000000',tstop='20200101 040000',zsini=-100.,qinf=0.,dtout=300.,dthisout=.4,dtmaxout=14400.,dtmax=1.,alpha=.4,huthresh=.01,hmin_cfl=.1,advection=1,viscosity=1,coriolis=0,baro=0,storehsubgrid=1,storehmean=1,storezvolume=1,storevel=1,storevelmax=1,storecumprcp=1,timestep_analysis=1,outputformat='net')
 sf.write_grid();sf.write_config();print('BUILD',folder.name,flush=True)
 sf.setup_subgrid(datasets_dep=[{'elevtn':str(P/'hydraulic_obstacle_0_50m.tif' if urban else OLD/'terrain_sampling_0_50m.tif'),'reproj_method':'nearest'}],datasets_rgh=[{'manning':str(P/'manning_0_50m.tif')}] if urban else [],nr_subgrid_pixels=int(dx/.5),nrmax=1600,nlevels=64,max_gradient=99999.,manning_land=.04,manning_sea=.04,huthresh=.01,write_dep_tif=False)
 sf.write_subgrid();sf.write();sections(m,dx,folder);by,bx=np.where(m==1);ii=len(by)//2;(folder/'sfincs.obs').write_text(f'{l+(bx[ii]+.5)*dx:.2f} {t-(by[ii]+.5)*dx:.2f} "diagnostic_land"\n');sf.setup_config(obsfile='sfincs.obs',crsfile='sfincs.crs');sf.write_config()
 dump(folder/'builder_complete.json',{'dx':dx,'urban':urban,'source_sha256':template_source_hashes(dx,urban),'subgrid_pixels':int(dx/.5),'nlevels':64,'versions':{p:importlib.metadata.version(p) for p in ['hydromt-sfincs','hydromt','pandas','numpy','rasterio']},'input_ground_aux_is_used_by_engine':False})
 print('BUILT',folder.name,flush=True)

def build(dx,urban=True):
 import fcntl
 folder=paths(dx,urban);folder.mkdir(exist_ok=True)
 with (folder/'build.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  _build(dx,urban)

def _run(dx,urban=True,names=None):
 template=paths(dx,urban);c=contract(dx);pts=boundary(dx);loc=gpd.GeoDataFrame(index=pts.index,geometry=[Point(x,y) for x,y in zip(pts.x,pts.y)],crs=6677);spec=json.loads((BASE/'model_spec.json').read_text());l,b,r,t=c['bounds']
 for name,rain,rise in [q for q in (CASES if urban else CASES[:1]) if names is None or q[0] in names]:
  folder=RUN/(('urban' if urban else 'terrain64')+f'_{name}_{dx}m');folder.mkdir(exist_ok=True)
  if (folder/'execution.json').exists():
   ex=json.loads((folder/'execution.json').read_text())
   if ex.get('returncode')==0 and ex.get('completion_marker'):print('Reuse completed',folder.name,flush=True);continue
  if (folder/'sfincs_map.nc').exists():raise RuntimeError('Preserve interrupted outputs before retry')
  for p in template.iterdir():
   if p.suffix in ['.ind','.msk','.nc','.obs','.inp','.crs','.dep'] and p.name not in ['sfincs_map.nc','sfincs_his.nc']:shutil.copy2(p,folder/p.name)
  shutil.copy2(template/'flux_sections.json',folder/'flux_sections.json');sf=SfincsModel(root=folder,mode='r+',write_gis=False,data_libs=[]);sf.read_config();sf.read_grid()
  tstage=pd.Timestamp('2020-01-01')+pd.to_timedelta([0,3600,5400,7200,10800,14400],unit='s');df=pd.DataFrame(pts.low_stage_m.to_numpy()[None,:]+rise*np.array([0,0,1,1,0,0])[:,None],index=tstage,columns=pts.index);sf.setup_waterlevel_forcing(timeseries=df,locations=loc,buffer=0,merge=False)
  seconds=np.array([0,3600,3601,7200,7201,14400]);tt=pd.Timestamp('2020-01-01')+pd.to_timedelta(seconds,unit='s');rate=np.array([0,0,rain,rain,0,0]);sf.setup_precip_forcing(timeseries=pd.DataFrame({'rain':rate},index=tt))
  if urban and c['roof_donors']:
   donors=c['roof_donors'];ids=np.arange(1,len(donors)+1);gl=gpd.GeoDataFrame(index=ids,geometry=[Point(l+(q['col']+.5)*dx,t-(q['row']+.5)*dx) for q in donors],crs=6677);area=np.array([q['area_m2'] for q in donors]);qdf=pd.DataFrame(rate[:,None]*area[None,:]/3.6e6,index=tt,columns=ids);sf.setup_discharge_forcing(timeseries=qdf,locations=gl,buffer=None,merge=False)
  sf.write_forcing(fmt='%.9f');sf.write_config();contract_=dict(c);contract_.pop('roof_donors');contract_.update({'name':name,'urban':urban,'rain_mm_h':rain,'stage_rise_m':rise,'original_rain_interior_volume_m3':c['original_interior_area_m2']*rain/1000,'roof_source_volume_m3':c['roof_source_area_m2']*rain/1000 if urban else 0,'nlevels':64,'synthetic':True,'active_measures':False,'dtmax':1.,'alpha':.4,'history_interval':.4,'baseline_vs_geometry_control_has_same_tables_nlevels':True});
  if not urban:contract_.update({'active_interior_area_m2':c['original_interior_area_m2'],'removed_fully_building_cells':0,'roof_source_area_m2':0,'roof_max_donor_distance_m':0.,'building_barrier_m':None,'road_manning_n':.04,'terrain_control_without_building_or_road_constraints':True})
  dump(folder/'scenario_contract.json',contract_)
  inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.suffix in ['.inp','.ind','.msk','.nc','.obs','.crs','.dep','.bnd','.bzs','.precip','.src','.dis']};command=['/usr/local/bin/docker','run','--rm','--platform','linux/amd64','--network','none','--cpus',os.environ.get('UDT_SFINCS_THREADS','4'),'--memory','4g','-e','OMP_NUM_THREADS='+os.environ.get('UDT_SFINCS_THREADS','4'),'--mount',f'type=bind,src={folder},dst=/data',spec['solver']['docker_image']]
  print('RUN',folder.name,flush=True);start=time.monotonic()
  with (folder/'solver_stdout.txt').open('w') as log:res=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
  log=(folder/'solver_stdout.txt').read_text(errors='replace');finished='Simulation finished' in log or 'Simulation is finished' in log;dump(folder/'execution.json',{'returncode':res.returncode,'completion_marker':finished,'runtime_seconds':time.monotonic()-start,'command':command,'input_sha256':inputs,'official_solver':spec['solver']})
  if res.returncode or not finished:raise RuntimeError('Solver failed; inspect '+str(folder))
  print('DONE',folder.name,round(time.monotonic()-start,1),flush=True)
def run(dx,urban=True,names=None):
 import fcntl
 for name,rain,rise in (CASES if urban else CASES[:1]):
  if names is not None and name not in names:continue
  folder=RUN/(('urban' if urban else 'terrain64')+f'_{name}_{dx}m');folder.mkdir(exist_ok=True)
  with (folder/'run.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   _run(dx,urban,[name])

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','run','all']);ap.add_argument('--dx',type=int,choices=[10,20],default=20);ap.add_argument('--terrain-control',action='store_true');ap.add_argument('--case',choices=[q[0] for q in CASES]);a=ap.parse_args();urban=not a.terrain_control
 if not urban and a.dx!=20:raise ValueError('The matching pure-terrain control is20m only')
 if a.action in ['build','all']:build(a.dx,urban)
 if a.action in ['run','all']:
  if a.dx==10 and urban:
   from concurrent.futures import ProcessPoolExecutor
   with ProcessPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(run,a.dx,urban,[case[0]]) for case in CASES if a.case is None or case[0]==a.case]
    for future in futures:future.result()
  else:run(a.dx,urban,[a.case] if a.case else None)
