"""Pinned official SFINCS runs for matched whole-domain connectivity comparison."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,time
import numpy as np
import netCDF4
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'data/processed/three_ward_sfincs_framework';RUN=BASE/'adaptive_connectivity_comparison';PREVIOUS=BASE/'fixed_runoff_comparison';P=ROOT/'data/processed/three_ward_adaptive_connectivity'
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def config(path,updates,remove=()):
 d={}
 for line in path.read_text().splitlines():
  if '=' in line:
   k,v=line.split('=',1);d[k.strip()]=v.strip()
 for k in remove:d.pop(k,None)
 d.update({k:str(v) for k,v in updates.items()});path.write_text(''.join(f'{k:24s} = {v}\n' for k,v in d.items()))
def run(case,pilot=False):
 import fcntl
 folder=RUN/(('pilot_' if pilot else 'rain60_low_')+case);folder.mkdir(exist_ok=True)
 with (folder/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (folder/'execution.json').exists():
   e=json.loads((folder/'execution.json').read_text());assert e['returncode']==0 and e['completion_marker'];print('Reuse',folder.name,flush=True);return
  assert not (folder/'sfincs_map.nc').exists(),'Preserve interrupted outputs before rerunning'
  template=RUN/(case+'_template');assert (template/'builder_complete.json').exists()
  if not pilot:assert json.loads((RUN/('pilot_'+case)/'forcing_validation.json').read_text())['passed']
  for p in template.iterdir():
   if p.suffix in ['.ind','.msk','.nc','.obs','.inp','.crs','.dep','.thd'] or p.name=='flux_sections.json':shutil.copy2(p,folder/p.name)
  if case=='corrected_10m':
   previous=PREVIOUS/('pilot_10m' if pilot else 'rain60_low_storage253_10m')
   for name in ['roof_surface_precip.nc','sfincs.bnd','sfincs.bzs']:shutil.copy2(previous/name,folder/name)
   updates={'netamprfile':'roof_surface_precip.nc','bndfile':'sfincs.bnd','bzsfile':'sfincs.bzs','ampr_block':0,'dtwnd':1.,'tstop':'20200101 000100' if pilot else '20200101 040000','dtout':60. if pilot else 300.,'dtmaxout':60. if pilot else 14400.}
   config(folder/'sfincs.inp',updates,remove=['precipfile','srcfile','disfile'])
   sc=json.loads((previous/'scenario_contract.json').read_text());sc.update({'connectivity_corrected':True,'case':case,'nlevels':253,'synthetic':True,'active_measures':False,'same_raw_terrain':True})
  else:
   from prepare_three_ward_adaptive_forcing import forcing
   sc=forcing(folder,pilot)
  dump(folder/'scenario_contract.json',sc)
  inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix in ['.inp','.ind','.msk','.nc','.obs','.crs','.dep','.bnd','.bzs','.thd']}
  spec=json.loads((BASE/'model_spec.json').read_text());command=['/usr/local/bin/docker','run','--rm','--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','-e','OMP_NUM_THREADS=6','--mount',f'type=bind,src={folder},dst=/data',spec['solver']['docker_image']]
  # Serialize only this project experiment's official-engine containers.
  announced=False
  while True:
   ids=subprocess.check_output(['/usr/local/bin/docker','ps','-q'],text=True).split();busy=False
   if ids:
    active=json.loads(subprocess.check_output(['/usr/local/bin/docker','inspect',*ids],text=True))
    busy=any(any(str(q.get('Source','')).startswith(str(RUN)) for q in a.get('Mounts',[])) for a in active)
   if not busy:break
   if not announced:print('WAIT for existing project solver to release memory',flush=True);announced=True
   time.sleep(15)
  start=time.monotonic();print('RUN',folder.name,flush=True)
  with (folder/'solver_stdout.txt').open('w') as log:res=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
  log=(folder/'solver_stdout.txt').read_text(errors='replace');finished='Simulation finished' in log or 'Simulation is finished' in log
  ex={'returncode':res.returncode,'completion_marker':finished,'runtime_seconds':time.monotonic()-start,'command':command,'input_sha256':inputs,'official_solver':spec['solver']};dump(folder/'execution.json',ex);assert res.returncode==0 and finished,log[-1200:]
  print('DONE',folder.name,round(ex['runtime_seconds'],1),flush=True)
  if pilot:
   with netCDF4.Dataset(folder/'sfincs_map.nc') as d:
    if case=='corrected_10m':
     observed=np.flipud(np.ma.asarray(d['cumprcp'][-1]).filled(0));m=np.flipud(np.asarray(d['msk'][:]));import rasterio
     with rasterio.open(ROOT/'data/processed/three_ward_fixed_runoff/contributing_area_10m.tif') as r:a=r.read(1)
     expected=a/100*.001;sel=m==1;engine=float(observed[sel].sum(dtype=np.float64)*100);target=float(a.sum()*.001);error=float(np.max(abs(observed[sel]-expected[sel])))
    else:
     with np.load(P/'adaptive_cell_geometry.npz') as q:m=q['mask'];area=q['size_m']**2;contribution=q['contributing_area_m2']
     observed=np.asarray(np.ma.asarray(d['cumprcp'][-1]).filled(0)).ravel();sel=m==1;expected=contribution/area*.001;assert len(observed)==len(m);engine=float(np.dot(observed[sel],area[sel]));target=float(contribution[sel].sum()*.001);error=float(np.max(abs(observed[sel]-expected[sel])))
    status='pending'
   with netCDF4.Dataset(folder/'sfincs_his.nc') as d:status=float(np.asarray(d['status'][:]).ravel()[0])
   record={'passed':error<1e-6 and abs(engine-target)/target<1e-5 and status==0,'max_cell_precip_depth_error_m':error,'engine_precip_volume_m3':engine,'expected_precip_volume_m3':target,'relative_volume_error':abs(engine-target)/target,'engine_status':status};dump(folder/'forcing_validation.json',record);assert record['passed'],record;print('PILOT',record,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('case',choices=['corrected_10m','adaptive_10_5m']);ap.add_argument('--pilot',action='store_true');a=ap.parse_args();run(a.case,a.pilot)
