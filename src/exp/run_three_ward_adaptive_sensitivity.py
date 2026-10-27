"""Reuse immutable corrected geometries for matched rain/river-stage scenarios."""
from pathlib import Path
import json,hashlib,os,shutil,subprocess,time,fcntl,argparse
import numpy as np
import netCDF4
from run_three_ward_adaptive_connectivity import config,dump
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/processed/three_ward_sfincs_framework'
SOURCE=BASE/'adaptive_connectivity_comparison'
RUN=BASE/'adaptive_sensitivity'
CASES={'rain100_low':(100.,0.),'rain60_raised':(60.,1.5),'dry_raised':(0.,1.5)}
GRIDS=['corrected_10m','adaptive_10_5m']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def prepare(name,grid):
 rain,rise=CASES[name];source=SOURCE/('rain60_low_'+grid);folder=RUN/(name+'_'+grid)
 folder.mkdir(parents=True,exist_ok=True)
 assert json.loads((source/'execution.json').read_text())['completion_marker']
 if (folder/'forcing_validation.json').exists():
  assert json.loads((folder/'forcing_validation.json').read_text())['passed'];return folder
 assert not (folder/'sfincs_map.nc').exists()
 ex=json.loads((source/'execution.json').read_text());identical={}
 for filename,sha in ex['input_sha256'].items():
  p=source/filename;dest=folder/filename
  assert digest(p)==sha
  if filename in ['sfincs.inp','sfincs.bzs','roof_surface_precip.nc']:shutil.copy2(p,dest)
  else:
   if not dest.exists():os.link(p,dest)
   assert digest(dest)==sha;identical[filename]=sha
 shutil.copy2(source/'flux_sections.json',folder/'flux_sections.json')
 with netCDF4.Dataset(folder/'roof_surface_precip.nc','r+') as d:
  d.set_auto_mask(False)
  for i in range(len(d['time'])):d['Precipitation'][i]=d['Precipitation'][i]*(rain/60.)
  d.setncattr('scenario_note','Synthetic sensitivity; original spatial roof weights unchanged')
 stages=np.loadtxt(source/'sfincs.bzs');assert np.allclose(stages[:,1:],stages[0,1:])
 stage_times=np.array([0,3600,5400,7200,10800,14400],float)
 written=np.c_[stage_times,stages[0,1:][None,:]+rise*np.array([0,0,1,1,0,0])[:,None]]
 np.savetxt(folder/'sfincs.bzs',written,fmt='%.9f')
 assert np.allclose(written[:,1:]-stages[0,1:],rise*np.array([0,0,1,1,0,0])[:,None],atol=1e-8)
 sc=json.loads((source/'scenario_contract.json').read_text())
 sc.update(name=name,rain_mm_h=rain,stage_rise_m=rise,effective_rain_depth_mm=rain,original_rain_interior_volume_m3=38976800*rain/1000,geometry_source=str(source.relative_to(ROOT)),geometry_inputs_identical_sha256=identical,river_stage_times_s=stage_times.tolist(),river_stage_increment_m=(rise*np.array([0,0,1,1,0,0])).tolist(),rain_and_roof_allocation_matched_to_baseline=True)
 dump(folder/'scenario_contract.json',sc)
 # Rewritten forcing must retain every spatial weight, coordinate and time.
 with netCDF4.Dataset(source/'roof_surface_precip.nc') as a,netCDF4.Dataset(folder/'roof_surface_precip.nc') as b:
  a.set_auto_mask(False);b.set_auto_mask(False)
  for key in ['time','x','y']:assert np.array_equal(a[key][:],b[key][:])
  maxerror=max(float(np.max(abs(b['Precipitation'][i][:]-a['Precipitation'][i][:]*(rain/60.)))) for i in range(len(a['time'])))
  assert maxerror==0
 dump(folder/'forcing_validation.json',{'passed':True,'baseline_every_cell_pilot_previously_passed':True,'spatial_weight_scaling_max_error_mm_h':maxerror,'river_stage_profile_verified':True,'geometry_unchanged':True})
 return folder
def run(name,grid):
 folder=prepare(name,grid)
 if (folder/'execution.json').exists():
  e=json.loads((folder/'execution.json').read_text());assert e['returncode']==0 and e['completion_marker'];print('REUSE',folder.name,flush=True);return
 assert not (folder/'sfincs_map.nc').exists(),'Preserve interrupted output before retrying'
 inputs={p.name:digest(p) for p in folder.iterdir() if p.suffix in ['.inp','.ind','.msk','.nc','.obs','.crs','.dep','.bnd','.bzs','.thd']}
 spec=json.loads((BASE/'model_spec.json').read_text());command=['/usr/local/bin/docker','run','--rm','--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','-e','OMP_NUM_THREADS=6','--mount',f'type=bind,src={folder},dst=/data',spec['solver']['docker_image']]
 while True:
  ids=subprocess.check_output(['/usr/local/bin/docker','ps','-q'],text=True).split()
  active=json.loads(subprocess.check_output(['/usr/local/bin/docker','inspect',*ids],text=True)) if ids else []
  if not any(any(str(m.get('Source','')).startswith(str(BASE)) for m in a.get('Mounts',[])) for a in active):break
  time.sleep(15)
 start=time.monotonic();print('RUN',folder.name,flush=True)
 with (folder/'solver_stdout.txt').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
 log=(folder/'solver_stdout.txt').read_text(errors='replace');finished='Simulation finished' in log or 'Simulation is finished' in log
 e={'returncode':r.returncode,'completion_marker':finished,'runtime_seconds':time.monotonic()-start,'command':command,'input_sha256':inputs,'official_solver':spec['solver']};dump(folder/'execution.json',e)
 assert r.returncode==0 and finished,log[-2000:]
 print('DONE',folder.name,round(e['runtime_seconds'],1),flush=True)
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'batch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for name in CASES:
   for grid in GRIDS:run(name,grid)
if __name__=='__main__':main()
