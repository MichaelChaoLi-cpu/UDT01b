"""Qualify the production image and stage two explicitly hypothetical river stages."""
from pathlib import Path
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import json,sys,hashlib,shutil,subprocess
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_historical_radar_experiment_v1'
R=ROOT/'data/exp/estimation/three_ward_historical_radar_experiment_v1'
RAIN=ROOT/'data/processed/tokyo_historical_radar_20240821_v1'
OLD=ROOT/'data/processed/three_ward_rain_step_integral_v1'
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from load_three_ward_minimum_scenarios import load_geometry

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def qualify():
 R.mkdir(parents=True,exist_ok=True)
 image=json.loads((P/'build-metadata.json').read_text())['containerimage.digest']
 checks=[]
 def check(n,ok):checks.append({'check':n,'passed':bool(ok)});assert ok,n;print('PASS',n,flush=True)
 old_binding=json.loads((OLD/'solver_binding.json').read_text())
 fixtures=['positivity_fixture','adaptive_positivity_fixture','adaptive_missing_child_fixture','rain_integral_fixture','radar_block_rain_fixture_v1','radar_reader_fixture_v1']
 for binary in fixtures:
  for threads in [1,4]:
   folder=P/'fixtures'/(binary+'_'+str(threads));folder.mkdir(parents=True,exist_ok=True)
   args=[]
   if binary=='radar_reader_fixture_v1':
    with Dataset(folder/'input.nc','w') as nc:
     for k,n in [('time',4),('y',4),('x',4)]:nc.createDimension(k,n)
     v=nc.createVariable('time','f8',('time',));v.units='minutes since 2024-08-21 06:00:00';v[:]=np.array([0,1,300,32400])/60
     for k,data in [('x',-11370+np.arange(4)*10),('y',-42450+np.arange(4)*10)]:
      v=nc.createVariable(k,'f8',(k,));v.units='m';v[:]=data
     v=nc.createVariable('Precipitation','f4',('time','y','x'));v.unit='mm.hr-1';v.units='mm h-1';v[:]=np.arange(4)[:,None,None]*100+np.arange(1,17).reshape(1,4,4)
    args=['input.nc']
   cmd=['docker','run','--rm','--platform','linux/amd64','--network','none','--env',f'OMP_NUM_THREADS={threads}','--mount',f'type=bind,source={folder},target=/data','--entrypoint','/opt/udt-build/'+binary,image,*args]
   if (folder/'execution.json').exists():
    ex=json.loads((folder/'execution.json').read_text());stdout=(folder/'stdout.txt').read_text()
   else:
    res=subprocess.run(cmd,text=True,capture_output=True);stdout=res.stdout
    (folder/'stdout.txt').write_text(stdout);(folder/'stderr.txt').write_text(res.stderr);ex={'returncode':res.returncode,'command':cmd};dump(folder/'execution.json',ex)
   check(binary+'_'+str(threads),ex['returncode']==0 and 'PASS' in stdout)
  check(binary+'_serial_parallel_identical',(P/'fixtures'/(binary+'_1')/'stdout.txt').read_bytes()==(P/'fixtures'/(binary+'_4')/'stdout.txt').read_bytes())
  if binary in fixtures[:4]:
   key={'positivity_fixture':'regular','adaptive_positivity_fixture':'adaptive','adaptive_missing_child_fixture':'missing_child','rain_integral_fixture':'rain'}[binary]
   check(binary+'_frozen_regression',(P/'fixtures'/(binary+'_1')/'stdout.txt').read_bytes()==(OLD/'link_tests'/(key+'1')/'stdout.txt').read_bytes())
 binding=json.loads((P/'source_binding.json').read_text())
 for name in binding['preserved_modules']:check('unchanged_'+name,sha(P/'context'/name)==sha(OLD/'context'/name))
 for name,h in binding['context_sha256'].items():check('context_hash_'+name,sha(P/'context'/name)==h)
 check('increment_and_full_patch_identity',sha(P/'full_solver.patch')==binding['patch_sha256'] and sha(P/'block_rain_increment.patch')==binding['increment_patch_sha256'])
 dump(R/'solver_qualification.json',dict(all_passed=True,checks=checks,image_id=image,hydraulic_end_to_end_qualified=False))
 dump(P/'solver_binding.json',dict(image_id=image,image_tag='udt01b/sfincs-cpu:galibier-historical-radar-block-v1',patch_sha256=binding['patch_sha256'],source_commit=binding['source_commit'],unit_and_reader_qualified=True,hydraulic_end_to_end_qualified=False))
 return image


def prepare(image):
 if (P/'run_manifest.json').exists():print('Retain already staged inputs');return
 rainqa=json.loads((RAIN/'spatial_forcing_qa.json').read_text());assert rainqa['missing_values_over_catchment']==0 and rainqa['saturated_values_over_catchment']==0
 assert sha(RAIN/'forcing_intervals_10m.nc')==rainqa['output_sha256']
 old=json.loads((ROOT/'data/processed/three_ward_reference_setup_pilot_v1/qualification_manifest.json').read_text())['runs'][1]
 source=ROOT/old['folder'];sb=json.loads((P/'solver_binding.json').read_text());geo=load_geometry()
 export=P/'forcing';export.mkdir(exist_ok=False)
 knots=np.arange(109)*300
 with Dataset(RAIN/'forcing_intervals_10m.nc') as src,Dataset(export/'historical_rain.nc','w') as nc:
  assert np.array_equal(src['time_bounds'][:]-src['time_bounds'][0,0],np.column_stack([knots[:-1],knots[1:]]))
  ny,nx=1014,772
  for k,n in [('time',109),('y',ny),('x',nx)]:nc.createDimension(k,n)
  v=nc.createVariable('time','f8',('time',));v.units='minutes since 2024-08-21 06:00:00';v[:]=knots/60
  for k,data in [('x',-11370+np.arange(nx)*10),('y',-42450+np.arange(ny)*10)]:
   v=nc.createVariable(k,'f8',(k,));v.units='m';v[:]=data
  v=nc.createVariable('Precipitation','f4',('time','y','x'),fill_value=-9999,zlib=True,complevel=2,chunksizes=(1,128,128));v.unit='mm.hr-1';v.units='mm h-1'
  for i in range(108):v[i]=np.flipud(np.pad(src['urban_effective_rainfall'][i].data,1))
  v[108]=np.zeros((ny,nx),'f4')
  nc.timezone='All model reference dates use UTC. 06:00 UTC corresponds to 15:00 JST.'
  nc.coordinate_convention='Galibier lower-left corner convention; reader adds half cell; input rows south-first.'
  nc.interval_convention='Record k applies on [time_k,time_k+300); original END-stamped radar representative intensity shifted to interval START without smoothing or resampling.'
  nc.source_sha256=sha(RAIN/'forcing_intervals_10m.nc');nc.required_solver_amprblock=1
  nc.no_rain_after_source_window='The last timestamp at32400s is a zero-valued sentinel, not another observed interval.'
 # Native file geometry and all108 recipient fields must be exactly preserved.
 with Dataset(export/'historical_rain.nc') as dst,Dataset(RAIN/'forcing_intervals_10m.nc') as src:
  for i in range(108):assert np.array_equal(np.flipud(dst['Precipitation'][i])[1:-1,1:-1],src['urban_effective_rainfall'][i])
  assert np.all(dst['Precipitation'][108]==0)
 stage_times=np.array([0,10800,12600,14400,18000,32400],float)
 increment=np.array([0,0,1.5,1.5,0,0],float)
 runs=[]
 for case in ['radar_low','radar_raised']:
  dest=P/'runs'/case;dest.mkdir(parents=True,exist_ok=False)
  for name,h in old['input_sha256'].items():
   assert sha(source/name)==h,(name,'Frozen geometry/input changed')
   if name in ['roof_surface_precip.nc','sfincs.bzs','sfincs.inp']:continue
   if name=='sfincs_subgrid.nc':(dest/name).hardlink_to(source/name)
   else:shutil.copyfile(source/name,dest/name)
  (dest/'historical_rain.nc').hardlink_to(export/'historical_rain.nc')
  config=(source/'sfincs.inp').read_text()
  changes={'tref':'20240821 060000','tstart':'20240821 060000','tstop':'20240821 150000','dtmaxout':'32400.0','dthisout':'0.0001','netamprfile':'historical_rain.nc','ampr_block':'1','amprblock':'1'}
  lines=[]
  for line in config.splitlines():
   key=line.partition('=')[0].strip()
   lines.append(f'{key:25s}= {changes[key]}' if key in changes else line)
  (dest/'sfincs.inp').write_text('\n'.join(lines)+'\n')
  inc=increment if case=='radar_raised' else np.zeros_like(increment)
  stage=geo['Boundary Low Stage'][None,:]+inc[:,None]
  np.savetxt(dest/'sfincs.bzs',np.column_stack([stage_times,stage]),fmt='%.10f')
  geometry_sha=old['identity']['geometry_sha256']
  forcing_sha=hashlib.sha256((sha(dest/'historical_rain.nc')+sha(dest/'sfincs.bzs')).encode()).hexdigest()
  ident=dict(old['identity']);ident.update(run_id=case+'__hist20240821_10m',parent_id='HIST-RADAR-TOKYO-20240821',family_id='HIST-RADAR-TOKYO-20240821',split='external_historical_candidate',forcing_sha256=forcing_sha,solver_digest=image,patch_sha256=sb['patch_sha256'])
  dump(dest/'udt_run_identity.json',ident);(dest/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in ident.items())+'\n')
  names=list(old['input_sha256']);names=[n for n in names if n!='roof_surface_precip.nc']+['historical_rain.nc','udt_run_identity.txt']
  run=dict(case_id=case,run_id=ident['run_id'],folder=str(dest.relative_to(ROOT)),identity=ident,image_id=image,input_sha256={n:sha(dest/n) for n in names},
    large_readonly_source=old['large_readonly_source'],horizon_s=32400,dtout_s=300,split=ident['split'],stage_times_s=stage_times.tolist(),stage_increment_m=inc.tolist(),
    measured_rain=True,observed_river_stage=False,formal_library_member=False,training=False,sealed_test=False)
  runs.append(run)
 contract=dict(event='2024-08-21 Tokyo',parent_id='HIST-RADAR-TOKYO-20240821',same_parent_for_two_boundary_descendants=True,
  no_formal_split_assignment=True,reference_policy_source='src/preprocessing/reference_label_policy.py',reference_policy_sha256=sha(ROOT/'src/preprocessing/reference_label_policy.py'),
  numerical_tolerances_and_ground_mask_unchanged=True,
  explicit_experiment_overrides={'horizon_s':32400,'map_records':109,'map_interval_s':300,'history_request_interval_s':0.0001,'history_analysis_clock':'Join each record to the actual output-clock ledger, not nominal schedule','amprblock':1,'rain_time_reference_utc':'2024-08-21T06:00:00Z','output_scope':'historical rainfall, scenario river stages; not official four-hour training-library release'},
  stage_hypothesis={'low':'Existing per-boundary-point low-stage assumptions, NOT observed event hydrograph.',
    'raised':'Add common triangular/trapezoidal0→1.5m rise18:00–18:30JST, plateau18:30–19:00, fall19:00–20:00 to all7 existing water groups; zero outside. Same amplitude/durations as legacy pressure control, shifted+2hours to coincide with storm core. Not actual tide or operational river management.'},
  drainage_hypothesis='No pipes, infiltration or pumps. All roof runoff sent instantly to surface using frozen source-recipient map. Potential surface accumulation under this assumption; not a calibrated real-event reconstruction.',
  output_limits='Peak ground mean depth bounds,300s snapshots and all-step boundary exchanges; no certified onset/duration/particle pathways or affected population.',
  maximum_full_runs=2,low_run_must_pass_before_raised=True,retries='Never restart an incomplete/failed hydraulic attempt automatically.',
  rain_input_sha256=sha(RAIN/'forcing_intervals_10m.nc'),roof_map_sha256=json.loads((RAIN/'source_recipient_map_contract.json').read_text())['matrix_sha256'])
 dump(P/'experiment_contract.json',contract);dump(P/'run_manifest.json',dict(runs=runs,maximum_runs=2,training=False,sealed_test=False))
 protected={}
 for rel in ['src/preprocessing/reference_label_policy.py','data/processed/three_ward_common_depth_labels/common_support_10m.npz','data/processed/three_ward_minimum_scenarios_v1/catalog.json','data/processed/three_ward_minimum_scenarios_v1/run_manifest.json','data/processed/tokyo_historical_radar_20240821_v1/radar_intervals.npz','data/processed/tokyo_historical_radar_20240821_v1/forcing_intervals_10m.nc','data/processed/three_ward_reference_expansion_12_4_v1/selection.json']:
  p=ROOT/rel
  if p.exists():protected[rel]=sha(p)
 for p in (ROOT/'data/processed/three_ward_minimum_scenarios_v1').rglob('*'):
  if p.is_file():protected[str(p.relative_to(ROOT))]=sha(p)
 dump(P/'protected_sources.json',protected)
 print('STAGED exactly2 historical rain / scenario stage runs; inputs byte-verified',flush=True)

if __name__=='__main__':prepare(qualify())
