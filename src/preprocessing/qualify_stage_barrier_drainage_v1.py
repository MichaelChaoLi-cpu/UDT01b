"""Production interfaces,unchanged no-op replay and opt-in full-grid short control."""
import os,sys,json,shutil,subprocess,datetime as dt,time
from pathlib import Path
import numpy as np,pandas as pd
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_barrier_drainage_inputs_v1 import P,R,NEW,OLD,BASEIMAGE
from observed_load_common_v1 import sha,dump
TAG='udt01b/sfincs-cpu:barrier-drainage-v1';PYTHON='/private/tmp/udt01b-sfincs-builder/bin/python'
def docker(*args):return subprocess.run(['docker',*args],capture_output=True,text=True,check=True)
def clockstop(text,seconds):
 values={k.strip():v.strip()for k,v in(line.split('=',1)for line in text.splitlines()if'='in line)};start=dt.datetime.strptime(values['tstart'].strip(),'%Y%m%d %H%M%S');stop=start+dt.timedelta(seconds=seconds);out=[]
 for line in text.splitlines():
  k=line.split('=',1)[0].strip()
  if k=='tstop':line='tstop = '+stop.strftime('%Y%m%d %H%M%S')
  if k=='dtmaxout':line='dtmaxout = '+str(seconds)
  out.append(line)
 return '\n'.join(out)+'\n'
def baselines():
 result={}
 for folder in [OLD,NEW]:
  for run in json.loads((folder/'run_manifest.json').read_text())['runs']:
   if run['case_id']in['E20250911__H__U','E20250911__L__U','E20240618__H__U','E20240618__L__U']:result[run['case_id']]=run
 assert len(result)==4;return result

def stage(src,case,measure,image,short=0):
 folder=P/('qualification'if short else 'runs')/case;folder.mkdir(parents=True,exist_ok=False);s=ROOT/src['folder']
 for n in src['input_sha256']:
  if n=='sfincs_subgrid.nc':continue
  shutil.copy2(s/n,folder/n)
 (folder/'sfincs_subgrid.nc').symlink_to(src['large_readonly_source'])
 text=(folder/'sfincs.inp').read_text()
 if measure in ['barrier','combined']:
  shutil.copy2(P/'measures/barriers.weir',folder/'barriers.weir');text+='weirfile = barriers.weir\n'
 if measure in ['drainage','combined']:shutil.copy2(P/'measures/udt_pumps.dat',folder/'udt_pumps.dat')
 if short:text=clockstop(text,short)
 (folder/'sfincs.inp').write_text(text);identity=src['identity'].copy();identity.update(run_id=case+'__physical_measures_10m',split='external_physical_measures',solver_digest=image,patch_sha256=json.loads((P/'source_binding.json').read_text())['measure_patch_sha256']);names=list(src['input_sha256'])
 if measure in ['barrier','combined']:names+=['barriers.weir']
 if measure in ['drainage','combined']:names+=['udt_pumps.dat']
 identity['geometry_sha256']=sha(folder/'sfincs.inp') if measure!='none' else src['identity']['geometry_sha256'];identity['forcing_sha256']=src['identity']['forcing_sha256'];dump(folder/'udt_run_identity.json',identity);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v)for k,v in identity.items())+'\n');inputs={n:sha(folder/n)for n in names};run=src.copy();run.update(case_id=case,baseline_case=src['case_id'],measure=measure,folder=str(folder.relative_to(ROOT)),run_id=identity['run_id'],split=identity['split'],identity=identity,image_id=image,input_sha256=inputs,reused=False,horizon_s=short or 32400);return run

def command(run,name):
 folder=ROOT/run['folder'];return ['docker','run','--rm','--name',name,'--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount',f'type=bind,source={folder},target=/data','--mount',f'type=bind,source={run["large_readonly_source"]},target=/data/sfincs_subgrid.nc,readonly','--mount',f'type=bind,source={folder}/historical_rain.nc,target=/data/historical_rain.nc,readonly','--entrypoint','/usr/local/bin/sfincs',run['image_id']]

def main():
 assert not(P/'run_manifest.json').exists(),'Preserve existing staging';image=docker('image','inspect',TAG,'--format','{{.Id}}').stdout.strip();binding=json.loads((P/'source_binding.json').read_text());dump(P/'solver_binding.json',dict(image_id=image,image_tag=TAG,patch_sha256=binding['measure_patch_sha256'],source_commit=binding['source_commit'],unit_qualified=False,hydraulic_prefix_qualified=False));checks=[]
 def check(n,ok,detail=None):checks.append(dict(check=n,passed=bool(ok),detail=detail));dump(R/'solver_qualification_progress.json',dict(checks=checks));assert ok,n;print('PASS',n,flush=True)
 # Production limiter,exact rain and reader profiles in both thread settings.
 template=next(iter(baselines().values()))['identity']
 for binary in ['pump_fixture','positivity_fixture','adaptive_positivity_fixture','adaptive_missing_child_fixture','rain_integral_fixture','radar_block_rain_fixture_v1','radar_reader_fixture_v1','radar_budget_profile_fixture_v2']:
  outputs=[]
  for threads in [1,4]:
   f=P/'qualification/interfaces'/f'{binary}_{threads}'
   if(f/'execution.json').exists():
    ex=json.loads((f/'execution.json').read_text());stdout=(f/'stdout.txt').read_text();check(binary+'_'+str(threads),ex['returncode']==0 and image in ex['command']and 'PASS'in stdout);outputs.append(stdout);continue
   f.mkdir(parents=True,exist_ok=False);args=[]
   if binary=='radar_reader_fixture_v1':
    with Dataset(f/'input.nc','w')as ds:
     for k,n in [('time',4),('y',4),('x',4)]:ds.createDimension(k,n)
     v=ds.createVariable('time','f8',('time',));v.units='minutes since 2024-08-21 06:00:00';v[:]=np.array([0,1,300,32400])/60
     for k,vals in [('x',-11370+np.arange(4)*10),('y',-42450+np.arange(4)*10)]:v=ds.createVariable(k,'f8',(k,));v.units='m';v[:]=vals
     v=ds.createVariable('Precipitation','f4',('time','y','x'));v.unit='mm.hr-1';v.units='mm h-1';v[:]=np.arange(4)[:,None,None]*100+np.arange(1,17).reshape(1,4,4)
    args=['input.nc']
   if binary=='radar_budget_profile_fixture_v2':
    identity=template.copy();identity.update(run_id='measure_budget_fixture',solver_digest=image,patch_sha256=binding['measure_patch_sha256']);(f/'udt_run_identity.txt').write_text('\n'.join(k+'='+v for k,v in identity.items())+'\n');args=['block']
   cmd=['docker','run','--rm','--platform','linux/amd64','--network','none','--env',f'OMP_NUM_THREADS={threads}','--mount',f'type=bind,source={f},target=/data','--entrypoint','/opt/udt-build/'+binary,image,*args];res=subprocess.run(cmd,capture_output=True,text=True);(f/'stdout.txt').write_text(res.stdout);(f/'stderr.txt').write_text(res.stderr);dump(f/'execution.json',dict(command=cmd,returncode=res.returncode));check(binary+'_'+str(threads),res.returncode==0 and'PASS'in res.stdout);outputs.append(res.stdout)
  check(binary+'_thread_replay',outputs[0]==outputs[1])
 native=load_np(P/'measures/spatial_contract.npz');pumps=pd.read_csv(P/'measures/pump_sites.csv');ad=load_np(NEW/'motor_common_support.npz')['common_admitted'];check('all_pump_donors_admitted',ad.ravel()[native['native_grid_north_index'][pumps.source_native_cell.to_numpy()-1]].all())
 src=baselines()['E20250911__H__U'];runs=[]
 for case,arm,img in [('noop_original','none',BASEIMAGE),('noop_new','none',image),('combined_new','combined',image)]:
  run=stage(src,case,arm,img,1800);runs.append(run);folder=ROOT/run['folder'];cmd=command(run,'udt-measure-qualification-'+case);start=time.monotonic()
  with(folder/'solver_stdout.txt').open('w')as so,(folder/'solver_stderr.txt').open('w')as se:res=subprocess.run(cmd,stdout=so,stderr=se,timeout=900)
  log=(folder/'sfincs.log').read_text(errors='replace');check(case+'_completion',res.returncode==0 and any(t in log for t in['Simulation finished','Simulation completed','Closing off SFINCS']));led=pd.read_csv(folder/'udt_step_budget.csv');check(case+'_positive',led.negative_volume_sum.eq(0).all()and led.minimum_native_volume_since_initialization.min()>=0);scale=np.maximum.reduce([led.rain_applied_cumulative.to_numpy()+led.boundary_in_cumulative.to_numpy()+led.other_source_cumulative.to_numpy(),led.boundary_out_cumulative.to_numpy()+led.other_sink_cumulative.to_numpy(),np.ones(len(led))]);check(case+'_budget',bool(np.all(abs(led.budget_residual)<=np.maximum(1,.01*scale))and abs(led.budget_residual.iloc[-1])<=max(1,.001*scale[-1])));dump(folder/'execution.json',dict(returncode=res.returncode,completion_marker=True,runtime_seconds=time.monotonic()-start,command=cmd,solver_image_id=img));print('PREFIX COMPLETE',case,flush=True)
 with Dataset(ROOT/runs[0]['folder']/'sfincs_map.nc')as a,Dataset(ROOT/runs[1]['folder']/'sfincs_map.nc')as b:
  for k in ['zs','subgrid_volume','u','v']:check('no_op_exact_'+k,np.array_equal(np.ma.asarray(a[k][:]).filled(np.nan),np.ma.asarray(b[k][:]).filled(np.nan),equal_nan=True))
 a=pd.read_csv(ROOT/runs[0]['folder']/'udt_step_budget.csv');b=pd.read_csv(ROOT/runs[1]['folder']/'udt_step_budget.csv');check('no_op_budget_and_clock_exact',np.array_equal(a.to_numpy(),b.to_numpy()))
 combined=ROOT/runs[2]['folder'];trace=pd.read_csv(combined/'udt_pump_steps.csv');check('actual_pump_exercised',trace.transferred_m3.sum()>0,float(trace.transferred_m3.sum()));check('pump_capacity_and_prior_storage',bool((trace.transferred_m3<=trace.requested_m3+1e-12).all()and(trace.transferred_m3<=trace.source_prior_volume_m3+1e-12).all()));led=pd.read_csv(combined/'udt_step_budget.csv');group=trace.groupby('actual_end_time_s').transferred_m3.sum().reindex(led.actual_state_time[1:]).to_numpy();check('pump_ledger_exact',np.max(abs(np.r_[0,np.cumsum(group)]-led.other_sink_cumulative))<1e-7 and led.other_source_cumulative.eq(0).all())
 with Dataset(combined/'sfincs_his.nc')as ds:
  x=np.asarray(ds['structure_x'][:]);y=np.asarray(ds['structure_y'][:]);h=np.asarray(ds['structure_height'][:]);faces=pd.read_csv(P/'measures/barrier_faces.csv');mid=np.c_[(faces.x1+faces.x2)/2,(faces.y1+faces.y2)/2]
  from scipy.spatial import cKDTree
  dist,ix=cKDTree(mid).query(np.c_[x,y]);check('barrier_snapping_exact',len(x)==len(faces)and len(np.unique(ix))==len(faces)and dist.max()<1e-5 and np.max(abs(h-faces.crest_m.to_numpy()[ix]))<1e-4,dict(planned=len(faces),runtime=len(x)))
 check('original_sources_unchanged',all(sha(ROOT/n)==v for n,v in json.loads((P/'protected_sources.json').read_text()).items()))
 val=dict(passed=True,checks=checks,source_patch_sha256=binding['measure_patch_sha256'],image_id=image,no_op_exact_replay=True,original_quality_thresholds_unchanged=True);dump(R/'solver_qualification.json',val);dump(P/'solver_binding.json',dict(image_id=image,image_tag=TAG,patch_sha256=binding['measure_patch_sha256'],source_commit=binding['source_commit'],unit_qualified=True,hydraulic_prefix_qualified=True,no_op_exact=True))
 staged=[];queue=[]
 for event in ['E20250911','E20240618']:
  for level in ['H','L']:
   base=baselines()[event+'__'+level+'__U'];z=base.copy();z.update(case_id=base['case_id']+'__none',baseline_case=base['case_id'],measure='none',reused=True);staged.append(z)
   for arm in ['barrier','drainage','combined']:
    case=base['case_id']+'__'+arm;r=stage(base,case,arm,image);staged.append(r);queue.append(case)
 dump(P/'run_manifest.json',dict(protocol_sha256=sha(P/'protocol.json'),runs=staged,new_execution_order=queue,budget=json.loads((P/'protocol.json').read_text())['budget']));dump(R/'input_validation.json',dict(passed=True,baselines=4,new_runs=12,qualified_image=image,all_inputs_hashed=True));print('12 PHYSICAL RUNS STAGED',flush=True)
def load_np(p):return dict(np.load(p,allow_pickle=False))
if __name__=='__main__':main()
