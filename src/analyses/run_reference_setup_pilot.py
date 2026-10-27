"""Execute the authorized finite stage; at most two CPU processes with measured memory headroom."""
from pathlib import Path
import sys,json,time,subprocess,concurrent.futures,argparse,re
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_reference_setup_pilot import OUT,REPORT,CONTROLS,PILOT
from prepare_sfincs_cpu_build import dump

def execute(run):
 p=ROOT/run['folder'];name='udt01b-refpilot-'+run['case_id'].replace('_','-')
 if(p/'execution.json').exists():
  e=json.loads((p/'execution.json').read_text());assert e['returncode']==0 and e['completion_marker'];print('RETAIN',run['case_id'],flush=True);return
 assert not(p/'sfincs.log').exists(),'Preserve incomplete attempt'
 cmd=['docker','run','--rm','--name',name,'--platform','linux/amd64','--network','none','--cpus','4','--memory','7g','--env','OMP_NUM_THREADS=4','--mount',f'type=bind,source={p},target=/data','--mount','type=bind,source='+run['large_readonly_source']+',target=/data/sfincs_subgrid.nc,readonly','--entrypoint','/usr/local/bin/sfincs',run['image_id']]
 dump(p/'launch.json',{'command':cmd,'authorization':'MILI-D-20261015-017: two legacy controls then conditional eight regular-grid pilot inputs; no training, sealed or remaining runs','start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});print('START',run['phase'],run['case_id'],flush=True);t=time.monotonic()
 with(p/'solver_stdout.txt').open('w')as so,(p/'solver_stderr.txt').open('w')as se:r=subprocess.run(cmd,stdout=so,stderr=se)
 log=(p/'sfincs.log').read_text(errors='replace')if(p/'sfincs.log').exists()else'';done=any(s in log for s in ['Simulation finished','Simulation completed','Closing off SFINCS'])
 dump(p/'execution.json',{'returncode':r.returncode,'completion_marker':done,'runtime_seconds':time.monotonic()-t,'command':cmd,'input_sha256':run['input_sha256'],'solver_image_id':run['image_id'],'solver_image_digest':run['image_digest'],'phase':run['phase'],'new_parent_event':run['new_parent_event'],'library_descendant':run['library_descendant'],'training':False,'sealed_response':False});print('FINISH',run['case_id'],r.returncode,done,round(time.monotonic()-t,1),flush=True);assert r.returncode==0 and done,'No automatic rerun'

def memory_used():
 r=subprocess.run(['docker','stats','--no-stream','--format','{{.MemUsage}}'],capture_output=True,text=True,check=True);values=[]
 for line in r.stdout.splitlines():
  s=line.split('/')[0].strip();m=re.fullmatch(r'([0-9.]+)([A-Za-z]+)',s)
  if not m:return None
  factor={'GiB':2**30,'MiB':2**20,'KiB':2**10,'B':1,'GB':10**9,'MB':10**6}.get(m[2]);
  if factor is None:return None
  values.append(float(m[1])*factor)
 return values

def main(phase):
 out=ROOT/OUT;report=ROOT/REPORT;manifest=json.loads((out/(phase+'_manifest.json')).read_text());runs=manifest['runs'];expected=CONTROLS if phase=='qualification'else PILOT;assert[r['case_id']for r in runs]==expected
 if phase=='pilot':
  for c in CONTROLS:
   a=json.loads((report/'qualification'/c/'audit.json').read_text());assert a['all_checks_passed']and a['admission']['technical_reference']
 queue=list(runs);active={};telemetry=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
  while queue or active:
   for future,run in list(active.items()):
    if future.done():future.result();del active[future]
   if queue and len(active)<2:
    ready=not active
    if active and all((ROOT/r['folder']/'udt_step_budget.csv').exists()for r in active.values()):
     used=memory_used()
     if used is not None:
      projected=sum(used)+max(3.0*2**30,max(used,default=0)+.3*2**30);ready=projected<6.4*2**30
      telemetry.append({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'docker_used_bytes':used,'projected_bytes':projected,'parallel_launch_allowed':ready})
    if ready:
     r=queue.pop(0);active[pool.submit(execute,r)]=r
   dump(report/(phase+'_execution_progress.json'),{'active':[r['case_id']for r in active.values()],'queued':[r['case_id']for r in queue],'completed':[r['case_id']for r in runs if(ROOT/r['folder']/'execution.json').exists()],'max_parallel':2,'memory_launch_budget_GiB':6.4,'training':False})
   if queue or active:time.sleep(10)
 dump(report/(phase+'_resource_decisions.json'),{'launch_checks':telemetry,'max_parallel':2,'training':False});print('PHASE COMPLETE',phase,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['qualification','pilot']);main(p.parse_args().phase)
