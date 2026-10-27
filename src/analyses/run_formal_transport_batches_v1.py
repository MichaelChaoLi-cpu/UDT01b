"""Finite audited serial campaign; no retry, no threshold relaxation, pause-aware caps."""
from pathlib import Path
import sys,os,json,time,subprocess,shutil,csv,math,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_formal_transport_experiments_v1 import P,R,D,IMAGE,BUILDER,load,read,dump,sha

def atomic(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def docker(*args,check=True):return subprocess.run(['docker',*args],capture_output=True,text=True,check=check)
def tail_record(path):
 if not path.exists():return None
 with path.open('rb') as f:
  header=f.readline().decode().strip();f.seek(0,2);n=f.tell();f.seek(max(0,n-131072));tail=f.read().decode(errors='replace')
 if not tail.endswith('\n'):tail=tail.rsplit('\n',1)[0]
 names=[x.strip() for x in header.split(',')]
 for line in reversed(tail.splitlines()):
  vals=line.split(',')
  if len(vals)==len(names):
   try:return {k:float(v.strip()) for k,v in zip(names,vals)}
   except ValueError:continue
 return None
def output_bytes(manifest):
 total=0
 for r in manifest['runs']:
  if r['reused']:continue
  f=ROOT/r['folder'];inputs=set(r['input_sha256'])|{'udt_run_identity.json'}
  for p in f.iterdir():
   if p.is_file() and p.name not in inputs:total+=p.stat().st_size
 return total

def main():
 manifest=json.loads((P/'run_manifest.json').read_text());protocol=json.loads((D/'protocol.json').read_text());budget=manifest['budget'];assert sha(D/'protocol.json')==manifest['protocol_sha256'];assert json.loads((R/'input_validation.json').read_text())['all_passed'];assert json.loads((R/'plan_validation.json').read_text())['all_passed']
 if (R/'campaign_state.json').exists():raise RuntimeError('Campaign state already exists; no automatic restart')
 assert docker('image','inspect',IMAGE,'--format','{{.Id}}').stdout.strip()==IMAGE
 sources=json.loads((P/'protected_sources.json').read_text());codepaths=[ROOT/'src/analyses'/n for n in ['run_formal_transport_batches_v1.py','audit_formal_transport_run_v1.py','report_formal_transport_matrix_v1.py']]+[ROOT/'src/preprocessing/prepare_formal_transport_experiments_v1.py'];codebind={str(p.relative_to(ROOT)):sha(p) for p in codepaths};dump(P/'execution_code_binding.json',codebind)
 runby={r['case_id']:r for r in manifest['runs']};state=dict(status='running',campaign_pid=os.getpid(),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),current_case=None,current_batch=None,completed_new_cases=[],reused_cases=protocol['reused_cases'],queue=manifest['new_execution_order'],active_wall_seconds=0.,completed_batches=[],hydraulic_cases_started=0,budget=budget,training=False,sealed_tests_opened=False,people_counted=False)
 spent=0.;lasttick=time.monotonic();container=None;child=None
 def save(**extra):
  state.update(extra);atomic(R/'campaign_state.json',state)
 def report():
  with (R/'live_report.log').open('a') as f:subprocess.run([BUILDER,str(ROOT/'src/analyses/report_formal_transport_matrix_v1.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
 def resource_check():
  if state['active_wall_seconds']>=budget['maximum_campaign_active_wall_s']:raise RuntimeError('Campaign active-wall budget exhausted')
  if shutil.disk_usage(P).free<budget['minimum_free_gib']*1024**3:raise RuntimeError('Minimum free-space gate failed')
  if output_bytes(manifest)>budget['maximum_new_output_gib']*1024**3:raise RuntimeError('New-output storage budget exhausted')
  if (P/'STOP').exists():raise RuntimeError('User STOP flag detected')
 try:
  save();report()
  for batch in protocol['batches']:
   save(current_batch=batch['batch_id'])
   for case in batch['new_cases']:
    resource_check();assert all(sha(ROOT/p)==h for p,h in codebind.items()),'Execution code changed';assert all(sha(ROOT/p)==h for p,h in sources.items()),'Protected source changed'
    r=runby[case];f=ROOT/r['folder'];assert not any((f/n).exists() for n in ['launch.json','execution.json','sfincs.log','sfincs_map.nc']),'Preserve previous attempt; no automatic restart';assert all(sha(f/n)==h for n,h in r['input_sha256'].items())
    container='udt01b-formal-'+case.replace('__','-').lower();cmd=['docker','run','--rm','--name',container,'--platform','linux/amd64','--network','none','--cpus',str(budget['cpus']),'--memory',str(budget['memory_gib'])+'g','--env','OMP_NUM_THREADS='+str(budget['cpus']),'--mount',f'type=bind,source={f},target=/data','--mount',f"type=bind,source={r['large_readonly_source']},target=/data/sfincs_subgrid.nc,readonly",'--mount',f'type=bind,source={f}/historical_rain.nc,target=/data/historical_rain.nc,readonly','--entrypoint','/usr/local/bin/sfincs',IMAGE]
    start=time.monotonic();runactive=0.;stalled=0.;last_state=None;paused=False;lasttick=time.monotonic();save(current_case=case,case_active_wall_seconds=0.,latest_native_time_s=None,hydraulic_cases_started=state['hydraulic_cases_started']+1)
    dump(f/'launch.json',dict(command=cmd,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=manifest['protocol_sha256'],budget=budget));print('START',batch['batch_id'],case,flush=True)
    so=(f/'solver_stdout.txt').open('w');se=(f/'solver_stderr.txt').open('w');child=subprocess.Popen(cmd,cwd=f,stdout=so,stderr=se);report()
    while child.poll() is None:
     time.sleep(10);now=time.monotonic();elapsed=now-lasttick;lasttick=now
     if (P/'PAUSE').exists():
      if not paused:docker('pause',container);paused=True;save(status='paused_by_user');report()
      if (P/'STOP').exists():raise RuntimeError('User STOP while paused')
      continue
     if paused:docker('unpause',container);paused=False;save(status='running');report();continue
     runactive+=elapsed;state['active_wall_seconds']+=elapsed;save(case_active_wall_seconds=runactive);resource_check()
     if runactive>budget['maximum_run_active_wall_s']:raise RuntimeError('Per-run active-wall budget exhausted')
     row=tail_record(f/'udt_step_budget.csv')
     if row is not None:
      t=row['actual_state_time'];values=list(row.values());assert all(math.isfinite(x) for x in values),'Nonfinite live ledger';assert row['negative_native_count']==0 and row['negative_volume_sum']==0 and row['minimum_native_volume_since_initialization']>=0,'Negative native volume'
      scale=max(row['rain_applied_cumulative']+row['boundary_in_cumulative'],row['boundary_out_cumulative'],1.);assert abs(row['budget_residual'])<=max(1.,.01*scale),'Live prefix water-balance gate'
      if t==last_state:stalled+=elapsed
      else:stalled=0.;last_state=t
      if stalled>900:raise RuntimeError('Native clock stalled900s')
      save(latest_native_time_s=t,latest_step_index=int(row['step_index']),latest_budget_residual_m3=row['budget_residual'])
     elif runactive>900:raise RuntimeError('No diagnostic clock within900s')
    so.close();se.close();rc=child.returncode;log=(f/'sfincs.log').read_text(errors='replace') if (f/'sfincs.log').exists() else '';done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS']);dump(f/'execution.json',dict(returncode=rc,completion_marker=done,runtime_seconds=runactive,elapsed_with_pauses_s=time.monotonic()-start,solver_image_id=IMAGE,input_sha256=r['input_sha256'],command=cmd,training=False,sealed_test=False));assert rc==0 and done,'Hydraulic process failed; preserve attempt';child=None;container=None
    save(status='auditing');audit=R/'execution'/case;audit.mkdir(parents=True,exist_ok=True);auditstart=time.monotonic()
    with (audit/'audit.log').open('w') as out:result=subprocess.run([BUILDER,str(ROOT/'src/analyses/audit_formal_transport_run_v1.py'),'execution',case],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    state['active_wall_seconds']+=time.monotonic()-auditstart
    assert result.returncode==0 and json.loads((audit/'audit.json').read_text())['all_checks_passed'],'Unchanged numerical gates failed; hold queue'
    state['completed_new_cases'].append(case);save(status='running',queue=[x for x in manifest['new_execution_order'] if x not in state['completed_new_cases']]);report();print('QUALIFIED',case,flush=True)
   state['completed_batches'].append(batch['batch_id']);save();dump(R/(batch['batch_id']+'_completion.json'),dict(numerical_audits_passed=True,cases=batch['new_cases'],no_automatic_reference_release=True));report()
  save(status='all_hydraulics_qualified',current_case=None,current_batch=None,queue=[],end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));report();print('CAMPAIGN QUALIFIED',flush=True)
 except BaseException as exc:
  if container:
   docker('unpause',container,check=False);docker('stop','--time','10',container,check=False)
  if child:
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.terminate()
  save(status='held',error=str(exc),automatic_retry=False,preserved_attempt=True);(R/'campaign_error.txt').write_text(traceback.format_exc());report();print('HELD',str(exc),flush=True);raise
if __name__=='__main__':main()
