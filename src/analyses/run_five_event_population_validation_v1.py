"""Serial gated six-run campaign with automatic population and intervention stages.
PAUSE suspends the container; STOP halts. Every failed attempt is preserved.
"""
from pathlib import Path
import os,sys,json,time,subprocess,shutil,math,traceback,signal
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('NUMBA_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_five_event_population_validation_v1 import P,R,IMAGE,BUILDER,dump,sha
from run_cross_event_protection_batches_v1 import docker,tail_record,output_bytes

def atomic(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def main():
 m=json.loads((P/'run_manifest.json').read_text());protocol=json.loads((P/'protocol.json').read_text());budget=m['budget'];assert sha(P/'protocol.json')==m['protocol_sha256'];assert json.loads((R/'plan_validation.json').read_text())['all_passed'];assert not(R/'campaign_state.json').exists(),'No automatic restart; preserve campaign state';assert docker('image','inspect',IMAGE,'--format','{{.Id}}').stdout.strip()==IMAGE
 names=['src/preprocessing/prepare_five_event_population_validation_v1.py','src/analyses/audit_five_event_population_run_v1.py','src/analyses/validate_five_event_population_inputs_v1.py','src/analyses/run_five_event_population_validation_v1.py','src/analyses/report_five_event_population_validation_v1.py','src/analyses/postprocess_five_event_population_v1.py','src/analyses/run_five_event_transport_measures_v1.py','src/analyses/audit_cross_event_protection_run_v1.py','src/analyses/run_cross_event_protection_batches_v1.py','src/analyses/observed_load_common_v1.py','src/analyses/integrate_night_population_transport_v1.py','src/analyses/run_combined_transport_measures_v1.py','src/preprocessing/reference_label_policy.py','src/preprocessing/prepare_historical_surface_coverage_v2.py']
 codebind={n:sha(ROOT/n)for n in names};dump(P/'execution_code_binding.json',codebind);sources=json.loads((P/'protected_sources.json').read_text());runby={r['case_id']:r for r in m['runs']};state=dict(status='running',campaign_pid=os.getpid(),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),current_case=None,current_batch=None,completed_new_cases=[],queue=m['new_execution_order'],active_wall_seconds=0.,completed_batches=[],hydraulic_cases_started=0,budget=budget,population_integrated=False,traffic_complete=False,training=False,sealed_tests_opened=False);container=None;child=None
 def save(**extra):state.update(extra);atomic(R/'campaign_state.json',state)
 def report():
  with(R/'live_report.log').open('a')as f:subprocess.run([BUILDER,str(ROOT/'src/analyses/report_five_event_population_validation_v1.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
 def resources():
  if state['active_wall_seconds']>=budget['maximum_campaign_active_wall_s']:raise RuntimeError('Hydraulic active-wall budget exhausted')
  if shutil.disk_usage(P).free<budget['minimum_free_gib']*1024**3:raise RuntimeError('Minimum free-space gate failed')
  if output_bytes(m)>budget['maximum_new_output_gib']*1024**3:raise RuntimeError('Output storage budget exhausted')
  if(P/'STOP').exists():raise RuntimeError('User STOP')
 def unchanged():
  assert all(sha(ROOT/n)==h for n,h in codebind.items()),'Execution code changed'
  assert all(sha(ROOT/n)==h for n,h in sources.items()),'Protected source changed'
 def pause_boundary():
  while(P/'PAUSE').exists():
   save(status='paused_by_user_at_stage_boundary');report()
   if(P/'STOP').exists():raise RuntimeError('User STOP while paused')
   time.sleep(10)
 def post(name,script,validation):
  pause_boundary();resources();unchanged();save(status=name,current_case=None);report();start=time.monotonic();log=R/(name+'.log')
  with log.open('w')as f:
   proc=subprocess.Popen([BUILDER,str(ROOT/'src/analyses'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   active=0.;tick=time.monotonic();paused=False
   while proc.poll()is None:
    time.sleep(10);now=time.monotonic();delta=now-tick;tick=now
    if(P/'STOP').exists():
     if paused:proc.send_signal(signal.SIGCONT)
     proc.terminate();raise RuntimeError('User STOP during '+name)
    if(P/'PAUSE').exists():
     if not paused:proc.send_signal(signal.SIGSTOP);paused=True;save(status='paused_by_user_'+name);report()
     continue
    if paused:proc.send_signal(signal.SIGCONT);paused=False;save(status=name);report();continue
    active+=delta
    if active>budget['postprocessing_active_wall_s']:proc.terminate();raise RuntimeError('Postprocessing budget exhausted:'+name)
    report()
   assert proc.returncode==0 and json.loads(validation.read_text())['passed'],name+' failed; see '+str(log)
  unchanged();return time.monotonic()-start
 try:
  save();report()
  for batch in protocol['batches']:
   save(current_batch=batch['batch_id'])
   for case in batch['new_cases']:
    pause_boundary();resources();unchanged();r=runby[case];folder=ROOT/r['folder'];assert not any((folder/n).exists()for n in ['launch.json','execution.json','sfincs.log','sfincs_map.nc']),'Existing attempt preserved';assert all(sha(folder/n)==h for n,h in r['input_sha256'].items())
    container='udt01b-five-event-'+case.replace('__','-').lower();cmd=['docker','run','--rm','--name',container,'--platform','linux/amd64','--network','none','--cpus',str(budget['cpus']),'--memory',str(budget['memory_gib'])+'g','--env','OMP_NUM_THREADS='+str(budget['cpus']),'--mount',f'type=bind,source={folder},target=/data','--mount',f"type=bind,source={r['large_readonly_source']},target=/data/sfincs_subgrid.nc,readonly",'--mount',f'type=bind,source={folder}/historical_rain.nc,target=/data/historical_rain.nc,readonly','--entrypoint','/usr/local/bin/sfincs',IMAGE];start=time.monotonic();active=0.;stalled=0.;lastnative=None;paused=False;lastreport=0.;tick=time.monotonic();save(status='running',current_case=case,case_active_wall_seconds=0.,latest_native_time_s=None,hydraulic_cases_started=state['hydraulic_cases_started']+1);dump(folder/'launch.json',dict(command=cmd,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=m['protocol_sha256'],budget=budget));print('START',batch['batch_id'],case,flush=True)
    with(folder/'solver_stdout.txt').open('w')as so,(folder/'solver_stderr.txt').open('w')as se:
     child=subprocess.Popen(cmd,cwd=folder,stdout=so,stderr=se);report()
     while child.poll()is None:
      time.sleep(10);now=time.monotonic();delta=now-tick;tick=now
      if(P/'PAUSE').exists():
       if not paused:docker('pause',container);paused=True;save(status='paused_by_user');report()
       if(P/'STOP').exists():raise RuntimeError('User STOP while paused')
       continue
      if paused:docker('unpause',container);paused=False;save(status='running');report();continue
      active+=delta;state['active_wall_seconds']+=delta;save(case_active_wall_seconds=active);resources()
      if active>budget['maximum_run_active_wall_s']:raise RuntimeError('Per-run active-wall budget exhausted')
      row=tail_record(folder/'udt_step_budget.csv')
      if row is not None:
       native=row['actual_state_time'];assert all(math.isfinite(v)for v in row.values()),'Nonfinite live ledger';assert row['negative_native_count']==0 and row['negative_volume_sum']==0 and row['minimum_native_volume_since_initialization']>=0,'Negative native volume';scale=max(row['rain_applied_cumulative']+row['boundary_in_cumulative'],row['boundary_out_cumulative'],1.);assert abs(row['budget_residual'])<=max(1.,.01*scale),'Live prefix water-balance gate';stalled=stalled+delta if native==lastnative else 0.;lastnative=native
       if stalled>900:raise RuntimeError('Native clock stalled900s')
       save(latest_native_time_s=native,latest_step_index=int(row['step_index']),latest_budget_residual_m3=row['budget_residual'])
      elif active>900:raise RuntimeError('No diagnostic clock within900s')
      if active-lastreport>=60:report();lastreport=active
    rc=child.returncode;log=(folder/'sfincs.log').read_text(errors='replace')if(folder/'sfincs.log').exists()else'';done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS']);dump(folder/'execution.json',dict(returncode=rc,completion_marker=done,runtime_seconds=active,elapsed_with_pauses_s=time.monotonic()-start,solver_image_id=IMAGE,input_sha256=r['input_sha256'],command=cmd,training=False,sealed_test=False));assert rc==0 and done,'Hydraulic execution failed';child=None;container=None;save(status='auditing');report();a=R/'execution'/case;a.mkdir(parents=True,exist_ok=True)
    with(a/'audit.log').open('w')as out:result=subprocess.run([BUILDER,str(ROOT/'src/analyses/audit_five_event_population_run_v1.py'),'execution',case],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    audit_result=json.loads((a/'audit.json').read_text())if(a/'audit.json').exists()else{};assert result.returncode==0 and audit_result.get('all_checks_passed')and audit_result.get('admission',{}).get('technical_reference'),'Original numerical or label-support admission failed;queue held'
    state['completed_new_cases'].append(case);save(status='running',queue=[c for c in m['new_execution_order']if c not in state['completed_new_cases']]);report();print('QUALIFIED',case,flush=True)
   state['completed_batches'].append(batch['batch_id']);save();report()
  save(status='all_hydraulics_qualified',current_case=None,current_batch=None,queue=[]);report()
  seconds=post('population_postprocessing','postprocess_five_event_population_v1.py',R/'population/validation.json');save(population_integrated=True,population_stage_seconds=seconds);report()
  seconds=post('traffic_experiments','run_five_event_transport_measures_v1.py',R/'traffic/validation.json');save(traffic_complete=True,traffic_stage_seconds=seconds,status='qualified_complete',end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));report();dump(R/'delivery_validation.json',dict(passed=True,hydraulic_cases=6,extra_rain_parents=3,population_integrated=True,traffic_complete=True,original_gates_preserved=True,real_observed_accuracy_established=False,training=False,sealed_tests_opened=False));print('CAMPAIGN,POPULATION AND FOLLOW-ON EXPERIMENTS QUALIFIED',flush=True)
 except BaseException as exc:
  if container:docker('unpause',container,check=False);docker('stop','--time','10',container,check=False)
  if child:
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.terminate()
  save(status='held',error=str(exc),automatic_retry=False,preserved_attempt=True);(R/'campaign_error.txt').write_text(traceback.format_exc());report();print('HELD',str(exc),flush=True);raise
if __name__=='__main__':main()
