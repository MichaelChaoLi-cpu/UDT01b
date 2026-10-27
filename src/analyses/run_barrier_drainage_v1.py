"""Serial12-run physical-measure campaign;retains independent existing traffic jobs."""
import os,sys,time,json,subprocess,shutil,math,traceback,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from qualify_stage_barrier_drainage_v1 import P,R,command,PYTHON
from observed_load_common_v1 import sha,dump
from run_cross_event_protection_batches_v1 import tail_record

def main():
 assert json.loads((R/'solver_qualification.json').read_text())['passed'];assert json.loads((R/'input_validation.json').read_text())['passed'];assert not(R/'campaign_state.json').exists(),'No automatic restart';manifest=json.loads((P/'run_manifest.json').read_text());protocol=json.loads((P/'protocol.json').read_text());budget=manifest['budget'];assert sha(P/'protocol.json')==manifest['protocol_sha256'];sources=json.loads((P/'protected_sources.json').read_text());names=['src/analyses/'+n for n in ['run_barrier_drainage_v1.py','audit_barrier_drainage_run_v1.py','postprocess_barrier_drainage_v1.py','run_barrier_drainage_traffic_v1.py','report_barrier_drainage_v1.py']]+['src/preprocessing/qualify_stage_barrier_drainage_v1.py','src/preprocessing/reference_label_policy.py','src/analyses/audit_cross_event_protection_run_v1.py'];bindings={n:sha(ROOT/n)for n in names};dump(P/'execution_code_binding.json',bindings);state=dict(status='running',pid=os.getpid(),completed_new_cases=[],current_case=None,active_wall_seconds=0.,new_runs=12,baseline_reused=4,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training=False,sealed_tests=False);container=None;child=None
 def save(**kw):
  state.update(kw);tmp=R/'campaign_state.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(R/'campaign_state.json')
 def unchanged():
  assert all(sha(ROOT/n)==h for n,h in sources.items()),'Protected source changed';assert all(sha(ROOT/n)==h for n,h in bindings.items()),'Execution code changed'
 def report():
  with(R/'report_execution.log').open('a')as out:subprocess.run([PYTHON,str(ROOT/'src/analyses/report_barrier_drainage_v1.py')],stdout=out,stderr=subprocess.STDOUT)
 def check():
  if(P/'STOP').exists():raise RuntimeError('User STOP')
  assert shutil.disk_usage(P).free>=budget['minimum_free_gib']*1024**3,'Free space gate';assert state['active_wall_seconds']<=budget['maximum_campaign_active_wall_s'],'Active campaign budget';size=sum(f.stat().st_size for f in(P/'runs').rglob('*')if f.is_file()and not f.is_symlink());assert size<=budget['maximum_new_output_gib']*1024**3,'Output storage budget'
 try:
  save();report()
  for case in manifest['new_execution_order']:
   check();unchanged()
   while(P/'PAUSE').exists():save(status='paused_at_boundary');time.sleep(5);check()
   run=next(r for r in manifest['runs']if r['case_id']==case);f=ROOT/run['folder'];assert all(sha(f/n)==h for n,h in run['input_sha256'].items());assert not(f/'launch.json').exists();container='udt-physical-'+case.replace('__','-').lower();cmd=command(run,container);begun=time.monotonic();active=0.;tick=begun;paused=False;last_report=0.;last_native=None;stalled=0.;save(status='running',current_case=case,case_active_wall_seconds=0.,latest_native_time_s=0.);dump(f/'launch.json',dict(command=cmd,protocol_sha256=manifest['protocol_sha256']));print('START PHYSICAL',case,flush=True)
   with(f/'solver_stdout.txt').open('w')as so,(f/'solver_stderr.txt').open('w')as se:
    child=subprocess.Popen(cmd,stdout=so,stderr=se)
    while child.poll()is None:
     time.sleep(10);now=time.monotonic();delta=now-tick;tick=now
     if(P/'PAUSE').exists():
      if not paused:subprocess.run(['docker','pause',container],check=True,capture_output=True);paused=True;save(status='paused_by_user');report()
      if(P/'STOP').exists():raise RuntimeError('User STOP while paused')
      continue
     if paused:subprocess.run(['docker','unpause',container],check=True,capture_output=True);paused=False;save(status='running');continue
     active+=delta;state['active_wall_seconds']+=delta;check();assert active<budget['maximum_run_active_wall_s'],'Per-run budget';row=tail_record(f/'udt_step_budget.csv')
     if row:
      stalled=stalled+delta if row['actual_state_time']==last_native else 0.;last_native=row['actual_state_time'];assert stalled<900,'Native clock stalled900s'
      assert all(math.isfinite(v)for v in row.values());assert row['negative_native_count']==0 and row['negative_volume_sum']==0 and row['minimum_native_volume_since_initialization']>=0,'Negative volume';scale=max(row['rain_applied_cumulative']+row['boundary_in_cumulative']+row['other_source_cumulative'],row['boundary_out_cumulative']+row['other_sink_cumulative'],1.);assert abs(row['budget_residual'])<=max(1.,.01*scale),'Prefix water budget';save(latest_native_time_s=row['actual_state_time'],latest_budget_residual_m3=row['budget_residual'],case_active_wall_seconds=active)
     if row is None and active>900:raise RuntimeError('No actual-clock ledger within900s')
     if active-last_report>60:report();last_report=active
   log=(f/'sfincs.log').read_text(errors='replace');done=any(s in log for s in ['Simulation finished','Simulation completed','Closing off SFINCS']);dump(f/'execution.json',dict(returncode=child.returncode,completion_marker=done,runtime_seconds=active,solver_image_id=run['image_id'],command=cmd));assert child.returncode==0 and done,'Solver execution failed';child=None;container=None;save(status='auditing');report();audit=R/'execution'/case;audit.mkdir(parents=True,exist_ok=True)
   with(audit/'audit.log').open('w')as out:res=subprocess.run([PYTHON,str(ROOT/'src/analyses/audit_barrier_drainage_run_v1.py'),case],stdout=out,stderr=subprocess.STDOUT)
   a=json.loads((audit/'audit.json').read_text())if(audit/'audit.json').exists()else{};assert res.returncode==0 and a.get('all_checks_passed')and a.get('admission',{}).get('technical_reference'),'Original/measure admission failed;queue held';state['completed_new_cases'].append(case);save(status='running');report();print('QUALIFIED PHYSICAL',case,flush=True)
  unchanged();save(status='road_population_postprocessing',current_case=None);report()
  for name in ['postprocess_barrier_drainage_v1.py','run_barrier_drainage_traffic_v1.py']:
   save(status='road_population_postprocessing'if name.startswith('postprocess')else'traffic_postprocessing');report()
   with(R/(name+'.log')).open('w')as out:
    child=subprocess.Popen([PYTHON,str(ROOT/'src/analyses'/name)],stdout=out,stderr=subprocess.STDOUT);active=0.;tick=time.monotonic();paused=False
    while child.poll()is None:
     time.sleep(5);now=time.monotonic();delta=now-tick;tick=now
     if(P/'STOP').exists():
      if paused:child.send_signal(signal.SIGCONT)
      child.terminate();raise RuntimeError('User STOP in postprocessing')
     if(P/'PAUSE').exists():
      if not paused:child.send_signal(signal.SIGSTOP);paused=True;save(status='paused_postprocessing');report()
      continue
     if paused:child.send_signal(signal.SIGCONT);paused=False;save(status='postprocessing');continue
     active+=delta;assert active<budget['postprocessing_active_wall_s'],'Postprocessing active budget'
    assert child.returncode==0,name+' failed';child=None
   report()
  save(status='qualified_complete',end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));report();dump(R/'delivery_validation.json',dict(passed=True,new_hydraulics=12,reused_baselines=4,experimental_arms=4,rain_parents=2,original_gates_preserved=True,physical_effectiveness_observed=False));print('PHYSICAL MEASURE CAMPAIGN QUALIFIED',flush=True)
 except BaseException as exc:
  if container:subprocess.run(['docker','unpause',container],capture_output=True);subprocess.run(['docker','stop','--time','10',container],capture_output=True)
  if child:
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:child.terminate()
  save(status='held',reason=str(exc),automatic_retry=False);(R/'campaign_error.txt').write_text(traceback.format_exc());report();raise
if __name__=='__main__':main()
