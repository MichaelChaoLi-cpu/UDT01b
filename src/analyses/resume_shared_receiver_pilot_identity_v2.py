#!/usr/bin/env python3
"""Serial two-run conditional drainage pilot;gates,automatic summaries,no automatic retries."""
from pathlib import Path
import sys,os,json,time,subprocess,shutil,math,traceback,signal
from shared_receiver_pilot_common_v1 import ROOT,P,R,PY,VENV,sha,dump,tail,command

def main():
 assert json.loads((R/'preflight_validation.json').read_text())['passed'];assert json.loads((R/'identity_repair_v2/validation.json').read_text())['passed'];assert not(R/'identity_repair_v2/resume_started.json').exists(),'Authorized resume is single-use'
 manifest=json.loads((P/'run_manifest.json').read_text());budget=manifest['budget'];assert sha(P/'protocol.json')==manifest['protocol_sha256'];assert len(manifest['runs'])==2
 original_binding=json.loads((P/'execution_code_binding.json').read_text());assert all(sha(ROOT/f)==h for f,h in original_binding.items()),'Original pinned code changed'
 scripts=['src/analyses/'+n for n in ['shared_receiver_pilot_common_v1.py','run_shared_receiver_pilot_v1.py','audit_shared_receiver_pilot_v1.py','postprocess_shared_receiver_pilot_v1.py','report_shared_receiver_pilot_identity_v2.py','audit_cross_event_protection_run_v1.py']]+['src/preprocessing/reference_label_policy.py','src/preprocessing/barrier_drainage_v1/pump_routines.inc'];scripts += ['src/analyses/'+n for n in ['shared_receiver_identity_v2.py','audit_shared_receiver_pilot_identity_v2.py','report_shared_receiver_pilot_identity_v2.py','resume_shared_receiver_pilot_identity_v2.py']]
 binding={s:sha(ROOT/s) for s in scripts};dump(P/'execution_code_binding_resume_identity_v2.json',binding)
 protected=json.loads((P/'protected_sources.json').read_text());state=json.loads((R/'campaign_state.json').read_text());assert state['status']=='held' and state['new_runs_started']==1 and state['completed_cases']==[], 'Resume only the diagnosed high-stage identity hold';state.pop('error',None);state.update(status='resuming_identity_repair',pid=os.getpid(),identity_repair='identity_repair_v2',resume_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));dump(R/'identity_repair_v2/resume_started.json',{'pid':os.getpid(),'utc':state['resume_utc'],'new_hydraulic_retries':0,'retained_high_result':True});container=None;child=None
 def save(**kw):state.update(kw);dump(R/'campaign_state.json',state)
 def report():
  with (R/'report.log').open('a') as out:subprocess.run([str(VENV),str(ROOT/'src/analyses/report_shared_receiver_pilot_identity_v2.py')],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=60)
 def unchanged():assert all(sha(ROOT/f)==h for f,h in protected.items()) and all(sha(ROOT/f)==h for f,h in binding.items()),'Protected input or code changed'
 def resources():
  if (P/'STOP').exists():raise RuntimeError('User STOP')
  assert shutil.disk_usage(P).free>=budget['minimum_free_gib']*1024**3,'Free space stop';assert state['active_wall_seconds']<=budget['maximum_campaign_active_wall_s'],'Campaign active-wall stop'
  size=sum(f.stat().st_size for f in (P/'runs').rglob('*') if f.is_file() and not f.is_symlink());assert size<=budget['maximum_new_output_gib']*1024**3,'Output storage stop'
 def phase_run(cmd,stream,limit):
  nonlocal child
  child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);active=0.;tick=time.monotonic();paused=False;base_status=state['status']
  while child.poll() is None:
   time.sleep(5);now=time.monotonic();elapsed=now-tick;tick=now
   if (P/'STOP').exists():
    if paused:child.send_signal(signal.SIGCONT)
    child.terminate();raise RuntimeError('User STOP in audit or postprocessing')
   if (P/'PAUSE').exists():
    if not paused:child.send_signal(signal.SIGSTOP);paused=True;save(status='paused_'+base_status);report()
    continue
   if paused:child.send_signal(signal.SIGCONT);paused=False;save(status=base_status);report();continue
   active+=elapsed;resources();assert active<=limit,'Audit or postprocessing active-wall stop'
  rc=child.returncode;child=None;assert rc==0,'Audit or postprocessing failed'
 try:
  save();report()
  for run in manifest['runs']:
   unchanged();resources()
   while (P/'PAUSE').exists():save(status='paused_between_cases');report();time.sleep(10);resources()
   folder=ROOT/run['folder']
   if (folder/'launch.json').exists():
    assert run['stage']=='H' and run['case_id']==manifest['runs'][0]['case_id']
    existing=json.loads((folder/'execution.json').read_text());assert existing['returncode']==0 and existing['completion_marker']
    save(status='auditing_existing_high',current_case=run['case_id']);report();out=R/'execution'/run['case_id']
    with (out/'audit_identity_v2.log').open('w') as stream:phase_run([str(PY),str(ROOT/'src/analyses/audit_shared_receiver_pilot_identity_v2.py'),run['case_id']],stream,budget['maximum_audit_wall_s'])
    a=json.loads((out/'audit.json').read_text());assert a['all_checks_passed'] and a['admission']['technical_reference'] and json.loads((out/'pump_budget_summary.json').read_text())['passed']
    state['completed_cases'].append(run['case_id']);save(status='qualified');report();print('EXISTING_HIGH_QUALIFIED_NO_HYDRAULIC_REPLAY',run['case_id'],flush=True);continue
   assert all(sha(folder/f)==h for f,h in run['input_sha256'].items())
   container='udt-shared-receiver-'+run['stage'].lower();cmd=command(run,container);dump(folder/'launch.json',{'command':cmd,'protocol_sha256':manifest['protocol_sha256'],'maximum_active_wall_s':7200});active=0.;start=time.monotonic();tick=start;last_report=0.;previous=None;stalled=0.;paused=False
   save(status='running',current_case=run['case_id'],latest_native_time_s=0.,case_active_wall_seconds=0.,new_runs_started=state['new_runs_started']+1);report();print('START_SHARED_RECEIVER',run['case_id'],flush=True)
   with (folder/'solver_stdout.txt').open('w') as so,(folder/'solver_stderr.txt').open('w') as se:
    child=subprocess.Popen(cmd,stdout=so,stderr=se)
    while child.poll() is None:
     time.sleep(10);now=time.monotonic();elapsed=now-tick;tick=now;resources()
     if (P/'PAUSE').exists():
      if not paused:subprocess.run(['docker','pause',container],check=True,capture_output=True);paused=True;save(status='paused_by_user');report()
      continue
     if paused:subprocess.run(['docker','unpause',container],check=True,capture_output=True);paused=False;save(status='running');report();continue
     active+=elapsed;state['active_wall_seconds']+=elapsed;assert active<=budget['maximum_run_active_wall_s'],'Per-run active-wall stop';row=tail(folder/'udt_step_budget.csv')
     if row:
      assert all(math.isfinite(v) for v in row.values());assert row['negative_native_count']==row['negative_volume_sum']==0 and row['minimum_native_volume_since_initialization']>=0,'Negative volume detected'
      scale=max(row['rain_applied_cumulative']+row['boundary_in_cumulative']+row['other_source_cumulative'],row['boundary_out_cumulative']+row['other_sink_cumulative'],1.);assert abs(row['budget_residual'])<=max(1.,.01*scale),'Original live prefix budget failed'
      stalled=stalled+elapsed if row['actual_state_time']==previous else 0.;previous=row['actual_state_time'];assert stalled<=budget['no_clock_progress_stop_s'],'No native clock progress900s'
      save(latest_native_time_s=row['actual_state_time'],latest_step_index=int(row['step_index']),latest_budget_residual_m3=row['budget_residual'],actual_pump_sink_m3=row['other_sink_cumulative'],case_active_wall_seconds=active)
     else:
      save(case_active_wall_seconds=active)
      if active>900:raise RuntimeError('No actual-clock ledger within900s')
     if active-last_report>=30:report();last_report=active
   rc=child.returncode;child=None;container=None;log=(folder/'sfincs.log').read_text(errors='replace') if (folder/'sfincs.log').exists() else '';done=any(s in log for s in ['Simulation finished','Simulation completed','Closing off SFINCS']);dump(folder/'execution.json',{'returncode':rc,'completion_marker':done,'runtime_seconds':active,'elapsed_including_pauses_s':time.monotonic()-start,'solver_image_id':run['image_id'],'command':cmd,'input_sha256':run['input_sha256']});assert rc==0 and done,'Solver did not complete'
   save(status='auditing',latest_native_time_s=32400.);report();out=R/'execution'/run['case_id'];out.mkdir(parents=True,exist_ok=True)
   with (out/'audit.log').open('w') as stream:phase_run([str(PY),str(ROOT/'src/analyses/audit_shared_receiver_pilot_identity_v2.py'),run['case_id']],stream,budget['maximum_audit_wall_s'])
   a=json.loads((out/'audit.json').read_text());assert a['all_checks_passed'] and a['admission']['technical_reference'] and json.loads((out/'pump_budget_summary.json').read_text())['passed'];state['completed_cases'].append(run['case_id']);save(status='qualified');report();print('QUALIFIED_SHARED_RECEIVER',run['case_id'],flush=True)
  unchanged();resources();save(status='road_transfer_postprocessing',current_case=None);report()
  with (R/'postprocessing.log').open('w') as stream:phase_run([str(PY),str(ROOT/'src/analyses/postprocess_shared_receiver_pilot_v1.py')],stream,budget['maximum_postprocess_wall_s'])
  assert json.loads((R/'postprocessing_validation.json').read_text())['passed'];unchanged();resources();dump(R/'delivery_validation.json',{'passed':True,'new_hydraulic_runs':2,'reused_baselines':2,'original_numerical_rules_preserved':True,'independent_variable_pump_and_shared_pool_budgets':True,'roads_and_off_target_changes_summarized':True,'new_population_counts':False,'training':False});save(status='qualified_complete',current_case=None,end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));report();print('SHARED_RECEIVER_PILOT_COMPLETE',flush=True)
 except BaseException as e:
  if container:
   subprocess.run(['docker','unpause',container],capture_output=True);subprocess.run(['docker','stop','--time','10',container],capture_output=True)
  if child:
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.terminate()
  save(status='held',error=str(e),automatic_retry=False);(R/'campaign_error.txt').write_text(traceback.format_exc())
  try:report()
  except Exception:pass
  raise
if __name__=='__main__':main()
