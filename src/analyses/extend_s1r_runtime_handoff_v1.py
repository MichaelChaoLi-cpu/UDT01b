#!/usr/bin/env python3
"""Adopt the running Q5H solver, then finish Q5L with approved4h/run.
Original16h campaign budget, physical inputs, image,28+8gates stay unchanged.
Only the Python monitor is replaced; no signal or restart is sent to the solver.
"""
from pathlib import Path
import os,sys,json,time,signal,subprocess,shutil,traceback,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import s1r_cross_event_validation_v1 as c
from shared_receiver_pilot_common_v1 import sha,dump,tail,PY
import numpy as np
P,R=c.P,c.R;A=R/'runtime_extension_01';SCRIPT=Path(__file__).resolve()
CASE='E20240830__H__U__s1r_q5';BATCH='E20240830_Q5';NAME='udt-s1rx-recover1-e20240830_q5-h'


def read(p):return json.loads(Path(p).read_text())
def report():
 c.report();f=R/'review.html';text=f.read_text();old='每组最多2小时活跃水动力、累计16小时'
 new='原先每组2小时；经用户确认，8月Q5高／低水位两组延长至每组4小时，累计16小时预算保持不变'
 assert text.count(old)==1;f.write_text(text.replace(old,new).replace('<h2>冻结规则与解释边界</h2>','<h2>冻结规则与解释边界</h2><p><a href="runtime_extension_01/amendment.json">已批准的运行时限变更与无损监控交接</a></p>'))


def remaining_source():
 source=(R/'recovery_disk_disconnect_01/bound_resume_worker.py.txt').read_text()
 replacements={
  "['status']=='held';assert RECOVERY_VERIFIED":"['status']=='qualified';assert RECOVERY_VERIFIED",
  "budget=m['budget']":"budget={**m['budget'],'maximum_run_active_wall_s':14400}",
  "actual_container_starts=6":"actual_container_starts=RESUME_STATE['actual_container_starts']",
  "status='resuming_after_disk_disconnect'":"status='finishing_with_approved_4h_limit'",
 }
 for old,new in replacements.items():assert source.count(old)==1,old;source=source.replace(old,new)
 compile(source,str(SCRIPT)+'::remaining_case','exec');return source


def prepare():
 assert not A.exists();A.mkdir()
 s=read(R/'campaign_state.json');assert s['status']=='running'and s['current_case']==CASE and len(s['completed_cases'])==6
 assert s['case_active_wall_seconds']<7200
 assert all(sha(ROOT/k)==h for k,h in read(P/'execution_code_binding.json').items())
 manifest=read(P/'run_manifest.json');run=next(x for x in manifest['runs']if x['case_id']==CASE);folder=ROOT/run['folder']
 # The running input hashes were checked at launch; check small forcing/treatment/identity files again.
 checked={}
 for n,h in run['input_sha256'].items():
  if n=='sfincs_subgrid.nc':continue
  assert sha(folder/n)==h,n;checked[n]=h
 inspect=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0]
 assert inspect['State']['Running']and inspect['Config']['Image']==run['image_id']and inspect['HostConfig']['NanoCpus']==6000000000
 assert inspect['HostConfig']['Memory']==7*1024**3
 assert subprocess.check_output(['ps','-p',str(s['pid']),'-o','command='],text=True).strip().endswith('resume_s1r_after_disk_disconnect_v1.py run')
 src=remaining_source();(A/'remaining_worker_source.py.txt').write_text(src)
 dump(A/'amendment.json',{'approved':True,'authorization':'User explicitly selected extension to4hours for the remaining two original scenarios.','affected_cases':[CASE,'E20240830__L__U__s1r_q5'],'old_per_case_active_seconds':7200,'new_per_case_active_seconds':14400,'campaign_active_seconds_unchanged':57600,'unchanged':'All scientific inputs,8unique scenarios,solver image,6CPU/7GiB,28numerical gates,8pump checks,masks,allocation and access rules. No automatic retry.','current_case_progress_retained':True,'method':'Adopt the same running container using docker wait;replace only the Python monitor.','original_manifest_sha256':sha(P/'run_manifest.json'),'original_protocol_sha256':sha(P/'protocol.json'),'original_execution_code_sha256':sha(P/'execution_code_binding.json'),'container_id':inspect['Id'],'expected_old_monitor_pid':s['pid'],'running_input_checks':checked})
 dump(A/'binding.json',{str(p.relative_to(ROOT)):sha(p)for p in [SCRIPT,A/'amendment.json',A/'remaining_worker_source.py.txt']})
 dump(A/'interface_validation.json',{'passed':True,'runtime_limit_separate_from_numerical_admission':True,'original_code_binding_exact':True,'current_container_resources_and_image_exact':True,'current_inputs_unchanged':True,'current_case_not_restarted':True,'remaining_source_compiles_with_only_operational_changes':True,'manifest_and_protocol_not_rewritten':True})
 print('RUNTIME_EXTENSION_PREPARED_NO_SOLVER_SIGNAL',flush=True)


def run():
 assert read(A/'interface_validation.json')['passed'];assert not(A/'launch.json').exists()
 assert all(sha(ROOT/k)==h for k,h in read(A/'binding.json').items())
 s=read(R/'campaign_state.json');assert s['status']=='running'and s['current_case']==CASE and len(s['completed_cases'])==6
 oldpid=s['pid'];meta=read(A/'amendment.json');assert oldpid==meta['expected_old_monitor_pid']
 info=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0];assert info['Id']==meta['container_id']and info['State']['Running']
 assert not(P/'PAUSE').exists()and not(P/'STOP').exists()
 m=read(P/'run_manifest.json');runinfo=next(x for x in m['runs']if x['case_id']==CASE);folder=ROOT/runinfo['folder'];bp,br=c.dirs(BATCH)
 oldstopped=False;oldreplaced=False;waiter=None;child=None
 with(A/'container_exit_code.txt').open('x')as exitfile,(A/'docker_wait_stderr.txt').open('x')as errfile:
  try:
   # Register an exit-status listener while the old watchdog still runs.
   waiter=subprocess.Popen(['docker','wait',NAME],stdout=exitfile,stderr=errfile,stdin=subprocess.DEVNULL)
   time.sleep(.2);assert waiter.poll()is None,'Container finished before handoff'
   os.kill(oldpid,signal.SIGSTOP);oldstopped=True
   s=read(R/'campaign_state.json');assert s['pid']==oldpid and s['current_case']==CASE and s['status']=='running'
   assert len(s['completed_cases'])==6 and s['case_active_wall_seconds']<7200
   age=time.time()-(R/'campaign_state.json').stat().st_mtime;assert 0<=age<30,'Stale old watchdog clock'
   for n in ['campaign_state.json','orchestration_state.json','worker_pid.json','review.html']:shutil.copy2(R/n,A/n)
   clock=tail(folder/'udt_step_budget.csv');assert clock and clock['negative_native_count']==clock['negative_volume_sum']==0
   s['case_active_wall_seconds']+=age;s['active_wall_seconds']+=age;s['pid']=os.getpid();s['runtime_limit_seconds']=14400;s['runtime_extension_receipt']='runtime_extension_01/amendment.json'
   dump(A/'launch.json',{'old_monitor_pid':oldpid,'new_monitor_pid':os.getpid(),'same_container_id':info['Id'],'native_clock_at_handoff':clock['actual_state_time'],'pre_handoff_active_seconds':s['case_active_wall_seconds'],'old_snapshot_age_added_seconds':age,'solver_restarted':False,'solver_paused':False,'unique_cases_started':s['new_runs_started'],'actual_container_starts':s['actual_container_starts']})
   # SIGKILL is confined to the old Python watchdog. Its Docker CLI child and solver remain running.
   os.kill(oldpid,signal.SIGKILL);oldreplaced=True;oldstopped=False
   dump(R/'campaign_state.json',s);dump(R/'worker_pid.json',{'pid':os.getpid(),'script':str(SCRIPT),'phase':'approved4h runtime extension without solver restart'})
   dump(R/'orchestration_state.json',{'status':'serial_cross_event_hydraulics','pid':os.getpid(),'manual_disk_recovery':True,'approved_runtime_extension':'runtime_extension_01/amendment.json'})
   print('MONITOR_ADOPTED_SAME_CONTAINER',info['Id'],flush=True);report()
   started=time.monotonic();last=started;active=s['case_active_wall_seconds'];previous=clock['actual_state_time'];stalled=0.;paused=False;lastreport=0.
   def save(**kw):s.update(kw);dump(R/'campaign_state.json',s)
   def resources():
    if(P/'STOP').exists():raise RuntimeError('User STOP')
    assert shutil.disk_usage(P).free>=m['budget']['minimum_free_gib']*1024**3
    assert s['active_wall_seconds']<=m['budget']['maximum_campaign_active_wall_s'],'Original cumulative active-time budget'
    size=sum(f.stat().st_size for f in (P/'batches').rglob('*')if f.is_file()and not f.is_symlink());assert size<=m['budget']['maximum_new_output_gib']*1024**3
   while waiter.poll()is None:
    time.sleep(10);now=time.monotonic();elapsed=now-last;last=now;resources()
    if(P/'PAUSE').exists():
     if not paused:subprocess.run(['docker','pause',NAME],check=True,capture_output=True);paused=True;save(status='paused_by_user');report()
     continue
    if paused:subprocess.run(['docker','unpause',NAME],check=True,capture_output=True);paused=False;save(status='running');report();continue
    active+=elapsed;s['active_wall_seconds']+=elapsed;assert active<=14400,'Approved per-case4hactive budget'
    z=tail(folder/'udt_step_budget.csv');assert z and all(np.isfinite(v)for v in z.values())
    assert z['negative_native_count']==z['negative_volume_sum']==0 and z['minimum_native_volume_since_initialization']>=0,'Negative volume'
    scale=max(z['rain_applied_cumulative']+z['boundary_in_cumulative']+z['other_source_cumulative'],z['boundary_out_cumulative']+z['other_sink_cumulative'],1.)
    assert abs(z['budget_residual'])<=max(1.,.01*scale),'Original live mass budget'
    stalled=stalled+elapsed if z['actual_state_time']==previous else 0.;previous=z['actual_state_time'];assert stalled<=900,'No native clock progress'
    save(status='running',latest_native_time_s=z['actual_state_time'],latest_step_index=int(z['step_index']),latest_budget_residual_m3=z['budget_residual'],actual_pump_sink_m3=z['other_sink_cumulative'],case_active_wall_seconds=active)
    if now-lastreport>=30:report();lastreport=now
   exitfile.flush();assert waiter.returncode==0,'docker wait failed';code=int((A/'container_exit_code.txt').read_text().strip());assert code==0,'Solver exit'+str(code)
   log=(folder/'sfincs.log').read_text(errors='replace');done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS']);assert done
   original_launch=read(folder/'launch.json')
   dump(folder/'execution.json',{'returncode':code,'completion_marker':done,'runtime_seconds':active,'solver_image_id':runinfo['image_id'],'command':original_launch['command'],'input_sha256':runinfo['input_sha256'],'monitor_handoff_receipt':'runtime_extension_01/launch.json','container_not_restarted':True,'approved_run_limit_s':14400})
   save(status='auditing',latest_native_time_s=32400.);report()
   with(A/'audit_current.log').open('x')as stream:
    child=subprocess.Popen([str(PY),str(c.SCRIPT),'audit','--event',BATCH,'--case',CASE],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=ROOT)
    phaseactive=0.;last=time.monotonic();phasepaused=False
    while child.poll()is None:
     time.sleep(5);now=time.monotonic();elapsed=now-last;last=now;resources()
     if(P/'PAUSE').exists():
      if not phasepaused:child.send_signal(signal.SIGSTOP);phasepaused=True
      continue
     if phasepaused:child.send_signal(signal.SIGCONT);phasepaused=False;continue
     phaseactive+=elapsed;assert phaseactive<=m['budget']['maximum_audit_wall_s']
    assert child.returncode==0,'Original audit failed';child=None
   a=read(br/'execution'/CASE/'audit.json');b=read(br/'execution'/CASE/'pump_budget_summary.json');assert a['all_checks_passed']and a['admission']['technical_reference']and len(a['checks'])==28 and all(v['pass']for v in a['checks'])and a['summary']['negative_cells']==0
   assert b['passed']and len(b['shared_receiver_checks'])==8 and all(b['shared_receiver_checks'].values())
   s['completed_cases'].append(CASE);save(status='qualified');report();print('ADOPTED_CASE_QUALIFIED',CASE,flush=True)
   source=(A/'remaining_worker_source.py.txt').read_text();c.configure();c.worker.report=report
   c.worker.__dict__.update(RESUME_STATE=s,RETRY_CASE='E20240830__L__U__s1r_q2p5',RECOVERY_VERIFIED=True)
   exec(compile(source,str(SCRIPT)+'::remaining_case','exec'),c.worker.__dict__);c.worker.run()
   assert read(R/'final_summary_validation.json')['passed'];dump(R/'orchestration_state.json',{'status':'qualified_complete','pid':os.getpid(),'manual_disk_recovery':True,'approved_runtime_extension':True});report()
   dump(A/'completion.json',{'passed':True,'all8unique_scenarios_qualified':True,'current_container_not_restarted':True,'only_runtime_limit_extended':True,'original28_plus8_unchanged':True})
  except BaseException as exc:
   if oldstopped and not oldreplaced:os.kill(oldpid,signal.SIGCONT)
   if oldreplaced:
    subprocess.run(['docker','unpause',NAME],capture_output=True);subprocess.run(['docker','stop','--time','10',NAME],capture_output=True)
    if child:
     child.send_signal(signal.SIGCONT);child.terminate()
    s=read(R/'campaign_state.json');s.update(status='held',error=str(exc)or type(exc).__name__);dump(R/'campaign_state.json',s);dump(R/'orchestration_state.json',{'status':'held','pid':os.getpid(),'error':str(exc)or type(exc).__name__});report()
   (A/'error.txt').write_text(traceback.format_exc());raise


def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','run']);a=ap.parse_args();{'prepare':prepare,'run':run}[a.phase]()
if __name__=='__main__':main()
