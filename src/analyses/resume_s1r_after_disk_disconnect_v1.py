#!/usr/bin/env python3
"""One user-authorized recovery after reported external-volume disconnection.
Retain five qualified cases; isolate the partial sixth and restart it from time0.
Original scenario manifests, solver, physical inputs and28+8admission stay frozen.
"""
from pathlib import Path
import os,sys,json,shutil,subprocess,inspect,time,argparse,traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import s1r_cross_event_validation_v1 as c
import numpy as np
from netCDF4 import Dataset
from shared_receiver_pilot_common_v1 import sha,dump
P,R=c.P,c.R;SCRIPT=Path(__file__).resolve();REC=R/'recovery_disk_disconnect_01'
CASE='E20240830__L__U__s1r_q2p5';BATCH='E20240830_Q2p5'


def read(f):return json.loads(Path(f).read_text())
def assert_idle():
 assert not subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip(),'Another Docker calculation is active'
 state=read(R/'campaign_state.json');p=subprocess.run(['ps','-p',str(state['pid']),'-o','command='],capture_output=True,text=True)
 assert 's1r_cross_event_validation_v1.py run'not in p.stdout and str(SCRIPT)not in p.stdout,'Previous worker still active'
 return state


def recovery_source():
 # Generate from the unchanged, already-qualified worker with explicit control-flow guards.
 source=c.adapt_worker_source()
 before="assert not (R/'campaign_state.json').exists(),'Preserve existing attempt;no automatic restart'"
 after="assert json.loads((R/'campaign_state.json').read_text())['status']=='held';assert RECOVERY_VERIFIED"
 assert source.count(before)==1;source=source.replace(before,after)
 begin=source.index("state=dict(status='starting'");end=source.index(';container=None;child=None',begin)
 source=source[:begin]+"state=RESUME_STATE.copy();state.update(status='resuming_after_disk_disconnect',pid=os.getpid(),current_case=None,automatic_retry=False,manual_disk_recoveries=1,actual_container_starts=6,resume_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));state.pop('error',None)"+source[end:]
 replacements={
  'for event in EVENTS:\n   bp,br=dirs(event)':"for event in EVENTS:\n   if event in state['completed_events']:continue\n   bp,br=dirs(event)",
  "if x['batch']==event]":"if x['batch']==event and x['case_id']not in state['completed_cases']]",
  "new_runs_started=state['new_runs_started']+1":"new_runs_started=state['new_runs_started']+(0 if r['case_id']==RETRY_CASE else 1),actual_container_starts=state['actual_container_starts']+1",
  "container='udt-s1rx-'+event.lower()+'-'+r['stage'].lower()":"container='udt-s1rx-recover1-'+event.lower()+'-'+r['stage'].lower()",
 }
 for old,new in replacements.items():assert source.count(old)==1,old;source=source.replace(old,new)
 compile(source,str(SCRIPT)+'::manual_resume','exec')
 return source


def verify():
 assert not REC.exists(),'Preserve recovery verification'
 REC.mkdir();state=assert_idle();assert state['status']=='held'and state['current_case']==CASE and len(state['completed_cases'])==5 and state['new_runs_started']==6
 assert state['completed_events']==['E20250911_Q2p5','E20250911_Q5']
 assert shutil.disk_usage(P).free>=100*1024**3
 memo={};checked={};problems=[]
 def checksum(f):
  f=Path(f);st=f.stat();key=(str(f.resolve()),st.st_size,st.st_mtime_ns)
  if key not in memo:memo[key]=sha(f)
  return memo[key]
 for filename in ['protected_sources.json','execution_code_binding.json']:
  for name,h in read(P/filename).items():
   try:
    actual=checksum(ROOT/name);checked[name]=actual
    if actual!=h:problems.append({'file':name,'issue':'sha256_mismatch','expected':h,'actual':actual})
   except OSError as ex:problems.append({'file':name,'issue':str(ex)})
 dump(REC/'frozen_input_verification.json',{'passed':not problems,'checked_paths':len(checked),'unique_physical_files_hashed':len(memo),'failures':problems})
 assert not problems,'Frozen input or code mismatch;no recovery staging allowed'
 manifest=read(P/'run_manifest.json');remaining=[x['case_id']for x in manifest['runs']if x['case_id']not in state['completed_cases']]
 assert remaining==[CASE,'E20240830__H__U__s1r_q5','E20240830__L__U__s1r_q5']
 assert [b for b in c.BATCHES if b not in state['completed_events']]==['E20240830_Q2p5','E20240830_Q5']
 images={x['image_id']for x in manifest['runs']};assert len(images)==1;image=next(iter(images))
 assert subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()==image
 retained=[];output_bindings={}
 for run in manifest['runs']:
  if run['case_id']not in state['completed_cases']:continue
  bp,br=c.dirs(run['batch']);case=run['case_id'];ar=br/'execution'/case;a=read(ar/'audit.json');b=read(ar/'pump_budget_summary.json');folder=ROOT/run['folder'];execution=read(folder/'execution.json')
  assert a['all_checks_passed']and len(a['checks'])==28 and all(x['pass']for x in a['checks'])and a['admission']['technical_reference']and a['summary']['negative_cells']==0
  assert b['passed']and len(b['shared_receiver_checks'])==8 and all(b['shared_receiver_checks'].values())
  assert execution['returncode']==0 and execution['completion_marker']and execution['solver_image_id']==image and execution['input_sha256']==run['input_sha256']
  with np.load(bp/'execution'/(case+'_output_clock.npz'))as z:
   clock=z['Actual Map Time'];assert len(clock)==109 and np.max(abs(clock-np.arange(109)*300))<=.001
  with Dataset(folder/'sfincs_map.nc')as ds:
   assert ds['zs'].shape[0]==109;last=np.ma.asarray(ds['zs'][-1]);assert not np.isinf(last.compressed()).any()
  for f in [ar/'audit.json',ar/'pump_budget_summary.json',folder/'sfincs_map.nc',folder/'execution.json',bp/'execution'/(case+'_output_clock.npz')]:output_bindings[str(f.relative_to(ROOT))]=checksum(f)
  retained.append({'case':case,'original_checks_passed':28,'pump_checks_passed':8,'negative_cells':0,'map_records':109,'last_map_readable':True})
 # Already-qualified road/access products retain their own source bindings and readable archives.
 for batch in state['completed_events']:
  bp,br=c.dirs(batch)
  for name in ['postprocessing_validation.json','expanded_transfer_validation.json','access/validation.json']:assert read(br/name)['passed'];output_bindings[str((br/name).relative_to(ROOT))]=checksum(br/name)
  for name in ['input_bindings.json','qualified_output_bindings.json']:
   for f,h in read(bp/'access'/name).items():assert checksum(ROOT/f)==h,(batch,f)
  with np.load(bp/'common_road_support.npz')as z:assert z['qualified_roads'].sum()==28064
 for event in ['E20240731','E20250911']:assert read(R/'comparisons'/event/'validation.json')['passed']
 dump(REC/'retained_output_bindings.json',output_bindings)
 # Explicitly exercise skip/count logic without executing any hydraulic operation.
 simulated=dict(state);simulated['completed_cases']=state['completed_cases'].copy();counts=[]
 for case in remaining:
  simulated['new_runs_started']+=0 if case==CASE else 1;simulated['completed_cases'].append(case);counts.append(simulated['new_runs_started'])
 assert counts==[6,7,8]and len(set(simulated['completed_cases']))==8
 source=recovery_source();(REC/'bound_resume_worker.py.txt').write_text(source)
 dump(REC/'verification.json',{'passed':True,'user_authorized_recovery':True,'cause_attribution':'User reports physical external-drive disconnection;container exit139 was observed.','frozen_input_and_original_code_sha256_exact':True,'retained_five_cases':retained,'completed_batch_bindings_unchanged':True,'remaining_cases':remaining,'unique_scenario_counts_after_recovery':counts,'one_explicit_manual_restart_not_automatic_retry':True,'partial_case_restarts_from_zero_not_snapshot_stitching':True,'same_solver_image':image,'original_numerical_rules_unchanged':True,'free_gib':shutil.disk_usage(P).free/1024**3,'cumulative_active_budget_preserved_s':state['active_wall_seconds'],'device':P.stat().st_dev})
 print('DISK_RECOVERY_VERIFIED_5_RETAINED_3_REMAINING',flush=True)


def stage():
 receipt=read(REC/'verification.json');assert receipt['passed'];state=assert_idle()
 assert not(REC/'staging_validation.json').exists();assert state['status']=='held'and len(state['completed_cases'])==5
 for n in ['campaign_state.json','orchestration_state.json','review.html','worker_pid.json','worker_resume_01.log']:shutil.copy2(R/n,REC/n)
 manifest=read(P/'run_manifest.json');run=next(x for x in manifest['runs']if x['case_id']==CASE);folder=ROOT/run['folder'];archive=P/'interrupted_attempts/disk_disconnect_01'/CASE;archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists()
 assert not(folder/'execution.json').exists(),'Interrupted case unexpectedly has a completion receipt'
 assert not(c.dirs(BATCH)[1]/'execution'/CASE/'audit.json').exists()
 folder.rename(archive);folder.mkdir()
 for name,h in run['input_sha256'].items():
  src=archive/name
  if src.is_symlink():(folder/name).symlink_to(src.resolve(strict=True))
  else:shutil.copy2(src,folder/name)
  assert sha(folder/name)==h,(CASE,name)
 assert {x.name for x in folder.iterdir()}==set(run['input_sha256'])
 dump(REC/'recovery_code_binding.json',{str(SCRIPT.relative_to(ROOT)):sha(SCRIPT),str((P/'run_manifest.json').relative_to(ROOT)):sha(P/'run_manifest.json'),str((P/'protocol.json').relative_to(ROOT)):sha(P/'protocol.json'),str((REC/'bound_resume_worker.py.txt').relative_to(ROOT)):sha(REC/'bound_resume_worker.py.txt')})
 dump(REC/'staging_validation.json',{'passed':True,'archived_incomplete_attempt':str(archive.relative_to(ROOT)),'fresh_folder_only_original_hash_exact_inputs':True,'completed_case_outputs_not_moved':True,'scenario_manifest_and_original_source_code_not_modified':True,'extra_authorized_attempts':1,'unique_scenarios_still8':True,'numerical_admission_and_physical_inputs_unchanged':True,'no_new_scenarios_or_quota_levels':True,'automatic_retries':0})
 print('DISK_RECOVERY_STAGED_WITHOUT_REPLACING_INPUTS',flush=True)


def run():
 assert read(REC/'verification.json')['passed']and read(REC/'staging_validation.json')['passed']
 assert not(REC/'launch.json').exists(),'Recovery launched already;no automatic restart'
 state=assert_idle();assert state['status']=='held'and state['current_case']==CASE
 assert all(sha(ROOT/k)==v for k,v in read(REC/'recovery_code_binding.json').items())
 assert all(sha(ROOT/k)==v for k,v in read(P/'execution_code_binding.json').items())
 # Original worker rechecks every protected input before each launch; this extra guard protects reused results.
 assert all(sha(ROOT/k)==v for k,v in read(REC/'retained_output_bindings.json').items())
 source=(REC/'bound_resume_worker.py.txt').read_text();c.configure()
 c.worker.__dict__.update(RESUME_STATE=state,RETRY_CASE=CASE,RECOVERY_VERIFIED=True)
 exec(compile(source,str(SCRIPT)+'::manual_resume','exec'),c.worker.__dict__)
 dump(REC/'launch.json',{'pid':os.getpid(),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'recovery_script_sha256':sha(SCRIPT),'user_reported_cause':'external hard drive disconnected','original_image_and_inputs':True,'completed_cases_retained':5,'remaining_unique_cases':3,'one_explicit_restart_allowed':True})
 dump(R/'orchestration_state.json',{'status':'serial_cross_event_hydraulics','pid':os.getpid(),'manual_disk_recovery':True,'recovery_receipt':'recovery_disk_disconnect_01/staging_validation.json'})
 try:
  c.worker.run();assert read(R/'final_summary_validation.json')['passed'];dump(R/'orchestration_state.json',{'status':'qualified_complete','pid':os.getpid(),'manual_disk_recovery':True});c.report()
  dump(REC/'completion.json',{'passed':True,'five_prior_cases_reused':True,'three_remaining_unique_cases_qualified':True,'one_restarted_case':CASE,'unique_scenarios':8,'actual_container_attempts':9,'original_rules_unchanged':True})
 except BaseException as exc:
  try:
   dump(R/'orchestration_state.json',{'status':'held','pid':os.getpid(),'error':str(exc)or type(exc).__name__,'manual_disk_recovery':True});(REC/'recovery_error.txt').write_text(traceback.format_exc());c.report()
  except OSError:
   Path('/private/tmp/udt01b_s1r_disk_recovery_error.txt').write_text(traceback.format_exc())
  raise


def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['verify','stage','run']);a=ap.parse_args()
 {'verify':verify,'stage':stage,'run':run}[a.phase]()
if __name__=='__main__':main()
