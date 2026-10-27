#!/usr/bin/env python3
"""Serial high-then-low gated observational replay with original physics/image."""
from pathlib import Path
import sys,os,json,hashlib,time,subprocess,shutil,math,traceback
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_ground_flux_replay_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG;PY=ROOT/'.frame-control/runtimes/physical-measures-py312/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def docker(*args,check=True):return subprocess.run(['docker',*args],capture_output=True,text=True,check=check)
def tail(path):
 if not path.exists():return None
 with path.open('rb') as f:
  names=f.readline().decode().strip().split(',');f.seek(0,2);length=f.tell();f.seek(max(0,length-131072));s=f.read().decode(errors='replace')
 if not s.endswith('\n'):s=s.rsplit('\n',1)[0]
 for line in reversed(s.splitlines()):
  v=line.split(',')
  if len(v)==len(names):
   try:return {k:float(x) for k,x in zip(names,v)}
   except ValueError:pass
 return None

def main():
 manifest=json.loads((P/'run_manifest.json').read_text());assert not (R/'campaign_state.json').exists() or '--resume-qualified' in sys.argv,'No automatic overwrite/retry';assert json.loads((R/'mapping_validation.json').read_text())['passed'];assert json.loads((R/'input_validation.json').read_text())['passed'];budget=manifest['budget'];image=manifest['image_id'];assert docker('image','inspect',image,'--format','{{.Id}}').stdout.strip()==image
 paths=[ROOT/'src/analyses'/n for n in ['run_ground_flux_replay_v1.py','audit_ground_flux_replay_v1.py','audit_cross_event_protection_run_v1.py']]+[ROOT/'src/preprocessing/reference_label_policy.py']
 code={str(x.relative_to(ROOT)):sha(x) for x in paths};atomic(P/'execution_code_binding.json',code)
 state={'status':'starting','pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'current_case':None,'completed_cases':[],'queue':manifest['execution_order'],'new_hydraulic_replays_started':0,'new_forcing_or_measure_scenarios':0,'training':False,'people_counted':False};container=None;child=None
 if (R/'campaign_state.json').exists():
  assert '--resume-qualified' in sys.argv
  previous=json.loads((R/'campaign_state.json').read_text())
  assert previous['status']=='held' and previous['new_hydraulic_replays_started']==1
  first=manifest['runs'][0];second=manifest['runs'][1]
  qualified=json.loads((R/'execution'/first['case_id']/'replay_validation.json').read_text())
  assert qualified['passed'] and qualified['original_numerical_gates_passed']
  assert not (ROOT/second['folder']/'launch.json').exists(),'No automatic hydraulic retry'
  state=previous
  state.update(status='starting_qualified_continuation',pid=os.getpid(),completed_cases=[first['case_id']],queue=[second['case_id']],error=None,resumed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reporting_audit_correction='reporting_comparison_correction.json')
 def report(**kwargs):
  state.update(kwargs);atomic(R/'campaign_state.json',state)
  status=state['status'];case=state.get('current_case') or '';seconds=state.get('latest_native_time_s',0) or 0;progress=min(100.,100*seconds/32400)
  results=''
  for c in state['completed_cases']:
   f=R/'execution'/c/'replay_validation.json';v=json.loads(f.read_text());results+=f'<li>{c}: 原地图、时钟及B流量逐位一致；新增地表格面{v["new_ground_face_count"]}；原数值准入通过。</li>'
  page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>控制区地表流量诊断重放</title><style>body{{font:16px/1.8 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#203b49}}aside{{background:#edf4f7;padding:18px}}progress{{width:100%}}pre{{white-space:pre-wrap}}</style><h1>控制区地表流量诊断重放</h1><aside>状态：{status}；已通过{len(state['completed_cases'])}/2组。当前：{case}</aside><p>当前计算时钟 {seconds:.1f}/32400秒（{progress:.1f}%），这是模型时间进度，不是耗时预测。</p><progress value="{progress}" max="100"></progress><p>仅追加观测断面，保持原雨型、屋面分配、几何、水位和求解器镜像。高水位组通过后才运行低水位组，不扩展到其余8组。旧数值准入规则不变。</p><ul>{results}</ul><p>局部差额将区分可观测的格面乘积算术差与地图写出量化界；未独立记录的逐单元组合舍入仍保留未归因，不能直接推导泵数。</p><p><a href="campaign_state.json">实时状态</a> · <a href="input_validation.json">输入核验</a> · <a href="mapping_validation.json">运行时断面映射核验</a></p><pre>{json.dumps(state,ensure_ascii=False,indent=2)}</pre></html>'''
  (R/'review.html').write_text(page)
 def resources():
  if (P/'STOP').exists():raise RuntimeError('User STOP')
  if shutil.disk_usage(P).free<budget['minimum_free_gib']*1024**3:raise RuntimeError('Free disk gate')
  size=sum(f.stat().st_size for f in P.rglob('*') if f.is_file())
  if size>budget['maximum_new_output_gib']*1024**3:raise RuntimeError('Output size ceiling')
 try:
  report()
  for run in manifest['runs']:
   if run['case_id'] in state['completed_cases']:continue
   while (P/'PAUSE').exists():report(status='paused_at_boundary');time.sleep(10);resources()
   resources();assert all(sha(ROOT/n)==h for n,h in code.items()),'Audit code changed'
   folder=ROOT/run['folder'];assert not any((folder/n).exists() for n in ['launch.json','execution.json','sfincs.log','sfincs_map.nc']),'Preserve old attempt'
   assert all(sha(folder/n)==h for n,h in run['input_sha256'].items())
   original=pd.read_csv(ROOT/run['original_folder']/'udt_step_budget.csv',float_precision='round_trip').set_index('step_index')
   container='udt01b-ground-replay-'+run['stage'].lower();cmd=['docker','run','--rm','--name',container,'--platform','linux/amd64','--network','none','--cpus',str(budget['cpus']),'--memory',str(budget['memory_gib'])+'g','--env','OMP_NUM_THREADS='+str(budget['cpus']),'--mount',f'type=bind,source={folder},target=/data','--mount',f'type=bind,source={folder}/sfincs_subgrid.nc,target=/data/sfincs_subgrid.nc,readonly','--mount',f'type=bind,source={folder}/historical_rain.nc,target=/data/historical_rain.nc,readonly','--entrypoint','/usr/local/bin/sfincs',image]
   atomic(folder/'launch.json',{'command':cmd,'run_id':run['run_id'],'image':image,'max_active_wall_s':run['maximum_run_active_wall_s'],'cpus':6,'memory_gib':7});active=0.;tick=time.monotonic();start=tick;stalled=0.;last=None;paused=False
   report(status='running',current_case=run['case_id'],case_active_wall_seconds=0,latest_native_time_s=0,new_hydraulic_replays_started=state['new_hydraulic_replays_started']+1)
   print('START_REPLAY',run['case_id'],flush=True)
   with (folder/'solver_stdout.txt').open('w') as so,(folder/'solver_stderr.txt').open('w') as se:
    child=subprocess.Popen(cmd,cwd=folder,stdout=so,stderr=se)
    while child.poll() is None:
     time.sleep(10);now=time.monotonic();delta=now-tick;tick=now
     if (P/'PAUSE').exists():
      if not paused:docker('pause',container);paused=True;report(status='paused_by_user')
      resources();continue
     if paused:docker('unpause',container);paused=False;report(status='running');continue
     active+=delta;resources()
     if active>run['maximum_run_active_wall_s']:raise RuntimeError('Operational active wallclock ceiling')
     v=tail(folder/'udt_step_budget.csv')
     if v:
      assert all(math.isfinite(x) for x in v.values());assert v['negative_native_count']==v['negative_volume_sum']==0 and v['minimum_native_volume_since_initialization']>=0
      idx=int(v['step_index']);ref=original.loc[idx]
      assert all(ref[k]==x for k,x in v.items() if k!='step_index'),('Live original step budget changed',idx)
      native=v['actual_state_time'];stalled=stalled+delta if native==last else 0.;last=native
      if stalled>budget['no_clock_progress_stop_s']:raise RuntimeError('Actual clock stalled')
      report(case_active_wall_seconds=active,latest_native_time_s=native,latest_step_index=idx,latest_budget_residual_m3=v['budget_residual'],original_live_budget_exact=True)
     else:
      report(case_active_wall_seconds=active)
      if active>900:raise RuntimeError('No actual diagnostic clock within900s')
   rc=child.returncode;child=None;container=None;log=(folder/'sfincs.log').read_text(errors='replace') if (folder/'sfincs.log').exists() else '';done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS']);atomic(folder/'execution.json',{'returncode':rc,'completion_marker':done,'runtime_seconds':active,'elapsed_with_pauses_s':time.monotonic()-start,'solver_image_id':image,'input_sha256':run['input_sha256'],'command':cmd});assert rc==0 and done,'Replay execution failed'
   report(status='auditing',latest_native_time_s=32400);out=R/'execution'/run['case_id'];out.mkdir(parents=True,exist_ok=True)
   with (out/'full_audit.log').open('w') as f:subprocess.run([str(PY),str(ROOT/'src/analyses/audit_ground_flux_replay_v1.py'),run['case_id']],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
   assert json.loads((out/'replay_validation.json').read_text())['passed'];state['completed_cases'].append(run['case_id']);report(status='qualified',queue=[x for x in manifest['execution_order'] if x not in state['completed_cases']]);print('QUALIFIED_REPLAY',run['case_id'],flush=True)
  assert all(sha(ROOT/n)==h for n,h in json.loads((P/'protected_sources.json').read_text()).items()),'Original inputs changed'
  report(status='both_replays_qualified',current_case=None,queue=[])
  subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'src/analyses/package_ground_flux_replay_v1.py')],cwd=ROOT,check=True,timeout=300)
  print('BOTH_REPLAYS_AND_GROUND_BUDGETS_COMPLETE',flush=True)
 except BaseException as e:
  if container:
   docker('unpause',container,check=False);docker('stop','--time','10',container,check=False)
  if child:
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.terminate()
  report(status='held',error=str(e),automatic_retry=False,preserved_attempt=True);(R/'campaign_error.txt').write_text(traceback.format_exc());print('REPLAY_HELD',str(e),flush=True);raise
if __name__=='__main__':main()
