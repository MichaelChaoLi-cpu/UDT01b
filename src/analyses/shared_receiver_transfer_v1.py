#!/usr/bin/env python3
"""Four serial fixed-portfolio transfer runs, with unchanged numerical gates.

Reuse qualified modules via explicit output-directory bindings. No parent files,
forcing curves, numerical thresholds, population intervals or solver code edited.
"""
from pathlib import Path
import os,sys,json,time,shutil,subprocess,traceback,signal,argparse,html
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from shared_receiver_pilot_common_v1 import sha,dump,tail,command,PY,VENV
import numpy as np,pandas as pd
TAG='three_ward_shared_receiver_transfer_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
PRIOR=ROOT/'data/processed/three_ward_shared_receiver_pilot_v1'
EVENTS=['E20250911','E20240830'];SCRIPT=Path(__file__).resolve()

def dirs(event):return P/'batches'/event,R/'batches'/event

def phase_module(name,event):
 import importlib
 mod=importlib.import_module(name);bp,br=dirs(event);mod.P=bp;mod.R=br;return mod

def report():
 m=json.loads((P/'run_manifest.json').read_text());s=json.loads((R/'campaign_state.json').read_text()) if (R/'campaign_state.json').exists() else dict(status='preflight_passed',completed_cases=[],completed_events=[])
 rows=[]
 for r in m['runs']:
  rows.append(dict(Event=r['event_id'],Stage=r['stage'],State='qualified' if r['case_id'] in s['completed_cases'] else (s['status'] if s.get('current_case')==r['case_id'] else 'waiting'),Baseline=r['baseline_case']))
 refresh='<meta http-equiv="refresh" content="30">' if s['status'] not in ['held','qualified_complete'] else ''
 body='<h1>固定抽排方案：跨暴雨迁移试验</h1><aside>顺序：2025-09-11高／低水位 → 两组核验与后处理 → 2024-08-30高／低水位。固定62个模型取水位置、总预留2.5 m³/s、每具名接收池共享0.5 m³/s；位置数不是机器台数。原抽排启动、扬程和可用储量规则不变。</aside>'
 body+='<p><b>状态：'+html.escape(s['status'])+'</b>；数值通过 '+str(len(s['completed_cases']))+'/4组，事件后处理完成 '+str(len(s.get('completed_events',[])))+'/2。</p>'+pd.DataFrame(rows).to_html(index=False)
 if s.get('current_case'):body+='<p>当前：'+html.escape(s['current_case'])+'；模型时钟 '+f'{s.get("latest_native_time_s",0)/3600:.3f}'+'/9 h；本组活跃计算 '+f'{s.get("case_active_wall_seconds",0):.0f}'+' s。</p>'
 if s.get('error'):body+='<p class="warn">已停止扩跑并保留结果：'+html.escape(s['error'])+'</p>'
 body+='<h2>固定条件与停止规则</h2><p>相同10 m网格、建筑道路约束、每场原雷达雨型、屋面分配和水位曲线；复用各自无措施基线。使用已通过首轮试验的同一CPU求解器镜像及严格身份适配，保留原28项数值检查与8项抽排预算检查。任何失败停止，不自动重试或调整额度。</p><p>最多4个新水动力计算，串行6 CPU／7 GiB，每组活跃计算最多7200 s，水动力累计最多28800 s；每组审核最多3600 s，每事件道路后处理最多3600 s、人口可达性最多1800 s。新输出最多80 GiB，磁盘可用至少100 GiB。PAUSE暂停、STOP停止；不训练、不打开封存测试。</p>'
 for e in EVENTS:
  bp,br=dirs(e);f=br/'effects_summary.json'
  if f.exists():
   effect=json.loads(f.read_text());roads=pd.DataFrame(effect['road_effects']);sel=roads[(roads.Scope=='Three wards')&(roads['Depth Branch']=='lower')]
   body+='<h2>'+e+'：三区配对道路效果</h2>'+sel[['Stage','Baseline Peak Closed km','Drainage Peak Closed km','Closure Reduction km h','Closure Reduction Percent','Maximum Same Time Newly Closed km']].to_html(index=False,float_format=lambda v:f'{v:,.5f}')
  f=br/'access/summary.json'
  if f.exists():
   a=json.loads(f.read_text());cols=['stage','anchor_net_avoided_loss_lower','anchor_net_avoided_loss_upper','maximum_new_loss_lower','maximum_new_loss_upper']
   body+='<h3>固定基线影响时刻的常住人口通达性区间</h3>'+pd.DataFrame(a['main_population'])[cols].to_html(index=False)
   body+='<p>'+html.escape(a['next_step_judgment'])+'</p>'
  if (br/'delivery_validation.json').exists():body+='<p><a href="batches/'+e+'/effects_summary.json">Road / transfer effects</a> · <a href="batches/'+e+'/access/summary.json">Access / population summary</a> · <a href="batches/'+e+'/delivery_validation.json">Batch validation</a></p>'
 body+='<h2>比较口径</h2><p>每场内部采用相同时间和共同质量支持，恢复与新增封路、人口恢复与新增损失分别列账。跨场同时披露合格道路长度和支持是否一致；不得把覆盖差异当作天气或措施效应。人口沿用2020年官方住地人口及0／10／25 m接入候选区间，不是当天实际受困人数。交通为原转向图的条件可达性和固定时刻参考路径成本，不是真实拥堵排队。</p><p>接收河道为规定水位，未模拟实际新增河道蓄水或排口许可。已暂停benchmark补搜；本轮检验跨事件的情景稳定性，不宣称实测验证或独立预测精度。</p><p><a href="../../../processed/'+TAG+'/protocol.json">Protocol / budget</a> · <a href="preflight_validation.json">Preflight</a> · <a href="campaign_state.json">Live state</a> · <a href="../three_ward_shared_receiver_access_v1/review.html">July31 reference results</a></p>'
 style='body{font:16px/1.75 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#233a4d;background:#f5f7fa}h1{font-size:28px}aside{background:#e4eef4;padding:16px}table{display:block;max-width:100%;overflow:auto;background:#fff;border-collapse:collapse;font-size:13px}th,td{padding:9px;border:1px solid #d7dfe7;text-align:left}.warn{background:#fff0d8;padding:14px}a{color:#196c9d}'
 tmp=R/'review.html.tmp';tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+refresh+'<title>Shared Receiver Transfer Experiments</title><style>'+style+'</style>'+body+'</html>');tmp.replace(R/'review.html')

def prepare():
 assert not (P/'run_manifest.json').exists(),'Preserve staged transfer campaign'
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 priorR=ROOT/'data/exp/estimation'/PRIOR.name;assert json.loads((priorR/'delivery_validation.json').read_text())['passed'];assert json.loads((ROOT/'data/exp/estimation/three_ward_shared_receiver_access_v1/validation.json').read_text())['passed']
 from shared_receiver_identity_v2 import selftest
 identity_tests=selftest();assert identity_tests['passed']
 image=json.loads((PRIOR/'run_manifest.json').read_text())['pump_solver_image'];assert subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()==image
 assert not subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip(),'Existing Docker workload'
 assert shutil.disk_usage(P).free>=100*1024**3
 budget=dict(maximum_new_runs=4,concurrent_hydraulics=1,cpus=6,memory_gib=7,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=28800,no_clock_progress_stop_s=900,maximum_new_output_gib=80,minimum_free_gib=100,maximum_audit_wall_s=3600,maximum_postprocess_wall_s=3600,maximum_access_wall_s=1800,automatic_retries=0)
 old_protocol=json.loads((PRIOR/'protocol.json').read_text());protocol={k:v for k,v in old_protocol.items() if k not in ['status','question','new_hydraulic_runs','baseline_cases','execution','reuse_baselines','budget','new_people_counts','not_independent_validation']}
 protocol.update(status='prepared_for_authorized_transfer',decision='MILI-D-20261020-022',question='Does the exact previously allocated shared-receiver portfolio retain road and access benefits in two additional historical rainfall parents?',events=EVENTS,event_order=EVENTS,new_hydraulic_runs=4,reuse_baselines=4,stage_order=['H','L'],execution='September11 H then L, then qualified road/access postprocessing;only then August30 H then L and postprocessing. Failure holds whole queue.',frozen_portfolio_source=str(PRIOR.relative_to(ROOT)),allocation_event='E20240731',not_independent_validation='Different rainfall parents without retuning;not observed historical validation and not sealed-test evaluation.',new_people_counts='Existing official resident-stock candidate-location method only after each two-case batch qualifies',budget=budget)
 dump(P/'protocol.json',protocol)
 originals=json.loads((ROOT/'data/processed/three_ward_cross_event_protection_v1/run_manifest.json').read_text())['runs'];ib=json.loads((ROOT/'data/processed/three_ward_barrier_drainage_v1/solver_binding.json').read_text());source_pump=PRIOR/'runs/E20240731__H__U__shared_r1000_q0p5/udt_pumps.dat';pump_hash=sha(source_pump);q=np.loadtxt(source_pump,skiprows=1,ndmin=2);assert q.shape==(62,5) and abs(q[:,2].sum()-2.5)<1e-12
 frozen_geo=json.loads((PRIOR/'run_manifest.json').read_text())['runs'][0]['identity']['geometry_sha256'];allruns=[];protected={};memo={}
 def checksum(f):
  f=Path(f);k=(str(f.resolve()),f.stat().st_size,f.stat().st_mtime_ns)
  if k not in memo:memo[k]=sha(f)
  return memo[k]
 for e in EVENTS:
  bp,br=dirs(e);bp.mkdir(parents=True,exist_ok=True);br.mkdir(parents=True,exist_ok=True);(bp/'execution').mkdir(exist_ok=True);batch=[]
  for name in ['pump_sites.csv','pool_allowances.csv']:shutil.copy2(PRIOR/name,bp/name);protected[str((bp/name).relative_to(ROOT))]=checksum(bp/name)
  for stage in ['H','L']:
   base=next(r for r in originals if r['case_id']==e+'__'+stage+'__U');s=ROOT/base['folder'];ar=ROOT/'data/exp/estimation/three_ward_cross_event_protection_v1/execution'/base['case_id']/'audit.json';a=json.loads(ar.read_text());assert a['all_checks_passed'] and a['admission']['technical_reference'];assert base['identity']['geometry_sha256']==frozen_geo
   case=base['case_id']+'__shared_r1000_q0p5_transfer';folder=bp/'runs'/case;folder.mkdir(parents=True,exist_ok=False)
   for name,h in base['input_sha256'].items():
    assert checksum(s/name)==h,(base['case_id'],name);protected[str((s/name).relative_to(ROOT))]=h
    if name=='sfincs_subgrid.nc':(folder/name).symlink_to(Path(base['large_readonly_source']).resolve())
    else:shutil.copy2(s/name,folder/name)
   shutil.copy2(source_pump,folder/'udt_pumps.dat');assert checksum(folder/'udt_pumps.dat')==pump_hash
   identity=base['identity'].copy();identity.update(run_id=case+'__shared_receiver_10m',split='external_shared_receiver_transfer',solver_digest=image,patch_sha256=ib['patch_sha256'],treatment_sha256=pump_hash)
   dump(folder/'udt_run_identity.json',identity);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v) for k,v in identity.items())+'\n')
   names=list(base['input_sha256'])+['udt_pumps.dat','udt_run_identity.json'];run=base.copy();run.update(case_id=case,baseline_case=base['case_id'],original_folder=base['folder'],measure='shared_static_drainage',portfolio=old_protocol['portfolio'],folder=str(folder.relative_to(ROOT)),run_id=identity['run_id'],split=identity['split'],identity=identity,image_id=image,input_sha256={n:checksum(folder/n) for n in names},reused=False);batch.append(run);allruns.append(run)
   assert (folder/'sfincs.inp').read_bytes()==(s/'sfincs.inp').read_bytes()
   for name in ['sfincs_map.nc','execution.json','udt_output_clock.csv']:protected[str((s/name).relative_to(ROOT))]=checksum(s/name)
   protected[str(ar.relative_to(ROOT))]=checksum(ar)
  dump(bp/'protocol.json',{**protocol,'events':[e],'new_hydraulic_runs':2,'baseline_cases':[r['baseline_case'] for r in batch]})
  dump(bp/'run_manifest.json',dict(runs=batch,protocol_sha256=checksum(bp/'protocol.json'),budget=budget));protected[str((bp/'protocol.json').relative_to(ROOT))]=checksum(bp/'protocol.json');protected[str((bp/'run_manifest.json').relative_to(ROOT))]=checksum(bp/'run_manifest.json')
  # Preflight canonical native echo and treatment binding without changing solver receipts.
  from shared_receiver_identity_v2 import CORE,identity_receipt_check
  f=bp/'qualification/identity';f.mkdir(parents=True);run=batch[0]
  for n in ['udt_pumps.dat','udt_run_identity.json','udt_run_identity.txt']:shutil.copy2(ROOT/run['folder']/n,f/n)
  (f/'udt_diagnostic_identity.txt').write_text('\n'.join(k+'='+str(run['identity'][k]) for k in CORE)+'\n');assert identity_receipt_check(f,run)
 dump(P/'run_manifest.json',dict(runs=allruns,protocol_sha256=checksum(P/'protocol.json'),budget=budget,pump_solver_image=image,execution_order=[r['case_id'] for r in allruns]))
 for f in [P/'protocol.json',P/'run_manifest.json',source_pump,PRIOR/'pump_sites.csv',PRIOR/'pool_allowances.csv']:protected[str(f.relative_to(ROOT))]=checksum(f)
 dump(P/'protected_sources.json',protected)
 scripts=[SCRIPT]+[ROOT/'src/analyses'/n for n in ['shared_receiver_pilot_common_v1.py','audit_shared_receiver_pilot_identity_v3.py','shared_receiver_identity_v2.py','audit_cross_event_protection_run_v1.py','postprocess_shared_receiver_pilot_v1.py','shared_receiver_access_effects_v1.py','temporal_transport_common_v1.py','run_temporal_road_population_v1.py','integrate_night_population_transport_v1.py','postprocess_five_event_population_v1.py']]+[ROOT/'src/preprocessing/reference_label_policy.py',ROOT/'src/preprocessing/barrier_drainage_v1/pump_routines.inc']
 dump(P/'execution_code_binding.json',{str(f.relative_to(ROOT)):checksum(f) for f in scripts})
 # Qualified production fixture;no real-case replay during preparation.
 fixture=P/'qualification/pump_fixture_6';fixture.mkdir(parents=True)
 z=subprocess.run(['docker','run','--rm','--platform','linux/amd64','--network','none','--cpus','6','--memory','7g','--env','OMP_NUM_THREADS=6','--mount',f'type=bind,source={fixture},target=/data','--entrypoint','/opt/udt-build/pump_fixture',image],capture_output=True,text=True,timeout=120);(fixture/'stdout.txt').write_text(z.stdout);(fixture/'stderr.txt').write_text(z.stderr);assert z.returncode==0 and 'PASS' in z.stdout
 dump(R/'preflight_validation.json',dict(passed=True,baselines_original_gates_passed=4,frozen_pump_bytes_same_as_july=True,frozen_pump_sha256=pump_hash,source_intake_count=62,reserved_total_rate_m3_s=float(q[:,2].sum()),no_op_and_pump_fixture_passed=True,identity_tamper_tests=identity_tests,no_new_rain_geometry_or_stage_inputs=True,geometry_identity_same_as_july=True,image_id=image,new_hydraulic_runs_started=0,budget=budget));report();print('TRANSFER_PREFLIGHT_PASSED',flush=True)

def audit(event,case):phase_module('audit_shared_receiver_pilot_identity_v3',event).main(case)

def road_postprocess(event):
 mod=phase_module('postprocess_shared_receiver_pilot_v1',event);mod.main();bp,br=dirs(event);roads=pd.read_csv(bp/'road_trajectories.csv');rows=[]
 for keys,g in roads.groupby(['Case','Stage','Depth Branch','Scope']):
  b=g[g.Arm.eq('baseline')].sort_values('Frame');d=g[g.Arm.eq('drainage')].sort_values('Frame');assert np.array_equal(b.Frame,d.Frame);t=b['Analysis Time'].to_numpy()/3600;net=b['Closed Road Length'].to_numpy()-d['Closed Road Length'].to_numpy();rest=d['Same Time Restored Road Length'].to_numpy();new=d['Same Time Newly Closed Road Length'].to_numpy();assert np.max(abs(net-rest+new))<1e-7;i=int(np.argmax(b['Closed Road Length'].to_numpy()));integral=float(np.trapezoid(b['Closed Road Length'],t));delta=float(np.trapezoid(net,t));row=dict(zip(['Case','Stage','Depth Branch','Scope'],keys));row.update({'Baseline Peak Closed km':float(b['Closed Road Length'].max()),'Drainage Peak Closed km':float(d['Closed Road Length'].max()),'Baseline Closure km h':integral,'Closure Reduction km h':delta,'Closure Reduction Percent':100*delta/integral if integral else None,'Baseline Peak Hour':float(t[i]),'Restored At Baseline Peak km':float(rest[i]),'Newly Closed At Baseline Peak km':float(new[i]),'Maximum Same Time Newly Closed km':float(new.max()),'Maximum Same Time Restored km':float(rest.max()),'Common Qualified Road km':float(b['Common Qualified Road Length'].iloc[0]),'Unknown Road km':float(b['Unknown Road Length'].iloc[0])});rows.append(row)
 pd.DataFrame(rows).to_csv(bp/'paired_road_effects.csv',index=False);transfer=pd.read_csv(bp/'off_target_head_changes.csv');shifts=transfer.groupby(['Case','Stage','Scope'],as_index=False).agg({'Outside Target Cells Certain Head Rise Area':'max','Outside Target Cells Certain Head Fall Area':'max'});shifts.to_csv(bp/'off_target_summary.csv',index=False)
 dump(br/'effects_summary.json',dict(passed=True,road_effects=rows,off_target=shifts.to_dict('records'),same_time_checks=True,limits=['Paired common support within event;cross-event coverage explicitly reported.','Fixed-stage receiving boundary omits real added river storage.','No observed historical accuracy claim.']))
 dump(br/'delivery_validation.json',dict(passed=True,new_hydraulic_runs=2,reused_baselines=2,original_numerical_rules_preserved=True,independent_variable_pump_and_shared_pool_budgets=True,roads_and_off_target_changes_summarized=True));dump(br/'campaign_state.json',dict(status='qualified_complete',completed_cases=[r['case_id'] for r in json.loads((bp/'run_manifest.json').read_text())['runs']]))

def access(event):
 mod=phase_module('shared_receiver_access_effects_v1',event);bp,br=dirs(event);ap=bp/'access';ar=br/'access';ap.mkdir(exist_ok=True);ar.mkdir(exist_ok=True)
 for n in ['node_cache','population','anchor_nodes']:(ap/n).mkdir(exist_ok=True)
 mod.P=ap;mod.R=ar;mod.HP=bp;mod.HR=br;mod.report=lambda:None
 proto=json.loads((ROOT/'data/processed/three_ward_shared_receiver_access_v1/protocol.json').read_text());anchor=pd.read_csv(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/reference_states.csv');a=anchor[(anchor.event_id==event)&anchor.selection_reasons.str.contains('H_noaction_closed_length_peak')];assert len(a)==1
 proto.update(decision='MILI-D-20261020-022',parent_event=event,anchor_frame=int(a.iloc[0].frame),anchor_time_s=float(a.iloc[0].analysis_elapsed_s),frozen_portfolio_selection_event='E20240731');dump(ap/'protocol.json',proto)
 binding=mod.bindings();binding={k:v for k,v in binding.items() if not k.startswith('data/processed/three_ward_shared_receiver_pilot_v1/')};binding.update({str(f.relative_to(ROOT)):sha(f) for f in [SCRIPT,bp/'protocol.json',bp/'run_manifest.json',ap/'protocol.json']});dump(ap/'input_bindings.json',binding)
 from temporal_transport_common_v1 import Reachability,population_inputs
 blocks,pop,normal,infos=population_inputs();eng=Reachability();assert np.array_equal(eng.direct(np.zeros(len(mod.load(mod.TEMP/'clipped_road_lengths.npz')['qualified']),bool)),normal);assert len(blocks)==4746 and pop.sum()==496345;assert mod.routing_selfcheck()
 dump(ar/'preflight_validation.json',dict(passed=True,original_normal_graph_replayed=True,frozen_baseline_anchor=True,resident_stock_unchanged=True));mod.run();assert json.loads((ar/'validation.json').read_text())['passed']

def child_phase(phase,event,case=None):
 if phase=='audit':audit(event,case)
 elif phase=='post':road_postprocess(event)
 elif phase=='access':access(event)
 else:raise ValueError(phase)

def run():
 assert json.loads((R/'preflight_validation.json').read_text())['passed'];assert not (R/'campaign_state.json').exists(),'Preserve existing attempt;no automatic restart'
 m=json.loads((P/'run_manifest.json').read_text());budget=m['budget'];protected=json.loads((P/'protected_sources.json').read_text());code=json.loads((P/'execution_code_binding.json').read_text());state=dict(status='starting',pid=os.getpid(),completed_cases=[],completed_events=[],current_case=None,new_runs_started=0,active_wall_seconds=0.,automatic_retry=False,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));container=None;child=None
 def put(**kw):state.update(kw);dump(R/'campaign_state.json',state)
 def unchanged():assert all(sha(ROOT/k)==h for k,h in code.items()) and all(sha(ROOT/k)==h for k,h in protected.items()),'Protected source/code changed'
 def resources():
  if (P/'STOP').exists():raise RuntimeError('User STOP')
  assert shutil.disk_usage(P).free>=budget['minimum_free_gib']*1024**3
  assert state['active_wall_seconds']<=budget['maximum_campaign_active_wall_s']
  size=sum(f.stat().st_size for f in (P/'batches').rglob('*') if f.is_file() and not f.is_symlink());assert size<=budget['maximum_new_output_gib']*1024**3
 def phase(what,event,limit,case=None):
  nonlocal child
  out=dirs(event)[1];name=what+('_'+case if case else '')+'.log';cmd=[str(PY),str(SCRIPT),what,'--event',event]+(['--case',case] if case else [])
  with (out/name).open('w') as stream:
   child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);active=0.;tick=time.monotonic();paused=False
   while child.poll() is None:
    time.sleep(5);now=time.monotonic();elapsed=now-tick;tick=now;resources()
    if (P/'PAUSE').exists():
     if not paused:child.send_signal(signal.SIGSTOP);paused=True
     continue
    if paused:child.send_signal(signal.SIGCONT);paused=False;continue
    active+=elapsed;assert active<=limit,'Postprocessing/audit budget'
   rc=child.returncode;child=None;assert rc==0,'Failed '+what+' '+event
 try:
  put();report()
  for event in EVENTS:
   bp,br=dirs(event)
   for r in [x for x in m['runs'] if x['event_id']==event]:
    unchanged();resources()
    while (P/'PAUSE').exists():put(status='paused_between_cases');report();time.sleep(10);resources()
    folder=ROOT/r['folder'];assert not (folder/'launch.json').exists();assert all(sha(folder/k)==h for k,h in r['input_sha256'].items());container='udt-transfer-'+event[1:]+'-'+r['stage'].lower();cmd=command(r,container);dump(folder/'launch.json',dict(command=cmd,protocol_sha256=m['protocol_sha256']));active=0.;tick=time.monotonic();previous=None;stalled=0.;paused=False;last_report=0.
    put(status='running',current_case=r['case_id'],new_runs_started=state['new_runs_started']+1,latest_native_time_s=0.,case_active_wall_seconds=0.);report();print('START_TRANSFER',r['case_id'],flush=True)
    with (folder/'solver_stdout.txt').open('w') as so,(folder/'solver_stderr.txt').open('w') as se:
     child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=so,stderr=se)
     while child.poll() is None:
      time.sleep(10);now=time.monotonic();elapsed=now-tick;tick=now;resources()
      if (P/'PAUSE').exists():
       if not paused:subprocess.run(['docker','pause',container],capture_output=True,check=True);paused=True;put(status='paused_by_user');report()
       continue
      if paused:subprocess.run(['docker','unpause',container],capture_output=True,check=True);paused=False;put(status='running');report();continue
      active+=elapsed;state['active_wall_seconds']+=elapsed;assert active<=budget['maximum_run_active_wall_s'],'Per-run active budget';z=tail(folder/'udt_step_budget.csv')
      if z:
       assert all(np.isfinite(v) for v in z.values());assert z['negative_native_count']==z['negative_volume_sum']==0 and z['minimum_native_volume_since_initialization']>=0,'Negative volume'
       scale=max(z['rain_applied_cumulative']+z['boundary_in_cumulative']+z['other_source_cumulative'],z['boundary_out_cumulative']+z['other_sink_cumulative'],1.);assert abs(z['budget_residual'])<=max(1.,.01*scale),'Original live budget'
       stalled=stalled+elapsed if z['actual_state_time']==previous else 0.;previous=z['actual_state_time'];assert stalled<=900,'No native clock progress';put(latest_native_time_s=z['actual_state_time'],latest_step_index=int(z['step_index']),latest_budget_residual_m3=z['budget_residual'],actual_pump_sink_m3=z['other_sink_cumulative'],case_active_wall_seconds=active)
      else:
       put(case_active_wall_seconds=active);assert active<=900,'No native clock ledger'
      if active-last_report>=30:report();last_report=active
    rc=child.returncode;child=None;container=None;log=(folder/'sfincs.log').read_text(errors='replace');done=any(x in log for x in ['Simulation finished','Simulation completed','Closing off SFINCS']);dump(folder/'execution.json',dict(returncode=rc,completion_marker=done,runtime_seconds=active,solver_image_id=r['image_id'],command=cmd,input_sha256=r['input_sha256']));assert rc==0 and done,'Solver incomplete'
    put(status='auditing',latest_native_time_s=32400.);report();phase('audit',event,budget['maximum_audit_wall_s'],r['case_id']);a=json.loads((br/'execution'/r['case_id']/'audit.json').read_text());b=json.loads((br/'execution'/r['case_id']/'pump_budget_summary.json').read_text());assert a['all_checks_passed'] and a['admission']['technical_reference'] and b['passed'];state['completed_cases'].append(r['case_id']);put(status='qualified');report();print('TRANSFER_CASE_QUALIFIED',r['case_id'],flush=True)
   unchanged();put(status='road_transfer_postprocessing',current_case=None);report();phase('post',event,budget['maximum_postprocess_wall_s']);put(status='access_population_postprocessing');report();phase('access',event,budget['maximum_access_wall_s']);assert json.loads((br/'access/validation.json').read_text())['passed'];state['completed_events'].append(event);put(status='event_qualified');report();print('TRANSFER_EVENT_QUALIFIED',event,flush=True)
  unchanged();supports={e:np.load(dirs(e)[0]/'common_road_support.npz')['qualified_roads'] for e in EVENTS};same=np.array_equal(supports[EVENTS[0]],supports[EVENTS[1]])
  dump(R/'delivery_validation.json',dict(passed=True,new_hydraulic_runs=4,original_numerical_rules_preserved=True,full_fixed_portfolio_transfer=True,two_new_rainfall_parents=True,matched_road_and_access_results=True,cross_transfer_road_support_identical=same,coverage_note='Compare within-event paired effects;coverage disclosed if cross-event supports differ.',training=False,sealed_test=False));put(status='qualified_complete',current_case=None,end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));report();print('TRANSFER_CAMPAIGN_COMPLETE',flush=True)
 except BaseException as exc:
  if container:
   subprocess.run(['docker','unpause',container],capture_output=True);subprocess.run(['docker','stop','--time','10',container],capture_output=True)
  if child:
   try:child.send_signal(signal.SIGCONT);child.terminate();child.wait(timeout=30)
   except Exception:pass
  put(status='held',error=str(exc),automatic_retry=False);(R/'campaign_error.txt').write_text(traceback.format_exc());report();raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','run','report','audit','post','access']);ap.add_argument('--event',choices=EVENTS);ap.add_argument('--case');a=ap.parse_args()
 if a.phase=='prepare':prepare()
 elif a.phase=='run':run()
 elif a.phase=='report':report()
 else:assert a.event;child_phase(a.phase,a.event,a.case)
if __name__=='__main__':main()
