#!/usr/bin/env python3
"""Bounded July H/L S1 fixed-layout allocation contrast.

Reuse the frozen S1 positions,rates as weights and qualified weighted-pool image.
Only within-pool allocation changes. Internal contrast O1 selects the qualified
operator/auditor interface;the scientific portfolio is S1R,not the old62-site O1.
No capacity,other-parent,traffic-assignment,population or training expansion.
"""
from pathlib import Path
import os,sys,json,shutil,subprocess,inspect,argparse,html,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
sys.path.append(str(ROOT/'.frame-control/runtimes/physical-measures-py312/lib/python3.12/site-packages'))
import numpy as np,pandas as pd
from shared_receiver_pilot_common_v1 import sha,dump,PY
from shared_receiver_control_support_v1 import treatment_hash,identity_receipt_check,pool_rules,independent_fill,expected_requested_and_transfer
from shared_receiver_identity_v2 import CORE,selftest
import shared_receiver_controls_v1 as control
import shared_receiver_transfer_v1 as worker
TAG='three_ward_s1_reallocation_pilot_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
OLD=ROOT/'data/processed/three_ward_shared_receiver_controls_v1/batches/S1'
OLDR=ROOT/'data/exp/estimation/three_ward_shared_receiver_controls_v1/batches/S1'
IMAGE_BINDING=ROOT/'data/processed/three_ward_shared_receiver_controls_v1/solver_binding.json'
SCRIPT=Path(__file__).resolve();DECISION='MILI-D-20261022-007'

def dirs(_='O1'):return P/'batches/O1',R/'batches/O1'

def qualify():
 assert not (P/'run_manifest.json').exists(),'Preserve staged inputs'
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 assert not (R/'qualification.json').exists(),'Preserve existing qualification'
 image=json.loads(IMAGE_BINDING.read_text())
 assert image['unit_qualified']and image['hydraulic_prefix_qualified']
 assert subprocess.check_output(['docker','image','inspect',image['image_id'],'--format','{{.Id}}'],text=True).strip()==image['image_id']
 original=ROOT/'data/exp/estimation/three_ward_shared_receiver_controls_v1/solver_qualification.json'
 assert json.loads(original.read_text())['passed']
 checks=[];stdout=[]
 for threads in [1,6]:
  f=P/'qualification'/f'weighted_pool_fixture_{threads}';f.mkdir(parents=True,exist_ok=False)
  cmd=['docker','run','--rm','--platform','linux/amd64','--network','none','--cpus','6','--memory','1g','--env',f'OMP_NUM_THREADS={threads}','--mount',f'type=bind,source={f},target=/data','--entrypoint','/opt/udt-build/weighted_pool_fixture',image['image_id']]
  z=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(f/'stdout.txt').write_text(z.stdout);(f/'stderr.txt').write_text(z.stderr)
  assert z.returncode==0 and 'PASS'in z.stdout;stdout.append(z.stdout)
  data=pd.read_csv(f/'weighted_fixture_cases.csv');n=0
  for _,g in data.groupby('case'):
   expect=independent_fill(g.weight.to_numpy(),g.limit.to_numpy()[None,:],np.array([g.cap.iloc[0]]))[0]
   assert np.max(abs(expect-g.amount.to_numpy()))<2e-14;n+=1
  checks.append({'threads':threads,'independent_fixture_vectors':n,'passed':True})
 assert stdout[0]==stdout[1]
 assert selftest()['passed']
 dump(R/'qualification.json',{'passed':True,'image_id':image['image_id'],'checks':checks,'thread_outputs_exact':True,'original_solver_qualification_sha256':sha(original),'source_and_solver_unchanged':True,'no_production_hydraulic_run_started':True})
 print('S1R_OPERATOR_QUALIFIED',flush=True)

def prepare():
 assert not (P/'run_manifest.json').exists(),'Preserve campaign'
 assert json.loads((R/'qualification.json').read_text())['passed']
 assert not subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip(),'Existing Docker workload'
 assert shutil.disk_usage(P).free>=100*1024**3
 image=json.loads(IMAGE_BINDING.read_text());oldruns=json.loads((OLD/'run_manifest.json').read_text())['runs']
 assert json.loads((OLDR.parent.parent/'final_summary_validation.json').read_text())['passed']
 bp,br=dirs();bp.mkdir(parents=True,exist_ok=True);br.mkdir(parents=True,exist_ok=True);(bp/'execution').mkdir(exist_ok=True)
 sites=pd.read_csv(OLD/'pump_sites.csv').sort_values('Pump ID');assert len(sites)==319 and sites['Source Native Cell One Based'].is_unique
 assert np.allclose(sites.groupby('Receiving Pool')['Reserved Rate'].sum(),.5,rtol=0,atol=1e-12)
 # Freeze source/outlet indices,original weights and trigger/lift bytes exactly.
 shutil.copy2(OLD/'pump_sites.csv',bp/'pump_sites.csv')
 shutil.copy2(ROOT/oldruns[0]['folder']/'udt_pumps.dat',bp/'udt_pumps.dat')
 spec=np.loadtxt(bp/'udt_pumps.dat',skiprows=1,ndmin=2)
 assert spec.shape==(319,5)and np.all(spec[:,3]==.05)and np.all(spec[:,4]==5.)
 allowances=pd.read_csv(OLD/'pool_allowances.csv')
 for pool,g in sites.groupby('Receiving Pool'):
  m=allowances['Receiving Pool'].eq(pool);allowances.loc[m,'Activated Model Intakes']=len(g)
  assert float(allowances.loc[m,'Scenario Shared Allowance'].iloc[0])==.5
 allowances.to_csv(bp/'pool_allowances.csv',index=False)
 pools=sorted(sites['Receiving Pool'].unique());mapping={p:i+1 for i,p in enumerate(pools)}
 with (bp/'udt_pool_rules.dat').open('w')as f:
  f.write('UDT_WEIGHTED_POOL_V1\n5\n'+''.join(f'{mapping[p]} 0.5\n'for p in pools)+'319\n')
  for row in sites.to_dict('records'):f.write(f"{row['Pump ID']} {mapping[row['Receiving Pool']]} 0.5\n")
 groups,units,caps=pool_rules(bp,319);assert np.all(units==.5)and all(v==.5 for v in caps.values())
 # Test the exact319-site metadata with mixed wet,dry,head-blocked and storage-limited states.
 rng=np.random.default_rng(20261022);prior=rng.uniform(0,.5,(9,319));depth=rng.uniform(0,.8,(9,319));head=rng.uniform(0,7,(9,319));dt=np.full(9,.4)
 limits=np.where((depth>=.05)&(head<=5),np.minimum(dt[:,None]*.5,prior/(1+64*np.finfo(np.float32).eps)),0.)
 for g in caps:
  m=groups==g;result=independent_fill(spec[m,2],limits[:,m],caps[g]*dt)
  assert np.all(result>=0)and np.all(result<=limits[:,m]+1e-14)
  assert np.allclose(result.sum(axis=1),np.minimum(limits[:,m].sum(axis=1),caps[g]*dt),rtol=0,atol=1e-13)
 budget=dict(maximum_new_runs=2,concurrent_hydraulics=1,cpus=6,memory_gib=7,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=14400,no_clock_progress_stop_s=900,maximum_new_output_gib=60,minimum_free_gib=100,maximum_audit_wall_s=3600,maximum_postprocess_wall_s=3600,automatic_retries=0)
 protocol={'decision':DECISION,'event':'E20240731','public_portfolio':'S1R: S1 layout with within-pool reallocation','internal_operator_identifier':'O1 means weighted-pool interface only;not the previous62-site O1 portfolio','scientific_reference':'exact frozen S1 static319-location deployment','single_changed_factor':'static reservations to work-conserving weighted capped within-pool allocation','frozen':'same319 source and receiver cells,original reservation weights,unit ceiling0.5m3/s,trigger0.05m,lift5m,geometry,radar forcing,roof redistribution,stage curves and109maps','pool_count':5,'pool_allowance_m3_s':.5,'total_allowance_m3_s':2.5,'same_capacity_not_same_economic_cost':True,'new_hydraulic_runs':2,'stage_order':['H','L'],'baseline_and_static_reused':True,'solver_image':image['image_id'],'solver_static_equivalence':'qualified unchanged static path in weighted image;no solver source change','budget':budget,'original_checks':'28 numerical and8 pump invariants;independent allocation reconstruction unchanged','main_comparison':'remaining closure versus frozen S1;no action retained for absolute context;new closure and head rise versus each reference separately','depth_branches':['lower main','upper sensitivity'],'population_or_traffic_assignment':False,'training':False,'sealed_test':False,'capacity_expansion_enabled':False,'other_parent_expansion_enabled':False,'receivers':'prescribed stages;added river storage and downstream acceptance unmodeled','selection':'July-derived layout,within-selection-parent development contrast;not independent observed validation'}
 dump(P/'protocol.json',protocol);dump(bp/'protocol.json',protocol)
 runs=[];protected={};memo={}
 def checksum(f):
  f=Path(f);key=(str(f.resolve()),f.stat().st_size,f.stat().st_mtime_ns)
  if key not in memo:memo[key]=sha(f)
  return memo[key]
 for prior in sorted(oldruns,key=lambda x:['H','L'].index(x['stage'])):
  stage=prior['stage'];source=ROOT/prior['folder'];case=f'E20240731__{stage}__U__s1r';folder=bp/'runs'/case;folder.mkdir(parents=True,exist_ok=False)
  oldaudit=json.loads((OLDR/'execution'/prior['case_id']/'audit.json').read_text());assert oldaudit['all_checks_passed']and oldaudit['summary']['negative_cells']==0
  names=[]
  for name,h in prior['input_sha256'].items():
   if name in ['udt_run_identity.txt','udt_run_identity.json']:continue
   assert checksum(source/name)==h
   if name=='sfincs_subgrid.nc':(folder/name).symlink_to(Path(prior['large_readonly_source']).resolve())
   else:shutil.copy2(source/name,folder/name)
   names.append(name);protected[str((source/name).relative_to(ROOT))]=h
  assert checksum(folder/'udt_pumps.dat')==checksum(bp/'udt_pumps.dat')
  shutil.copy2(bp/'udt_pool_rules.dat',folder/'udt_pool_rules.dat');names.append('udt_pool_rules.dat')
  ident=prior['identity'].copy();ident.update(run_id=case+'__same_layout_10m',split='external_same_s1_layout_allocation',solver_digest=image['image_id'],patch_sha256=image['patch_sha256'],treatment_sha256=treatment_hash(folder,['udt_pumps.dat','udt_pool_rules.dat']))
  dump(folder/'udt_run_identity.json',ident);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v)for k,v in ident.items())+'\n');names+=['udt_run_identity.json','udt_run_identity.txt']
  run={**prior,'case_id':case,'run_id':ident['run_id'],'folder':str(folder.relative_to(ROOT)),'contrast':'O1','scientific_portfolio':'S1R','static_reference_case':prior['case_id'],'static_reference_folder':prior['folder'],'measure':'same_s1_layout_reallocation','treatment_files':['udt_pumps.dat','udt_pool_rules.dat'],'image_id':image['image_id'],'identity':ident,'split':ident['split'],'input_sha256':{n:checksum(folder/n)for n in names}}
  for name in prior['input_sha256']:
   if not name.startswith('udt_run_identity'):assert run['input_sha256'][name]==prior['input_sha256'][name]
  fixture=bp/'qualification/identity'/case;fixture.mkdir(parents=True)
  for name in run['treatment_files']+['udt_run_identity.json','udt_run_identity.txt']:shutil.copy2(folder/name,fixture/name)
  (fixture/'udt_diagnostic_identity.txt').write_text('\n'.join(k+'='+str(ident[k])for k in CORE)+'\n')
  assert identity_receipt_check(fixture,run)
  for name in run['treatment_files']:
   data=(fixture/name).read_bytes();(fixture/name).write_bytes(data+b'\n');assert not identity_receipt_check(fixture,run);(fixture/name).write_bytes(data)
  assert identity_receipt_check(fixture,run)
  for name,h in run['input_sha256'].items():protected[str((folder/name).relative_to(ROOT))]=h
  for name in ['sfincs_map.nc','udt_output_clock.csv','execution.json']:protected[str((ROOT/prior['original_folder']/name).relative_to(ROOT))]=checksum(ROOT/prior['original_folder']/name)
  for name in ['run_manifest.json','pump_sites.csv','common_road_support.npz','paired_road_effects.csv']:
   protected[str((OLD/name).relative_to(ROOT))]=checksum(OLD/name)
  for arm in ['baseline','drainage']:protected[str((OLD/'exposure'/f"{prior['case_id']}__{arm}.npz").relative_to(ROOT))]=checksum(OLD/'exposure'/f"{prior['case_id']}__{arm}.npz")
  runs.append(run)
 for path in [P,bp]:dump(path/'run_manifest.json',{'runs':runs,'budget':budget,'protocol_sha256':checksum(path/'protocol.json'),'execution_order':[x['case_id']for x in runs]})
 for f in [P/'run_manifest.json',P/'protocol.json',bp/'run_manifest.json',bp/'protocol.json',bp/'pool_allowances.csv',bp/'pump_sites.csv',bp/'udt_pumps.dat',bp/'udt_pool_rules.dat',IMAGE_BINDING]:protected[str(f.relative_to(ROOT))]=checksum(f)
 dump(P/'protected_sources.json',protected)
 code=[SCRIPT]+[ROOT/'src/analyses'/n for n in ['shared_receiver_controls_v1.py','shared_receiver_transfer_v1.py','shared_receiver_control_support_v1.py','shared_receiver_pilot_common_v1.py','audit_shared_receiver_pilot_identity_v3.py','audit_cross_event_protection_run_v1.py','shared_receiver_identity_v2.py','postprocess_shared_receiver_pilot_v1.py']]+[ROOT/'src/preprocessing/reference_label_policy.py']
 dump(P/'execution_code_binding.json',{str(f.relative_to(ROOT)):checksum(f)for f in code})
 dump(R/'preflight_validation.json',{'passed':True,'new_cases':2,'source_and_receiver_cells_and_weights_byte_exact':True,'frozen_sites':319,'same_total_allowance_m3_s':2.5,'same5pool_allowances_m3_s':.5,'same_trigger_lift_and_standalone_ceiling':True,'same_rain_geometry_roof_stage_input_hashes':True,'only_operator_and_required_identity_metadata_changed':True,'319_site_pool_limit_selfcheck':True,'identity_and_tamper_tests':True,'population_not_recomputed':True,'other_parent_or_capacity_expansion':False,'budget':budget})
 (R/'README.md').write_text('# Same-layout S1 reallocation pilot\n\nDecision MILI-D-20261022-007. First bounded July H/L batch only. Public portfolio S1R keeps319frozen S1intake/outlet cells and all original weights; only switches static allocation to the already-qualified weighted pool operator. Internal contrast O1 selects the operator/auditor interface and does not refer to the previous62site portfolio. Same2.5m3/s total allowance and five0.5pools;standalone0.5ceiling,0.05mtrigger,5mlift unchanged. No capacity or other-parent expansion,training,population/traffic assignment or sealed-test access.\n\nRun qualify,prepare,run once using this script. Uses the existing physical NetCDF runtime,never installs packages. Production source/rain/stage/geometry inputs are exact copies or read-only mounts;new identities bind both treatment files. Existing no-action and S1static outputs are reused. Original28+8checks remain. First numerical failure holds the queue without automatic retries. After both qualify,road effects and native head-rise transfer are automatically compared against both no action and S1static. Higher or lower benefit is admissible;adverse effects are reported rather than erased.\n\nSerial6CPU/7GiB,max2h active per run,4hhydraulics total;each audit/postphase1h;60GiBnew output ceiling,100GiBfree floor. PAUSE/STOP files under the processed root control the worker. These are hypothetical prescribed-stage scenarios,not real receiver-capacity or economic-equivalence claims. Later parents/capacity and figure updates remain disabled.\n')
 report();print('S1R_PREFLIGHT_PASSED_2_CASES',flush=True)

def configure():
 control.P=P;control.R=R;control.ARMS=['O1'];control.SCRIPT=SCRIPT
 worker.P=P;worker.R=R;worker.EVENTS=['O1'];worker.SCRIPT=SCRIPT;worker.report=report
 worker.audit=control.audit;worker.phase_module=control.post_module
 text=inspect.getsource(worker.run)
 replacements={"container='udt-transfer-'+event[1:]+'-'+r['stage'].lower()":"container='udt-s1-reallocation-'+r['stage'].lower()", "if x['event_id']==event":"if x['contrast']==event", "phase('post',event,budget['maximum_postprocess_wall_s']);put(status='access_population_postprocessing');report();phase('access',event,budget['maximum_access_wall_s']);assert json.loads((br/'access/validation.json').read_text())['passed'];":"phase('post',event,budget['maximum_postprocess_wall_s']);", "same=np.array_equal(supports[EVENTS[0]],supports[EVENTS[1]])":"same=np.array_equal(supports[EVENTS[0]],np.load(STATIC_REFERENCE/'common_road_support.npz')['qualified_roads']);assert same,'Road support changed'", "new_hydraulic_runs=4":"new_hydraulic_runs=2", "full_fixed_portfolio_transfer=True,two_new_rainfall_parents=True":"same_s1_layout_allocation_contrast=True,new_rainfall_parents=False", "matched_road_and_access_results=True":"matched_road_results=True,population_not_recomputed=True"}
 for before,after in replacements.items():assert text.count(before)==1,before;text=text.replace(before,after)
 worker.__dict__['STATIC_REFERENCE']=OLD
 exec(compile(text,str(SCRIPT),'exec'),worker.__dict__)

def finalize():
 # Implemented before launch and frozen together with the worker.
 bp,br=dirs();runs=json.loads((P/'run_manifest.json').read_text())['runs']
 assert len(runs)==2
 for run in runs:
  a=json.loads((br/'execution'/run['case_id']/'audit.json').read_text());b=json.loads((br/'execution'/run['case_id']/'pump_budget_summary.json').read_text())
  assert a['all_checks_passed']and len(a['checks'])==28 and all(x['pass']for x in a['checks'])and a['summary']['negative_cells']==0
  assert b['passed']and len(b['shared_receiver_checks'])==8 and all(b['shared_receiver_checks'].values())
 with np.load(bp/'common_road_support.npz')as z:qualified=z['qualified_roads'];common=z['common_admitted']
 with np.load(OLD/'common_road_support.npz')as z:assert np.array_equal(qualified,z['qualified_roads'])and np.array_equal(common,z['common_admitted'])
 clip=dict(np.load(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz'))
 comparisons=[];trajectory=[]
 for run in runs:
  case=run['case_id'];ref=run['static_reference_case'];stage=run['stage'];newactual=json.loads((br/'execution'/case/'pump_budget_summary.json').read_text())['actual_transferred_m3'];oldactual=json.loads((OLDR/'execution'/ref/'pump_budget_summary.json').read_text())['actual_transferred_m3']
  with np.load(bp/'exposure'/f'{case}__drainage.npz')as z:new=z['road_state_codes'];qi=z['road_indices'];t=z['analysis_times']
  with np.load(bp/'exposure'/f'{case}__baseline.npz')as z:noaction=z['road_state_codes'];assert np.array_equal(qi,z['road_indices'])and np.array_equal(t,z['analysis_times'])
  with np.load(OLD/'exposure'/f'{ref}__drainage.npz')as z:static=z['road_state_codes'];assert np.array_equal(qi,z['road_indices'])and np.array_equal(t,z['analysis_times'])
  with np.load(OLD/'exposure'/f'{ref}__baseline.npz')as z:assert np.array_equal(noaction,z['road_state_codes'])
  for bi,bound in enumerate(['lower','upper']):
   b=new[:,bi]==2
   for reference,a0 in [('S1 static',static),('No action',noaction)]:
    a=a0[:,bi]==2
    for wi,scope in enumerate(['Three wards','Chiyoda','Chuo','Minato']):
     w=(clip['union_length_m']if wi==0 else clip['ward_length_m'][wi-1])[qi]/1000
     oldlength=a@w;newlength=b@w;restore=(a&~b)@w;added=(~a&b)@w
     integ=lambda x:float(np.trapezoid(x,t)/3600)
     net=integ(oldlength-newlength);rr=integ(restore);nn=integ(added);assert abs(net-rr+nn)<1e-8
     comparisons.append({'case':case,'stage':stage,'branch':bound,'scope':scope,'reference':reference,'reference_closure_km_h':integ(oldlength),'remaining_closure_km_h':integ(newlength),'net_avoided_km_h':net,'restored_km_h':rr,'newly_closed_km_h':nn,'peak_newly_closed_km':float(added.max()),'net_reduction_percent':100*net/integ(oldlength)if integ(oldlength)>0 else None,'S1R_actual_extraction_m3':newactual,'S1_static_actual_extraction_m3':oldactual})
     for ti in range(109):trajectory.append({'case':case,'stage':stage,'branch':bound,'scope':scope,'reference':reference,'frame':ti,'elapsed_s':float(t[ti]),'reference_closed_km':float(oldlength[ti]),'remaining_closed_km':float(newlength[ti]),'restored_km':float(restore[ti]),'newly_closed_km':float(added[ti])})
 pd.DataFrame(comparisons).to_csv(P/'road_comparison.csv',index=False);pd.DataFrame(trajectory).to_csv(P/'road_comparison_trajectories.csv',index=False)
 from netCDF4 import Dataset
 ward=np.load(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz')['Ward Code'].ravel();active=np.zeros(common.size,bool);active[pd.read_csv(bp/'pump_sites.csv')['Grid Index'].to_numpy()]=True
 head=[]
 for run in runs:
  with Dataset(ROOT/run['static_reference_folder']/'sfincs_map.nc')as a,Dataset(ROOT/run['folder']/'sfincs_map.nc')as b:
   bed0=np.flipud(np.ma.asarray(a['zb'][:],dtype=float).filled(np.nan)).ravel();bed1=np.flipud(np.ma.asarray(b['zb'][:],dtype=float).filled(np.nan)).ravel();assert np.array_equal(bed0,bed1,equal_nan=True)
   unions={s:np.zeros(common.size,bool)for s in ['Three wards','Chiyoda','Chuo','Minato']};peaks={s:0 for s in unions};outside={s:0 for s in unions}
   for ti in range(109):
    x=np.flipud(np.ma.asarray(a['zs'][ti],dtype=float).filled(np.nan)).ravel();y=np.flipud(np.ma.asarray(b['zs'][ti],dtype=float).filled(np.nan)).ravel();increase=np.where(np.isfinite(y),y,bed1)-np.where(np.isfinite(x),x,bed0+.01)
    for code,scope in [(0,'Three wards'),(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
     use=common.ravel()&((ward>0)if code==0 else ward==code);hit=use&(increase>=.10);unions[scope]|=hit;peaks[scope]=max(peaks[scope],int(hit.sum()));outside[scope]=max(outside[scope],int((hit&~active).sum()))
   for scope in unions:head.append({'case':run['case_id'],'stage':run['stage'],'reference':'S1 static','scope':scope,'peak_simultaneous_certain_head_rise_m2':peaks[scope]*100,'any_time_certain_head_rise_union_m2':int(unions[scope].sum())*100,'peak_outside_actual_intakes_m2':outside[scope]*100,'threshold_m':.10,'dry_censor_bound_m':.01})
 pd.DataFrame(head).to_csv(P/'head_transfer_vs_s1_static.csv',index=False)
 for name in ['postprocessing_validation.json','expanded_transfer_validation.json']:assert json.loads((br/name).read_text())['passed']
 dump(R/'final_summary_validation.json',{'passed':True,'two_cases_original_28_plus_8_checks':True,'same_common_road_support':True,'same_noaction_states':True,'same_time_direct_comparison_against_static_S1':True,'head_transfer_vs_both_references':True,'new_hydraulic_runs':2,'capacity_changed':False,'other_events_enabled':False,'population_recomputed':False,'training':False,'sealed_test':False})
 print('S1R_COMPARISON_COMPLETE',flush=True)

def report():
 if not(P/'run_manifest.json').exists():return
 manifest=json.loads((P/'run_manifest.json').read_text());state=json.loads((R/'campaign_state.json').read_text())if(R/'campaign_state.json').exists()else {'status':'preflight_passed','completed_cases':[],'new_runs_started':0}
 if state['status']=='qualified_complete'and not(R/'final_summary_validation.json').exists():finalize()
 rows=[]
 for run in manifest['runs']:rows.append({'Stage':run['stage'],'Portfolio':'S1R: same319locations,within-pool reallocation','State':'qualified'if run['case_id']in state['completed_cases']else state['status']if state.get('current_case')==run['case_id']else'waiting'})
 refresh='<meta http-equiv="refresh" content="30">'if state['status']not in ['held','qualified_complete']else''
 body='<h1>S1原位置：固定分配与池内调配对照</h1><aside><b>首批仅7月高／低水位两组。</b>319个原取水位置、原接收位置、原预留权重、每池0.5m³/s和总2.5m³/s不变，只切换到已核验的池内调配。复用S1静态与无措施结果；不增加容量、不自动扩到其他暴雨。</aside>'
 body+=f"<p><b>状态：{html.escape(state['status'])}</b>；数值核验通过 {len(state['completed_cases'])}/2，新启动 {state.get('new_runs_started',0)}/2。</p>"+pd.DataFrame(rows).to_html(index=False)
 if state.get('current_case'):body+=f"<p>当前：{html.escape(state['current_case'])}；实际模型时钟 {state.get('latest_native_time_s',0)/3600:.3f}/9 h；活跃计算 {state.get('case_active_wall_seconds',0):.0f}秒。</p>"
 if state.get('error'):body+='<p class="warn">队列已保留并停止扩跑：'+html.escape(state['error'])+'</p>'
 body+='<h2>比较与停止条件</h2><p>原雷达雨型、屋面分配、10m网格、城市几何、水位边界、质量掩膜、109帧、0.30m封路规则保持不变。原28项数值核验及8项抽排核验不放宽；任一失败停止队列，不自动重试或调参。串行6CPU/7GiB，每组活跃计算最多2小时，两组合计4小时；每次审核或后处理最多1小时。新增输出上限60GiB，剩余空间至少100GiB。PAUSE/STOP文件可暂停或停止。</p><p>内部O1标识仅用于选择已有调配算子与审核接口，<b>本实验不是原62位置的O1方案</b>。公开方案标识为S1R，即S1位置加池内调配。取水位置数不是实体机器台数；同额度不等于同成本。接收河道为规定水位，未模拟新增河道蓄水和真实下游接收能力。</p>'
 if(P/'road_comparison.csv').exists():
  d=pd.read_csv(P/'road_comparison.csv');d=d[d.scope.eq('Three wards')&d.branch.eq('lower')];body+='<h2>三区道路对照：主水深分支</h2>'+d.to_html(index=False,float_format=lambda v:f'{v:,.5f}')
  body+='<p>reference=S1 static表示同布局调配的增量；reference=No action表示相对无措施的总体收益。km h是长度—时间积分，不是唯一道路公里数。恢复与新增封路分列，不用区域净收益抵消局部副作用。</p>'
 if(P/'head_transfer_vs_s1_static.csv').exists():body+='<h2>相对S1静态方案的水位抬升</h2>'+pd.read_csv(P/'head_transfer_vs_s1_static.csv').to_html(index=False)+'<p>原生水头升高≥0.10m的最大同时面积和任意时刻并集分别报告，不与微小的道路封路阈值穿越混为一谈。相对无措施结果另见该批expanded_head_transfer_summary.csv。</p>'
 body+='<p>首批通过后自动汇总并停在本批。没有新的人口或需求分配、模型训练、封存测试访问，也不覆盖既有论文图和AnaSOP。</p><p><a href="../../../processed/'+TAG+'/protocol.json">Protocol / budget</a> · <a href="preflight_validation.json">Preflight</a> · <a href="qualification.json">Operator checks</a> · <a href="campaign_state.json">Live state</a></p>'
 style='body{font:16px/1.8 system-ui;max-width:1280px;margin:28px auto;padding:0 22px;background:#f8fafb;color:#243c47}aside{background:#e7f0f5;padding:18px}table{display:block;max-width:100%;overflow:auto;border-collapse:collapse;font-size:13px;background:white}td,th{border:1px solid #ccd7df;padding:8px;text-align:left}.warn{background:#fff1d9;padding:16px}a{color:#176a96}'
 tmp=R/'review.html.tmp';tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+refresh+'<title>S1 Same-Layout Allocation Pilot</title><style>'+style+'</style>'+body+'</html>');tmp.replace(R/'review.html')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['qualify','prepare','run','audit','post','report']);ap.add_argument('--event',choices=['O1']);ap.add_argument('--case');a=ap.parse_args();configure()
 if a.phase=='qualify':qualify()
 elif a.phase=='prepare':prepare()
 elif a.phase=='run':worker.run()
 elif a.phase=='audit':control.audit('O1',a.case)
 elif a.phase=='post':control.post('O1')
 else:report()
if __name__=='__main__':main()
