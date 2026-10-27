#!/usr/bin/env python3
"""Predeclared O1/S1 H/L experiments. Local serial worker, fail closed, no retries."""
from pathlib import Path
import os,sys,json,time,shutil,subprocess,argparse,inspect,html
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
import numpy as np,pandas as pd
from shared_receiver_pilot_common_v1 import sha,dump,PY,VENV,command
from shared_receiver_control_support_v1 import treatment_hash,identity_receipt_check,independent_fill,expected_requested_and_transfer
import shared_receiver_transfer_v1 as worker
TAG='three_ward_shared_receiver_controls_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
OLD=ROOT/'data/processed/three_ward_shared_receiver_pilot_v1';A=ROOT/'data/processed/three_ward_shared_receiver_allocation_v1';C=ROOT/'data/processed/three_ward_intake_receiver_candidates_v1';SCRIPT=Path(__file__).resolve()
ARMS=['O1','S1'];POOLS={17:'Sumida',71:'Kamejima',79:'Furukawa',111:'Furukawa',106:'Nihonbashi',110:'Kanda',51:'Unidentified'}
def dirs(arm):return P/'batches'/arm,R/'batches'/arm

def make_layout():
 bp,br=dirs('S1');bp.mkdir(parents=True,exist_ok=True);br.mkdir(parents=True,exist_ok=True)
 src=pd.read_parquet(C/'source_candidates_preprocessed.parquet');link=pd.read_parquet(C/'intake_receiver_links_preprocessed.parquet');recv=pd.read_parquet(C/'receiver_candidates_preprocessed.parquet')
 with np.load(C/'candidate_index.npz') as z:
  grid=z['source_grid'];target=z['target_storage'];si=z['link_source'];ri=z['link_receiver'];distance=z['link_distance'].reshape(7,-1);eligible=z['source_eligible'];native=z['source_native'];rn=z['receiver_native']
 n=len(grid);codes=link['Receiver Group'].to_numpy().reshape(7,n)[:,0];assert np.array_equal(si,np.tile(np.arange(n),7));allpass=np.ones((7,n),bool);maxex=np.zeros(n);positive_any=np.zeros(n,bool)
 for stage in ['H','L']:
  with np.load(C/f'E20240731__{stage}__U_saved_states.npz') as z:
   ex=np.maximum(z['source_volume']-target,0);ex[:,src['Ward Code'].to_numpy()<=0]=0.;positive=ex>1e-6
   lift=np.maximum(z['receiver_stage'][:,ri]-z['source_stage'][:,si],0).reshape(109,7,n)
   passing=(lift+float(z['receiver_alignment_bound'])<=5.)&((z['source_stage']-z['source_minimum'])>=.05)[:,None,:]
   allpass&=(~positive[:,None,:]|passing).all(axis=0);maxex=np.maximum(maxex,ex.max(axis=0));positive_any|=positive.any(axis=0)
 allowed=allpass&eligible[None,:]&positive_any[None,:]&(codes!=51)[:,None]&(distance<=1000.)
 gm=pd.read_parquet(ROOT/'data/processed/three_ward_protection_service_demand_plan_v1/road_policy_membership_preprocessed.parquet');cg=pd.read_parquet(ROOT/'data/processed/three_ward_protection_requirements_register_v1/road_control_cell_group_links_preprocessed.parquet')
 joined=cg.merge(gm[['Road Group','Nonincident Anchor Loss Under Group Probe','Backbone Candidate']],on='Road Group',validate='many_to_one')
 prior=joined.groupby('Grid Index').agg(dependency=('Nonincident Anchor Loss Under Group Probe','max'),backbone=('Backbone Candidate','max')).reindex(grid).fillna(0)
 dep=prior.dependency.to_numpy();back=prior.backbone.to_numpy(dtype=int);request=maxex/1800;rate=np.zeros(n);outlet=np.full(n,-1,int);remain={p:.5 for p in set(POOLS.values()) if p!='Unidentified'}
 order=np.lexsort((grid,-request,-back,-dep))
 for i in order:
  for j in sorted(np.flatnonzero(allowed[:,i]),key=lambda j:(distance[j,i],codes[j])):
   pool=POOLS[int(codes[j])];q=min(.5,request[i],max(0.,remain[pool]))
   if q<=1e-9:continue
   rate[i]=q;outlet[i]=j;remain[pool]=max(0.,remain[pool]-q);break
 chosen=np.flatnonzero(rate>0);rows=[]
 for k,i in enumerate(chosen,1):
  j=outlet[i];r=ri[j*n+i]
  rows.append({'Pump ID':k,'Portfolio':'S1_transport_priority_static','Source Candidate':src['Source Candidate'].iloc[i],'Grid Index':int(grid[i]),'Ward Code':int(src['Ward Code'].iloc[i]),'Source Native Cell One Based':int(native[i])+1,'Receiver Native Cell One Based':int(rn[r])+1,'Receiver Candidate':recv['Receiver Candidate'].iloc[r],'Receiver Group':int(codes[j]),'Receiving Pool':POOLS[int(codes[j])],'Reserved Rate':rate[i],'Standalone Rate Ceiling':.5,'Straight Distance Lower Bound':distance[j,i],'Reference Max Local Rate Request':request[i],'Reference Max Local Excess':maxex[i],'Selected Location Meaning':'native modeled intake;not machine count','Field Deployment Verified':False,'Dependency Priority':int(dep[i]),'Backbone Priority':bool(back[i])})
 sites=pd.DataFrame(rows);assert len(sites)>0 and len(sites)<=10000 and sites['Source Native Cell One Based'].is_unique
 sites.to_csv(bp/'pump_sites.csv',index=False)
 with (bp/'udt_pumps.dat').open('w') as f:
  f.write(str(len(sites))+'\n')
  for row in rows:f.write(f"{row['Source Native Cell One Based']} {row['Receiver Native Cell One Based']} {row['Reserved Rate']:.17e} 0.05 5.0\n")
 assert sum(rate)<=2.5+1e-12 and all(remain[p]>=0 for p in remain)
 dump(br/'allocation_validation.json',{'passed':True,'derived_native_intakes':len(sites),'reserved_total_m3_s':float(sum(rate)),'pool_reserved_m3_s':sites.groupby('Receiving Pool')['Reserved Rate'].sum().to_dict(),'selection_events':['E20240731__H__U','E20240731__L__U'],'priority':'nonincident dependency,backbone,max local request,gridID','unchanged_ground_link_and_head_trigger_screens':True,'not_official_pumps':True})
 return sites

def prepare():
 assert not(P/'run_manifest.json').exists(),'Preserve existing campaign'
 assert json.loads((R/'solver_qualification.json').read_text())['passed']
 assert json.loads((R/'qualification/auditor_regression/regression_validation.json').read_text())['passed']
 new=json.loads((P/'solver_binding.json').read_text());old=json.loads((ROOT/'data/processed/three_ward_barrier_drainage_v1/solver_binding.json').read_text())
 originals=json.loads((OLD/'run_manifest.json').read_text())['runs'];frozen=pd.read_csv(OLD/'pump_sites.csv');made=make_layout()
 budget=dict(maximum_new_runs=4,concurrent_hydraulics=1,cpus=6,memory_gib=7,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=28800,no_clock_progress_stop_s=900,maximum_new_output_gib=100,minimum_free_gib=100,maximum_audit_wall_s=3600,maximum_postprocess_wall_s=3600,maximum_access_wall_s=1800,automatic_retries=0)
 plan=json.loads((ROOT/'data/exp/estimation/three_ward_measure_bottleneck_diagnosis_v1/next_round_plan.json').read_text())
 protocol={'source_plan_is_design_snapshot':True,'execution_governed_by_top_level_protocol':True,'status':'authorized_prepared','decision':'MILI-D-20261021-001','events':['E20240731'],'contrasts':['O1','S1'],'stage_order':['H','L'],'plan':plan,'budget':budget,'new_hydraulic_runs':4,'off_target_definition':'outside13214candidate road-control cells,not merely active intakes','original_28_numerical_gates':'unchanged','pump_checks':'8 invariants,independent dynamic rule reconstruction forO1;static rate/storage reconstruction forS1','receivers':'prescribed stages;real additional receiving storage not modeled','training':False,'sealed_test':False,'C1_enabled':False,'later_event_expansion_enabled':False}
 dump(P/'protocol.json',protocol);runs=[];protected={};memo={}
 def checksum(f):
  f=Path(f);key=(str(f.resolve()),f.stat().st_size,f.stat().st_mtime_ns)
  if key not in memo:memo[key]=sha(f)
  return memo[key]
 for arm in ARMS:
  bp,br=dirs(arm);bp.mkdir(parents=True,exist_ok=True);br.mkdir(parents=True,exist_ok=True);(bp/'execution').mkdir(exist_ok=True)
  if arm=='O1':
   shutil.copy2(OLD/'pump_sites.csv',bp/'pump_sites.csv');shutil.copy2(ROOT/originals[0]['folder']/'udt_pumps.dat',bp/'udt_pumps.dat')
  shutil.copy2(OLD/'pool_allowances.csv',bp/'pool_allowances.csv');sites=pd.read_csv(bp/'pump_sites.csv');pnames=sorted(sites['Receiving Pool'].unique());pmap={p:i+1 for i,p in enumerate(pnames)}
  if arm=='O1':
   with (bp/'udt_pool_rules.dat').open('w') as f:
    f.write('UDT_WEIGHTED_POOL_V1\n'+str(len(pmap))+'\n')
    for name,k in pmap.items():f.write(f'{k} 0.5\n')
    f.write(str(len(sites))+'\n')
    for row in sites.to_dict('records'):f.write(f"{row['Pump ID']} {pmap[row['Receiving Pool']]} 0.5\n")
  armruns=[]
  for prior in originals:
   stage=prior['stage'];base=ROOT/prior['original_folder'];case=f'E20240731__{stage}__U__{arm.lower()}';folder=bp/'runs'/case;folder.mkdir(parents=True,exist_ok=False)
   image=new if arm=='O1' else old
   for name,h in prior['input_sha256'].items():
    if name in ['udt_pumps.dat','udt_run_identity.json','udt_run_identity.txt']:continue
    assert checksum(ROOT/prior['folder']/name)==h
    if name=='sfincs_subgrid.nc':(folder/name).symlink_to(Path(prior['large_readonly_source']).resolve())
    else:shutil.copy2(ROOT/prior['folder']/name,folder/name)
   files=['udt_pumps.dat']+(['udt_pool_rules.dat'] if arm=='O1' else [])
   for name in files:shutil.copy2(bp/name,folder/name)
   ident=prior['identity'].copy();ident.update(run_id=case+'__controls_10m',split='external_shared_receiver_controls',patch_sha256=image['patch_sha256'],solver_digest=image['image_id'],treatment_sha256=treatment_hash(folder,files))
   dump(folder/'udt_run_identity.json',ident);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v) for k,v in ident.items())+'\n')
   names=[n for n in prior['input_sha256'] if n not in ['udt_pumps.dat','udt_run_identity.json']]+files+['udt_run_identity.json']
   run={**prior,'case_id':case,'contrast':arm,'treatment_files':files,'measure':'shared_control_'+arm,'folder':str(folder.relative_to(ROOT)),'run_id':ident['run_id'],'split':ident['split'],'identity':ident,'image_id':image['image_id'],'input_sha256':{n:checksum(folder/n) for n in dict.fromkeys(names)}}
   assert (folder/'sfincs.inp').read_bytes()==(base/'sfincs.inp').read_bytes()
   # Bind full existing baseline outputs and audits; never overwrite or rerun them.
   for name in ['sfincs_map.nc','udt_output_clock.csv','execution.json']:protected[str((base/name).relative_to(ROOT))]=checksum(base/name)
   for name,h in run['input_sha256'].items():protected[str((folder/name).relative_to(ROOT))]=h
   from shared_receiver_identity_v2 import CORE,selftest
   assert selftest()['passed']
   fixture=bp/'qualification/identity'/case;fixture.mkdir(parents=True,exist_ok=False)
   for name in files+['udt_run_identity.txt','udt_run_identity.json']:shutil.copy2(folder/name,fixture/name)
   (fixture/'udt_diagnostic_identity.txt').write_text('\n'.join(k+'='+str(ident[k]) for k in CORE)+'\n')
   assert identity_receipt_check(fixture,run)
   for name in files:
    raw=(fixture/name).read_bytes();(fixture/name).write_bytes(raw+b'\n');assert not identity_receipt_check(fixture,run);(fixture/name).write_bytes(raw)
   assert identity_receipt_check(fixture,run)
   armruns.append(run);runs.append(run)
  dump(bp/'protocol.json',{**protocol,'contrast':arm,'source_site_count':len(sites),'reserved_rate':float(sites['Reserved Rate'].sum())})
  dump(bp/'run_manifest.json',{'runs':armruns,'budget':budget,'protocol_sha256':checksum(bp/'protocol.json')})
  for f in [bp/'protocol.json',bp/'run_manifest.json',bp/'pump_sites.csv',bp/'pool_allowances.csv']:protected[str(f.relative_to(ROOT))]=checksum(f)
 dump(P/'run_manifest.json',{'runs':runs,'protocol_sha256':checksum(P/'protocol.json'),'budget':budget,'execution_order':[r['case_id'] for r in runs]})
 for f in [P/'protocol.json',P/'run_manifest.json',P/'solver_binding.json',ROOT/'docs/AnaSOP.md',ROOT/'data/exp/estimation/three_ward_measure_bottleneck_diagnosis_v1/next_round_plan.json']:protected[str(f.relative_to(ROOT))]=checksum(f)
 for f in list(C.glob('*.npz'))+[C/'source_candidates_preprocessed.parquet',C/'intake_receiver_links_preprocessed.parquet',C/'receiver_candidates_preprocessed.parquet',ROOT/'data/processed/three_ward_protection_service_demand_plan_v1/road_policy_membership_preprocessed.parquet',ROOT/'data/processed/three_ward_protection_requirements_register_v1/road_control_cell_group_links_preprocessed.parquet']:
  protected[str(f.relative_to(ROOT))]=checksum(f)
 dump(P/'protected_sources.json',protected)
 code=[SCRIPT,ROOT/'src/analyses/shared_receiver_control_support_v1.py',ROOT/'src/preprocessing/prepare_shared_receiver_control_solver_v1.py',ROOT/'src/preprocessing/shared_receiver_controls_v1/weighted_pool.inc',ROOT/'src/preprocessing/shared_receiver_controls_v1/weighted_pool_fixture.f90']
 for name in ['shared_receiver_transfer_v1.py','shared_receiver_pilot_common_v1.py','audit_shared_receiver_pilot_identity_v3.py','shared_receiver_identity_v2.py','audit_cross_event_protection_run_v1.py','postprocess_shared_receiver_pilot_v1.py','shared_receiver_access_effects_v1.py','temporal_transport_common_v1.py','run_temporal_road_population_v1.py','integrate_night_population_transport_v1.py']:code.append(ROOT/'src/analyses'/name)
 code.append(ROOT/'src/preprocessing/reference_label_policy.py');dump(P/'execution_code_binding.json',{str(f.relative_to(ROOT)):checksum(f) for f in code})
 assert shutil.disk_usage(P).free>100*1024**3
 dump(R/'preflight_validation.json',{'passed':True,'new_cases':4,'predeclared_order':['O1H','O1L','S1H','S1L'],'O1_same_layout':True,'O1_operator_qualified':True,'S1_derived_native_intakes':len(made),'S1_reserved_total_m3_s':float(made['Reserved Rate'].sum()),'original_forcing_and_geometry_exact':True,'C1_or_later_events_enabled':False,'budget':budget})
 (R/'README.md').write_text('''# O1 / S1 separated controls\n\nDecision MILI-D-20261021-001. No prespecified physical machine count. O1 keeps62native intakes and redistributes each0.5m3/s pool at each actual step; S1 derives transport-priority intakes and keeps the original static operator. All four runs retain the original July H/L forcing, geometry, roofs, masks,109frames and28 numerical gates. Dynamic rates are independently reconstructed before the explicit-source/sink numerical audit. Fixed-stage receivers omit added river storage.\n\nPrepare/qualify: `.venv/bin/python src/analyses/shared_receiver_controls_v1.py qualify`, then `prepare`; run `run` once. Worker refuses existing attempts; no automatic retries. A batch of two stages must qualify before road and access summaries. Serial6CPU/7GiB;2h per-run soft budget is enforced as a conservative stop;100GiB new-output budget and100GiB free floor. PAUSE and STOP files in the processed root control the serial worker. C1 and later event expansion are disabled. `review.html` reads saved progress. Completion automatically builds matched road, resident-hour and transfer summaries; no unique stranded people or real congestion accuracy claim.\n''')
 report();print('FOUR_CONTROL_PREFLIGHT_PASSED',len(made),flush=True)

def qualify():
 from netCDF4 import Dataset
 from qualify_stage_barrier_drainage_v1 import clockstop
 new=json.loads((P/'solver_binding.json').read_text());image=new['image_id'];oldimage=new['prior_image_id'];checks=[]
 def check(name,ok):
  checks.append({'check':name,'passed':bool(ok)});dump(R/'qualification_progress.json',{'checks':checks});assert ok,name;print('PASS',name,flush=True)
 for binary in ['weighted_pool_fixture','pump_fixture','positivity_fixture','adaptive_positivity_fixture','adaptive_missing_child_fixture','rain_integral_fixture','radar_block_rain_fixture_v1']:
  outputs=[]
  for threads in [1,6]:
   f=P/'qualification/interfaces'/f'{binary}_{threads}';f.mkdir(parents=True,exist_ok=False)
   cmd=['docker','run','--rm','--platform','linux/amd64','--network','none','--env',f'OMP_NUM_THREADS={threads}','--mount',f'type=bind,source={f},target=/data','--entrypoint','/opt/udt-build/'+binary,image]
   z=subprocess.run(cmd,capture_output=True,text=True,timeout=120);(f/'stdout.txt').write_text(z.stdout);(f/'stderr.txt').write_text(z.stderr);check(binary+'_'+str(threads),z.returncode==0 and 'PASS' in z.stdout);outputs.append(z.stdout)
   if binary=='weighted_pool_fixture':
    df=pd.read_csv(f/'weighted_fixture_cases.csv')
    for k,g in df.groupby('case'):
     expect=independent_fill(g.weight.to_numpy(),g.limit.to_numpy()[None,:],np.array([g.cap.iloc[0]]))[0]
     assert np.max(abs(expect-g.amount.to_numpy()))<2e-14
    check('independent_200_vectors_'+str(threads),True)
    # All 5 real calls have one complete per-step source/sink trace with known receiver IDs.
    tr=pd.read_csv(f/'udt_pump_steps.csv');check('real_sidecar_trace_'+str(threads),len(tr)==15 and np.array_equal(tr.pump_id.to_numpy(),np.tile([1,2,3],5)))
  check(binary+'_thread_replay',outputs[0]==outputs[1])
 # Same real input, new static path must be bitwise identical to the old image.
 base=json.loads((OLD/'run_manifest.json').read_text())['runs'][0];results=[]
 for name,img,dynamic in [('prior_static',oldimage,False),('new_static',image,False),('new_dynamic',image,True)]:
  f=P/'qualification/prefix'/name;f.mkdir(parents=True,exist_ok=False);source=ROOT/base['folder']
  for n in base['input_sha256']:
   if n=='sfincs_subgrid.nc':(f/n).symlink_to(Path(base['large_readonly_source']).resolve())
   else:shutil.copy2(source/n,f/n)
  (f/'sfincs.inp').write_text(clockstop((f/'sfincs.inp').read_text(),600))
  if dynamic:
   sites=pd.read_csv(OLD/'pump_sites.csv');names=sorted(sites['Receiving Pool'].unique());mp={p:i+1 for i,p in enumerate(names)}
   with (f/'udt_pool_rules.dat').open('w') as s:
    s.write('UDT_WEIGHTED_POOL_V1\n5\n'+''.join(f'{i} 0.5\n' for i in range(1,6))+'62\n')
    for row in sites.to_dict('records'):s.write(f"{row['Pump ID']} {mp[row['Receiving Pool']]} 0.5\n")
  ident=base['identity'].copy();ident.update(run_id=name,solver_digest=img,patch_sha256=new['patch_sha256'] if img==image else base['identity']['patch_sha256'])
  (f/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v) for k,v in ident.items())+'\n')
  run={**base,'folder':str(f.relative_to(ROOT)),'image_id':img};cmd=command(run,'udt-control-prefix-'+name)
  with (f/'stdout.txt').open('w') as out:res=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=600)
  check(name+'_completion',res.returncode==0 and 'Closing off SFINCS' in (f/'sfincs.log').read_text())
  led=pd.read_csv(f/'udt_step_budget.csv');check(name+'_positivity',led.negative_native_count.eq(0).all() and led.minimum_native_volume_since_initialization.ge(0).all())
  scale=np.maximum.reduce([led.rain_applied_cumulative.to_numpy()+led.boundary_in_cumulative.to_numpy(),led.boundary_out_cumulative.to_numpy()+led.other_sink_cumulative.to_numpy(),np.ones(len(led))]);check(name+'_budget',np.all(abs(led.budget_residual)<=np.maximum(1,.01*scale)) and abs(led.budget_residual.iloc[-1])<=max(1,.001*scale[-1]))
  results.append(f)
 with Dataset(results[0]/'sfincs_map.nc') as a,Dataset(results[1]/'sfincs_map.nc') as b:
  for k in ['zs','subgrid_volume','u','v']:check('static_path_exact_'+k,np.array_equal(np.ma.asarray(a[k][:]).filled(np.nan),np.ma.asarray(b[k][:]).filled(np.nan),equal_nan=True))
 check('static_clock_budget_exact',np.array_equal(pd.read_csv(results[0]/'udt_step_budget.csv').to_numpy(),pd.read_csv(results[1]/'udt_step_budget.csv').to_numpy()))
 # Verify full-grid O1 trace against independent allocation and actual source/sink ledger.
 f=results[2];sites=pd.read_csv(OLD/'pump_sites.csv');pump=np.loadtxt(f/'udt_pumps.dat',skiprows=1,ndmin=2);tr=pd.read_csv(f/'udt_pump_steps.csv');n=len(pump);led=pd.read_csv(f/'udt_step_budget.csv');matrix={k:tr[k].to_numpy().reshape(-1,n) for k in ['source_prior_volume_m3','head_m','depth_m']};dt=led.actual_step_dt.to_numpy()[1:];req,v=expected_requested_and_transfer({'contrast':'O1'},pump,sites,matrix,dt,f,tr)
 check('full_grid_dynamic_allocation_exact',np.max(abs(v-tr.transferred_m3.to_numpy().reshape(-1,n)))<1e-12)
 check('full_grid_dynamic_source_sink',np.max(abs(np.cumsum(tr.transferred_m3.to_numpy().reshape(-1,n).sum(axis=1))-led.other_sink_cumulative.to_numpy()[1:]))<1e-7)
 new.update(unit_qualified=True,hydraulic_prefix_qualified=True);dump(P/'solver_binding.json',new);dump(R/'solver_qualification.json',{'passed':True,'checks':checks,'image_id':image,'prefix_horizon_s':600,'static_path_exact':True,'original_rules_changed':False});print('CONTROL_SOLVER_QUALIFIED',flush=True)

def audit(arm,case):
 import audit_shared_receiver_pilot_identity_v3 as original
 text=inspect.getsource(original);text=text.replace('from shared_receiver_identity_v2 import identity_receipt_check','from shared_receiver_control_support_v1 import identity_receipt_check,expected_requested_and_transfer')
 text=text.replace('assert len(pump)==n==62','assert len(pump)==n')
 anchor="expected=dt[target,None]*pump[None,:,2];req=matrix['requested_m3'];v=matrix['transferred_m3'];available=matrix['source_prior_volume_m3'];assert np.max(abs(req-expected),initial=0.)<1e-12"
 assert text.count(anchor)==1
 text=text.replace(anchor,"expected,wanted=expected_requested_and_transfer(run,pump,sites,matrix,dt[target],folder,frame);req=matrix['requested_m3'];v=matrix['transferred_m3'];available=matrix['source_prior_volume_m3'];assert np.max(abs(req-expected),initial=0.)<1e-12 and np.max(abs(v-wanted),initial=0.)<1e-12")
 text=text.replace('cum=0.;step=0;', 'active_seconds=np.zeros(n);trigger_seconds=np.zeros(n);head_seconds=np.zeros(n);cum=0.;step=0;',1)
 text=text.replace('total_by_site+=v.sum(axis=0);', 'active_seconds+=np.sum((v>0)*dt[target,None],axis=0);trigger_seconds+=np.sum((matrix["depth_m"]<pump[None,:,3])*dt[target,None],axis=0);head_seconds+=np.sum((matrix["head_m"]>pump[None,:,4])*dt[target,None],axis=0);total_by_site+=v.sum(axis=0);',1)
 text=text.replace("sites['Actual Transferred Volume']=total_by_site;", "sites['Active Native Seconds']=active_seconds;sites['Trigger Blocked Native Seconds']=trigger_seconds;sites['Head Blocked Native Seconds']=head_seconds;sites['Actual Transferred Volume']=total_by_site;",1)
 text=text.replace("'variable_reserved_rates_exact':True","'allocation_rule_exact':True")
 text=text.replace('shared_receiver_identity_v2:exact10nativeKeys+hashBoundTreatment','control_identity:exact10nativeKeys+hashBoundAllTreatmentFiles')
 env={'__name__':'control_auditor','__file__':original.__file__};exec(compile(text,original.__file__,'exec'),env);env['P'],env['R']=dirs(arm);env['main'](case)

def access(arm):
 import shared_receiver_access_effects_v1 as mod
 bp,br=dirs(arm);ap=bp/'access';ar=br/'access';ap.mkdir(exist_ok=True);ar.mkdir(exist_ok=True)
 for name in ['node_cache','population','anchor_nodes']:(ap/name).mkdir(exist_ok=True)
 mod.P=ap;mod.R=ar;mod.HP=bp;mod.HR=br;mod.report=lambda:None
 proto=json.loads((ROOT/'data/processed/three_ward_shared_receiver_access_v1/protocol.json').read_text());proto.update(decision='MILI-D-20261021-001',contrast=arm);dump(ap/'protocol.json',proto)
 binding=mod.bindings();binding={k:v for k,v in binding.items() if not k.startswith('data/processed/three_ward_shared_receiver_pilot_v1/')};binding.update({str(f.relative_to(ROOT)):sha(f) for f in [SCRIPT,bp/'protocol.json',bp/'run_manifest.json',ap/'protocol.json']});dump(ap/'input_bindings.json',binding)
 from temporal_transport_common_v1 import Reachability,population_inputs
 blocks,pop,normal,infos=population_inputs();eng=Reachability();assert np.array_equal(eng.direct(np.zeros(len(mod.load(mod.TEMP/'clipped_road_lengths.npz')['qualified']),bool)),normal);assert pop.sum()==496345 and len(blocks)==4746 and mod.routing_selfcheck()
 dump(ar/'preflight_validation.json',dict(passed=True,original_normal_graph_replayed=True,original_population_and_anchor=True));mod.run()

def post_module(name,arm):
 import postprocess_shared_receiver_pilot_v1 as parent
 import types
 assert name=='postprocess_shared_receiver_pilot_v1'
 text=inspect.getsource(parent)
 anchor="target[load(D/'three_ward_intake_receiver_candidates_v1/candidate_index.npz')['source_grid']]=True"
 assert text.count(anchor)==1
 text=text.replace(anchor,anchor+";active_intakes=np.zeros_like(target);active_intakes[pd.read_csv(P/'pump_sites.csv')['Grid Index'].to_numpy()]=True;assert np.all(~active_intakes|target)")
 anchor="'Interpretation':'10m cell area km2,conservative dry censor bound;not measured flood area'"
 assert text.count(anchor)==1
 text=text.replace(anchor,"'All Supported Certain Head Rise Area':float(np.count_nonzero(use&(increase>=.10))*.0001),'Outside Actual Intakes Certain Head Rise Area':float(np.count_nonzero(use&~active_intakes&(increase>=.10))*.0001),'Within Candidate Control Area Certain Head Rise Area':float(np.count_nonzero(use&target&(increase>=.10))*.0001),"+anchor)
 env={'__name__':'control_postprocess','__file__':parent.__file__};exec(compile(text,parent.__file__,'exec'),env);env['P'],env['R']=dirs(arm)
 return types.SimpleNamespace(**env)

def post(arm):
 worker.road_postprocess(arm);bp,br=dirs(arm)
 df=pd.read_csv(bp/'off_target_head_changes.csv')
 assert np.max(np.abs(df['All Supported Certain Head Rise Area']-df['Within Candidate Control Area Certain Head Rise Area']-df['Outside Target Cells Certain Head Rise Area']))<1e-10
 cols=['All Supported Certain Head Rise Area','Outside Actual Intakes Certain Head Rise Area','Within Candidate Control Area Certain Head Rise Area','Outside Target Cells Certain Head Rise Area']
 df.groupby(['Case','Stage','Scope'],as_index=False)[cols].max().to_csv(bp/'expanded_head_transfer_summary.csv',index=False)
 dump(br/'expanded_transfer_validation.json',{'passed':True,'original_candidate_exclusion_unchanged':True,'supplementary_whole_domain_and_actual_intake_exclusion_added':True,'area_unit':'km2','threshold_m':.10,'dry_censor_bound_m':.01,'all_equals_inside_plus_outside_candidate_each_frame':True,'downstream_receiver_added_storage_not_modeled':True})

def report():
 m=json.loads((P/'run_manifest.json').read_text()) if (P/'run_manifest.json').exists() else {'runs':[]};s=json.loads((R/'campaign_state.json').read_text()) if (R/'campaign_state.json').exists() else {'status':'prepared'}
 body='<h1>同预算调配与布局：四组分离对照</h1><aside><b>状态：'+html.escape(s['status'])+'</b><br>已通过：'+str(len(s.get('completed_cases',[])))+'/4；当前：'+html.escape(str(s.get('current_case') or '—'))+'</aside>'
 if 'latest_native_time_s' in s:body+=f'<p>当前原生计算时刻：{s["latest_native_time_s"]/3600:.2f} / 9.00 h；该时刻不是实际等待时间。</p>'
 if s.get('error'):body+='<p class="warn">已暂停队列：'+html.escape(s['error'])+'</p>'
 body+='<p>O1保留原62个原生取水单元，在同一接收池内调配闲置预留；S1按交通依赖优先重新选择位置，保留静态抽排规则。每池假设额度0.5m³/s，共2.5m³/s；位置数量由分配产生，不是设备台数。同预算指总流量额度相同，不意味着设备、管线或经济成本相同。两组分别跑原7月H/L，不叠加、不扩展C1。</p>'
 for arm in ARMS:
  bp,br=dirs(arm);body+='<h2>'+arm+'</h2>'
  if (bp/'pump_sites.csv').exists():
   sites=pd.read_csv(bp/'pump_sites.csv');body+=f'<p>{len(sites)}个原生取水位置，静态预留／调配权重合计{sites["Reserved Rate"].sum():.6f}m³/s。</p>'
  if (bp/'paired_road_effects.csv').exists():
   d=pd.read_csv(bp/'paired_road_effects.csv');d=d[d['Depth Branch'].eq('lower')&d.Scope.eq('Three wards')];body+=d[['Stage','Closure Reduction km h','Closure Reduction Percent','Maximum Same Time Newly Closed km']].to_html(index=False)
  if (bp/'expanded_head_transfer_summary.csv').exists():
   d=pd.read_csv(bp/'expanded_head_transfer_summary.csv');d=d[d.Scope.eq('Three wards')];body+='<p>三区水头抬升≥0.10m的最大同时面积（km²），分别列全域、实际取水位置之外及原候选控制区之外：</p>'+d.to_html(index=False)
  if (bp/'access/paired_population_summary.csv').exists():
   d=pd.read_csv(bp/'access/paired_population_summary.csv');d=d[d.bound.eq('lower')&d.buffer_m.eq(10)&d.scope.eq('Three wards')];body+=d[['stage','net_avoided_loss_lower_resident_hours','net_avoided_loss_upper_resident_hours']].to_html(index=False)
 if s['status']=='qualified_complete':
  finalize();body+='<p><a href="comparison_summary.json">相对原静态部署的统一汇总</a></p>'
 body+='<p>保留原28项数值门槛，独立核验每一步的身份、时钟、守恒正性和共享额度。数值核验不等于历史道路实测验证。人口为2020年住地情景及原位置区间；路径成本不是真实拥堵延误。目标区外面积沿用13,214个候选道路控制单元以外的范围，并非仅排除取水单元；规定接收水位未模拟新增河道蓄水，不能排除下游风险。</p><p><a href="campaign_state.json">进度JSON</a> · <a href="preflight_validation.json">预检查</a> · <a href="solver_qualification.json">新接口核验</a> · <a href="README.md">复现与暂停方式</a></p>'
 style='body{font:16px/1.8 system-ui;max-width:1200px;margin:30px auto;padding:0 22px;color:#203b4a;background:#f6f8fa}aside{background:#e3eef2;padding:18px}.warn{background:#ffefcf}table{display:block;overflow:auto;background:white;border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccd9df;padding:9px;text-align:left}'
 tmp=R/'review.html.tmp';tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+('' if s['status'] in ['qualified_complete','held'] else '<meta http-equiv="refresh" content="30">')+'<title>O1 / S1 Four Controls</title><style>'+style+'</style>'+body+'</html>');tmp.replace(R/'review.html')

def finalize():
 if (R/'comparison_summary.json').exists():return
 # No outcome-selected branch or shortened window; exact support is required.
 supports=[np.load(OLD/'common_road_support.npz')['qualified_roads']]+[np.load(dirs(a)[0]/'common_road_support.npz')['qualified_roads'] for a in ARMS]
 assert all(np.array_equal(supports[0],z) for z in supports[1:]),'Support changed;hold cross-arm interpretation for harmonization'
 road=pd.read_csv(OLD/'paired_road_effects.csv');pop=pd.read_csv(ROOT/'data/processed/three_ward_shared_receiver_access_v1/paired_population_summary.csv');rows=[]
 for arm in ARMS:
  bp,br=dirs(arm);rr=pd.read_csv(bp/'paired_road_effects.csv');pp=pd.read_csv(bp/'access/paired_population_summary.csv')
  for stage in ['H','L']:
   r=rr[rr.Stage.eq(stage)&rr['Depth Branch'].eq('lower')&rr.Scope.eq('Three wards')].iloc[0];b=road[road.Stage.eq(stage)&road['Depth Branch'].eq('lower')&road.Scope.eq('Three wards')].iloc[0]
   p=pp[pp.stage.eq(stage)&pp.bound.eq('lower')&pp.buffer_m.eq(10)&pp.scope.eq('Three wards')].iloc[0]
   rows.append({'contrast':arm,'stage':stage,'closure_reduction_vs_noaction_km_h':float(r['Closure Reduction km h']),'incremental_closure_reduction_vs_static_km_h':float(r['Closure Reduction km h']-b['Closure Reduction km h']),'newly_closed_vs_noaction_max_km':float(r['Maximum Same Time Newly Closed km']),'paired_vs_noaction_resident_hours_lower':float(p.net_avoided_loss_lower_resident_hours),'paired_vs_noaction_resident_hours_upper':float(p.net_avoided_loss_upper_resident_hours)})
 # Reconstruct treatment-vs-static paired candidate changes, not marginal-bound subtraction.
 from temporal_transport_common_v1 import population_inputs
 from run_temporal_road_population_v1 import envelope,signed_envelope
 blocks,stock,normal,infos=population_inputs();ward=blocks.ward.to_numpy(dtype=str)
 selectors=[np.ones(len(stock),bool)]+[ward==w for w in ['13101','13102','13103']];scopes=['Three wards','Chiyoda','Chuo','Minato']
 oldaccess=ROOT/'data/processed/three_ward_shared_receiver_access_v1';temp=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
 endpoints=np.load(ROOT/'data/processed/three_ward_normal_vehicle_repair_v1/multi_endpoint_components.npz')['native_node_index']
 poprows=[];endrows=[];continuity=[]
 members=pd.read_parquet(ROOT/'data/processed/three_ward_protection_requirements_register_v1/road_group_members_preprocessed.parquet')
 gm=pd.read_parquet(ROOT/'data/processed/three_ward_protection_service_demand_plan_v1/road_policy_membership_preprocessed.parquet').set_index('Road Group')['Priority Evidence'].to_dict()
 clip=np.load(temp/'clipped_road_lengths.npz');weights=clip['union_length_m']/1000
 def node(access,key):
  f=access/'node_cache'/(str(key)+'.npz')
  if not f.exists():f=temp/'reachability_cache'/(str(key)+'.npz')
  with np.load(f) as z:return np.unpackbits(z['node_ok_packed'],count=len(normal)).astype(bool)
 for arm in ARMS:
  bp,br=dirs(arm)
  for stage in ['H','L']:
   case=f'E20240731__{stage}__U__{arm.lower()}';oldcase=f'E20240731__{stage}__U__shared_r1000_q0p5'
   with np.load(oldaccess/'population'/(oldcase+'.npz')) as z:oldkeys=z['node_cache_keys'];oldtimes=z['analysis_times'];oldids=z['block_ids']
   with np.load(bp/'access/population'/(case+'.npz')) as z:newkeys=z['node_cache_keys'];times=z['analysis_times'];assert np.array_equal(z['block_ids'],oldids) and np.array_equal(times,oldtimes)
   local=[]
   for ti in range(109):
    for bi,bound in enumerate(['lower','upper']):
     a=node(oldaccess,oldkeys[ti,bi,1]);b=node(bp/'access',newkeys[ti,bi,1]);delta=a.astype('i1')-b.astype('i1');delta[~normal]=0
     for buf,info in infos.items():
      low,high=signed_envelope(delta,info,stock);ra,rb=envelope(normal&~a&b,info,stock);na,nb=envelope(normal&a&~b,info,stock)
      for sel,scope in zip(selectors,scopes):
       local.append(dict(contrast=arm,stage=stage,frame=ti,elapsed_s=float(times[ti]),bound=bound,buffer_m=buf,scope=scope,net_lower=int(-high[sel].sum()),net_upper=int(-low[sel].sum()),restored_lower=int(stock[sel&ra].sum()),restored_upper=int(stock[sel&rb].sum()),new_loss_lower=int(stock[sel&na].sum()),new_loss_upper=int(stock[sel&nb].sum())))
     endrows.append(dict(contrast=arm,stage=stage,frame=ti,elapsed_s=float(times[ti]),bound=bound,restored_endpoints_vs_static=int((normal[endpoints]&~a[endpoints]&b[endpoints]).sum()),newly_lost_endpoints_vs_static=int((normal[endpoints]&a[endpoints]&~b[endpoints]).sum())))
   poprows.extend(local)
   with np.load(OLD/'exposure'/(oldcase+'__drainage.npz')) as z:oldcodes=z['road_state_codes'];qi=z['road_indices']
   with np.load(bp/'exposure'/(case+'__drainage.npz')) as z:newcodes=z['road_state_codes'];assert np.array_equal(qi,z['road_indices'])
   lookup={int(k):i for i,k in enumerate(qi)}
   for group,g in members.groupby('Road Group'):
    ii=np.array([lookup[int(k)] for k in g['Road Index'] if int(k) in lookup],dtype=int)
    if not len(ii):continue
    for bi,bound in enumerate(['lower','upper']):
     a=oldcodes[:,bi,ii]==2;b=newcodes[:,bi,ii]==2
     # Advanced indexing keeps time first here; assert shape rather than assuming.
     assert a.shape==(109,len(ii))
     continuity.append(dict(contrast=arm,stage=stage,bound=bound,road_group=group,priority_evidence=gm[group],all_registered_members_qualified=(len(ii)==len(g)),known_group_open_time_gain_h=float(np.trapezoid(a.any(axis=1).astype(int)-b.any(axis=1).astype(int),times)/3600),net_restored_edge_km_h=float(np.trapezoid((a.astype(int)-b.astype(int))@weights[qi[ii]],times)/3600)))
 paired=pd.DataFrame(poprows);assert (paired.net_lower<=paired.net_upper).all()
 paired.to_csv(P/'paired_population_vs_static_trajectories.csv',index=False)
 summary=[]
 for keys,g in paired.groupby(['contrast','stage','bound','buffer_m','scope']):
  g=g.sort_values('frame');row=dict(zip(['contrast','stage','bound','buffer_m','scope'],keys))
  for k in ['net_lower','net_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper']:row[k+'_resident_hours']=float(np.trapezoid(g[k],g.elapsed_s)/3600)
  summary.append(row)
 pd.DataFrame(summary).to_csv(P/'paired_population_vs_static_summary.csv',index=False);pd.DataFrame(endrows).to_csv(P/'endpoint_access_vs_static_trajectories.csv',index=False);pd.DataFrame(continuity).to_csv(P/'group_continuity_vs_static.csv',index=False)
 dump(R/'comparison_summary.json',{'passed':True,'road_common_support_exact':True,'rows':rows,'population':'Direct candidate-level pairing against static deployment retained at0/10/25m;not marginal interval subtraction.','paired_population_vs_static':summary,'transfer':'See per-arm off_target_summary.csv;outside13214candidate road-control cells,fixed receivers omit downstream added storage.','next_expansion_automatic':False})
 pump_rows=[]
 for arm in ARMS:
  bp,br=dirs(arm)
  for run in json.loads((bp/'run_manifest.json').read_text())['runs']:
   site=pd.read_csv(br/'execution'/run['case_id']/'actual_pump_site_totals.csv');site['contrast']=arm;site['stage']=run['stage'];pump_rows.append(site)
 totals=pd.concat(pump_rows,ignore_index=True);totals.to_csv(P/'actual_pump_site_comparison.csv',index=False)
 for group,name in [('Receiving Pool','pool'),('Ward Code','ward')]:
  totals.groupby(['contrast','stage',group],as_index=False)[['Actual Transferred Volume','Active Native Seconds','Trigger Blocked Native Seconds','Head Blocked Native Seconds']].sum().to_csv(P/('pump_'+name+'_operating_summary.csv'),index=False)
 pd.DataFrame(rows).to_csv(P/'comparison_summary.csv',index=False)

def configure_worker():
 worker.P=P;worker.R=R;worker.EVENTS=ARMS;worker.SCRIPT=SCRIPT;worker.report=report;worker.audit=audit;worker.access=access;worker.phase_module=post_module
 # Existing runner dispatches phase functions using contrast identifiers; paths are isolated.
 oldcommand=worker.command
 worker.command=lambda run,name:oldcommand(run,'udt-controls-'+run['contrast'].lower()+'-'+run['stage'].lower())
 # Worker stop handles the same name returned above via the original local variable.
 # Keep the generated name in the old routine identical to the wrapper's expectation.
 text=inspect.getsource(worker.run);text=text.replace("container='udt-transfer-'+event[1:]+'-'+r['stage'].lower()", "container='udt-controls-'+event.lower()+'-'+r['stage'].lower()")
 text=text.replace("if x['event_id']==event","if x['contrast']==event")
 text=text.replace('full_fixed_portfolio_transfer=True,two_new_rainfall_parents=True','separated_allocation_and_layout_controls=True,rainfall_parents=1')
 env=worker.__dict__;exec(compile(text,str(SCRIPT),'exec'),env)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['qualify','prepare','run','audit','post','access','report']);ap.add_argument('--event',choices=ARMS);ap.add_argument('--case');a=ap.parse_args();configure_worker()
 if a.phase=='qualify':qualify()
 elif a.phase=='prepare':prepare()
 elif a.phase=='run':worker.run()
 elif a.phase=='audit':audit(a.event,a.case)
 elif a.phase=='post':post(a.event)
 elif a.phase=='access':access(a.event)
 else:report()
if __name__=='__main__':main()
