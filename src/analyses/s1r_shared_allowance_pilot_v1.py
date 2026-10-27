#!/usr/bin/env python3
"""July H/L receiving-allowance contrast at fixed S1R layout and allocation.
Only five pool caps double0.5->1.0m3/s,total2.5->5.0;unit caps stay0.5.
Reuse qualified weighted solver and original28+8gates;no automatic expansion.
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
TAG='three_ward_s1r_shared_allowance_pilot_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
OLD=ROOT/'data/processed/three_ward_s1_reallocation_pilot_v1/batches/O1'
OLDR=ROOT/'data/exp/estimation/three_ward_s1_reallocation_pilot_v1/batches/O1'
IMAGE_BINDING=ROOT/'data/processed/three_ward_shared_receiver_controls_v1/solver_binding.json'
SCRIPT=Path(__file__).resolve();DECISION='MILI-D-20261022-008'

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
 print('S1R_Q5_OPERATOR_QUALIFIED',flush=True)

def prepare():
 assert not (P/'run_manifest.json').exists(),'Preserve campaign'
 assert json.loads((R/'qualification.json').read_text())['passed']
 assert json.loads((OLDR.parent.parent/'final_summary_validation.json').read_text())['passed']
 assert json.loads((OLDR.parent.parent/'campaign_state.json').read_text())['status']=='qualified_complete'
 assert not subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip(),'Existing Docker workload'
 assert shutil.disk_usage(P).free>=100*1024**3
 image=json.loads(IMAGE_BINDING.read_text());oldruns=json.loads((OLD/'run_manifest.json').read_text())['runs']
 bp,br=dirs();bp.mkdir(parents=True,exist_ok=True);br.mkdir(parents=True,exist_ok=True);(bp/'execution').mkdir(exist_ok=True)
 sites=pd.read_csv(OLD/'pump_sites.csv').sort_values('Pump ID');assert len(sites)==319 and sites['Source Native Cell One Based'].is_unique
 shutil.copy2(OLD/'pump_sites.csv',bp/'pump_sites.csv')
 source=ROOT/oldruns[0]['folder'];shutil.copy2(source/'udt_pumps.dat',bp/'udt_pumps.dat')
 spec=np.loadtxt(bp/'udt_pumps.dat',skiprows=1,ndmin=2)
 assert spec.shape==(319,5)and np.all(spec[:,3]==.05)and np.all(spec[:,4]==5.)
 assert np.allclose(sites.groupby('Receiving Pool')['Reserved Rate'].sum(),.5,rtol=0,atol=1e-12)
 previous_groups,previous_units,previous_caps=pool_rules(source,319)
 assert len(previous_caps)==5 and all(v==.5 for v in previous_caps.values())and np.all(previous_units==.5)
 oldlines=(source/'udt_pool_rules.dat').read_text().splitlines(keepends=True)
 assert oldlines[0].strip()=='UDT_WEIGHTED_POOL_V1'and int(oldlines[1])==5
 newlines=oldlines[:2]+[f'{i} 1.0\n'for i in range(1,6)]+oldlines[7:]
 (bp/'udt_pool_rules.dat').write_text(''.join(newlines))
 groups,units,caps=pool_rules(bp,319)
 assert np.array_equal(groups,previous_groups)and np.array_equal(units,previous_units)
 assert all(caps[g]==2*previous_caps[g]for g in caps)
 assert newlines[:2]==oldlines[:2]and newlines[7:]==oldlines[7:]
 names=sorted(sites['Receiving Pool'].unique());assert len(names)==5
 poolrows=[]
 for i,name in enumerate(names,1):
  m=sites['Receiving Pool'].eq(name)
  assert np.array_equal(np.flatnonzero(groups==i),np.flatnonzero(m))
  poolrows.append({'Receiving Pool':name,'Scenario Shared Allowance':1.,'Previous Shared Allowance':.5,'Frozen Weight Sum':float(sites.loc[m,'Reserved Rate'].sum()),'Activated Model Intakes':int(m.sum()),'Standalone Rate Ceiling':.5,'Allowance Meaning':'hypothetical shared receiving ceiling;not surveyed river or installed pump capacity'})
 poolrows.append({'Receiving Pool':'Unidentified','Scenario Shared Allowance':0.,'Previous Shared Allowance':0.,'Frozen Weight Sum':0.,'Activated Model Intakes':0,'Standalone Rate Ceiling':.5,'Allowance Meaning':'withheld unresolved identity'})
 pd.DataFrame(poolrows).to_csv(bp/'pool_allowances.csv',index=False)
 # Independently check319-site unit/storage/trigger limits at the new pool caps.
 rng=np.random.default_rng(20261022);dt=np.full(9,.4);prior=rng.uniform(0,.5,(9,319));depth=rng.uniform(0,.8,(9,319));head=rng.uniform(0,7,(9,319))
 limits=np.where((depth>=.05)&(head<=5),np.minimum(dt[:,None]*units[None,:],prior/(1+64*np.finfo(np.float32).eps)),0.)
 for g,cap in caps.items():
  m=groups==g;filled=independent_fill(spec[m,2],limits[:,m],cap*dt)
  assert np.all(filled>=0)and np.all(filled<=limits[:,m]+1e-14)
  assert np.allclose(filled.sum(axis=1),np.minimum(limits[:,m].sum(axis=1),cap*dt),rtol=0,atol=1e-13)
 budget=dict(maximum_new_runs=2,concurrent_hydraulics=1,cpus=6,memory_gib=7,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=14400,no_clock_progress_stop_s=900,maximum_new_output_gib=60,minimum_free_gib=100,maximum_audit_wall_s=3600,maximum_postprocess_wall_s=3600,automatic_retries=0)
 protocol={'decision':DECISION,'event':'E20240731','public_portfolio':'S1R-Q5:shared receiving allowance5m3/s','scientific_reference':'qualified S1R-Q2.5 at the same319locations and same weighted allocation','internal_operator_identifier':'O1 selects weighted-pool interface;not the old62-site portfolio','single_changed_factor':'five shared receiving caps0.5->1.0m3/s,total2.5->5.0','frozen':'same319source/outlet cells,original weights,standalone0.5m3/s ceiling,0.05mtrigger,5mlift,allocation,solver,geometry,rain,roofs,stage curves,masks and109maps','pool_count':5,'pool_allowance_m3_s':1.,'total_allowance_m3_s':5.,'previous_total_allowance_m3_s':2.5,'quota_is_not_installed_capacity':True,'quota_is_not_downstream_capacity_evidence':True,'new_hydraulic_runs':2,'stage_order':['H','L'],'existing_Q2p5_and_noaction_reused':True,'solver_image':image['image_id'],'budget':budget,'original_checks':'28 numerical and8 pump invariants;generic declared-pool-cap reconstruction unchanged','main_comparison':'same-time remaining closure and actual extraction versus S1R-Q2.5;No action as contextual reference;restored and new closures separate','head_screen':'native certain head rise>=0.10m versus Q2.5 and no action;simultaneous peak and any-time union separate','frozen_adverse_followup':{'north_up_grid_index':254457,'source':'prior July high-stage S1R head-rise screen','track_all109frames':True,'not_a_new_selection_target':True},'depth_branches':['lower main','upper sensitivity'],'other_parent_expansion_enabled':False,'further_capacity_levels_enabled':False,'population_or_traffic_assignment':False,'training':False,'sealed_test':False,'receivers':'prescribed stages;added river storage and downstream acceptance unmodeled','selection':'July-derived layout;conditional development comparison,not untouched testing or observed flood reconstruction'}
 dump(P/'protocol.json',protocol);dump(bp/'protocol.json',protocol)
 runs=[];protected={};memo={}
 def checksum(f):
  f=Path(f);k=(str(f.resolve()),f.stat().st_size,f.stat().st_mtime_ns)
  if k not in memo:memo[k]=sha(f)
  return memo[k]
 for prior in sorted(oldruns,key=lambda x:['H','L'].index(x['stage'])):
  stage=prior['stage'];src=ROOT/prior['folder'];case=f'E20240731__{stage}__U__s1r_q5';folder=bp/'runs'/case;folder.mkdir(parents=True,exist_ok=False)
  assert prior['image_id']==image['image_id']
  a=json.loads((OLDR/'execution'/prior['case_id']/'audit.json').read_text());b=json.loads((OLDR/'execution'/prior['case_id']/'pump_budget_summary.json').read_text())
  assert a['all_checks_passed']and len(a['checks'])==28 and b['passed']and len(b['shared_receiver_checks'])==8
  entries=[]
  for name,h in prior['input_sha256'].items():
   if name in ['udt_run_identity.txt','udt_run_identity.json']:continue
   assert checksum(src/name)==h;protected[str((src/name).relative_to(ROOT))]=h
   if name=='sfincs_subgrid.nc':(folder/name).symlink_to(Path(prior['large_readonly_source']).resolve())
   else:shutil.copy2(src/name,folder/name)
   entries.append(name)
  assert checksum(folder/'udt_pumps.dat')==checksum(bp/'udt_pumps.dat')
  shutil.copy2(bp/'udt_pool_rules.dat',folder/'udt_pool_rules.dat')
  ident=prior['identity'].copy();ident.update(run_id=case+'__same_layout_shared_quota_10m',split='external_same_s1r_layout_quota',treatment_sha256=treatment_hash(folder,['udt_pumps.dat','udt_pool_rules.dat']))
  dump(folder/'udt_run_identity.json',ident);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v)for k,v in ident.items())+'\n');entries+=['udt_run_identity.json','udt_run_identity.txt']
  run={**prior,'case_id':case,'run_id':ident['run_id'],'folder':str(folder.relative_to(ROOT)),'contrast':'O1','scientific_portfolio':'S1R-Q5','quota_reference_case':prior['case_id'],'quota_reference_folder':prior['folder'],'measure':'same_s1r_layout_shared_quota_5','identity':ident,'split':ident['split'],'input_sha256':{n:checksum(folder/n)for n in entries}}
  run.pop('static_reference_case',None);run.pop('static_reference_folder',None)
  for name in prior['input_sha256']:
   if name not in ['udt_pool_rules.dat','udt_run_identity.txt','udt_run_identity.json']:assert run['input_sha256'][name]==prior['input_sha256'][name],name
  staged=(folder/'udt_pool_rules.dat').read_text().splitlines(keepends=True);previous=(src/'udt_pool_rules.dat').read_text().splitlines(keepends=True)
  assert staged[:2]==previous[:2]and staged[7:]==previous[7:]
  fixture=bp/'qualification/identity'/case;fixture.mkdir(parents=True)
  for name in run['treatment_files']+['udt_run_identity.json','udt_run_identity.txt']:shutil.copy2(folder/name,fixture/name)
  (fixture/'udt_diagnostic_identity.txt').write_text('\n'.join(k+'='+str(ident[k])for k in CORE)+'\n');assert identity_receipt_check(fixture,run)
  for name in run['treatment_files']:
   raw=(fixture/name).read_bytes();(fixture/name).write_bytes(raw+b'\n');assert not identity_receipt_check(fixture,run);(fixture/name).write_bytes(raw)
  assert identity_receipt_check(fixture,run)
  for name,h in run['input_sha256'].items():protected[str((folder/name).relative_to(ROOT))]=h
  for base in [src,ROOT/prior['original_folder']]:
   for name in ['sfincs_map.nc','udt_output_clock.csv','execution.json']:protected[str((base/name).relative_to(ROOT))]=checksum(base/name)
  for arm in ['baseline','drainage']:protected[str((OLD/'exposure'/f"{prior['case_id']}__{arm}.npz").relative_to(ROOT))]=checksum(OLD/'exposure'/f"{prior['case_id']}__{arm}.npz")
  runs.append(run)
 for path in [P,bp]:dump(path/'run_manifest.json',{'runs':runs,'budget':budget,'protocol_sha256':checksum(path/'protocol.json'),'execution_order':[r['case_id']for r in runs]})
 for f in [P/'run_manifest.json',P/'protocol.json',bp/'run_manifest.json',bp/'protocol.json',bp/'pool_allowances.csv',bp/'pump_sites.csv',bp/'udt_pumps.dat',bp/'udt_pool_rules.dat',IMAGE_BINDING,OLD/'common_road_support.npz',OLD/'run_manifest.json',OLD/'pump_sites.csv',OLDR.parent.parent/'final_summary_validation.json']:
  protected[str(f.relative_to(ROOT))]=checksum(f)
 dump(P/'protected_sources.json',protected)
 code=[SCRIPT]+[ROOT/'src/analyses'/n for n in ['shared_receiver_controls_v1.py','shared_receiver_transfer_v1.py','shared_receiver_control_support_v1.py','shared_receiver_pilot_common_v1.py','audit_shared_receiver_pilot_identity_v3.py','audit_cross_event_protection_run_v1.py','shared_receiver_identity_v2.py','postprocess_shared_receiver_pilot_v1.py']]+[ROOT/'src/preprocessing/reference_label_policy.py']
 dump(P/'execution_code_binding.json',{str(f.relative_to(ROOT)):checksum(f)for f in code})
 dump(R/'preflight_validation.json',{'passed':True,'new_cases':2,'same319positions_and_outlets':True,'same_udt_pumps_bytes_and_weights':True,'same_standalone_cap_0p5_trigger_and_head':True,'same_rain_geometry_roof_stage_and_solver':True,'only5pool_caps_and_required_identity_changed':True,'new_shared_pool_caps':caps,'new_total_allowance_m3_s':5.,'319site_independent_pool_limit_checks':True,'identity_tamper_tests':True,'original_28_plus_8_rules_preserved':True,'other_parent_or_further_capacity_expansion':False,'budget':budget})
 (R/'README.md').write_text('# S1R same-layout receiving-allowance pilot\n\nDecision MILI-D-20261022-008. July H/L only. S1R-Q5 retains319intake/outlet cells,all weights,weighted allocator,standalone0.5m3/s ceiling,trigger0.05m,lift5m,forcing and geometry. Only five shared receiving caps change0.5to1.0m3/s,total2.5to5. The values are hypothetical allowances,not surveyed receiver or installed pump capacities. Frozen Q2.5/no-action runs are reused.\n\nUse qualify,prepare,run once with this script. Original28numerical+8pump checks use the declared generic caps without changed tolerances. Serial6CPU/7GiB,2hactive per run,4hhydraulics total,1hper audit/postphase,60GiBnew output and100GiBfree floor. First failure holds the queue,no automatic retry. PAUSE/STOP under the processed root control execution. Completion independently pairs all109frames and both depth branches against Q2.5 and no action,including new closure and native head transfer. Peak simultaneous and any-time union areas stay separate. Extra road benefit per1000extra m3 is descriptive,not economic efficiency. Prior adverse grid254457 is followed through allframes without selecting new deployment. No other parents,additional quota levels,population,traffic assignment,ML or sealed tests start automatically. Prior outputs,AnaSOP and accepted figures remain unchanged.\n')
 report();print('S1R_Q5_PREFLIGHT_PASSED_2_CASES',flush=True)

def configure():
 control.P=P;control.R=R;control.ARMS=['O1'];control.SCRIPT=SCRIPT
 worker.P=P;worker.R=R;worker.EVENTS=['O1'];worker.SCRIPT=SCRIPT;worker.report=report
 worker.audit=control.audit;worker.phase_module=control.post_module
 text=inspect.getsource(worker.run)
 replacements={"container='udt-transfer-'+event[1:]+'-'+r['stage'].lower()":"container='udt-s1-capacity-q5-'+r['stage'].lower()", "if x['event_id']==event":"if x['contrast']==event", "phase('post',event,budget['maximum_postprocess_wall_s']);put(status='access_population_postprocessing');report();phase('access',event,budget['maximum_access_wall_s']);assert json.loads((br/'access/validation.json').read_text())['passed'];":"phase('post',event,budget['maximum_postprocess_wall_s']);", "same=np.array_equal(supports[EVENTS[0]],supports[EVENTS[1]])":"same=np.array_equal(supports[EVENTS[0]],np.load(STATIC_REFERENCE/'common_road_support.npz')['qualified_roads']);assert same,'Road support changed'", "new_hydraulic_runs=4":"new_hydraulic_runs=2", "full_fixed_portfolio_transfer=True,two_new_rainfall_parents=True":"same_s1r_shared_allowance_contrast=True,new_rainfall_parents=False", "matched_road_and_access_results=True":"matched_road_results=True,population_not_recomputed=True"}
 for before,after in replacements.items():assert text.count(before)==1,before;text=text.replace(before,after)
 worker.__dict__['STATIC_REFERENCE']=OLD
 exec(compile(text,str(SCRIPT),'exec'),worker.__dict__)

def finalize():
 bp,br=dirs();runs=json.loads((P/'run_manifest.json').read_text())['runs'];assert len(runs)==2
 for run in runs:
  a=json.loads((br/'execution'/run['case_id']/'audit.json').read_text());b=json.loads((br/'execution'/run['case_id']/'pump_budget_summary.json').read_text())
  assert a['all_checks_passed']and len(a['checks'])==28 and all(x['pass']for x in a['checks'])and a['summary']['negative_cells']==0
  assert b['passed']and len(b['shared_receiver_checks'])==8 and all(b['shared_receiver_checks'].values())
 with np.load(bp/'common_road_support.npz')as z:qualified=z['qualified_roads'];common=z['common_admitted']
 with np.load(OLD/'common_road_support.npz')as z:assert np.array_equal(qualified,z['qualified_roads'])and np.array_equal(common,z['common_admitted'])
 clip=dict(np.load(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz'))
 comparisons=[];trajectory=[]
 for run in runs:
  case=run['case_id'];ref=run['quota_reference_case'];stage=run['stage'];newactual=json.loads((br/'execution'/case/'pump_budget_summary.json').read_text())['actual_transferred_m3'];oldactual=json.loads((OLDR/'execution'/ref/'pump_budget_summary.json').read_text())['actual_transferred_m3']
  with np.load(bp/'exposure'/f'{case}__drainage.npz')as z:new=z['road_state_codes'];qi=z['road_indices'];t=z['analysis_times']
  with np.load(bp/'exposure'/f'{case}__baseline.npz')as z:noaction=z['road_state_codes'];assert np.array_equal(qi,z['road_indices'])and np.array_equal(t,z['analysis_times'])
  with np.load(OLD/'exposure'/f'{ref}__drainage.npz')as z:old=z['road_state_codes'];assert np.array_equal(qi,z['road_indices'])and np.array_equal(t,z['analysis_times'])
  with np.load(OLD/'exposure'/f'{ref}__baseline.npz')as z:assert np.array_equal(noaction,z['road_state_codes'])
  for bi,bound in enumerate(['lower','upper']):
   b=new[:,bi]==2
   for reference,a0 in [('S1R-Q2.5',old),('No action',noaction)]:
    a=a0[:,bi]==2
    for wi,scope in enumerate(['Three wards','Chiyoda','Chuo','Minato']):
     w=(clip['union_length_m']if wi==0 else clip['ward_length_m'][wi-1])[qi]/1000
     ref_length=a@w;newlength=b@w;restore=(a&~b)@w;added=(~a&b)@w;integ=lambda v:float(np.trapezoid(v,t)/3600)
     net=integ(ref_length-newlength);rr=integ(restore);nn=integ(added);assert abs(net-rr+nn)<1e-8
     extra=newactual-oldactual
     comparisons.append({'case':case,'stage':stage,'branch':bound,'scope':scope,'reference':reference,'reference_closure_km_h':integ(ref_length),'remaining_closure_km_h':integ(newlength),'net_avoided_km_h':net,'restored_km_h':rr,'newly_closed_km_h':nn,'peak_newly_closed_km':float(added.max()),'net_reduction_percent':100*net/integ(ref_length)if integ(ref_length)>0 else None,'Q5_actual_extraction_m3':newactual,'Q2p5_actual_extraction_m3':oldactual,'extra_extraction_m3':extra,'descriptive_incremental_km_h_per1000extra_m3':1000*net/extra if reference=='S1R-Q2.5'and extra>0 else None})
     for ti in range(109):trajectory.append({'case':case,'stage':stage,'branch':bound,'scope':scope,'reference':reference,'frame':ti,'elapsed_s':float(t[ti]),'reference_closed_km':float(ref_length[ti]),'remaining_closed_km':float(newlength[ti]),'restored_km':float(restore[ti]),'newly_closed_km':float(added[ti])})
 pd.DataFrame(comparisons).to_csv(P/'road_comparison.csv',index=False);pd.DataFrame(trajectory).to_csv(P/'road_comparison_trajectories.csv',index=False)
 from netCDF4 import Dataset
 ward=np.load(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz')['Ward Code'].ravel();active=np.zeros(common.size,bool);active[pd.read_csv(bp/'pump_sites.csv')['Grid Index'].to_numpy()]=True
 head=[];head_frames=[];follow=[];adverse_samples=[];frozen=254457
 for run in runs:
  with Dataset(ROOT/run['quota_reference_folder']/'sfincs_map.nc')as a,Dataset(ROOT/run['folder']/'sfincs_map.nc')as b,Dataset(ROOT/run['original_folder']/'sfincs_map.nc')as c:
   beds=[np.flipud(np.ma.asarray(ds['zb'][:],dtype=float).filled(np.nan)).ravel()for ds in [a,b,c]]
   assert all(np.array_equal(beds[0],z,equal_nan=True)for z in beds[1:])
   references={'S1R-Q2.5':(a,beds[0]),'No action':(c,beds[2])}
   unions={key:np.zeros(common.size,bool)for ref in references for key in [(ref,s)for s in ['Three wards','Chiyoda','Chuo','Minato']]};peaks={k:0 for k in unions};outside={k:0 for k in unions};newwetunion={k:np.zeros(common.size,bool)for k in unions}
   clocks=np.load(bp/'execution'/(run['case_id']+'_output_clock.npz'))['Actual Map Time'];analysis=clocks.copy();analysis[0]=0;analysis[-1]=32400
   for ti in range(109):
    y=np.flipud(np.ma.asarray(b['zs'][ti],dtype=float).filled(np.nan)).ravel();states={ref:np.flipud(np.ma.asarray(ds['zs'][ti],dtype=float).filled(np.nan)).ravel()for ref,(ds,bed)in references.items()}
    row={'case':run['case_id'],'stage':run['stage'],'frame':ti,'analysis_time_s':float(analysis[ti]),'grid_index':frozen,'Q5_depth_m':float(max(0,y[frozen]-beds[1][frozen]))if np.isfinite(y[frozen])else None}
    for ref,(ds,bed)in references.items():
     x=states[ref];increase=np.where(np.isfinite(y),y,beds[1])-np.where(np.isfinite(x),x,bed+.01);row[ref+'_depth_m']=float(max(0,x[frozen]-bed[frozen]))if np.isfinite(x[frozen])else None
     for code,scope in [(0,'Three wards'),(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
      use=common.ravel()&((ward>0)if code==0 else ward==code);hit=use&(increase>=.10);key=(ref,scope);unions[key]|=hit;newwet=hit&~np.isfinite(x)&np.isfinite(y);newwetunion[key]|=newwet
      peaks[key]=max(peaks[key],int(hit.sum()));outside[key]=max(outside[key],int((hit&~active).sum()))
      head_frames.append({'case':run['case_id'],'stage':run['stage'],'reference':ref,'scope':scope,'frame':ti,'analysis_time_s':float(analysis[ti]),'certain_head_rise_m2':int(hit.sum())*100,'newly_wet_subset_of_head_screen_m2':int(newwet.sum())*100})
      if code==0:
       for ix in np.flatnonzero(hit):adverse_samples.append({'case':run['case_id'],'stage':run['stage'],'reference':ref,'frame':ti,'analysis_time_s':float(analysis[ti]),'grid_index':int(ix),'ward_code':int(ward[ix]),'cell_area_m2':100,'certain_head_rise_m':float(increase[ix]),'reference_saved_wet':bool(np.isfinite(x[ix])),'reference_depth_m':float(max(0,x[ix]-bed[ix]))if np.isfinite(x[ix])else None,'Q5_depth_m':float(max(0,y[ix]-beds[1][ix]))if np.isfinite(y[ix])else None,'screen_meaning':'certain head increase>=0.10m,not total new inundation'})
    follow.append(row)
   for key in unions:
    ref,scope=key;head.append({'case':run['case_id'],'stage':run['stage'],'reference':ref,'scope':scope,'peak_simultaneous_certain_head_rise_m2':peaks[key]*100,'any_time_certain_head_rise_union_m2':int(unions[key].sum())*100,'peak_outside_actual_intakes_m2':outside[key]*100,'any_time_newly_wet_subset_of_head_screen_m2':int(newwetunion[key].sum())*100,'threshold_m':.10,'dry_censor_bound_m':.01})
 pd.DataFrame(head).to_csv(P/'head_transfer_comparison.csv',index=False);pd.DataFrame(head_frames).to_csv(P/'head_transfer_trajectories.csv',index=False);pd.DataFrame(follow).to_csv(P/'prior_adverse_location_followup.csv',index=False)
 pd.DataFrame(adverse_samples,columns=['case','stage','reference','frame','analysis_time_s','grid_index','ward_code','cell_area_m2','certain_head_rise_m','reference_saved_wet','reference_depth_m','Q5_depth_m','screen_meaning']).to_csv(P/'adverse_head_samples.csv',index=False)
 for name in ['postprocessing_validation.json','expanded_transfer_validation.json']:assert json.loads((br/name).read_text())['passed']
 dump(R/'final_summary_validation.json',{'passed':True,'two_cases_original_28_plus_8_checks':True,'same_common_road_support':True,'same_noaction_states':True,'same_time_direct_comparison_against_Q2p5':True,'head_transfer_vs_both_references':True,'prior_adverse_location_all109frames_tracked':True,'new_hydraulic_runs':2,'only_shared_caps_doubled':True,'unit_ceiling_changed':False,'weights_layout_operator_changed':False,'other_events_enabled':False,'population_recomputed':False,'training':False,'sealed_test':False})
 print('S1R_Q5_COMPARISON_COMPLETE',flush=True)

def report():
 if not(P/'run_manifest.json').exists():return
 manifest=json.loads((P/'run_manifest.json').read_text());state=json.loads((R/'campaign_state.json').read_text())if(R/'campaign_state.json').exists()else {'status':'preflight_passed','completed_cases':[],'new_runs_started':0}
 if state['status']=='qualified_complete'and not(R/'final_summary_validation.json').exists():finalize()
 rows=[{'Stage':r['stage'],'Portfolio':'S1R-Q5: same319locations,same allocation,5m3/s shared quota','State':'qualified'if r['case_id']in state['completed_cases']else state['status']if state.get('current_case')==r['case_id']else'waiting'}for r in manifest['runs']]
 refresh='<meta http-equiv="refresh" content="30">'if state['status']not in ['held','qualified_complete']else''
 body='<h1>S1R容量对照：共享额度2.5与5 m³/s</h1><aside><b>首批仅7月高／低水位两组。</b>固定319个取水和接收位置、原权重、池内调配规则和单点0.5 m³/s上限，只将五个接收池的共享额度各从0.5提高到1.0 m³/s。复用现有2.5 m³/s和无措施结果；不自动扩到其他暴雨或更高额度。</aside>'
 body+=f"<p><b>状态：{html.escape(state['status'])}</b>；数值核验通过 {len(state['completed_cases'])}/2，新启动 {state.get('new_runs_started',0)}/2。</p>"+pd.DataFrame(rows).to_html(index=False)
 if state.get('current_case'):body+=f"<p>当前：{html.escape(state['current_case'])}；实际模型时钟 {state.get('latest_native_time_s',0)/3600:.3f}/9 h；活跃计算 {state.get('case_active_wall_seconds',0):.0f}秒。</p>"
 if state.get('error'):body+='<p class="warn">已停止扩跑并保留结果：'+html.escape(state['error'])+'</p>'
 body+='<h2>唯一改变的条件与核验</h2><p>本次放宽的是假设共享接收额度，不代表实际安装了更大的泵或确认河道可以接收这些水。单点上限仍0.5 m³/s，原权重仍只用于池内分配；取水文件保持逐字节一致。接收水位、启动水深0.05 m、扬程上限5 m、原雷达雨型、屋面分配、10 m城市几何、质量掩膜、109帧和0.30 m封路阈值不变。</p><p>原28项数值和8项抽排检查不放宽，逐步独立核对声明的1.0 m³/s池限额及原单点／储量／扬程约束。串行6 CPU/7 GiB，每组活跃计算最多2小时，两组累计4小时；单次审核及后处理最多1小时。新输出上限60 GiB，剩余空间至少100 GiB。失败停止，不自动重试；PAUSE/STOP文件可暂停或停止。</p>'
 if(P/'road_comparison.csv').exists():
  d=pd.read_csv(P/'road_comparison.csv');d=d[d.scope.eq('Three wards')&d.branch.eq('lower')];body+='<h2>三区封路与实际抽排量</h2>'+d.to_html(index=False,float_format=lambda v:f'{v:,.5f}')
  body+='<p>reference=S1R-Q2.5用于隔离共享额度增加的增量，No action是无措施总背景。km h是长度—时间积分，不是唯一公里数。恢复和新增封路分列；每增加1000 m³对应的额外道路收益是描述性比值，不是成本效率，也不能外推。</p>'
 if(P/'head_transfer_comparison.csv').exists():body+='<h2>局部水位抬升与可能风险转移</h2>'+pd.read_csv(P/'head_transfer_comparison.csv').to_html(index=False)+'<p>≥0.10 m原生水头抬升的最大同时面积和任意时刻并集分别报告。新增湿区子集只限于该水头筛查，不是完整新增积水面积。已有湿区加深不称为新增湿区。与取水位置之外、原候选控制区之外的筛查继续保留。</p>'
 body+='<p>上一轮千代田区出现局部加深的冻结单元会跟踪全部109帧。接收河道新增蓄水及真实下游容量仍未模拟，不能据此宣称没有下游风险。没有新增人口或交通分配、训练、封存测试访问，也不覆盖AnaSOP和既有图件。</p><p>首批两组核验后自动汇总，并停在本批；不因收益较小自动加额度，也不因出现真实局部副作用而删去结果。</p><p><a href="../../../processed/'+TAG+'/protocol.json">Protocol / budget</a> · <a href="preflight_validation.json">Preflight</a> · <a href="qualification.json">Operator checks</a> · <a href="campaign_state.json">Live state</a></p>'
 style='body{font:16px/1.8 system-ui;max-width:1280px;margin:28px auto;padding:0 22px;background:#f8fafb;color:#243c47}aside{background:#e7f0f5;padding:18px}table{display:block;max-width:100%;overflow:auto;border-collapse:collapse;font-size:13px;background:white}td,th{border:1px solid #ccd7df;padding:8px;text-align:left}.warn{background:#fff1d9;padding:16px}a{color:#176a96}'
 tmp=R/'review.html.tmp';tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+refresh+'<title>S1R Shared Receiving Allowance Pilot</title><style>'+style+'</style>'+body+'</html>');tmp.replace(R/'review.html')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['qualify','prepare','run','audit','post','report']);ap.add_argument('--event',choices=['O1']);ap.add_argument('--case');a=ap.parse_args();configure()
 if a.phase=='qualify':qualify()
 elif a.phase=='prepare':prepare()
 elif a.phase=='run':worker.run()
 elif a.phase=='audit':control.audit('O1',a.case)
 elif a.phase=='post':control.post('O1')
 else:report()
if __name__=='__main__':main()
