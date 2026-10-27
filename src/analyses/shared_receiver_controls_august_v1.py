#!/usr/bin/env python3
"""Frozen July O1/S1 -> August30 H/L; four serial, gated runs only."""
from pathlib import Path
import os,sys,json,time,shutil,subprocess,inspect,argparse,html
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
# Reuse the already-installed NetCDF binary for legacy helper imports.
EXTRA=ROOT/'.frame-control/runtimes/physical-measures-py312/lib/python3.12/site-packages'
sys.path.append(str(EXTRA));sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
import netCDF4,numpy as np,pandas as pd
import shared_receiver_controls_v1 as control
import shared_receiver_transfer_v1 as worker
from shared_receiver_pilot_common_v1 import sha,dump,VENV,PY,command
from shared_receiver_control_support_v1 import treatment_hash,identity_receipt_check
from shared_receiver_identity_v2 import CORE,selftest
EVENT='E20240830';ARMS=['O1','S1'];TAG='three_ward_shared_receiver_controls_august_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG;SCRIPT=Path(__file__).resolve()
JULY=ROOT/'data/processed/three_ward_shared_receiver_controls_v1';JR=ROOT/'data/exp/estimation'/JULY.name
STATIC=ROOT/'data/processed/three_ward_shared_receiver_transfer_v1/batches'/EVENT
SR=ROOT/'data/exp/estimation/three_ward_shared_receiver_transfer_v1/batches'/EVENT
DECISION='MILI-D-20261021-003'
SEPTEMBER=ROOT/'data/processed/three_ward_shared_receiver_controls_transfer_v1'
SEPTEMBER_R=ROOT/'data/exp/estimation'/SEPTEMBER.name
def dirs(arm):return P/'batches'/arm,R/'batches'/arm

def summary_adapter(event,static,p,r,output_dirs,static_suffix,static_access=None):
 """Only context/date/reference paths differ from the qualified July summary."""
 source=inspect.getsource(control.finalize)
 assert source.count('E20240731')==2
 source=source.replace('E20240731',event)
 source=source.replace("__shared_r1000_q0p5'",static_suffix+"'")
 anchor="ROOT/'data/processed/three_ward_shared_receiver_access_v1'"
 assert source.count(anchor)==1
 source=source.replace(anchor,"STATIC_ACCESS")
 # An otherwise-unused marginal population read also follows the right parent.
 source=source.replace("ROOT/'data/processed/three_ward_shared_receiver_access_v1/paired_population_summary.csv'","STATIC_ACCESS/'paired_population_summary.csv'")
 env=control.__dict__.copy();env.update(P=p,R=r,OLD=static,STATIC_ACCESS=(static_access if static_access is not None else static/'access'),dirs=output_dirs)
 exec(compile(source,'<bound_frozen_control_summary>','exec'),env)
 return env['finalize'],source

def direct_adapter(event,p):
 import finish_shared_receiver_controls_v1 as finish
 source=inspect.getsource(finish.direct_pair);assert source.count('E20240731')==2;source=source.replace('E20240731',event)
 env=finish.__dict__.copy();env.update(P=p)
 exec(compile(source,'<bound_direct_population_summary>','exec'),env)
 return env['direct_pair'],source

def selfcheck():
 """Reproduce the July summaries in scratch output from existing results only."""
 import pyarrow
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 q=P/'qualification/summary_regression';qr=R/'qualification/summary_regression';q.mkdir(parents=True,exist_ok=False);qr.mkdir(parents=True,exist_ok=False)
 old=ROOT/'data/processed/three_ward_shared_receiver_pilot_v1'
 def july_dirs(arm):return JULY/'batches'/arm,JR/'batches'/arm
 fn,_=summary_adapter('E20240731',old,q,qr,july_dirs,'__shared_r1000_q0p5',ROOT/'data/processed/three_ward_shared_receiver_access_v1');fn()
 for name in ['comparison_summary.csv','paired_population_vs_static_summary.csv','endpoint_access_vs_static_trajectories.csv','group_continuity_vs_static.csv','pump_pool_operating_summary.csv','pump_ward_operating_summary.csv']:
  a=pd.read_csv(q/name);b=pd.read_csv(JULY/name);pd.testing.assert_frame_equal(a,b,check_exact=False,atol=1e-8,rtol=1e-10)
 # Real August references exist under the explicit date and transfer suffix.
 for stage in ['H','L']:
  case=f'{EVENT}__{stage}__U__shared_r1000_q0p5_transfer'
  for f in [STATIC/'exposure'/(case+'__drainage.npz'),STATIC/'access/population'/(case+'.npz')]:assert f.exists()
 _,src=summary_adapter(EVENT,STATIC,P,R,dirs,'__shared_r1000_q0p5_transfer')
 assert 'E20240731' not in src and '__shared_r1000_q0p5_transfer' in src
 import shared_receiver_control_support_v1 as support
 # Existing production fixture confirms independent operator expectations still match.
 with np.load(JULY/'batches/O1/common_road_support.npz') as z:assert z['qualified_roads'].sum()==28064
 dump(R/'interface_validation.json',{'passed':True,'july_full_summary_regression_exact':True,'six_summary_tables_reproduced':True,'august_date_and_static_reference_suffix_checked':True,'identity_tamper_tests':selftest(),'runtime_bridge':{'netCDF4':netCDF4.__version__,'numpy':np.__version__,'pyarrow':pyarrow.__version__},'no_new_hydraulic_runs':True})
 q2=P/'qualification/september_regression';r2=R/'qualification/september_regression';q2.mkdir(parents=True,exist_ok=False);r2.mkdir(parents=True,exist_ok=False)
 def september_dirs(arm):return SEPTEMBER/'batches'/arm,SEPTEMBER_R/'batches'/arm
 old2=ROOT/'data/processed/three_ward_shared_receiver_transfer_v1/batches/E20250911'
 fn,_=summary_adapter('E20250911',old2,q2,r2,september_dirs,'__shared_r1000_q0p5_transfer');fn()
 names=['comparison_summary.csv','paired_population_vs_static_summary.csv','endpoint_access_vs_static_trajectories.csv','group_continuity_vs_static.csv','pump_pool_operating_summary.csv','pump_ward_operating_summary.csv']
 for name in names:
  assert (q2/name).read_bytes()==(SEPTEMBER/name).read_bytes(),name
 # Direct pairing writes into its bound output; bind the copied namespace to read-only source batches.
 for arm in ARMS:(q2/'batches'/arm).parent.mkdir(parents=True,exist_ok=True);(q2/'batches'/arm).symlink_to(SEPTEMBER/'batches'/arm,target_is_directory=True)
 dp,_=direct_adapter('E20250911',q2);dp()
 assert (q2/'s1_vs_o1_population_summary.csv').read_bytes()==(SEPTEMBER/'s1_vs_o1_population_summary.csv').read_bytes()
 proto=json.loads((STATIC/'access/protocol.json').read_text());assert proto['parent_event']==EVENT and proto['anchor_frame']==97
 from summarize_three_event_controls_v1 import selfcheck as synthesis_selfcheck
 synthesis_selfcheck(P/'qualification/synthesis',R/'qualification/synthesis')
 v=json.loads((R/'interface_validation.json').read_text());v.update(september_six_tables_byte_exact=True,september_direct_s1_o1_byte_exact=True,august_frozen_anchor_frame=97,three_event_summary_interface_passed=True);dump(R/'interface_validation.json',v)
 print('TRANSFER_INTERFACE_SELFCHECK_PASSED',flush=True)

def prepare():
 assert not (P/'run_manifest.json').exists(),'Preserve existing staging'
 validation=json.loads((R/'interface_validation.json').read_text());assert validation['passed'] and validation['three_event_summary_interface_passed'] and validation['september_direct_s1_o1_byte_exact']
 assert json.loads((JR/'final_summary_validation.json').read_text())['passed']
 assert json.loads((JR/'campaign_state.json').read_text())['status']=='qualified_complete'
 assert json.loads((SR/'access/validation.json').read_text())['passed']
 assert json.loads((SEPTEMBER_R/'final_summary_validation.json').read_text())['passed']
 assert json.loads((SEPTEMBER_R/'campaign_state.json').read_text())['status']=='qualified_complete'
 assert not subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip(),'Existing Docker workload'
 assert shutil.disk_usage(P).free>=100*1024**3
 oldcode=json.loads((JULY/'execution_code_binding.json').read_text());assert all(sha(ROOT/k)==v for k,v in oldcode.items())
 priorcode=json.loads((SEPTEMBER/'execution_code_binding.json').read_text());assert all(sha(ROOT/k)==v for k,v in priorcode.items())
 frozen=json.loads((JULY/'run_manifest.json').read_text())['runs'];refs=json.loads((STATIC/'run_manifest.json').read_text())['runs']
 budget={'maximum_new_runs':4,'concurrent_hydraulics':1,'cpus':6,'memory_gib':7,'maximum_run_active_wall_s':7200,'maximum_campaign_active_wall_s':28800,'no_clock_progress_stop_s':900,'maximum_new_output_gib':100,'minimum_free_gib':100,'maximum_audit_wall_s':3600,'maximum_postprocess_wall_s':3600,'maximum_access_wall_s':1800,'automatic_retries':0}
 protocol={'decision':DECISION,'event':EVENT,'allocation_event':'E20240731','contrasts':ARMS,'new_hydraulic_runs':4,'sequence':['O1H','O1L','S1H','S1L'],'frozen':'same bytes for July intakes,outlets,rates,trigger,lift,pool rules and same per-arm image;no retuning','shared_quota_m3_s':2.5,'quota_per_named_pool_m3_s':.5,'quota_is_not_financial_cost':True,'noaction_and_static_august_reused':True,'original_28_numerical_gates_and_8_pump_checks':'unchanged,independent allocation reconstruction','access':'original quality masks and0/10/25m candidate intervals;paired candidate differences,not marginal-bound subtraction','anchor':'frozen August H/noaction closure peak,frame97;identical anchor for both stages and treatments','off_target':'legacy outside13214candidate cells plus whole supported domain and outside actual intakes;>=0.10m certain head rise with0.01m dry censor bound','receiving':'prescribed water stages;does not simulate downstream added receiver storage','inference':'cross-event stability check on an already-inspected rainfall parent;not independent observed validation or sealed testing','other_events_or_C1_automatic_expansion':False,'training':False,'sealed_test':False,'budget':budget}
 dump(P/'protocol.json',protocol);allruns=[];protected={};memo={}
 def checksum(f):
  f=Path(f);key=(str(f.resolve()),f.stat().st_size,f.stat().st_mtime_ns)
  if key not in memo:memo[key]=sha(f)
  return memo[key]
 for arm in ARMS:
  bp,br=dirs(arm);bp.mkdir(parents=True,exist_ok=True);br.mkdir(parents=True,exist_ok=True);(bp/'execution').mkdir(exist_ok=True)
  template=next(q for q in frozen if q['contrast']==arm);files=template['treatment_files'];src=ROOT/template['folder']
  sites=pd.read_csv(JULY/'batches'/arm/'pump_sites.csv');assert len(sites)==(62 if arm=='O1' else 319)
  assert np.allclose(sites.groupby('Receiving Pool')['Reserved Rate'].sum(),.5,rtol=0,atol=1e-12)
  for name in ['pump_sites.csv','pool_allowances.csv']:
   origin=JULY/'batches'/arm/name;shutil.copy2(origin,bp/name);protected[str(origin.relative_to(ROOT))]=checksum(origin)
  for name in files:
   assert checksum(src/name)==template['input_sha256'][name];shutil.copy2(src/name,bp/name);protected[str((src/name).relative_to(ROOT))]=checksum(src/name)
  image=template['image_id'];assert subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()==image
  batch=[]
  for stage in ['H','L']:
   base=next(q for q in refs if q['stage']==stage);basefolder=ROOT/base['folder'];noaction=ROOT/base['original_folder']
   a=json.loads((SR/'execution'/base['case_id']/'audit.json').read_text());assert a['all_checks_passed'] and a['admission']['technical_reference']
   assert base['identity']['geometry_sha256']==template['identity']['geometry_sha256']
   case=f'{EVENT}__{stage}__U__{arm.lower()}';folder=bp/'runs'/case;folder.mkdir(parents=True,exist_ok=False)
   names=[]
   for name,h in base['input_sha256'].items():
    if name in ['udt_run_identity.txt','udt_run_identity.json','udt_pumps.dat']:continue
    assert checksum(basefolder/name)==h;protected[str((basefolder/name).relative_to(ROOT))]=h
    if name=='sfincs_subgrid.nc':(folder/name).symlink_to(Path(base['large_readonly_source']).resolve())
    else:shutil.copy2(basefolder/name,folder/name)
    names.append(name)
   for name in files:
    shutil.copy2(bp/name,folder/name);assert checksum(folder/name)==checksum(src/name);names.append(name)
   identity=base['identity'].copy();identity.update(run_id=case+'__frozen_controls_transfer_10m',split='external_frozen_controls_transfer',solver_digest=image,patch_sha256=template['identity']['patch_sha256'],treatment_sha256=treatment_hash(folder,files))
   dump(folder/'udt_run_identity.json',identity);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v) for k,v in identity.items())+'\n');names+=['udt_run_identity.txt','udt_run_identity.json']
   run={**base,'case_id':case,'contrast':arm,'treatment_files':files,'folder':str(folder.relative_to(ROOT)),'measure':'frozen_shared_control_'+arm,'run_id':identity['run_id'],'identity':identity,'split':identity['split'],'image_id':image,'input_sha256':{n:checksum(folder/n) for n in names}}
   assert (folder/'sfincs.inp').read_bytes()==(noaction/'sfincs.inp').read_bytes()
   for name,h in run['input_sha256'].items():protected[str((folder/name).relative_to(ROOT))]=h
   for name in ['sfincs_map.nc','udt_output_clock.csv','execution.json']:protected[str((noaction/name).relative_to(ROOT))]=checksum(noaction/name)
   fixture=bp/'qualification/identity'/case;fixture.mkdir(parents=True)
   for name in files+['udt_run_identity.txt','udt_run_identity.json']:shutil.copy2(folder/name,fixture/name)
   (fixture/'udt_diagnostic_identity.txt').write_text('\n'.join(k+'='+str(identity[k]) for k in CORE)+'\n');assert identity_receipt_check(fixture,run)
   for name in files:
    original=(fixture/name).read_bytes();(fixture/name).write_bytes(original+b'\n');assert not identity_receipt_check(fixture,run);(fixture/name).write_bytes(original)
   assert identity_receipt_check(fixture,run)
   batch.append(run);allruns.append(run)
  dump(bp/'protocol.json',{**protocol,'contrast':arm,'source_site_count':len(sites),'reserved_total_m3_s':float(sites['Reserved Rate'].sum())})
  dump(bp/'run_manifest.json',{'runs':batch,'budget':budget,'protocol_sha256':checksum(bp/'protocol.json')})
  for f in [bp/'protocol.json',bp/'run_manifest.json',bp/'pump_sites.csv',bp/'pool_allowances.csv']+[bp/n for n in files]:protected[str(f.relative_to(ROOT))]=checksum(f)
 dump(P/'run_manifest.json',{'runs':allruns,'budget':budget,'protocol_sha256':checksum(P/'protocol.json'),'execution_order':[x['case_id'] for x in allruns]})
 # Verify contrast dispatch, without confusing contrast IDs with the rainfall ID.
 selected=[x['case_id'] for a in ARMS for x in allruns if x['contrast']==a]
 assert selected==[f'{EVENT}__{s}__U__{a.lower()}' for a in ARMS for s in ['H','L']]
 for name in ['run_manifest.json','paired_road_effects.csv','common_road_support.npz','access/paired_population_summary.csv','access/protocol.json']:
  f=STATIC/name;protected[str(f.relative_to(ROOT))]=checksum(f)
 for f in [P/'protocol.json',P/'run_manifest.json',JULY/'run_manifest.json',ROOT/'docs/AnaSOP.md']:protected[str(f.relative_to(ROOT))]=checksum(f)
 access_binding=json.loads((STATIC/'access/input_bindings.json').read_text())
 for key,value in access_binding.items():
  assert checksum(ROOT/key)==value,('Frozen access evidence changed',key)
  protected[key]=value
 for base,reportbase in [(JULY,JR),(SEPTEMBER,SEPTEMBER_R)]:
  for name in ['comparison_summary.csv','paired_population_vs_static_summary.csv','s1_vs_o1_population_summary.csv']:
   f=base/name;protected[str(f.relative_to(ROOT))]=checksum(f)
  for arm in ARMS:
   for name in ['common_road_support.npz','paired_road_effects.csv','expanded_head_transfer_summary.csv']:
    f=base/'batches'/arm/name;protected[str(f.relative_to(ROOT))]=checksum(f)
  for name in ['final_summary_validation.json','campaign_state.json']:
   f=reportbase/name;protected[str(f.relative_to(ROOT))]=checksum(f)
 dump(P/'protected_sources.json',protected)
 scripts=[SCRIPT,ROOT/'src/analyses/summarize_three_event_controls_v1.py',ROOT/'src/analyses/shared_receiver_controls_transfer_v1.py']+[ROOT/'src/analyses'/n for n in ['shared_receiver_controls_v1.py','finish_shared_receiver_controls_v1.py','shared_receiver_transfer_v1.py','shared_receiver_control_support_v1.py','shared_receiver_identity_v2.py','shared_receiver_pilot_common_v1.py','audit_shared_receiver_pilot_identity_v3.py','audit_cross_event_protection_run_v1.py','postprocess_shared_receiver_pilot_v1.py','shared_receiver_access_effects_v1.py','temporal_transport_common_v1.py','run_temporal_road_population_v1.py','integrate_night_population_transport_v1.py','postprocess_five_event_population_v1.py']]+[ROOT/'src/preprocessing/reference_label_policy.py']
 dump(P/'execution_code_binding.json',{str(f.relative_to(ROOT)):checksum(f) for f in scripts})
 dump(R/'preflight_validation.json',{'passed':True,'new_cases':4,'same_treatment_files_as_july':True,'O1_native_intakes':62,'S1_native_intakes':319,'same_per_arm_image':True,'same_august_rain_geometry_stage_and_roof':True,'dispatch_order_checked':selected,'strict_treatment_tamper_tests_passed':True,'runtime_bridge_checked':True,'no_new_hydraulic_run_started':True,'budget':budget})
 (R/'README.md').write_text('''# Frozen O1 / S1 August transfer\n\nDecision MILI-D-20261021-003. Four9h runs: O1H,O1L,S1H,S1L. Exact July treatment bytes and per-arm image,62/319native intake locations,not machines. Total assumed flow allowance2.5m3/s,five0.5m3/s pools. No retuning or new layouts; no C1 or other event automatic expansion.\n\nUse `.venv/bin/python src/analyses/shared_receiver_controls_august_v1.py selfcheck`, then `prepare`, then `run` once. This script reuses installed NetCDF4 from the physical runtime to support legacy imports; no package installation. All simulation and audit gates retain original tolerances. Same109times,roof distribution,August stage/rain,geometry and uncertainty branches. Frozen population anchor is August frame97 from the frozen August no-action reference. Existing no-action and static August results are reused.\n\nThe worker refuses existing attempts,stops on first failed gate,and does not automatically retry. PAUSE/STOP files in the processed root control the serial worker. Maximum7200active seconds per run,28800for four hydraulics; serial6CPU/7GiB;100GiB output ceiling and100GiB free floor. Each audit/road phase<=3600s and each access phase<=1800s.\n\nAfter all four qualify,the same validated July summary functions are adapted only for date,namespace and exact static reference suffix. Population changes are paired at candidate/node level before block aggregation,including S1 vs O1;never subtract marginal interval endpoints. Complete summaries combine July, September and August roads, direct paired population effects, registered-group reopening and head changes. Complete summaries include head rises for the whole support and outside actual intakes,not just outside the candidate control area. All results remain conditional scenarios without observed-road validation or added receiving-river storage. This already-inspected rainfall parent is not an untouched test.\n''')
 report();print('FROZEN_TRANSFER_PREFLIGHT_PASSED',flush=True)

def access(arm):
 import shared_receiver_access_effects_v1 as mod
 from temporal_transport_common_v1 import Reachability,population_inputs
 bp,br=dirs(arm);ap=bp/'access';ar=br/'access';ap.mkdir(exist_ok=True);ar.mkdir(exist_ok=True)
 for name in ['node_cache','population','anchor_nodes']:(ap/name).mkdir(exist_ok=True)
 mod.P=ap;mod.R=ar;mod.HP=bp;mod.HR=br;mod.report=lambda:None
 proto=json.loads((STATIC/'access/protocol.json').read_text());assert proto['parent_event']==EVENT and proto['anchor_frame']==97
 proto.update(decision=DECISION,contrast=arm,frozen_treatment_selection_event='E20240731');dump(ap/'protocol.json',proto)
 bound=mod.bindings();bound={k:v for k,v in bound.items() if not k.startswith('data/processed/three_ward_shared_receiver_pilot_v1/')}
 for f in [SCRIPT,bp/'protocol.json',bp/'run_manifest.json',ap/'protocol.json']:bound[str(f.relative_to(ROOT))]=sha(f)
 dump(ap/'input_bindings.json',bound)
 blocks,pop,normal,infos=population_inputs();eng=Reachability();assert np.array_equal(eng.direct(np.zeros(len(mod.load(mod.TEMP/'clipped_road_lengths.npz')['qualified']),bool)),normal);assert len(blocks)==4746 and pop.sum()==496345 and mod.routing_selfcheck()
 dump(ar/'preflight_validation.json',{'passed':True,'normal_graph_exact':True,'population_stock_unchanged':True,'frozen_august_anchor_frame':97});mod.run()

def finalize():
 if (R/'final_summary_validation.json').exists():return
 runs=json.loads((P/'run_manifest.json').read_text())['runs'];audits=[]
 for run in runs:
  br=dirs(run['contrast'])[1];ar=br/'execution'/run['case_id'];a=json.loads((ar/'audit.json').read_text());b=json.loads((ar/'pump_budget_summary.json').read_text())
  assert a['all_checks_passed'] and len(a['checks'])==28 and all(v['pass'] for v in a['checks']) and a['admission']['technical_reference']
  assert b['passed'] and len(b['shared_receiver_checks'])==8 and all(b['shared_receiver_checks'].values())
  audits.append({'case':run['case_id'],'negative_cells':a['summary']['negative_cells'],'max_budget_residual_m3':a['summary']['max_budget_residual_m3']})
 for arm in ARMS:
  for n in ['postprocessing_validation.json','expanded_transfer_validation.json','access/validation.json']:assert json.loads((dirs(arm)[1]/n).read_text())['passed']
 fn,src=summary_adapter(EVENT,STATIC,P,R,dirs,'__shared_r1000_q0p5_transfer');fn()
 dp,dsrc=direct_adapter(EVENT,P);direct=dp()
 pair=pd.read_csv(P/'paired_population_vs_static_summary.csv');groups=pd.read_csv(P/'group_continuity_vs_static.csv');regional=pd.read_csv(P/'comparison_summary.csv')
 for row in regional.to_dict('records'):
  sub=groups[groups.contrast.eq(row['contrast'])&groups.stage.eq(row['stage'])&groups.bound.eq('lower')]
  assert abs(sub.net_restored_edge_km_h.sum()-row['incremental_closure_reduction_vs_static_km_h'])<1e-8
 assert (pair.net_lower_resident_hours<=pair.net_upper_resident_hours+1e-8).all()
 # Compare only within-event paired effects; display July, September and August side by side.
 road=[]
 for event,static,base in [('E20240731',ROOT/'data/processed/three_ward_shared_receiver_pilot_v1',JULY),('E20250911',ROOT/'data/processed/three_ward_shared_receiver_transfer_v1/batches/E20250911',SEPTEMBER),(EVENT,STATIC,P)]:
  for scheme,path in [('Static',static),('O1',base/'batches/O1'),('S1',base/'batches/S1')]:
   d=pd.read_csv(path/'paired_road_effects.csv');d=d[d.Scope.eq('Three wards')&d['Depth Branch'].eq('lower')]
   for x in d.to_dict('records'):road.append({'event':event,'scheme':scheme,'stage':x['Stage'],'closure_reduction_km_h':x['Closure Reduction km h'],'closure_reduction_percent':x['Closure Reduction Percent'],'max_new_closed_km':x['Maximum Same Time Newly Closed km']})
 pd.DataFrame(road).to_csv(P/'cross_event_road_comparison.csv',index=False)
 shifts=[]
 for arm in ARMS:
  d=pd.read_csv(P/'batches'/arm/'expanded_head_transfer_summary.csv');d=d[d.Scope.isin(['Three wards','Chiyoda','Chuo','Minato','Whole supported domain'])].copy();d['contrast']=arm;shifts.append(d)
 d=pd.concat(shifts,ignore_index=True);d['head_rise_m2']=1e6*d['All Supported Certain Head Rise Area'];d.to_csv(P/'final_head_transfer_summary.csv',index=False)
 (P/'adapted_summary_source.py').write_text(src);(P/'adapted_direct_pair_source.py').write_text(dsrc)
 primary=direct[direct.bound.eq('lower')&direct.buffer_m.eq(10)&direct.scope.eq('Three wards')]
 result=json.loads((R/'comparison_summary.json').read_text());result.update(event=EVENT,direct_s1_vs_o1_primary=primary.to_dict('records'),interpretation='Judge road and paired resident-hour effects separately;intervals crossing zero do not establish a population ranking. Same flow quota does not imply same cost. Fixed-stage receivers omit downstream added storage.');dump(R/'comparison_summary.json',result)
 from summarize_three_event_controls_v1 import synthesize
 synthesize(P,R)
 dump(R/'final_summary_validation.json',{'passed':True,'four_numerical_and_pump_audits':audits,'exact_common_support_with_static':True,'direct_population_pairing_checked':True,'group_partitions_reconcile':True,'no_retuning':True,'new_hydraulic_runs':4,'three_event_synthesis_passed':True,'summary_source_sha256':sha(P/'adapted_summary_source.py'),'direct_pair_source_sha256':sha(P/'adapted_direct_pair_source.py'),'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


def report():
 s=json.loads((R/'campaign_state.json').read_text()) if (R/'campaign_state.json').exists() else {'status':'preflight_passed','completed_cases':[]}
 if s['status']=='qualified_complete':finalize()
 body='<h1>冻结O1／S1：2024年8月30日跨事件对照</h1><aside><b>状态：'+html.escape(s['status'])+'</b>；已通过 '+str(len(s.get('completed_cases',[])))+'/4组。</aside>'
 if s.get('current_case'):body+='<p>当前：'+html.escape(s['current_case'])+f'；模拟时刻 {s.get("latest_native_time_s",0)/3600:.2f}／9.00h。</p>'
 if s.get('error'):body+='<p class="warn">暂停并保留结果：'+html.escape(s['error'])+'</p>'
 body+='<p>7月部署原样迁移：O1为62个原生取水位置、池内动态调配；S1为319个交通优先位置、静态预留。各接收池假设额度0.5m³/s，总额度2.5m³/s。位置数不是设备台数，同流量额度不是同经济成本。沿用8月原雷达雨型、屋面分配、10m网格及高低水位；不重新选点或调参。</p><p>顺序O1高→O1低→O1后处理→S1高→S1低→S1后处理。保留原28项数值门槛和8项逐步抽排核验；失败停止，不自动重试或扩大到其他事件。</p>'
 for arm in ARMS:
  bp,br=dirs(arm);body+='<h2>'+arm+'</h2>'
  f=bp/'paired_road_effects.csv'
  if f.exists():
   d=pd.read_csv(f);d=d[d.Scope.eq('Three wards')&d['Depth Branch'].eq('lower')];body+=d[['Stage','Closure Reduction km h','Closure Reduction Percent','Maximum Same Time Newly Closed km']].to_html(index=False)
  f=bp/'access/paired_population_summary.csv'
  if f.exists():
   d=pd.read_csv(f);d=d[d.scope.eq('Three wards')&d.bound.eq('lower')&d.buffer_m.eq(10)];body+=d[['stage','net_avoided_loss_lower_resident_hours','net_avoided_loss_upper_resident_hours','maximum_new_loss_upper']].to_html(index=False)
 if (R/'final_summary_validation.json').exists():
  body+='<h2>三场暴雨的道路稳定性</h2>'+pd.read_csv(P/'cross_event_road_comparison.csv').to_html(index=False,float_format=lambda v:f'{v:,.3f}')
  for n,title in [('paired_population_vs_static_summary.csv','相对8月原静态部署：直接配对居民·小时'),('s1_vs_o1_population_summary.csv','S1相对O1：直接配对居民·小时')]:
   d=pd.read_csv(P/n);d=d[d.scope.eq('Three wards')&d.bound.eq('lower')&d.buffer_m.eq(10)];cols=[k for k in ['contrast','stage','net_lower_resident_hours','net_upper_resident_hours','new_loss_lower_resident_hours','new_loss_upper_resident_hours'] if k in d];body+='<h2>'+title+'</h2>'+d[cols].to_html(index=False,float_format=lambda v:f'{v:,.2f}')
  d=pd.read_csv(P/'final_head_transfer_summary.csv');body+='<h2>积水转移：相对无措施的水头抬升≥0.10m</h2>'+d[['contrast','Stage','Scope','head_rise_m2']].to_html(index=False)
  body+='<h2>三场暴雨总结</h2><p><a href="three_event_synthesis.json">完整三场汇总与解释边界</a></p>'
  body+='<p><a href="comparison_summary.json">完整比较</a> · <a href="final_summary_validation.json">最终核验</a></p>'
 body+='<p>本场固定窗口为2024-08-30日本时间02:00—11:00。沿用原干初始地表；窗口前约42.94mm汇水面积加权重建雨量未施加，因此不复原前期蓄水。</p>'
 body+='<p>人口为2020年常住人口的住地情景，保留0／10／25m候选接入及水深上下界；区间不是独立受困人数或置信区间。交通为条件可达性及参考路径成本，非真实拥堵排队。转移统计包括全域、候选控制区外和实际取水位置外；规定河道水位仍未模拟新增受纳储水。已检查过这场历史降雨，本轮是跨事件稳定性检查，不宣称独立实测验证。</p>'
 body+='<p><a href="preflight_validation.json">预检查</a> · <a href="interface_validation.json">接口回归</a> · <a href="campaign_state.json">实时状态</a> · <a href="README.md">复现与预算</a> · <a href="../three_ward_shared_receiver_controls_v1/review.html">7月对照结果</a></p>'
 style='body{font:16px/1.8 system-ui;max-width:1250px;margin:30px auto;padding:0 22px;color:#203d4b;background:#f6f8fa}aside{padding:18px;background:#e6f0f4}table{display:block;max-width:100%;overflow:auto;background:white;border-collapse:collapse;font-size:13px}th,td{border:1px solid #cfdae1;padding:8px;text-align:left}.warn{background:#ffefce;padding:15px}'
 refresh='<meta http-equiv="refresh" content="30">' if s['status'] not in ['held','qualified_complete'] else ''
 tmp=R/'review.html.tmp';tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+refresh+'<title>Frozen Controls August Transfer</title><style>'+style+'</style>'+body+'</html>');tmp.replace(R/'review.html')


def configure():
 # Reuse exact qualified operator/auditor code without editing the July campaign.
 control.P=P;control.R=R;control.OLD=STATIC;control.SCRIPT=SCRIPT
 worker.P=P;worker.R=R;worker.EVENTS=ARMS;worker.SCRIPT=SCRIPT;worker.report=report;worker.audit=control.audit;worker.access=access;worker.phase_module=control.post_module
 text=inspect.getsource(worker.run)
 assert text.count("if x['event_id']==event")==1
 text=text.replace("if x['event_id']==event","if x['contrast']==event")
 text=text.replace("container='udt-transfer-'+event[1:]+'-'+r['stage'].lower()","container='udt-controls-aug-'+event.lower()+'-'+r['stage'].lower()")
 text=text.replace('full_fixed_portfolio_transfer=True,two_new_rainfall_parents=True','frozen_O1_S1_transfer=True,rainfall_parents=1')
 exec(compile(text,str(SCRIPT),'exec'),worker.__dict__)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['selfcheck','prepare','run','audit','post','access','report']);ap.add_argument('--event',choices=ARMS);ap.add_argument('--case');a=ap.parse_args()
 if a.phase=='selfcheck':selfcheck();return
 if a.phase=='prepare':prepare();return
 configure()
 if a.phase=='run':worker.run()
 elif a.phase=='audit':control.audit(a.event,a.case)
 elif a.phase=='post':control.post(a.event)
 elif a.phase=='access':access(a.event)
 else:report()
if __name__=='__main__':main()
