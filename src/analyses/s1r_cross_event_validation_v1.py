#!/usr/bin/env python3
"""Frozen S1R Q2.5/Q5: July access reuse, eight serial August/September runs.
No geometry/forcing/quality/OD changes. All direct population contrasts are paired
at candidate nodes before aggregation. Existing campaigns remain read-only.
"""
from pathlib import Path
import os,sys,json,shutil,subprocess,inspect,argparse,html,hashlib,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
sys.path.append(str(ROOT/'.frame-control/runtimes/physical-measures-py312/lib/python3.12/site-packages'))
import numpy as np,pandas as pd
from shared_receiver_pilot_common_v1 import sha,dump
from shared_receiver_control_support_v1 import treatment_hash,identity_receipt_check,pool_rules
from shared_receiver_identity_v2 import CORE,selftest
import shared_receiver_transfer_v1 as worker
import shared_receiver_controls_v1 as control
import s1r_shared_allowance_pilot_v1 as quota
TAG='three_ward_s1r_cross_event_validation_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
SCRIPT=Path(__file__).resolve();DECISION='MILI-D-20261023-001'
EVENTS=['E20250911','E20240830'];QUOTAS=['Q2p5','Q5'];BATCHES=[e+'_'+q for e in EVENTS for q in QUOTAS]
JULY={q:ROOT/'data/processed'/n/'batches/O1' for q,n in [('Q2p5','three_ward_s1_reallocation_pilot_v1'),('Q5','three_ward_s1r_shared_allowance_pilot_v1')]}
TEMPLATES={e:ROOT/'data/processed'/n/'batches/O1' for e,n in [('E20250911','three_ward_shared_receiver_controls_transfer_v1'),('E20240830','three_ward_shared_receiver_controls_august_v1')]}

def read(f):return json.loads(Path(f).read_text())
def rep(p):return ROOT/'data/exp/estimation'/p.relative_to(ROOT/'data/processed')
def dirs(batch):return P/'batches'/batch,R/'batches'/batch

def context(event,q):
 if event=='E20240731':
  hp=JULY[q];return hp,rep(hp),P/'july'/q/'access',R/'july'/q/'access'
 hp,hr=dirs(event+'_'+q);return hp,hr,hp/'access',hr/'access'

def check_audits(hp,hr):
 runs=read(hp/'run_manifest.json')['runs'];assert len(runs)==2
 for r in runs:
  a=read(hr/'execution'/r['case_id']/'audit.json');b=read(hr/'execution'/r['case_id']/'pump_budget_summary.json')
  assert a['all_checks_passed']and a['admission']['technical_reference']and len(a['checks'])==28 and all(x['pass']for x in a['checks'])
  assert b['passed']and len(b['shared_receiver_checks'])==8 and all(b['shared_receiver_checks'].values())and a['summary']['negative_cells']==0
 return runs

def qualify():
 # The exact qualified image is re-exercised on synthetic fixtures, not real weather.
 quota.P=P;quota.R=R;quota.qualify()
 from run_temporal_road_population_v1 import signed_envelope,envelope
 info={'ix':np.array([0,1]),'ptr':np.array([0,2]),'lens':np.array([2]),'complete':np.array([True])}
 # Mixed restoration/new loss demonstrates why marginal bounds cannot be subtracted.
 lo,hi=signed_envelope(np.array([-1,1]),info,np.array([100]));assert lo[0]==-100 and hi[0]==100
 for change in [-1,0,1]:
  a,b=signed_envelope(np.full(2,change),info,np.array([100]));assert a[0]==b[0]==100*change
 a,b=envelope(np.array([True,False]),info,np.array([100]));assert not a[0]and b[0]
 source=adapt_worker_source();assert "x['batch']==event"in source and 'new_hydraulic_runs=8'in source
 for q in QUOTAS:check_audits(JULY[q],rep(JULY[q]));assert read(rep(JULY[q]).parent.parent/'final_summary_validation.json')['passed']
 dump(R/'interface_validation.json',{'passed':True,'independent_weighted_fixture_threads':[1,6],'identity_selftest':selftest(),'signed_candidate_interval_fixture':True,'batch_dispatch_and_eight_run_binding':True,'july_four_existing_cases_original28_plus8':True,'no_hydraulic_runs_started':True})
 print('CROSS_EVENT_INTERFACES_QUALIFIED',flush=True)

def prepare():
 assert read(R/'qualification.json')['passed']and read(R/'interface_validation.json')['passed']
 assert not(P/'run_manifest.json').exists(),'Preserve prepared attempt'
 assert not subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).strip(),'Existing Docker workload'
 assert shutil.disk_usage(P).free>=220*1024**3,'120GiB output forecast plus100GiB free floor'
 budget=dict(maximum_new_runs=8,concurrent_hydraulics=1,cpus=6,memory_gib=7,maximum_run_active_wall_s=7200,maximum_campaign_active_wall_s=57600,no_clock_progress_stop_s=900,maximum_new_output_gib=120,minimum_free_gib=100,maximum_audit_wall_s=3600,maximum_postprocess_wall_s=3600,maximum_access_wall_s=1800,automatic_retries=0)
 protocol=dict(decision=DECISION,events=['E20240731']+EVENTS,july_new_hydraulics=0,new_transfer_hydraulics=8,order=['July Q2.5 access','July Q5 access','July direct paired summary']+BATCHES,stage_order=['H','L'],allocation_event='E20240731',frozen='319 intake/outlet cells,weights,weighted allocator,0.5m3/s standalone ceiling,0.05m trigger,5m lift,solver,geometry,masks;each event original radar/roofs/stages unchanged',contrast='Only five shared receiving caps differ:0.5 vs1.0m3/s;totals2.5 vs5.0. Hypothetical allowances,not surveyed river capacity or installed equipment.',road='Same109clocks,both depth branches,0.30m closure threshold;paired restored and new closures and km-hours separately',population='2020 nighttime496345 residents in4746 basic units;frozen0/10/25m access candidates,10m main;paired signed changes before aggregation;unresolved stock separate',traffic='Original turn-memory bidirectional road access to16conditional outer portals;unknown roads assumed available. Fixed reference-cost detour only;no OD assignment or real congestion queues.',head='Native >=0.10m certain head rise against no action and Q2.5;peak simultaneous and any-time union separate;receiving-stage storage remains unmodeled',inference='Three inspected rainfall parents;scenario stability,not observed historical flood validation or sealed testing',checks='Original28numerical and8pump rules unchanged;failure holds entire queue;no automatic retries or retuning',budget=budget,training=False,sealed_test=False,new_quota_levels=False,new_locations=False)
 dump(P/'protocol.json',protocol)
 protected={};memo={};allruns=[]
 def checksum(f):
  f=Path(f);k=(str(f.resolve()),f.stat().st_size,f.stat().st_mtime_ns)
  if k not in memo:memo[k]=sha(f)
  return memo[k]
 def bind(f):protected[str(Path(f).relative_to(ROOT))]=checksum(f)
 # Bind frozen July scientific sources and completed road products, not mutable caches.
 for q in QUOTAS:
  hp,hr,_,_=context('E20240731',q);runs=check_audits(hp,hr)
  for n in ['run_manifest.json','protocol.json','pump_sites.csv','pool_allowances.csv','udt_pumps.dat','udt_pool_rules.dat','common_road_support.npz','paired_road_effects.csv','off_target_summary.csv']:bind(hp/n)
  for n in ['delivery_validation.json','postprocessing_validation.json','expanded_transfer_validation.json']:bind(hr/n)
  for r in runs:
   for arm in ['baseline','drainage']:bind(hp/'exposure'/f"{r['case_id']}__{arm}.npz")
 for event in EVENTS:
  template=TEMPLATES[event];tr=rep(template);check_audits(template,tr);priorruns=read(template/'run_manifest.json')['runs']
  for q in QUOTAS:
   batch=event+'_'+q;bp,br=dirs(batch);bp.mkdir(parents=True,exist_ok=False);br.mkdir(parents=True,exist_ok=False);(bp/'execution').mkdir()
   frozen=JULY[q];frozenrun=read(frozen/'run_manifest.json')['runs'][0];image=frozenrun['image_id']
   assert subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()==image
   files=['udt_pumps.dat','udt_pool_rules.dat']
   for name in files+['pump_sites.csv','pool_allowances.csv']:shutil.copy2(frozen/name,bp/name)
   assert checksum(bp/'udt_pumps.dat')==checksum(JULY['Q2p5']/'udt_pumps.dat')
   sites=pd.read_csv(bp/'pump_sites.csv');assert len(sites)==319 and sites['Source Native Cell One Based'].is_unique
   groups,units,caps=pool_rules(bp,319);assert len(caps)==5 and np.all(units==.5)and all(v==(.5 if q=='Q2p5'else 1.)for v in caps.values())
   batchruns=[]
   for stage in ['H','L']:
    base=next(x for x in priorruns if x['stage']==stage);src=ROOT/base['folder'];noaction=ROOT/base['original_folder']
    assert base['identity']['geometry_sha256']==frozenrun['identity']['geometry_sha256']and base['image_id']==image
    case=f'{event}__{stage}__U__s1r_'+('q2p5'if q=='Q2p5'else'q5');folder=bp/'runs'/case;folder.mkdir(parents=True,exist_ok=False);names=[]
    for name,h in base['input_sha256'].items():
     if name in files+['udt_run_identity.txt','udt_run_identity.json']:continue
     assert checksum(src/name)==h,(case,name);bind(src/name)
     if name=='sfincs_subgrid.nc':(folder/name).symlink_to(Path(base['large_readonly_source']).resolve())
     else:shutil.copy2(src/name,folder/name)
     names.append(name)
    for name in files:shutil.copy2(bp/name,folder/name);names.append(name)
    ident=base['identity'].copy();ident.update(run_id=case+'__frozen_s1r_transfer_10m',split='external_frozen_s1r_quota_transfer',solver_digest=image,patch_sha256=frozenrun['identity']['patch_sha256'],treatment_sha256=treatment_hash(folder,files))
    dump(folder/'udt_run_identity.json',ident);(folder/'udt_run_identity.txt').write_text('\n'.join(k+'='+str(v)for k,v in ident.items())+'\n');names+=['udt_run_identity.txt','udt_run_identity.json']
    run={**base,'case_id':case,'batch':batch,'contrast':'O1','scientific_portfolio':'S1R-'+q,'treatment_files':files,'measure':'frozen_same_s1r_layout_shared_quota_'+q,'folder':str(folder.relative_to(ROOT)),'identity':ident,'run_id':ident['run_id'],'split':ident['split'],'image_id':image,'input_sha256':{n:checksum(folder/n)for n in names}}
    for n in ['static_reference_case','static_reference_folder','quota_reference_case','quota_reference_folder']:run.pop(n,None)
    if q=='Q5':
     refcase=f'{event}__{stage}__U__s1r_q2p5';run.update(quota_reference_case=refcase,quota_reference_folder=str((dirs(event+'_Q2p5')[0]/'runs'/refcase).relative_to(ROOT)))
    for name in names:
     if name not in files+['udt_run_identity.txt','udt_run_identity.json']:assert run['input_sha256'][name]==base['input_sha256'][name]
     bind(folder/name)
    assert(folder/'sfincs.inp').read_bytes()==(noaction/'sfincs.inp').read_bytes()
    for name in ['sfincs_map.nc','udt_output_clock.csv','execution.json']:bind(noaction/name)
    fixture=bp/'qualification/identity'/case;fixture.mkdir(parents=True)
    for name in files+['udt_run_identity.txt','udt_run_identity.json']:shutil.copy2(folder/name,fixture/name)
    (fixture/'udt_diagnostic_identity.txt').write_text('\n'.join(k+'='+str(ident[k])for k in CORE)+'\n');assert identity_receipt_check(fixture,run)
    for name in files:
     raw=(fixture/name).read_bytes();(fixture/name).write_bytes(raw+b'\n');assert not identity_receipt_check(fixture,run);(fixture/name).write_bytes(raw)
    assert identity_receipt_check(fixture,run);batchruns.append(run);allruns.append(run)
   dump(bp/'protocol.json',{**protocol,'parent_event':event,'public_portfolio':'S1R-'+q,'total_shared_allowance_m3_s':2.5 if q=='Q2p5'else 5.,'new_hydraulic_runs':2})
   dump(bp/'run_manifest.json',{'runs':batchruns,'budget':budget,'protocol_sha256':checksum(bp/'protocol.json')})
   for n in ['protocol.json','run_manifest.json','pump_sites.csv','pool_allowances.csv']+files:bind(bp/n)
 assert len(allruns)==8 and len({r['case_id']for r in allruns})==8
 # Pairing and frozen inputs are checked across both quotas for every parent/stage.
 for e in EVENTS:
  a,b=[dirs(e+'_'+q)[0]for q in QUOTAS]
  assert checksum(a/'udt_pumps.dat')==checksum(b/'udt_pumps.dat')
  x=(a/'udt_pool_rules.dat').read_text().splitlines();y=(b/'udt_pool_rules.dat').read_text().splitlines();assert x[:2]==y[:2]and x[7:]==y[7:]
 dump(P/'run_manifest.json',{'runs':allruns,'budget':budget,'protocol_sha256':checksum(P/'protocol.json'),'execution_order':[r['case_id']for r in allruns]})
 for f in [P/'run_manifest.json',P/'protocol.json',R/'qualification.json',R/'interface_validation.json']:bind(f)
 dump(P/'protected_sources.json',protected)
 code=[SCRIPT]+[ROOT/'src/analyses'/n for n in ['s1r_shared_allowance_pilot_v1.py','shared_receiver_transfer_v1.py','shared_receiver_controls_v1.py','shared_receiver_control_support_v1.py','shared_receiver_pilot_common_v1.py','audit_shared_receiver_pilot_identity_v3.py','shared_receiver_identity_v2.py','audit_cross_event_protection_run_v1.py','postprocess_shared_receiver_pilot_v1.py','shared_receiver_access_effects_v1.py','temporal_transport_common_v1.py','run_temporal_road_population_v1.py','integrate_night_population_transport_v1.py','postprocess_five_event_population_v1.py']]+[ROOT/'src/preprocessing/reference_label_policy.py']
 dump(P/'execution_code_binding.json',{str(f.relative_to(ROOT)):checksum(f)for f in code})
 dump(R/'preflight_validation.json',{'passed':True,'new_cases':8,'frozen319intakes_and_outlets':True,'same_unit_caps_trigger_lift_allocator':True,'each_parent_forcing_geometry_roofs_and_stages_byte_exact':True,'both_treatment_file_identity_tamper_checks_per_case':True,'original28_plus8_unchanged':True,'existing_july_no_hydraulic_replay':True,'budget':budget})
 (R/'README.md').write_text('# Fixed S1R access and cross-event validation\n\nDecision '+DECISION+'. Reuse July Q2.5/Q5 outputs for access first; then eight serial cases in September Q2.5 H/L,September Q5 H/L,August Q2.5 H/L,August Q5 H/L order. Each pair must pass original28numerical+8pump checks, road postprocessing and candidate-paired resident access before the next pair. Fixed319native model locations are not installed pumps. Shared receiving allowances2.5/5m3/s are hypotheses. No extra locations or quota levels. Unknown roads remain conditional available. Official2020nighttime496345residents and frozen0/10/25m candidate intervals,not observed stranded people. No OD congestion assignment,ML,sealed tests or raw-data edits.\n\nReproduce: .venv/bin/python src/analyses/s1r_cross_event_validation_v1.py qualify; then prepare; then run once. No automatic restarts. A failed case holds queue; PAUSE/STOP in processed campaign root. Source hashes bound before run; subgrid symlinked and mounted read-only. Resource and numerical gates are in protocol.json. Maximum8newhydraulics,serial6CPU/7GiB,2hactive per case,16hactive total;3600sper audit/road phase,1800sper access pair;120GiBnewoutput,100GiBfreefloor. Current expectation about1hactive per hydraulic case plus audit/postprocessing;not a guaranteed wall deadline. Completion pairs both quotas and no action at identical109times,retains newly closed roads,new resident access losses,and >=0.10m native head rise. Peak simultaneous and any-time union areas differ. Prescribed receiver stages omit real added downstream storage. Three inspected radar parents do not establish observed flood accuracy.\n')
 report();print('S1R_CROSS_EVENT_PREFLIGHT_8_CASES',flush=True)

def access(event,q):
 import shared_receiver_access_effects_v1 as mod
 from temporal_transport_common_v1 import Reachability,population_inputs
 hp,hr,ap,ar=context(event,q)
 assert not(ar/'validation.json').exists(),'Do not overwrite qualified access'
 ap.mkdir(parents=True,exist_ok=True);ar.mkdir(parents=True,exist_ok=True)
 for n in ['node_cache','population','anchor_nodes']:(ap/n).mkdir(exist_ok=True)
 mod.P=ap;mod.R=ar;mod.HP=hp;mod.HR=hr;mod.report=lambda:None
 proto=read(ROOT/'data/processed/three_ward_shared_receiver_access_v1/protocol.json')
 anchor=pd.read_csv(mod.TEMP/'reference_states.csv');a=anchor[anchor.event_id.eq(event)&anchor.selection_reasons.str.contains('H_noaction_closed_length_peak')];assert len(a)==1
 proto.update(decision=DECISION,parent_event=event,scientific_portfolio='S1R-'+q,anchor_frame=int(a.iloc[0].frame),anchor_time_s=float(a.iloc[0].analysis_elapsed_s),frozen_selection_event='E20240731')
 dump(ap/'protocol.json',proto)
 bound={k:v for k,v in mod.bindings().items()if not k.startswith('data/processed/three_ward_shared_receiver_pilot_v1/')}
 bound.update({str(f.relative_to(ROOT)):sha(f)for f in [SCRIPT,hp/'run_manifest.json',hp/'protocol.json',ap/'protocol.json']});dump(ap/'input_bindings.json',bound)
 blocks,pop,normal,infos=population_inputs();eng=Reachability();old=read(mod.TEMP/'source_bindings.json')['sha256']
 for n in ['data/processed/three_ward_normal_vehicle_repair_v1/turn_router.npz','data/processed/three_ward_motor_flood_access_v1/road_projection.npz']:assert old[n]==bound[n]
 assert np.array_equal(eng.direct(np.zeros(len(mod.load(mod.TEMP/'clipped_road_lengths.npz')['qualified']),bool)),normal)and len(blocks)==4746 and pop.sum()==496345 and mod.routing_selfcheck()
 dump(ar/'preflight_validation.json',{'passed':True,'normal_turn_graph_exact':True,'global_cache_graph_binding_verified':True,'population_stock_conserved':True,'anchor_frame':proto['anchor_frame'],'depth_and_population_masks_unchanged':True})
 mod.run();assert read(ar/'validation.json')['passed'];report()

def direct_population(event,out):
 from temporal_transport_common_v1 import population_inputs
 from run_temporal_road_population_v1 import signed_envelope,envelope
 blocks,stock,normal,infos=population_inputs();sels=[np.ones(len(stock),bool)]+[blocks.ward.to_numpy(str)==w for w in ['13101','13102','13103']];scopes=['Three wards','Chiyoda','Chuo','Minato'];temp=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1';rows=[];endpoints=[];cache={}
 def node(ap,key):
  key=str(key);ck=(str(ap),key)
  if ck not in cache:
   f=ap/'node_cache'/(key+'.npz')
   if not f.exists():f=temp/'reachability_cache'/(key+'.npz')
   with np.load(f)as z:
    assert hashlib.sha256(z['closed_packed'].tobytes()).hexdigest()==key
    cache[ck]=np.unpackbits(z['node_ok_packed'],count=len(normal)).astype(bool)
   if len(cache)>160:cache.pop(next(iter(cache)))
  return cache[ck]
 for stage in ['H','L']:
  contexts=[context(event,q)for q in QUOTAS];cases=[next(r['case_id']for r in read(c[0]/'run_manifest.json')['runs']if r['stage']==stage)for c in contexts];aa,bb=[c[2]for c in contexts]
  with np.load(aa/'population'/(cases[0]+'.npz'))as z:a=z['node_cache_keys'];ts=z['analysis_times'];actual=z['actual_times'];ids=z['block_ids']
  with np.load(bb/'population'/(cases[1]+'.npz'))as z:b=z['node_cache_keys'];assert np.array_equal(ts,z['analysis_times'])and np.array_equal(actual,z['actual_times'])and np.array_equal(ids,z['block_ids'])
  assert np.array_equal(ids,blocks.block_id.to_numpy(str))and len(ts)==109
  for ti in range(109):
   for bi,bound in enumerate(['lower','upper']):
    # No-action states must match exactly, not merely have similar marginal counts.
    assert a[ti,bi,0]==b[ti,bi,0]
    x=node(aa,a[ti,bi,1]);y=node(bb,b[ti,bi,1]);delta=x.astype('i1')-y.astype('i1');delta[~normal]=0
    for buf,info in infos.items():
     lo,hi=signed_envelope(delta,info,stock);ra,rb=envelope(normal&~x&y,info,stock);na,nb=envelope(normal&x&~y,info,stock)
     for sel,scope in zip(sels,scopes):rows.append(dict(event=event,stage=stage,bound=bound,buffer_m=buf,scope=scope,frame=ti,elapsed_s=float(ts[ti]),net_lower=int(-hi[sel].sum()),net_upper=int(-lo[sel].sum()),restored_lower=int(stock[sel&ra].sum()),restored_upper=int(stock[sel&rb].sum()),new_loss_lower=int(stock[sel&na].sum()),new_loss_upper=int(stock[sel&nb].sum())))
  # Direct vehicle endpoint comparison on a fixed common clock and cohort.
  anchor=read(aa/'protocol.json')['anchor_frame'];assert anchor==read(bb/'protocol.json')['anchor_frame']
  ep=np.load(ROOT/'data/processed/three_ward_normal_vehicle_repair_v1/multi_endpoint_components.npz')['native_node_index']
  for bound in ['lower','upper']:
   ns=[]
   for ap,case in [(aa,cases[0]),(bb,cases[1])]:
    with np.load(ap/'anchor_nodes'/(case+'__'+bound+'__drainage.npz'))as z:ns.append(dict(z))
   x,y=[z['bidirectional']for z in ns];valid=normal[ep];cohort=valid&x[ep]&y[ep]
   costs=[np.maximum(z['outward_reference_seconds'][ep],z['inward_reference_seconds'][ep])for z in ns];ratio=np.divide(costs[1],costs[0],out=np.full_like(costs[0],np.nan),where=cohort&np.isfinite(costs[0])&(costs[0]>0)&np.isfinite(costs[1]));use=cohort&np.isfinite(ratio)
   endpoints.append(dict(event=event,stage=stage,bound=bound,anchor_frame=anchor,normal_reachable_endpoints=int(valid.sum()),Q2p5_reachable=int((valid&x[ep]).sum()),Q5_reachable=int((valid&y[ep]).sum()),restored=int((valid&~x[ep]&y[ep]).sum()),newly_lost=int((valid&x[ep]&~y[ep]).sum()),common_reachable_cohort=int(cohort.sum()),median_reference_cost_ratio=float(np.median(ratio[use]))if use.any()else None,p95_reference_cost_ratio=float(np.quantile(ratio[use],.95))if use.any()else None,meaning='fixed reference-cost comparison,not traffic congestion'))
 df=pd.DataFrame(rows);assert(df.net_lower<=df.net_upper).all();assert(df.restored_lower<=df.restored_upper).all();assert(df.new_loss_lower<=df.new_loss_upper).all();df.to_csv(out/'Q5_vs_Q2p5_population_trajectories.csv',index=False)
 summary=[]
 anchor=read(aa/'protocol.json')['anchor_frame']
 for keys,g in df.groupby(['event','stage','bound','buffer_m','scope']):
  g=g.sort_values('frame');row=dict(zip(['event','stage','bound','buffer_m','scope'],keys));f=g[g.frame.eq(anchor)].iloc[0]
  for k in ['net_lower','net_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper']:row[k+'_resident_hours']=float(np.trapezoid(g[k],g.elapsed_s)/3600);row['anchor_'+k]=int(f[k])
  row['maximum_new_loss_lower']=int(g.new_loss_lower.max());row['maximum_new_loss_upper']=int(g.new_loss_upper.max());summary.append(row)
 pd.DataFrame(summary).to_csv(out/'Q5_vs_Q2p5_population_summary.csv',index=False);pd.DataFrame(endpoints).to_csv(out/'Q5_vs_Q2p5_vehicle_endpoint_summary.csv',index=False)
 return len(df)

def compare(event):
 cp=P/'comparisons'/event;cr=R/'comparisons'/event;cp.mkdir(parents=True,exist_ok=True);cr.mkdir(parents=True,exist_ok=True)
 assert not(cr/'validation.json').exists(),'Preserve qualified comparison'
 for q in QUOTAS:
  hp,hr,ap,ar=context(event,q);check_audits(hp,hr);assert read(ar/'validation.json')['passed']
 ap,ar,_,_=context(event,'Q2p5');bp,br,_,_=context(event,'Q5')
 dump(cp/'run_manifest.json',read(bp/'run_manifest.json'))
 # Reuse the exact July road and native-head comparator, with context-only binding.
 source=inspect.getsource(quota.finalize)
 replacements={"'new_hydraulic_runs':2":"'new_hydraulic_runs_started_by_summary':0,'qualified_Q5_cases':2,'qualified_Q2p5_cases':2","'other_events_enabled':False":"'fixed_three_parent_transfer':True","'population_recomputed':False":"'population_recomputed_separately':True"}
 for before,after in replacements.items():assert source.count(before)==1;source=source.replace(before,after)
 env=quota.__dict__.copy();env.update(P=cp,R=cr,OLD=ap,OLDR=ar,dirs=lambda _='O1':(bp,br));exec(compile(source,str(SCRIPT)+'::bound_road_head','exec'),env);env['finalize']()
 n=direct_population(event,cp)
 supports=[np.load(c[0]/'common_road_support.npz')['qualified_roads']for c in [context(event,q)for q in QUOTAS]];assert np.array_equal(*supports)and int(supports[0].sum())==28064
 # No-action reference, both quotas, depths, radii and ward-level outputs remain visible.
 pop=[];accessrows=[]
 for q in QUOTAS:
  hp,hr,ap,ar=context(event,q)
  for name in ['population_summary.csv','paired_population_summary.csv']:
   d=pd.read_csv(ap/name);d.insert(0,'portfolio',q);d.insert(0,'event',event)
   dest=cp/('all_'+name)
   if name=='population_summary.csv':pop.append(d)
  d=pd.read_csv(ap/'anchor_access_summary.csv');d.insert(0,'portfolio',q);d.insert(0,'event',event);accessrows.append(d)
 pd.concat(pop,ignore_index=True).to_csv(cp/'population_totals_all_arms.csv',index=False);pd.concat(accessrows,ignore_index=True).to_csv(cp/'vehicle_access_vs_noaction.csv',index=False)
 dump(cr/'validation.json',{'passed':True,'event':event,'four_original28_plus8_audits':True,'same_road_support':True,'noaction_node_states_identical':True,'direct_candidate_pairs_not_marginal_subtraction':True,'direct_population_trajectory_rows':n,'all109frames_both_depths_three_radii_four_scopes':True,'new_access_losses_and_head_transfer_retained':True,'population_total':496345,'hydraulics_started_by_summary':0,'congestion_assignment':False})
 report();print('S1R_PARENT_COMPLETE',event,flush=True)

def adapt_worker_source():
 source=inspect.getsource(worker.run)
 replacements={"if x['event_id']==event":"if x['batch']==event","container='udt-transfer-'+event[1:]+'-'+r['stage'].lower()":"container='udt-s1rx-'+event.lower()+'-'+r['stage'].lower()",'new_hydraulic_runs=4':'new_hydraulic_runs=8','full_fixed_portfolio_transfer=True,two_new_rainfall_parents=True':'fixed_S1R_two_quotas_transfer=True,two_additional_rainfall_parents=True',"same=np.array_equal(supports[EVENTS[0]],supports[EVENTS[1]])":"same=all(np.array_equal(v,JULY_SUPPORT)for v in supports.values());assert same,'Common road support changed'"}
 for before,after in replacements.items():assert source.count(before)==1,before;source=source.replace(before,after)
 return source

def configure():
 control.P=P;control.R=R;control.ARMS=BATCHES;control.SCRIPT=SCRIPT
 worker.P=P;worker.R=R;worker.EVENTS=BATCHES;worker.SCRIPT=SCRIPT;worker.report=report;worker.audit=control.audit;worker.phase_module=control.post_module
 worker.__dict__['JULY_SUPPORT']=np.load(JULY['Q2p5']/'common_road_support.npz')['qualified_roads']
 exec(compile(adapt_worker_source(),str(SCRIPT)+'::bound_serial_worker','exec'),worker.__dict__)

def report():
 R.mkdir(parents=True,exist_ok=True)
 s=read(R/'campaign_state.json')if(R/'campaign_state.json').exists()else dict(status='prepared',completed_cases=[],completed_events=[])
 if s.get('status')=='qualified_complete'and not(R/'final_summary_validation.json').exists():finalize()
 status=read(R/'orchestration_state.json')if(R/'orchestration_state.json').exists()else{}
 body='<h1>S1R：道路—车辆可达性—人口影响与跨暴雨对照</h1><p><b>'+html.escape(status.get('status',s['status']))+'</b>；新增水动力通过 '+str(len(s.get('completed_cases',[])))+'/8，已启动 '+str(s.get('new_runs_started',0))+'/8。</p>'
 body+='<aside>先用7月已有两档结果补车辆可达性和夜间人口区间，再依次运行9月Q2.5高/低、9月Q5高/低、8月同样四组。固定319个模型取水/接收位置、权重、调配规则和单点上限；唯一对照为假设共享接收额度2.5与5 m³/s。</aside>'
 if s.get('current_case'):body+='<p>当前：'+html.escape(s['current_case'])+f"；模型时钟 {s.get('latest_native_time_s',0)/3600:.3f}/9 h；活跃计算 {s.get('case_active_wall_seconds',0):.0f} s。</p>"
 if s.get('error')or status.get('error'):body+='<p class="warn">已停止并保留结果：'+html.escape(s.get('error')or status['error'])+'</p>'
 if(P/'run_manifest.json').exists():
  rows=[]
  for r in read(P/'run_manifest.json')['runs']:rows.append({'Event':r['event_id'],'Stage':r['stage'],'Portfolio':r['scientific_portfolio'],'State':'qualified'if r['case_id']in s.get('completed_cases',[])else s['status']if s.get('current_case')==r['case_id']else'waiting'})
  body+=pd.DataFrame(rows).to_html(index=False)
 for event in ['E20240731']+EVENTS:
  body+='<h2>'+event+'</h2>'
  for q in QUOTAS:
   hp,hr,ap,ar=context(event,q)
   if(ar/'state.json').exists():body+='<p>'+q+' access: '+html.escape(read(ar/'state.json')['status'])+'</p>'
  cp=P/'comparisons'/event;cr=R/'comparisons'/event
  if(cr/'validation.json').exists():
   d=pd.read_csv(cp/'road_comparison.csv');d=d[d.scope.eq('Three wards')&d.branch.eq('lower')];body+='<h3>道路长度—时间积分（km h）</h3>'+d[['stage','reference','reference_closure_km_h','remaining_closure_km_h','net_avoided_km_h','newly_closed_km_h']].to_html(index=False,float_format=lambda x:f'{x:,.4f}')
   d=pd.read_csv(cp/'Q5_vs_Q2p5_population_summary.csv');d=d[d.scope.eq('Three wards')&d.bound.eq('lower')&d.buffer_m.eq(10)];body+='<h3>Q5相对Q2.5的直接配对夜间人口区间</h3>'+d[['stage','net_lower_resident_hours','net_upper_resident_hours','new_loss_lower_resident_hours','new_loss_upper_resident_hours']].to_html(index=False,float_format=lambda x:f'{x:,.2f}')
   body+='<p>居民·小时不是独立受困人数。净变化在同一候选接入位置配对后汇总；不相减两个边际区间。0/10/25 m、上下水深分支和三区分区表均保留。</p>'
   d=pd.read_csv(cp/'head_transfer_comparison.csv');d=d[d.scope.eq('Three wards')];body+='<h3>局部副作用筛查</h3>'+d[['stage','reference','peak_simultaneous_certain_head_rise_m2','any_time_certain_head_rise_union_m2','any_time_newly_wet_subset_of_head_screen_m2']].to_html(index=False)
   body+='<p><a href="comparisons/'+event+'/validation.json">配对核验</a></p>'
 body+='<h2>冻结规则与解释边界</h2><p>原28项数值和8项抽排检查不变。串行6 CPU/7 GiB；每组最多2小时活跃水动力、累计16小时；每次审核/道路后处理上限1小时、每对人口可达性30分钟。新输出上限120 GiB、可用空间至少100 GiB；失败停止、不自动重试。可在processed根目录放置PAUSE或STOP。</p><p>保留836.96 km合格道路与277.92 km未知分支，未知按原可用假设处理。2020年夜间常住人口496,345人，无法确定接入的位置单列，不是当天实际受困人数；不开展OD拥堵分配。三场已检查的历史雷达降雨用于情景稳定性，不代表有真实浸水观测验证。规定河道水位没有模拟新增下游蓄水，不能据此排除真实下游风险。</p><p><a href="../../../processed/'+TAG+'/protocol.json">Protocol / budget</a> · <a href="preflight_validation.json">Preflight</a> · <a href="campaign_state.json">Live state</a> · <a href="README.md">Reproduction</a></p>'
 style='body{font:16px/1.8 system-ui;max-width:1260px;margin:30px auto;padding:0 22px;background:#f7f9fb;color:#243d4d}aside{padding:18px;background:#e5eff5}table{display:block;max-width:100%;overflow:auto;background:white;border-collapse:collapse;font-size:13px}td,th{border:1px solid #cfdce3;padding:8px;text-align:left}.warn{background:#fff0d4;padding:16px}'
 refresh='<meta http-equiv="refresh" content="30">'if s['status']not in ['held','qualified_complete']and status.get('status')!='held'else''
 tmp=R/'review.html.tmp';tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+refresh+'<title>Frozen S1R Cross-event Impact Chain</title><style>'+style+'</style>'+body+'</html>');tmp.replace(R/'review.html')

def finalize():
 for e in ['E20240731']+EVENTS:assert read(R/'comparisons'/e/'validation.json')['passed']
 for name in ['road_comparison.csv','Q5_vs_Q2p5_population_summary.csv','Q5_vs_Q2p5_vehicle_endpoint_summary.csv','population_totals_all_arms.csv','head_transfer_comparison.csv']:
  rows=[]
  for event in ['E20240731']+EVENTS:
   d=pd.read_csv(P/'comparisons'/event/name)
   if'event'not in d:d.insert(0,'event',event)
   rows.append(d)
  pd.concat(rows,ignore_index=True).to_csv(P/('three_parent_'+name),index=False)
 dump(R/'final_summary_validation.json',{'passed':True,'new_hydraulic_cases_qualified':8,'reused_july_cases':4,'parents':3,'all_original28_plus8':True,'direct_candidate_population_pairs':True,'new_losses_and_head_rises_retained':True,'training':False,'sealed_test':False,'congestion_assignment':False,'observed_flood_accuracy_claim':False})


def run(resume_pre_hydro=False):
 if resume_pre_hydro:
  assert (R/'recovery_01/repair_validation.json').exists()
  assert read(R/'orchestration_state.json')['status']=='held'
  assert not(R/'campaign_state.json').exists(),'Never restart a hydraulic attempt'
  assert not any((P/'batches').rglob('launch.json')),'No hydraulic launch may have occurred'
  assert all(sha(ROOT/k)==v for k,v in read(P/'execution_code_binding.json').items())
  assert all(sha(ROOT/k)==v for k,v in read(P/'protected_sources.json').items())
  for q in QUOTAS:assert read(context('E20240731',q)[3]/'validation.json')['passed']
 else:assert not(R/'orchestration_state.json').exists(),'No automatic restart'
 dump(R/'orchestration_state.json',{'status':'july_access_reuse','pid':os.getpid()});report()
 try:
  for q in QUOTAS:
   if not resume_pre_hydro:access('E20240731',q)
  compare('E20240731');dump(R/'orchestration_state.json',{'status':'serial_cross_event_hydraulics','pid':os.getpid()});report()
  worker.run();assert read(R/'final_summary_validation.json')['passed'];dump(R/'orchestration_state.json',{'status':'qualified_complete','pid':os.getpid()});report()
 except BaseException as exc:
  import traceback
  dump(R/'orchestration_state.json',{'status':'held','pid':os.getpid(),'error':str(exc)});(R/'orchestration_error.txt').write_text(traceback.format_exc());report();raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['qualify','prepare','run','resume-prehydro','audit','post','access','compare','report']);ap.add_argument('--event',choices=BATCHES+['E20240731']);ap.add_argument('--case');a=ap.parse_args()
 if a.phase=='qualify':qualify();return
 if a.phase=='prepare':prepare();return
 configure()
 if a.phase=='run':run()
 elif a.phase=='resume-prehydro':run(resume_pre_hydro=True)
 elif a.phase=='audit':control.audit(a.event,a.case)
 elif a.phase=='post':control.post(a.event)
 elif a.phase=='access':
  event,q=a.event.rsplit('_',1);access(event,q)
  if q=='Q5':compare(event)
 elif a.phase=='compare':compare(a.event)
 else:report()
if __name__=='__main__':main()
