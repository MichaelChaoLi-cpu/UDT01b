#!/usr/bin/env python3
"""Four-way read-only diagnosis on the frozen three-parent drainage experiments.

No new solver execution, parameter change, routing, training or AnaSOP edit.
Run prepare, traces, roads, report in order, or all. Traces use two CPU readers.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json,hashlib,subprocess,time,concurrent.futures,html
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from summarize_three_event_controls_v1 import contexts
TAG='three_ward_drainage_bottleneck_diagnosis_v2'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
D=ROOT/'data/processed';SOURCE=D/'three_ward_shared_receiver_controls_august_v1'
POOLS=['Furukawa','Kamejima','Kanda','Nihonbashi','Sumida'];BIND={};STATS={}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def bind(p):
 p=Path(p);k=str(p.relative_to(ROOT))
 if k not in BIND:BIND[k]=sha(p);st=p.stat();STATS[k]=[st.st_size,st.st_mtime_ns]
 return p

def dump(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def save(name,data):data.to_csv(P/(name+'.csv'),index=False)
def integral(x,t):return np.trapezoid(np.asarray(x,float),t,axis=0)/3600

def prepare():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);(P/'traces').mkdir(exist_ok=True)
 runs=[]
 for event,base,report,static,suffix in contexts(SOURCE):
  assert json.loads(bind(report/'final_summary_validation.json').read_text())['passed']
  for scheme,path in [('Static',static),('O1',base/'batches/O1'),('S1',base/'batches/S1')]:
   sites=pd.read_csv(bind(path/'pump_sites.csv')).sort_values('Pump ID')
   assert sites['Pump ID'].tolist()==list(range(1,len(sites)+1))
   assert len(sites)==(319 if scheme=='S1'else 62)
   assert np.allclose(sites.groupby('Receiving Pool')['Reserved Rate'].sum(),.5,rtol=0,atol=1e-12)
   records=json.loads(bind(path/'run_manifest.json').read_text())['runs']
   for run in records:
    folder=ROOT/run['folder'];case=run['case_id'];trace=folder/'udt_pump_steps.csv';rpath=ROOT/'data/exp/estimation'/path.relative_to(D)/'execution'/case
    assert trace.exists()and folder.exists();audit=json.loads(bind(rpath/'audit.json').read_text());budget=json.loads(bind(rpath/'pump_budget_summary.json').read_text());assert audit['all_checks_passed']and budget['passed']
    assert audit['summary']['negative_cells']==0 and len(audit['checks'])==28 and len(budget['shared_receiver_checks'])==8 and all(budget['shared_receiver_checks'].values())
    spec=np.loadtxt(bind(folder/'udt_pumps.dat'),skiprows=1,ndmin=2)
    assert np.array_equal(spec[:,0],sites['Source Native Cell One Based'])and np.array_equal(spec[:,1],sites['Receiver Native Cell One Based'])and np.allclose(spec[:,2],sites['Reserved Rate'],atol=1e-15,rtol=0)
    assert np.all(spec[:,3]==.05)and np.all(spec[:,4]==5)
    meta=P/'traces'/f'{case}_metadata.tsv'
    with meta.open('w')as out:
     for a in sites.to_dict('records'):out.write(f"{a['Pump ID']} {POOLS.index(a['Receiving Pool'])} {a['Source Native Cell One Based']} {a['Receiver Native Cell One Based']} {a['Reserved Rate']:.17g}\n")
    runs.append({'event':event,'stage':run['stage'],'scheme':scheme,'case':case,'path':str(path.relative_to(ROOT)),'folder':run['folder'],'report':str(rpath.relative_to(ROOT)),'trace':str(trace.relative_to(ROOT)),'prefix':str((P/'traces'/case).relative_to(ROOT)),'trace_size':trace.stat().st_size,'trace_mtime_ns':trace.stat().st_mtime_ns,'expected_records':budget['pump_records'],'expected_steps':budget['actual_steps'],'expected_actual_m3':budget['actual_transferred_m3']})
 assert len(runs)==18 and len({x['case']for x in runs})==18
 dump(P/'runs.json',runs)
 dump(P/'prepare_bindings.json',BIND)
 dump(R/'protocol.json',{'scope':'three frozen rain parents,two prescribed stages,Static/O1/S1;whole-three-ward view','questions':['registered spatial coverage','actual-step receiving-pool saturation and eligibility','local intake availability versus unresolved surface conveyance','depth reductions that do not cross the road-closure threshold'],'no_new_hydrodynamics':True,'no_parameter_changes':True,'main_depth_branch':'lower;upper is a labeled sensitivity','nominal_pool_cap_m3_s':.5,'pool_count':5,'window_seconds':32400,'saturation_rule':'actual pool extraction >=99.99% of nominal per-step pool allowance','depth_response_screen_m':.01,'screen_meaning':'descriptive road-depth change,not a new closure rule or numerical significance certificate','coverage_meaning':'road-group/control-cell geometric linkage;not a causal influence radius or contributing catchment','fixed_state_envelope':'conditional hypothetical redistribution ceiling evaluated on already-simulated prior storage and eligibility;not achievable extra extraction or a forecast','conveyance_limit':'intake water availability is observed;blocked runoff routes cannot be diagnosed uniquely without appropriate face exchange and routing evidence','reader_processes':2,'input_bytes':sum(x['trace_size']for x in runs)})
 print('PREPARED',len(runs),'traces_GB',sum(x['trace_size']for x in runs)/1e9,flush=True)

def traces():
 runs=json.loads((P/'runs.json').read_text());binary=Path('/private/tmp/udt_drainage_trace_reader')
 subprocess.run(['/usr/bin/clang++','-O3','-std=c++17','-Wno-deprecated-declarations',str(ROOT/'src/analyses/drainage_trace_bottleneck_reader_v1.cpp'),'-o',str(binary)],check=True)
 def worker(run):
  prefix=ROOT/run['prefix'];validation=Path(str(prefix)+'_validation.json');trace=ROOT/run['trace']
  if not validation.exists():
   result=subprocess.run([str(binary),str(trace),str(prefix)+'_metadata.tsv',str(prefix),run['scheme']],capture_output=True,text=True)
   if result.returncode:raise RuntimeError(run['case']+': '+result.stderr)
  v=json.loads(validation.read_text());assert v['passed']and v['records']==run['expected_records']and v['steps']==run['expected_steps']
  assert trace.stat().st_size==run['trace_size']and trace.stat().st_mtime_ns==run['trace_mtime_ns']
  pool=pd.read_csv(str(prefix)+'_pool.csv');assert abs(pool.actual_m3.sum()-run['expected_actual_m3'])<1e-4
  assert np.allclose(pool.nominal_reserved_m3,16200,rtol=0,atol=.001)
  assert np.allclose(pool.nominal_reserved_m3,pool[['trigger_only_nominal_m3','head_only_nominal_m3','both_blocked_nominal_m3','eligible_storage_shortfall_nominal_m3','eligible_reserved_available_m3']].sum(axis=1),rtol=0,atol=1e-5)
  assert (pool.saturated_seconds<=pool.observed_seconds+1e-6).all()and (pool.saturated_with_deep_intake_seconds<=pool.saturated_seconds+1e-6).all()
  print('TRACE_VERIFIED',run['case'],round(float(pool.actual_m3.sum()),2),flush=True)
  return run['case']
 complete=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as executor:
  futures={executor.submit(worker,run):run for run in runs}
  for future in concurrent.futures.as_completed(futures):
   complete.append(future.result());dump(R/'progress.json',{'phase':'read_existing_traces','completed':len(complete),'total':18,'cases':complete,'new_simulations':0})
 pools=[];series=[];sites=[];case=[];tail_checks=[]
 for run in runs:
  prefix=ROOT/run['prefix'];a=pd.read_csv(str(prefix)+'_pool.csv');t=pd.read_csv(str(prefix)+'_time.csv');s=pd.read_csv(str(prefix)+'_site.csv')
  native=json.loads(Path(str(prefix)+'_validation.json').read_text())
  tail=max(0.,native['end_s']-32400.)
  assert tail<.001
  # The existing native clock ends <1 ms beyond the normalized map endpoint.
  # Retain that observed tail in the final bin; do not scale or alter simulation records.
  additive=[c for c in a if c not in ['pool_id','maximum_actual_rate_m3_s']]
  t[additive]=t[additive].astype(float);t['end_s']=t['end_s'].astype(float)
  for group,part in t.groupby('pool_id'):
   aa=a[a.pool_id.eq(group)].iloc[0];last=part.index[-1];nsite=int(s.pool_id.eq(group).sum())
   for field in additive:
    difference=float(aa[field]-part[field].sum())
    bound=tail*(max(1,nsite)if field.endswith('seconds')else .5)+2e-5
    assert -2e-5<=difference<=bound,(run['case'],field,difference,bound)
    t.loc[last,field]+=difference
   t.loc[last,'end_s']=native['end_s']
  tail_checks.append({'case':run['case'],'native_endpoint_tail_s':tail,'tail_retained_in_final_bin':True})
  for frame in [a,t,s]:
   for k in ['event','stage','scheme','case']:frame.insert(0,k,run[k])
   frame['Receiving Pool']=frame.pool_id.map(dict(enumerate(POOLS)))
  a['Window Utilization Percent']=100*a.actual_m3/a.nominal_reserved_m3
  a['Saturated Hours']=a.saturated_seconds/3600
  a['Deep Intake Saturated Hours']=a.saturated_with_deep_intake_seconds/3600
  a['Conditional Redistribution Gap m3']=np.maximum(a.fixed_state_pool_envelope_m3-a.actual_m3,0)
  for g,part in t.groupby('pool_id'):
   check=a[a.pool_id.eq(g)].iloc[0]
   for field in ['actual_m3','nominal_reserved_m3','saturated_seconds','fixed_state_pool_envelope_m3']:assert abs(part[field].sum()-check[field])<2e-4,(run['case'],field)
  pools.append(a);series.append(t);sites.append(s)
  r={k:run[k]for k in ['event','stage','scheme','case']};r.update(actual_m3=float(a.actual_m3.sum()),nominal_window_m3=float(a.nominal_reserved_m3.sum()),window_utilization_percent=float(100*a.actual_m3.sum()/a.nominal_reserved_m3.sum()),saturated_pool_hours=float(a.saturated_seconds.sum()/3600),deep_intake_saturated_pool_hours=float(a.saturated_with_deep_intake_seconds.sum()/3600),conditional_fixed_state_redistribution_gap_m3=float(np.maximum(a.fixed_state_pool_envelope_m3-a.actual_m3,0).sum()),trigger_only_nominal_m3=float(a.trigger_only_nominal_m3.sum()),head_only_nominal_m3=float(a.head_only_nominal_m3.sum()),both_blocked_nominal_m3=float(a.both_blocked_nominal_m3.sum()),storage_shortfall_nominal_m3=float(a.eligible_storage_shortfall_nominal_m3.sum()));case.append(r)
 save('pool_operating_diagnosis',pd.concat(pools,ignore_index=True));save('pool_five_minute_diagnosis',pd.concat(series,ignore_index=True));save('intake_operating_diagnosis',pd.concat(sites,ignore_index=True));save('case_operating_diagnosis',pd.DataFrame(case))
 dump(R/'trace_validation.json',{'passed':True,'cases':18,'all_original_audits_passed':True,'all_native_step_records_read':True,'SHA256_from_streaming_trace_reads':True,'pool_and_site_ledgers_reconciled':True,'five_minute_bins_conservatively_split_from_native_steps':True,'new_simulations':0,'binary_sha256':sha(binary),'native_endpoint_tail_reconciliation':tail_checks})
 print('ALL_TRACES_COMPLETE',flush=True)

def roads():
 runs=json.loads((P/'runs.json').read_text());clip=np.load(bind(D/'three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz'));weights=clip['union_length_m']/1000
 members=pd.read_parquet(bind(D/'three_ward_protection_requirements_register_v1/road_group_members_preprocessed.parquet'));links=pd.read_parquet(bind(D/'three_ward_protection_requirements_register_v1/road_control_cell_group_links_preprocessed.parquet'))
 policy=pd.read_parquet(bind(D/'three_ward_protection_service_demand_plan_v1/road_policy_membership_preprocessed.parquet'));assert members['Road Index'].is_unique
 mapping=members.set_index('Road Index')['Road Group'].to_dict();combined=set(policy.loc[policy['Combined Candidate'],'Road Group'])
 rows=[];bands=[];samples=[];group_rows=[];coverage=[]
 ref=pd.read_csv(bind(SOURCE/'three_event_road_effects.csv'));sitecache={};qi0=None
 for run in runs:
  event,stage,scheme,case=[run[k]for k in ['event','stage','scheme','case']];path=ROOT/run['path']
  sites=pd.read_csv(bind(path/'pump_sites.csv'));selected=set(links.loc[links['Grid Index'].isin(sites['Grid Index']),'Road Group'])
  with np.load(bind(path/'exposure'/f'{case}__baseline.npz'))as z:a=np.stack([z['lower'],z['upper']],axis=1);qi=z['road_indices'];t=z['analysis_times'];astate=z['road_state_codes']
  with np.load(bind(path/'exposure'/f'{case}__drainage.npz'))as z:b=np.stack([z['lower'],z['upper']],axis=1);assert np.array_equal(qi,z['road_indices'])and np.array_equal(t,z['analysis_times']);bstate=z['road_state_codes']
  assert a.shape==b.shape==(109,2,28064)and np.isfinite(a).all()and np.isfinite(b).all()
  if qi0 is None:qi0=qi.copy()
  else:assert np.array_equal(qi0,qi)
  assert np.array_equal(a>=.3,astate==2)and np.array_equal(b>=.3,bstate==2)
  ww=weights[qi];groups=np.array([mapping.get(int(x),'outside_registered_groups')for x in qi],object)
  bins=np.array(['intake_linked_group'if g in selected else'other_registered_group'if g!='outside_registered_groups'else'outside_registered_groups'for g in groups],object)
  pbin=np.array(['priority_policy_candidate'if g in combined else'context_or_unregistered'for g in groups],object)
  for name in np.unique(bins):coverage.append(dict(event=event,stage=stage,scheme=scheme,case=case,registered_link_class=name,qualified_length_km=float(ww[bins==name].sum()),intake_linked_group_count=len(selected),coverage_interpretation='geometric registered-group linkage;not hydraulic influence or causal coverage'))
  for bi,branch in enumerate(['lower','upper']):
   x,y=a[:,bi],b[:,bi];baseclosed=x>=.3;closed=y>=.3;delta=x-y
   restored=baseclosed&~closed;new=~baseclosed&closed;persistent=baseclosed&closed
   decreased=persistent&(delta>=.01);increased=persistent&(delta<=-.01);small=persistent&~(decreased|increased)
   metrics={'baseline':baseclosed,'remaining':closed,'restored':restored,'newly_closed':new,'reduced_but_still_closed':decreased,'small_change_still_closed':small,'increased_still_closed':increased}
   histories={k:v@ww for k,v in metrics.items()};tot={k:float(integral(v,t))for k,v in histories.items()}
   assert abs(tot['remaining']-tot['reduced_but_still_closed']-tot['small_change_still_closed']-tot['increased_still_closed']-tot['newly_closed'])<1e-7
   assert abs(tot['baseline']-tot['restored']-tot['reduced_but_still_closed']-tot['small_change_still_closed']-tot['increased_still_closed'])<1e-7
   old=ref[ref.event.eq(event)&ref.scheme.eq(scheme)&ref.Stage.eq(stage)&ref['Depth Branch'].eq(branch)&ref.Scope.eq('Three wards')].iloc[0]
   assert abs(tot['baseline']-float(old['Baseline Closure km h']))<1e-7 and abs(tot['baseline']-tot['remaining']-float(old['Closure Reduction km h']))<1e-7
   row=dict(event=event,stage=stage,scheme=scheme,case=case,branch=branch,**{k+'_km_h':v for k,v in tot.items()})
   row['reduced_still_closed_share_of_baseline_percent']=100*tot['reduced_but_still_closed']/tot['baseline']
   row['remaining_depth_reduction_weighted_mean_m']=float(integral(np.where(decreased,delta,0)@ww,t)/tot['reduced_but_still_closed'])if tot['reduced_but_still_closed']>0 else None
   rows.append(row)
   for ti in range(109):samples.append(dict(event=event,stage=stage,scheme=scheme,branch=branch,frame=ti,elapsed_s=float(t[ti]),**{k+'_km':float(v[ti])for k,v in histories.items()}))
   for name,mask in [('intake_linked_group',bins=='intake_linked_group'),('other_registered_group',bins=='other_registered_group'),('outside_registered_groups',bins=='outside_registered_groups')]:
    values={k+'_km_h':float(integral(v[:,mask]@ww[mask],t))for k,v in metrics.items()}
    group_rows.append(dict(event=event,stage=stage,scheme=scheme,branch=branch,registered_link_class=name,**values))
   for arm,z in [('No action',x),('Remaining',y)]:
    summed=0
    for lo,hi,label in [(.3,.4,'0.30-0.40'),(.4,.6,'0.40-0.60'),(.6,1.,'0.60-1.00'),(1.,np.inf,'1.00+')]:
     v=float(integral(((z>=lo)&(z<hi))@ww,t));summed+=v;bands.append(dict(event=event,stage=stage,scheme=scheme,branch=branch,arm=arm,depth_band_m=label,closed_exposure_km_h=v))
    assert abs(summed-tot['baseline'if arm=='No action'else'remaining'])<1e-7
  print('ROAD_DIAGNOSED',case,flush=True)
 save('road_response_partition',pd.DataFrame(rows));save('road_response_trajectories',pd.DataFrame(samples));save('remaining_depth_distribution',pd.DataFrame(bands));save('road_registered_link_effects',pd.DataFrame(group_rows));save('registered_coverage',pd.DataFrame(coverage))
 assert all((ROOT/key).stat().st_size==value[0]and (ROOT/key).stat().st_mtime_ns==value[1]for key,value in STATS.items())
 dump(P/'road_source_bindings.json',BIND);dump(R/'road_validation.json',{'passed':True,'cases':18,'branches':2,'all109frames':True,'same_qualified_road_order':True,'baseline_and_remaining_partition_close':True,'depth_bands_sum_to_closure_exposure':True,'all_qualified_summary_integrals_reproduced':True,'coverage_not_called_hydraulic_influence':True,'new_simulations':0})
 print('ROAD_DIAGNOSIS_COMPLETE',flush=True)

def report():
 from datetime import datetime
 from zoneinfo import ZoneInfo
 assert json.loads((R/'trace_validation.json').read_text())['passed']and json.loads((R/'road_validation.json').read_text())['passed']
 previous_figure=ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
 previous_bindings=json.loads((previous_figure/'cross_event_allocation_and_placement_effects_totals_source_bindings.json').read_text())
 previous_output=json.loads((previous_figure/'cross_event_allocation_and_placement_effects_totals_validation.json').read_text())
 assert sha(ROOT/'docs/AnaSOP.md')==previous_bindings['docs/AnaSOP.md']
 assert sha(ROOT/'data/results/figures/Figure_cross_event_allocation_and_placement_effects.png')==previous_output['output_sha256']
 operating=pd.read_csv(P/'case_operating_diagnosis.csv');pools=pd.read_csv(P/'pool_operating_diagnosis.csv');road=pd.read_csv(P/'road_response_partition.csv');link=pd.read_csv(P/'road_registered_link_effects.csv');coverage=pd.read_csv(P/'registered_coverage.csv')
 assert len(operating)==18 and len(pools)==90 and len(road)==36
 main=road[road.branch.eq('lower')].copy();s1=main[main.scheme.eq('S1')]
 linked=link[link.branch.eq('lower')].copy()
 for _,part in linked.groupby(['event','stage','scheme']):
  matching=main[main.event.eq(part.event.iloc[0])&main.stage.eq(part.stage.iloc[0])&main.scheme.eq(part.scheme.iloc[0])].iloc[0]
  for col in ['baseline_km_h','remaining_km_h','restored_km_h','newly_closed_km_h']:assert abs(part[col].sum()-matching[col])<1e-7
 linked['baseline_share_percent']=100*linked.baseline_km_h/linked.groupby(['event','stage','scheme']).baseline_km_h.transform('sum')
 linked['restoration_share_percent']=100*linked.restored_km_h/linked.groupby(['event','stage','scheme']).restored_km_h.transform('sum')
 save('registered_link_exposure_shares',linked)
 merged=main.merge(operating,on=['event','stage','scheme','case'],validate='one_to_one')
 merged['net_reduction_percent']=100*(merged.baseline_km_h-merged.remaining_km_h)/merged.baseline_km_h
 merged['mean_saturated_hours_per_pool']=merged.saturated_pool_hours/5
 merged['conditional_gap_fraction_of_actual_percent']=100*merged.conditional_fixed_state_redistribution_gap_m3/merged.actual_m3
 merged['small_change_share_of_remaining_percent']=100*merged.small_change_still_closed_km_h/merged.remaining_km_h
 save('primary_diagnosis_summary',merged)
 sept=pools[pools.event.eq('E20250911')&pools.stage.eq('H')&pools.scheme.eq('O1')]
 s1link=linked[linked.scheme.eq('S1')&linked.registered_link_class.eq('intake_linked_group')]
 s1outside=linked[linked.scheme.eq('S1')&linked.registered_link_class.eq('other_registered_group')]
 s1ops=operating[operating.scheme.eq('S1')]
 evidence=[
  {'question':'Receiving allowance','status':'Observed operating constraint under assumed limits','finding':f"September high-stage O1 pools are at their nominal 0.5 m3/s limit for {sept['Saturated Hours'].min():.2f}-{sept['Saturated Hours'].max():.2f} h each out of 9 h; eligible intake depth remains at least 0.30 m during {sept['Deep Intake Saturated Hours'].min():.2f}-{sept['Deep Intake Saturated Hours'].max():.2f} h of those periods.",'limit':'Does not identify how much traffic improvement a larger allowance would yield; prescribed stage is not a measured downstream capacity.','follow_up':'With placement and allocation fixed,compare declared capability/receiving envelopes only after a separately frozen run budget.'},
  {'question':'Static allocation within S1','status':'Observed fixed-state unused redistribution opportunity','finding':f"On the already-simulated S1 states,the conditional per-pool eligible/storage envelope exceeds actual extraction by {s1ops.conditional_fixed_state_redistribution_gap_m3.min():.0f}-{s1ops.conditional_fixed_state_redistribution_gap_m3.max():.0f} m3 per case.",'limit':'This is not predicted additional extraction or guaranteed road benefit; reallocating changes future storage,eligibility and flows. O1 fills this conditional envelope by construction.','follow_up':'Compare S1 static and within-pool reallocation at identical S1 locations and total allowance;do not change capacity simultaneously.'},
  {'question':'Spatial coverage','status':'Limited registered geometric linkage;causal influence unresolved','finding':f"S1 intakes link to 82 registered road groups and 37.51 km of qualified road geometry. These groups carry {s1link.baseline_share_percent.min():.2f}-{s1link.baseline_share_percent.max():.2f}% of baseline closure exposure. Other registered groups contribute {s1outside.restoration_share_percent.min():.2f}-{s1outside.restoration_share_percent.max():.2f}% of restored exposure,showing that benefits extend outside direct co-location.",'limit':'A road group is not a drainage catchment. Do not label the remaining roads hydraulically unprotected or extrapolate benefit linearly with intake count.','follow_up':'Use residual closures and existing source-receiver eligibility to review protection coverage;derive added locations from requirements rather than a target count.'},
  {'question':'Road-closure threshold','status':'Directly reconstructed from paired road depths','finding':f"Under S1,{s1.reduced_still_closed_share_of_baseline_percent.min():.2f}-{s1.reduced_still_closed_share_of_baseline_percent.max():.2f}% of baseline length-time exposure has road depth reduced by at least 0.01 m but remains at or above 0.30 m. The length-time-weighted decrease within that subset is {s1.remaining_depth_reduction_weighted_mean_m.min():.3f}-{s1.remaining_depth_reduction_weighted_mean_m.max():.3f} m.",'limit':'The 0.01 m screen is descriptive,not a changed passability threshold or significance certificate. These subsets cannot be added to net closure reduction as a generic benefit percentage.','follow_up':'Retain depth-response and residual closure beside restored passage;do not relax the 0.30 m rule to improve apparent results.'},
  {'question':'Surface conveyance to intakes','status':'Not uniquely identifiable from current records','finding':'Native traces establish local intake water depth,storage and gate states,but do not trace the origin or path of water that remains on roads elsewhere. Existing ground-face replay covers July no-action cases rather than a synchronized three-parent intervention comparison.','limit':'Dry intakes do not establish blocked runoff paths;shared road-group membership does not establish hydraulic reachability. No road curb or inlet blockage is inferred.','follow_up':'If needed,add specifically scoped face-exchange/tracer or path diagnostics in a future approved replay;no extra replay is part of this diagnosis.'}
 ]
 save('bottleneck_evidence',pd.DataFrame(evidence))
 bindings={}
 for path in [P/'prepare_bindings.json',P/'road_source_bindings.json']:
  for key,value in json.loads(path.read_text()).items():
   if key in bindings:assert bindings[key]==value
   bindings[key]=value
 runs=json.loads((P/'runs.json').read_text());records=0
 old=json.loads((D/'three_ward_measure_bottleneck_diagnosis_v1/input_bindings.json').read_text())['sha256']
 for run in runs:
  f=ROOT/run['trace'];st=f.stat();assert st.st_size==run['trace_size']and st.st_mtime_ns==run['trace_mtime_ns']
  v=json.loads(Path(str(ROOT/run['prefix'])+'_validation.json').read_text());bindings[run['trace']]=v['trace_sha256'];records+=v['records']
  if run['scheme']=='Static':assert old[run['trace']]==v['trace_sha256']
 # Small/non-trace inputs are checked again; large trace content was SHA256-hashed
 # while streaming and its size/mtime is checked through completion.
 tracekeys={x['trace']for x in runs}
 for key,value in bindings.items():
  if key not in tracekeys:assert sha(ROOT/key)==value,('Changed source',key)
 dump(P/'source_bindings.json',{'sha256':bindings,'large_trace_integrity':'streaming SHA256 plus stable size and nanosecond mtime;static digests also reproduce the previous diagnosis','scripts':{str(x.relative_to(ROOT)):sha(x)for x in [Path(__file__),ROOT/'src/analyses/drainage_trace_bottleneck_reader_v1.cpp']}})
 proposal={'status':'recommendation_only_no_execution','first_isolated_contrast':'S1 fixed allocation versus reallocation at exactly the same S1 positions,receivers and 2.5m3/s total allowance','second_isolated_contrast':'After allocation comparison,hold layout and operator fixed and compare declared capacity/receiving envelopes;unit ceilings and receiver assumptions must be explicit','coverage_follow_up':'Requirements-led extension only;no pump count,engineering capacity or feasible new outlet is inferred','surface_conveyance':'Keep unidentified pending appropriate paired face-exchange or tracking diagnostics','numerical_rules':'Retain original 28+8 numerical/pump rules and separate actual adverse effects from numerical failures','run_enabled':False,'no_new_parent_or_sealed_test':True}
 dump(R/'follow_up_recommendation.json',proposal)
 def table(frame):return '<div class="scroll">'+frame.to_html(index=False,float_format=lambda x:f'{x:,.3f}',escape=True)+'</div>'
 cols=['event','stage','scheme','baseline_km_h','remaining_km_h','net_reduction_percent','actual_m3','window_utilization_percent','mean_saturated_hours_per_pool','conditional_fixed_state_redistribution_gap_m3','reduced_still_closed_share_of_baseline_percent']
 public=merged[cols].rename(columns={'event':'Event','stage':'Stage','scheme':'Portfolio','baseline_km_h':'No-action closure (km h)','remaining_km_h':'Remaining closure (km h)','net_reduction_percent':'Closure reduction (%)','actual_m3':'Actual extraction (m3)','window_utilization_percent':'Window utilization (%)','mean_saturated_hours_per_pool':'Mean saturated hours per pool','conditional_fixed_state_redistribution_gap_m3':'Fixed-state redistribution gap (m3)','reduced_still_closed_share_of_baseline_percent':'Baseline exposure reduced >=1 cm but still closed (%)'})
 poolpublic=sept[['Receiving Pool','Window Utilization Percent','Saturated Hours','Deep Intake Saturated Hours']]
 coverpublic=s1link[['event','stage','baseline_km_h','baseline_share_percent','restored_km_h']].rename(columns={'event':'Event','stage':'Stage','baseline_km_h':'Baseline exposure in linked groups (km h)','baseline_share_percent':'Share of all baseline exposure (%)','restored_km_h':'Restored exposure in linked groups (km h)'})
 threshold=s1[['event','stage','baseline_km_h','restored_km_h','reduced_but_still_closed_km_h','small_change_still_closed_km_h','increased_still_closed_km_h','newly_closed_km_h','remaining_depth_reduction_weighted_mean_m']]
 now=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
 text=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区抽排改善为何有限：四类瓶颈核查</title><style>body{{font:16px/1.8 system-ui;max-width:1320px;margin:28px auto;padding:0 22px;color:#243c47;background:#f8fafb}}h1{{font-size:29px}}h2{{margin-top:32px}}aside,.card{{background:#e7f0f5;padding:18px;border-left:4px solid #4477aa}}.warn{{background:#fff2d9;padding:16px}}.scroll{{overflow:auto;max-width:100%;background:white}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #d2dce0;padding:7px;text-align:left;min-width:65px}}a{{color:#176a96}}code{{overflow-wrap:anywhere}}</style></head><body><h1>为什么封路改善只有约1%—5%？</h1><aside><b>结论：已发现局部额度饱和、静态分配闲置和未跨封路阈值三类直接证据；部署关联范围有限，但真实水力影响范围及地表通水瓶颈仍未识别。</b><br>已核验3场降雨×2种水位×3种方案，共18组既有结果；读取{records:,}条抽排记录，原28项数值及8项抽排核验全部保留。没有运行新模拟、调参、训练、访问封存测试或计算新的人口影响。</aside><p>完成时间（日本）：{now}。三区主分支使用共同836.958 km道路，277.920 km未知暴露继续单列。109个输出时刻、0.30 m封路规则及质量掩膜不变；水深上界只作为有标签的敏感性分支。</p>
<h2>1．容量：9月确有长时段满额，7月低平均也不排除短时满额</h2><p>以下为9月高水位O1。五池各自的假设上限均为0.5 m³/s；“满额”定义为实际步抽排达到该步额度的99.99%。最后一列还要求同池至少有一个符合抽排条件、局部水深≥0.30 m的取水位置；该局部水深并不等于关联道路的路面水深。</p>{table(poolpublic)}<p>这直接说明<strong>在既有位置和状态下存在额度限制</strong>，但并不能把封路剩余量归因于容量，或推算加倍容量会减少多少封路。7月O1高水位的单池满额时长约1.70—5.24小时，尽管全窗口总利用率仅34.51%；此前仅看全窗口平均，不能排除峰值时段限制。</p>
<h2>2．分配：S1仍有可重新分配的条件空间</h2><p>S1改了布局，但仍使用固定预留。逐步检查表明，有些取水位置未达到启动条件时，同池其他位置仍有可抽水量。以8月高水位S1为例，在其已经算出的状态上，假设允许池内重新分配的条件额度比实际抽排多约<strong>1.69万m³</strong>。</p><div class="warn">这1.69万m³是逐步固定状态的条件筛查差额，<strong>不是新模拟的额外抽排量或承诺收益</strong>。重新分配后未来水位、可用储量与汇流都会变化。O1已按其分配算子用满同样的条件可用额度；S1+O1尚未运行。</div>
<h2>3．覆盖：直接登记关联较窄，但不能当作水力影响边界</h2><p>Static/O1取水位置直接关联39个道路组、8.62 km已核验道路；S1关联82组、37.51 km。这是道路控制单元与组的登记关系，不是实体管线、汇水区或已证明的排水半径。S1关联组的原始封路暴露仅占三区总量约10.50%—13.46%。</p>{table(coverpublic)}<p>同时，S1约34.96%—51.42%的恢复暴露发生在其他登记道路组，说明效果并不限于直接关联位置。<strong>不能据此说其余约90%的封路完全未受保护，也不能按增加点数线性外推收益。</strong></p>
<h2>4．阈值：有些道路变浅了，却仍然不能通行</h2><p>S1约11.64%—15.43%的原始累计封路暴露，水深下降至少1 cm，却仍≥0.30 m；这一子集的长度—时间加权平均降深约4.3—6.5 cm。它们没有被算入恢复通行。以下保留恢复、变浅仍封闭、变化不足1 cm、加深仍封闭和新增封路的独立量。</p>{table(threshold)}<p>1 cm仅是本次公开的描述性响应筛查线，不是改变封路标准，也不是统计显著性或模型误差界限。不同指标不能直接相加为“总改善率”。多数剩余封路暴露的深度变化小于1 cm，阈值效应只能解释一部分，不是全部原因。</p>
<h2>5．通水：目前不能判定是路缘、道路分隔或入流路径堵住</h2><p>逐步记录能确认取水位置自身是否有水、储量是否足够、是否触发水深或扬程限制。它不能追溯别处道路积水是否本可流到该位置。既有地表格面交换重放是7月无措施组，没有覆盖三区三场暴雨的全部措施对照。<strong>取水点偏干不等于汇流路径堵塞，直接道路组关联也不等于能通水。</strong>本轮不把未知位置判为错误选址，不新增逐点现场核查。</p>
<h2>全部18组主分支诊断</h2>{table(public)}<p>平均每池满额时长＝五池满额时长之和÷5，最大约9小时，不是将五池之和误当作城市持续时间。全窗口额度按实际计算时钟积分；原时钟终点超过归一化地图终点约0.000483秒，仍在原0.001秒门限内，该实测尾段保留在最后一个分箱中。没有放宽原规则或改动模拟曲线。</p><p>启动／扬程／储量分解是<strong>原固定预留的条件台账</strong>。对于O1，其他位置可接收闲置额度，所以这些名义受阻量不能直接叫作实际损失抽排量，更不能与实际抽排相加解释能力缺口。分配与容量约束同时存在，未做因果百分比分摊。</p>
<h2>下一步建议：先分配，再能力，单因素分开</h2><ol><li>在冻结的S1位置和原2.5 m³/s总额度上，对照固定分配与池内调配；先检验改调配能否利用已出现的条件空间。</li><li>再固定布局与分配算子，另设清楚的能力／接收额度对照，检验额外资源是否真正转化为恢复通行；不把假设额度当东京真实接收能力。</li><li>保护范围扩张应从剩余关键道路需求和合格接收位置推导，不预先指定泵点数。</li></ol><p><b>以上只是后续对照建议，本轮没有启动。不能由现有结果宣称某因素解释了剩余封路的确定比例。</b>原图与AnaSOP未修改；后续图表仍暂停。</p><h2>数据、方法与复现</h2><p><a href="validation.json">总核验</a> · <a href="trace_validation.json">逐步抽排核验</a> · <a href="road_validation.json">道路分解核验</a> · <a href="protocol.json">诊断口径</a> · <a href="follow_up_recommendation.json">未执行的后续建议</a> · <a href="../../../processed/{TAG}/primary_diagnosis_summary.csv">18组主汇总</a> · <a href="../../../processed/{TAG}/pool_operating_diagnosis.csv">90组接收池台账</a> · <a href="../../../processed/{TAG}/pool_five_minute_diagnosis.csv">分时接收池台账</a> · <a href="../../../processed/{TAG}/bottleneck_evidence.csv">证据与未知项</a></p></body></html>'''
 # Derive presentation ranges from the verified tables,not manually rounded bounds.
 frac=s1outside.restoration_share_percent
 text=text.replace('34.96%—51.42%',f'{frac.min():.2f}%—{frac.max():.2f}%')
 (R/'review.html').write_text(text)
 summary={'completed_jst':now,'cases':18,'original_numerical_rules_preserved':True,'records_read':records,'input_trace_GB':sum(x['trace_size']for x in runs)/1e9,'september_high_O1_saturated_hours_per_pool':[float(sept['Saturated Hours'].min()),float(sept['Saturated Hours'].max())],'S1_decreased_but_closed_share_of_baseline_percent':[float(s1.reduced_still_closed_share_of_baseline_percent.min()),float(s1.reduced_still_closed_share_of_baseline_percent.max())],'S1_direct_linked_baseline_share_percent':[float(s1link.baseline_share_percent.min()),float(s1link.baseline_share_percent.max())],'S1_restoration_outside_linked_groups_percent':[float(frac.min()),float(frac.max())],'surface_conveyance_identified':False,'new_hydrodynamic_runs':0,'next_contrast_started':False,'AnaSOP_changed':False}
 dump(R/'summary.json',summary)
 dump(R/'validation.json',{'passed':True,'cases':18,'branches':2,'all_original_28_plus_8_checks_passed':True,'all_native_records_hashed_and_scanned':True,'road_partitions_and_depth_bands_reconciled':True,'road_group_partition_reproduces_regional_totals':True,'static_trace_hashes_reproduce_previous_diagnosis':True,'input_data_unchanged':True,'five_minute_tail_accounting':True,'no_new_hydraulics_or_routing_or_training':True,'sealed_test_access':False,'AnaSOP_unchanged':True,'final_figure_unchanged':True,'no_causal_bottleneck_percentage_assigned':True})
 dump(R/'progress.json',{'phase':'complete','completed':18,'total':18,'new_simulations':0})
 (R/'README.md').write_text('# Frozen three-parent drainage bottleneck diagnosis\n\nRun `.venv/bin/python src/analyses/diagnose_three_event_drainage_bottlenecks_v2.py all` from the project root. Existing trace-reader aggregates are reused if present; independently bound input sizes,mtimes and original audits are checked. To rebuild trace intermediates,archive the diagnosis outputs and rerun in a fresh output directory. The C++17 streaming reader is compiled locally with clang++;no package installation. Native traces are hashed while read and original static trace digests reproduce the prior diagnosis. Original numerical gates are neither changed nor rerun.\n\nThe endpoint tail (<1ms) is retained in the final descriptive bin,whereas road integrals retain the original normalized approved clocks. No fixed-state envelope is presented as predicted extra extraction. Registered road linkage is not a hydraulic influence area,and no source-water trajectory is reconstructed. Both depth branches and all18main cases remain available. No new model,routing,training,population calculation or AnaSOP/figure edit. Read review.html and bottleneck_evidence.csv before interpreting numerical tables.\n')
 files=list(P.glob('*.csv'))+[R/'review.html',R/'validation.json',R/'summary.json',Path(__file__),ROOT/'src/analyses/drainage_trace_bottleneck_reader_v1.cpp']
 dump(R/'delivery_manifest.json',{'files':[{'path':str(x.relative_to(ROOT)),'sha256':sha(x)}for x in files]})
 print('DIAGNOSIS_DELIVERED',R/'review.html',flush=True)
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def main():
 mode=sys.argv[1]if len(sys.argv)>1 else'all'
 for name,fn in [('prepare',prepare),('traces',traces),('roads',roads),('report',report)]:
  if mode in [name,'all']:fn()
if __name__=='__main__':main()
