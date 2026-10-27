#!/usr/bin/env python3
"""Six September S1R fixed-demand traffic states;reuse qualified hydraulics.
Same original two-path restricted UE,request proxy,capacity and numerical gates.
"""
from pathlib import Path
import os,sys,json,time as real,datetime as dt,inspect,hashlib,argparse,types,traceback,gc,html
for k in ['OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/udt01b-s1r-traffic-numba')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-s1r-traffic-mpl')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import numpy as np,pandas as pd
from scipy.sparse import load_npz
import run_combined_transport_measures_v1 as parent
from shared_receiver_pilot_common_v1 import sha,dump
TAG='three_ward_s1r_september_traffic_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
SOURCE=ROOT/'data/processed/three_ward_s1r_cross_event_validation_v1';SR=ROOT/'data/exp/estimation'/SOURCE.name
D=ROOT/'data/processed';SCRIPT=Path(__file__).resolve();DECISION='MILI-D-20261023-002';EVENT='E20250911'
ARMS=['none','Q2p5','Q5'];CASES=[f'{EVENT}__{stage}__{arm}'for stage in ['H','L']for arm in ARMS]
START=real.monotonic();PAUSED=0.;CACHE={}


def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False)as z:return dict(z)
def clock():return real.monotonic()-PAUSED

def tick():
 global PAUSED
 if(P/'STOP').exists():raise RuntimeError('User STOP')
 if(P/'PAUSE').exists():
  begun=real.monotonic()
  while(P/'PAUSE').exists():
   if(P/'STOP').exists():raise RuntimeError('User STOP while paused')
   real.sleep(2)
  PAUSED+=real.monotonic()-begun
 assert clock()-START<14400,'Four-hour total traffic wall budget excluding explicit pause'


def check():
 tick();assert all(sha(ROOT/n)==h for n,h in read(P/'input_bindings.json')['sha256'].items()),'Frozen input/code changed'

def state(case,_arm):
 if case not in CACHE:
  x=read(P/'state_manifest.json')['states'];meta=next(v for v in x if v['case_id']==case)
  z=load(P/'states'/(case+'.npz'));pr=load(D/'three_ward_motor_flood_access_v1/road_projection.npz');base=load(D/'three_ward_normal_vehicle_choice_v1/choice_costs.npz')['travel_seconds'];cap=load(D/'three_ward_demand_protection_feedback_v1/capacity_assumptions.npz')['hourly_capacity']
  assert z['codes'].shape==(415955,);closed=z['codes']==2;slow=z['codes']==1;t=base.copy();capacity=cap.copy();m=slow[pr['arc_to_road']];t[m]*=2;capacity[m]*=.5
  assert not np.any(closed&slow)and np.isfinite(t).all()and(t>0).all()and np.isfinite(capacity).all()and(capacity>0).all()
  CACHE[case]=(closed,slow,t,capacity)
 return CACHE[case]


def paths_for(case,_arm,kind):return P/'paths'/(case+'__flood_'+kind+'.npz')
def require_optimizer_fixture():assert read(R/'optimizer_selfcheck.json')['passed']


def configure():
 # Bind qualified mathematics;only loops,output paths and operational time ceilings change.
 pathsrc=inspect.getsource(parent.generate_paths)
 swaps={"for arm in ['flood','targeted']:":"for arm in ['flood']:","if(P/'PAUSE').exists()or(P/'STOP').exists():raise RuntimeError('Pause before next origin')":"tick()",'time.monotonic()-begun>180 or time.monotonic()-start>900':'time.monotonic()-begun>600 or time.monotonic()-start>1500',"R/'path_campaign.json'":"R/(case+'_path_campaign.json')",'sweeps=4':'sweeps=2'}
 for a,b in swaps.items():assert pathsrc.count(a)==1,a;pathsrc=pathsrc.replace(a,b)
 optsrc=inspect.getsource(parent.optimize)
 swaps={'time.monotonic()-started>120':'time.monotonic()-started>600',"if(P/'PAUSE').exists()or(P/'STOP').exists():raise RuntimeError('Pause in optimization')":"tick()"}
 for a,b in swaps.items():assert optsrc.count(a)==1,a;optsrc=optsrc.replace(a,b)
 fitsrc=inspect.getsource(parent.fit)
 swaps={"arms=['normal']if case=='normal'else['flood','targeted']":"arms=['flood']","for mode in ['UE','SO']:":"for mode in ['UE']:",'optimizer_test()':'require_optimizer_fixture()','time.monotonic()-started>2700':'time.monotonic()-started>1500'}
 for a,b in swaps.items():assert fitsrc.count(a)==1,a;fitsrc=fitsrc.replace(a,b)
 env=parent.__dict__.copy();env.update(P=P,R=R,state=state,paths_for=paths_for,check=check,tick=tick,time=types.SimpleNamespace(monotonic=clock),REDUCTIONS=[0.],inherit=lambda *args:None,require_optimizer_fixture=require_optimizer_fixture)
 raw=parent.dijkstra
 def guarded_dijkstra(*args,**kw):tick();return raw(*args,**kw)
 env['dijkstra']=guarded_dijkstra
 source=pathsrc+'\n'+optsrc+'\n'+fitsrc;exec(compile(source,str(SCRIPT)+'::qualified_traffic_bindings','exec'),env)
 # Analytic fixture uses this very optimizer,not a separate implementation.
 exec(compile(inspect.getsource(parent.optimizer_test),str(SCRIPT)+'::optimizer_fixture','exec'),env)
 return env,source


def prepare():
 global START
 assert not(P/'protocol.json').exists(),'Preserve prepared experiment'
 for f in [P,R,P/'paths',P/'fits',P/'states']:f.mkdir(parents=True,exist_ok=True)
 assert read(SR/'final_summary_validation.json')['passed']and read(SR/'campaign_state.json')['status']=='qualified_complete'
 assert read(ROOT/'data/exp/estimation/three_ward_barrier_drainage_v1/traffic_validation.json')['passed']
 refs=pd.read_csv(D/'three_ward_five_event_temporal_analysis_v1/reference_states.csv');anchor=refs[refs.event_id.eq(EVENT)&refs.selection_reasons.str.contains('H_noaction_closed_length_peak')];assert len(anchor)==1
 frame=int(anchor.iloc[0].frame);assert frame==29
 sourcepaths={q:SOURCE/'batches'/(EVENT+'_'+q)for q in ['Q2p5','Q5']};supports=[load(v/'common_road_support.npz')for v in sourcepaths.values()]
 for key in ['qualified_roads','common_admitted']:assert np.array_equal(supports[0][key],supports[1][key])
 qualified=supports[0]['qualified_roads'];assert qualified.sum()==28064;indices=np.flatnonzero(qualified)
 clip=load(D/'three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz');assert len(qualified)==len(clip['union_length_m']);assert abs(clip['union_length_m'][qualified].sum()/1000-836.958326)<1e-5
 dem=load(D/'three_ward_normal_vehicle_repair_v1/multi_routing_demands.npz');q0=parent.original_request(dem);core=load(D/'three_ward_demand_protection_feedback_v1/protection_masks.npz')['core_endpoint_od']&dem['base'];assert np.isfinite(q0).all()and(q0>=0).all()and q0[core].sum()>0
 dependency=[SCRIPT,ROOT/'src/analyses/run_combined_transport_measures_v1.py',ROOT/'src/analyses/demand_protection_feedback_v1.py',ROOT/'src/analyses/observed_load_common_v1.py',D/'three_ward_normal_vehicle_repair_v1/turn_router.npz',D/'three_ward_normal_vehicle_repair_v1/multi_endpoint_components.npz',D/'three_ward_normal_vehicle_repair_v1/multi_routing_demands.npz',D/'three_ward_motor_flood_access_v1/road_projection.npz',D/'three_ward_normal_vehicle_choice_v1/choice_costs.npz',D/'three_ward_demand_protection_feedback_v1/capacity_assumptions.npz',D/'three_ward_demand_protection_feedback_v1/protection_masks.npz',D/'three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz',D/'three_ward_five_event_temporal_analysis_v1/reference_states.csv',ROOT/'data/exp/feasibility-check/three_ward_vehicle_demand_coverage_audit_v1/summary.json',SR/'final_summary_validation.json']
 staged=[];sensitivity=[]
 for stage in ['H','L']:
  ex={}
  for q,base in sourcepaths.items():
   manifest=read(base/'run_manifest.json');r=next(v for v in manifest['runs']if v['stage']==stage);report=SR/'batches'/(EVENT+'_'+q)
   a=read(report/'execution'/r['case_id']/'audit.json');b=read(report/'execution'/r['case_id']/'pump_budget_summary.json');assert a['all_checks_passed']and len(a['checks'])==28 and b['passed']and len(b['shared_receiver_checks'])==8
   dependency.extend([base/'run_manifest.json',base/'common_road_support.npz',report/'execution'/r['case_id']/'audit.json',report/'execution'/r['case_id']/'pump_budget_summary.json'])
   for arm in ['baseline','drainage']:
    f=base/'exposure'/(r['case_id']+'__'+arm+'.npz');dependency.append(f);ex[q+'_'+arm]=load(f)
  a,b=ex['Q2p5_baseline'],ex['Q5_baseline'];assert np.array_equal(a['road_state_codes'],b['road_state_codes'])and np.array_equal(a['actual_times'],b['actual_times'])
  chosen={'none':a,'Q2p5':ex['Q2p5_drainage'],'Q5':ex['Q5_drainage']}
  for arm,z in chosen.items():
   case=f'{EVENT}__{stage}__{arm}';assert np.array_equal(z['road_indices'],indices)and np.array_equal(z['actual_times'],a['actual_times'])and z['road_state_codes'].shape==(109,2,len(indices))
   assert abs(z['actual_times'][frame]-frame*300)<.001 and np.isin(z['road_state_codes'],[0,1,2]).all()
   code=np.zeros(len(qualified),np.uint8);code[indices]=z['road_state_codes'][frame,0];upper=code.copy();upper[indices]=z['road_state_codes'][frame,1]
   np.savez_compressed(P/'states'/(case+'.npz'),codes=code,qualified=qualified,upper_codes=upper,actual_time_s=z['actual_times'][frame],analysis_time_s=z['analysis_times'][frame])
   staged.append({'case_id':case,'event':EVENT,'stage':stage,'arm':arm,'frame':frame,'elapsed_s':float(z['analysis_times'][frame]),'actual_time_s':float(z['actual_times'][frame]),'event_snapshot_jst':'2025-09-11T15:40:00+09:00','demand_clock':'frozen17h survey-derived proxy,2xreference;controlled pressure case,not actual15:40demand'})
   for scope,w in [('Three wards',clip['union_length_m'])]+[(name,clip['ward_length_m'][i])for i,name in enumerate(['Chiyoda','Chuo','Minato'])]:sensitivity.append({'case':case,'scope':scope,'qualified_km':float(w[qualified].sum()/1000),'unknown_km':float(w[~qualified].sum()/1000),'lower_closed_km':float(w[code==2].sum()/1000),'upper_closed_km':float(w[upper==2].sum()/1000),'depth_branch_changed_road_km':float(w[upper!=code].sum()/1000),'upper_branch_traffic_fit_run':False})
 protocol={'decision':DECISION,'question':'Do the fixed S1R Q2.5/Q5 drainage portfolios improve vehicle request completion and common-OD costs under identical proxy demand?','parent':EVENT,'states':staged,'state_count':6,'hydraulics_reused':True,'new_hydraulics':0,'selection':'Frame29 chosen from frozen H/no-action road-closure peak,applied unchanged to both stages and all three portfolios. No treatment-selected peak.','demand':'Exactly original_request=2*period_weights[:,3]*base,as prior high-demand physical-traffic pilot. 17hgeneric small-vehicle driver proxy,not actual event-hour cars or people. No deferral,coarse/intrazonal recovery or remote-remote insertion.','missing_demand':'Prior held/coarse/intrazonal demand,unrepresented external through traffic and vehicle-class gaps remain. Multiplier2does not recover those gaps.','cost':'Same Ureference travel_seconds and hourly_capacity. Slow0.10..0.30m:reference cost2x,capacity0.5x. Closed>=0.30m:excluded. c=tref[1+0.15(flow/capacity)^4];tref is not measured free-flow time. No actual queue claim.','unknown':'Original conditional turn-memory network;unqualified bridge/tunnel/outside roads retain assumed normal availability,not verified safety. Strict qualified mask preserved.','solver':'Two-path restricted UE only. A=reference shortest feasible path;B=shortest at1xbase AON feedback costs capped50xreference for candidate generation only. Final objective uncapped. Same2000iterations,ftol1e-13,gtol1e-7,gap<=0.001,closedflow<1e-9,balance<1e-6,costloadrelative<1e-9. No full-network equilibrium claim.','comparisons':'Original retained request denominator conserved. Pair restored and newly unreachable requests at the same OD. Costs on original-positive core ODs reachable under ALL3portfolios within each stage;same weights and cohort,no marginal survivor comparisons. Baseline index100. Excluded baseline-reachable requests disclosed.','road_load':'v/C>1is modeled capacity pressure,not observed congestion. Aggregate max directional v/C per physical source road,weight by exact clipped road lengths once. Keep known and unknown layers separate.','depth':'Lower admitted depth branch primary. Upper road-state differences reported without adding traffic fits;not an uncertainty interval for traffic outcomes. Population candidates remain untouched.','budget':{'traffic_states':6,'new_path_sweeps':12,'max_seconds_per_sweep':600,'total_path_seconds':7200,'new_UE_fits':6,'max_seconds_per_fit':600,'total_fit_seconds':3600,'total_campaign_seconds':14400,'parallel_fits':1,'threads':1,'automatic_retries':0},'stop':'First failed source,flow,request,cost,gap or time gate holds queue. PAUSE/STOP honored at path origins and objective calls. No automatic retuning.','new_population_counts':False,'training':False,'sealed_test':False,'observed_validation':False}
 dump(P/'protocol.json',protocol);dump(P/'state_manifest.json',{'states':staged,'order':CASES})
 rows=[]
 for scope,sel in [('Whole10km context',np.ones(len(q0),bool)),('Core-endpoint requests',core)]:rows.append({'scope':scope,'positive_OD_records':int((sel&(q0>0)).sum()),'original_request_proxy':float(q0[sel].sum()),'deferred':0.,'meaning':'modeled fixed-hour driver request proxy,not unique vehicles or persons'})
 pd.DataFrame(rows).to_csv(P/'input_demand_ledger.csv',index=False);pd.DataFrame(sensitivity).to_csv(P/'depth_support_audit.csv',index=False)
 env,source=configure();(P/'bound_traffic_functions.py.txt').write_text(source);START=clock();env['optimizer_test']();assert read(R/'optimizer_selfcheck.json')['passed']
 # Source/bound-output binding prevents accidental mixing with old physical-pilot treatments.
 dependency+=list((P/'states').glob('*.npz'))+[P/'protocol.json',P/'state_manifest.json',P/'bound_traffic_functions.py.txt',P/'input_demand_ledger.csv',P/'depth_support_audit.csv']
 dump(P/'input_bindings.json',{'sha256':{str(f.relative_to(ROOT)):sha(f)for f in dict.fromkeys(dependency)}})
 dump(R/'preflight_validation.json',{'passed':True,'source_hydraulics_qualified':True,'same_mask28064roads':True,'baseline_codes_identical_across_quota_sources':True,'same_anchor29_all6states':True,'same_demand_capacity_reference_cost_and_turn_graph':True,'new_optimizer_matches_analytic_fixture':True,'existing_numerical_gates_unchanged':True,'source_snapshot_staged_without_native_map_changes':True,'primary_depth':'lower','new_hydraulics':0})
 dump(R/'campaign_state.json',{'status':'prepared','completed_cases':[],'current_case':None,'total_cases':6})
 (R/'README.md').write_text('# September fixed-demand S1R traffic comparison\n\nDecision '+DECISION+'. Six states only: H/L x no action,Q2.5,Q5. Reuse qualified hydraulics at frozen no-action Hpeak frame29,15:40JST,all portfolios/stages. Demand is the prior17h driver proxy x2,not actual15:40traffic. Original two-path restricted UE,capacity/reference-cost assumptions and numerical tolerances remain fixed. No deferral or automatic transfer to other rainfall parents. No new hydrodynamics,population estimation,ML,sealed tests or actual queue claims.\n\nRun with the already-installed .frame-control/runtimes/physical-measures-py312/bin/python src/analyses/s1r_september_traffic_v1.py prepare,then run once. No new package installation. Numba cache uses/private/tmp. Serial single-thread;12path sweeps at most600seconds each,6fits at most600seconds each,4hour total excluding explicit PAUSE. Source hashes bound before and after each phase. First numerical/budget failure holds;no automatic retries or tolerance relaxation. Common-cost cohort is paired across all3portfolios separately within H/L. Restored and newly unreachable requests separate;total network cost is not a welfare comparison when completed demand changes. Physical-road high-load lengths use clipped geometry and count each road once;unknown remains separate. All outputs are scenario results without empirical event-hour traffic validation.\n')
 render();print('S1R_TRAFFIC_PREFLIGHT_PASSED_6_STATES',flush=True)


def summarize():
 dem=load(D/'three_ward_normal_vehicle_repair_v1/multi_routing_demands.npz');q0=parent.original_request(dem);core=load(D/'three_ward_demand_protection_feedback_v1/protection_masks.npz')['core_endpoint_od']&dem['base'];pr=load(D/'three_ward_motor_flood_access_v1/road_projection.npz');clip=load(D/'three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz');scopew=[('Three wards',clip['union_length_m'])]+[(n,clip['ward_length_m'][i])for i,n in enumerate(['Chiyoda','Chuo','Minato'])];rows=[];comparisons=[];loadrows=[];cohorts=[]
 allfits={case:load(P/'fits'/(parent.key(case,'flood',0,'UE')+'.npz'))for case in CASES}
 for stage in ['H','L']:
  cases={arm:f'{EVENT}__{stage}__{arm}'for arm in ARMS};fits={arm:allfits[case]for arm,case in cases.items()};cohort=core&(q0>0)
  for z in fits.values():assert np.array_equal(z['original_requested'],q0)and np.array_equal(z['retained_requested'],q0);cohort&=z['reachable']
  assert q0[cohort].sum()>0;basecost=float(q0[cohort]@fits['none']['mean_od_cost'][cohort]/q0[cohort].sum())
  for ix in np.flatnonzero(cohort):cohorts.append({'stage':stage,'expanded_od_index':int(ix),'original_request_weight':float(q0[ix])})
  for arm,case in cases.items():
   z=fits[arm];closed,slow,tref,capacity=state(case,'flood');weights=q0[cohort];cost=z['mean_od_cost'][cohort];served=float(q0[core&z['reachable']].sum());failed=float(q0[core&~z['reachable']].sum());original=float(q0[core].sum());assert abs(original-served-failed)<1e-6
   vc=z['arc_flow']/capacity;maxvc=np.zeros(415955);np.maximum.at(maxvc,pr['arc_to_road'],vc);known=load(P/'states'/(case+'.npz'))['qualified'];coreuse=clip['union_length_m']>0;over=(maxvc>1)&~closed
   mean=float(weights@cost/weights.sum());base=fits['none'];change=cost-base['mean_od_cost'][cohort]
   rows.append({'case':case,'stage':stage,'portfolio':arm,'frame':29,'original_core_request_proxy':original,'served_core_request_proxy':served,'unreachable_core_request_proxy':failed,'completion_fraction_original':served/original,'common_cohort_original_request_proxy':float(weights.sum()),'baseline_reachable_excluded_from_cost_proxy':float(q0[core&base['reachable']&~cohort].sum()),'common_cohort_mean_cost_min':mean/60,'common_cohort_cost_index':100*mean/basecost,'common_cohort_over1min_worse_share':float(weights[change>60].sum()/weights.sum()),'total_network_model_vehicle_seconds':float(z['total_model_cost']),'total_network_cost_not_fixed_served_cohort':True})
   for scope,w in scopew:loadrows.append({'case':case,'stage':stage,'portfolio':arm,'scope':scope,'qualified_road_km':float(w[known].sum()/1000),'unknown_road_km':float(w[~known].sum()/1000),'closed_road_km':float(w[closed].sum()/1000),'slow_road_km':float(w[slow].sum()/1000),'high_load_known_road_km':float(w[known&over].sum()/1000),'high_load_unknown_road_km':float(w[~known&over].sum()/1000)})
   idx=np.flatnonzero(coreuse);pd.DataFrame({'road_index':idx,'clipped_three_ward_length_m':clip['union_length_m'][idx],'qualified':known[idx],'closed':closed[idx],'slow':slow[idx],'maximum_directional_vc':maxvc[idx],'high_load':over[idx]}).to_csv(P/(case+'_road_load.csv'),index=False)
  for reference,treatment in [('none','Q2p5'),('none','Q5'),('Q2p5','Q5')]:
   a,b=fits[reference],fits[treatment];restored=core&~a['reachable']&b['reachable'];newloss=core&a['reachable']&~b['reachable'];rest=float(q0[restored].sum());lost=float(q0[newloss].sum());net=float(q0[core&~a['reachable']].sum()-q0[core&~b['reachable']].sum());assert abs(net-rest+lost)<1e-6
   ca=a['mean_od_cost'][cohort];cb=b['mean_od_cost'][cohort];mean_a=float(q0[cohort]@ca/q0[cohort].sum());mean_b=float(q0[cohort]@cb/q0[cohort].sum());comparisons.append({'stage':stage,'reference':reference,'treatment':treatment,'restored_request_proxy':rest,'newly_unreachable_request_proxy':lost,'net_avoided_unreachable_request_proxy':net,'common_cohort_original_request_proxy':float(q0[cohort].sum()),'common_cohort_cost_change_min':(mean_b-mean_a)/60,'common_cohort_cost_change_percent':100*(mean_b/mean_a-1),'same_cohort_over1min_worse_request_share':float(q0[cohort][cb-ca>60].sum()/q0[cohort].sum()),'meaning':'fixed-proxy scenario costs,not actual queue delays or people'})
 pd.DataFrame(rows).to_csv(P/'traffic_outcomes.csv',index=False);pd.DataFrame(comparisons).to_csv(P/'paired_traffic_effects.csv',index=False);pd.DataFrame(loadrows).to_csv(P/'road_load_summary.csv',index=False);pd.DataFrame(cohorts).to_csv(P/'fixed_common_cohort.csv',index=False)
 fits=[read(R/(parent.key(case,'flood',0,'UE')+'_fit_validation.json'))for case in CASES];paths=[read(f)for f in R.glob('*_path_validation.json')];assert len(fits)==6 and len(paths)==12 and all(v['passed']for v in fits+paths)
 assert sum(v['seconds']for v in fits)<=3600 and sum(v['seconds']for v in paths)<=7200
 check();dump(R/'validation.json',{'passed':True,'traffic_states':6,'path_sweeps':12,'max_gap':max(v['gap']for v in fits),'max_component_flow_residual':max(v['balance_residual']for v in fits),'all_original_requests_accounted':True,'common_cohort_within_stage_all3arms':True,'paired_restoration_and_new_losses':True,'road_lengths_clipped_and_not_direction_doubled':True,'source_hashes_unchanged':True,'new_hydraulics':0,'actual_congestion_validated':False,'fit_seconds':sum(v['seconds']for v in fits),'path_seconds':sum(v['seconds']for v in paths)})
 dump(R/'summary.json',{'outcomes':rows,'comparisons':comparisons,'road_load':loadrows,'limitations':['Fixed17h small-vehicle proxy x2,not event-hour observations.','Same two-path restricted UE and conditional unknown-road assumptions.','Mean cost on within-stage common survivors;original request accounting separate.','Unrepresented demand and class/capacity gaps remain.','No dynamic queues,observed congestion,persons or daytime population.']})


def render():
 if not(R/'campaign_state.json').exists():return
 s=read(R/'campaign_state.json');text='<h1>9月暴雨：同需求下的S1R排水交通对照</h1><aside>高／低水位 × 无措施、Q2.5、Q5，共6组。复用已核验的积水；固定需求、道路容量、成本与比较时刻，不追加水动力。</aside><p><b>状态：'+html.escape(s['status'])+'</b>；完成 '+str(len(s['completed_cases']))+'/6组。</p>'
 if s.get('current_case'):text+='<p>当前：'+html.escape(s['current_case'])+'；'+html.escape(s.get('phase',''))+'</p>'
 if s.get('error'):text+='<p class="warn">已保留结果并停止：'+html.escape(s['error'])+'</p>'
 text+='<h2>本轮回答什么</h2><p>封路减少，是否让更多原有车辆请求完成？相同可达OD的模型成本是否下降？交通负荷是否移到其他道路？它们可能出现收益与代价并存，不能仅由道路净改善推断拥堵缓解。</p><h2>固定条件与口径</h2><p>两种水位均使用预先冻结的无措施高水位封路峰值第29帧，事件时刻2025-09-11约15:40JST。交通负荷沿用此前17时小型车驾驶出行代理×2，作为控制需求的压力情景，不是该日15:40实测车流。暂留OD、区内短途、外部穿越和车型缺口不靠倍数补齐。</p><p>仅在同一水位三种方案均可达、原需求为正的核心OD上比较成本，固定原权重，并单列原基线可达但被共同群体排除的请求。恢复和新增不可达请求在同一OD上配对；车辆请求不是人数。既有夜间人口区间保持原样。</p><p>原转向记忆网络，沿用未知道路按正常可用的条件假设。0.10–0.30m路段参考成本×2、容量×0.5；≥0.30m关闭。两条可行候选路径内的静态UE，间隙≤0.001，原成本、流量与账本检查不变。参考时耗不是实测自由流时耗，v/C>1表示模型容量压力，不是真实排队。</p>'
 if(R/'summary.json').exists():
  z=read(R/'summary.json');text+='<h2>车辆请求与同群体成本</h2>'+pd.DataFrame(z['outcomes']).to_html(index=False,float_format=lambda x:f'{x:,.4f}')+'<h2>同OD直接配对：改善与恶化</h2>'+pd.DataFrame(z['comparisons']).to_html(index=False,float_format=lambda x:f'{x:,.4f}')+'<h2>道路负荷：分开已知与未知</h2>'+pd.DataFrame(z['road_load']).to_html(index=False,float_format=lambda x:f'{x:,.4f}')
 text+='<h2>预算与停止条件</h2><p>串行、单线程；最多12次路径扫描（每次600s）与6次优化（每次600s），路径累计≤7200s、优化累计≤3600s、全流程≤4小时（明确PAUSE时间不计入）。数值或源文件检查失败停止，不自动重试、不放宽容差；不自动扩至7月、8月。</p><p>高负荷长度按三区裁剪后的物理道路计一次，使用该路各方向最大v/C，不将双向长度相加。主要交通拟合用原准入水深下界，上界道路差异单列，不将其误称交通结果的不确定性区间。没有新模拟、训练或封存测试，也不改AnaSOP。</p><p><a href="../../../processed/'+TAG+'/protocol.json">Protocol</a> · <a href="preflight_validation.json">Preflight</a> · <a href="campaign_state.json">Live state</a> · <a href="README.md">Reproduce</a></p>'
 if(R/'validation.json').exists():text+='<p><a href="validation.json">Final validation</a> · <a href="summary.json">Results JSON</a></p>'
 style='body{font:16px/1.8 system-ui;max-width:1280px;margin:28px auto;padding:0 22px;background:#f7fafb;color:#243e4b}aside{padding:18px;background:#e6f0f5}table{display:block;max-width:100%;overflow:auto;background:white;border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #cddce5;text-align:left}.warn{padding:18px;background:#fff0d4}'
 refresh='<meta http-equiv="refresh" content="30">'if s['status']not in ['qualified_complete','held']else''
 tmp=R/'review.html.tmp';tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+refresh+'<title>S1R September Fixed-Demand Traffic</title><style>'+style+'</style>'+text+'</html>');tmp.replace(R/'review.html')


def run():
 global START,PAUSED
 assert read(R/'preflight_validation.json')['passed'];s=read(R/'campaign_state.json');assert s['status']=='prepared'and not s['completed_cases'],'No automatic restart'
 START=real.monotonic();PAUSED=0.;env,_=configure();s.update(status='running',pid=os.getpid(),phase='starting',started_utc=dt.datetime.now(dt.timezone.utc).isoformat());dump(R/'campaign_state.json',s);render()
 try:
  for i,case in enumerate(CASES,1):
   check();s.update(current_case=case,phase='two feasible path pools');dump(R/'campaign_state.json',s);render();print('TRAFFIC_CASE_START',case,flush=True)
   env['CASES']=['unused',case];env['generate_paths']()
   s.update(phase='restricted UE fit');dump(R/'campaign_state.json',s);render();env['fit'](case)
   v=read(R/(parent.key(case,'flood',0,'UE')+'_fit_validation.json'));assert v['passed'];s['completed_cases'].append(case);s.update(phase='qualified',elapsed_s=clock()-START);dump(R/'campaign_state.json',s);render();CACHE.clear();gc.collect();print('TRAFFIC_CASE_QUALIFIED',case,i,'/6',flush=True)
  s.update(status='summarizing',current_case=None,phase='fixed-cohort comparisons');dump(R/'campaign_state.json',s);render();summarize();s.update(status='qualified_complete',phase='complete',elapsed_s=clock()-START,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat());dump(R/'campaign_state.json',s);render();print('S1R_TRAFFIC_ALL6_COMPLETE',flush=True)
 except BaseException as ex:
  s.update(status='held',error=str(ex)or type(ex).__name__,elapsed_s=clock()-START);dump(R/'campaign_state.json',s);(R/'error.txt').write_text(traceback.format_exc());render();raise


def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','run','report']);a=ap.parse_args();{'prepare':prepare,'run':run,'report':render}[a.phase]()
if __name__=='__main__':main()
