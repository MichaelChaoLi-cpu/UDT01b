"""Qualified shared-receiver outputs -> frozen vehicle access and resident intervals.

Separate postprocessing authorization MILI-D-20261020-020. Does not modify the
hydraulic protocol, quality rules, source data, solver or active runner.
"""
from pathlib import Path
import os,sys,json,time,datetime as dt,hashlib,argparse,html
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import numpy as np,pandas as pd
from temporal_transport_common_v1 import Reachability,population_inputs,counts
from run_temporal_road_population_v1 import envelope,signed_envelope
from integrate_night_population_transport_v1 import distances,selfcheck as routing_selfcheck
D=ROOT/'data/processed';P=D/'three_ward_shared_receiver_access_v1';R=ROOT/'data/exp/estimation'/P.name
HP=D/'three_ward_shared_receiver_pilot_v1';HR=ROOT/'data/exp/estimation'/HP.name
TEMP=D/'three_ward_five_event_temporal_analysis_v1';POP=D/'three_ward_population_block_stability_v1'

def load(p):
 with np.load(p,allow_pickle=False) as z:return dict(z)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def save(p,x):pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def gate():
 if (P/'STOP').exists() or (HP/'STOP').exists():raise RuntimeError('User STOP')
 while (P/'PAUSE').exists() or (HP/'PAUSE').exists():
  if (P/'STOP').exists() or (HP/'STOP').exists():raise RuntimeError('User STOP while paused')
  time.sleep(5)
def bindings():
 names=['src/analyses/shared_receiver_access_effects_v1.py','src/analyses/temporal_transport_common_v1.py','src/analyses/run_temporal_road_population_v1.py','src/analyses/integrate_night_population_transport_v1.py','src/analyses/postprocess_five_event_population_v1.py',
 'data/processed/three_ward_normal_vehicle_repair_v1/turn_router.npz','data/processed/three_ward_normal_vehicle_repair_v1/multi_network_endpoints.csv','data/processed/three_ward_motor_flood_access_v1/road_projection.npz',
 'data/processed/three_ward_night_population_transport_v1/normal_node_access.npz','data/processed/three_ward_night_population_transport_v1/block_access_support.npz',
 'data/processed/three_ward_normal_vehicle_choice_v1/choice_costs.npz','data/processed/three_ward_population_block_stability_v1/official_2020_block_count_baseline.csv',
 'data/processed/three_ward_population_block_stability_v1/population_input_admission_list.csv','data/processed/three_ward_population_block_stability_v1/block_population_intervals.npz',
 'data/processed/three_ward_population_block_stability_v1/admission_policy.json','data/processed/three_ward_five_event_temporal_analysis_v1/reference_states.csv','data/processed/three_ward_shared_receiver_pilot_v1/run_manifest.json','data/processed/three_ward_shared_receiver_pilot_v1/protocol.json']
 return {n:sha(ROOT/n) for n in names}
def same_sources():
 return all(sha(ROOT/n)==h for n,h in json.loads((P/'input_bindings.json').read_text()).items())

def prepare():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 for n in ['node_cache','population','anchor_nodes']:(P/n).mkdir(exist_ok=True)
 if (P/'protocol.json').exists():assert same_sources(),'Bound access inputs changed';return
 blocks,pop,normal,infos=population_inputs();engine=Reachability();old_bind=json.loads((TEMP/'source_bindings.json').read_text())['sha256'];graph_names=['data/processed/three_ward_normal_vehicle_repair_v1/turn_router.npz','data/processed/three_ward_motor_flood_access_v1/road_projection.npz'];assert all(n in old_bind and sha(ROOT/n)==old_bind[n] for n in graph_names),'Old cache graph binding mismatch';direct=engine.direct(np.zeros(len(load(TEMP/'clipped_road_lengths.npz')['qualified']),bool))
 assert routing_selfcheck() and np.array_equal(direct,normal),'Original turn-state normal graph mismatch'
 assert len(pop)==4746 and pop.sum()==496345
 for a in [-1,0,1]:
  for b in [-1,0,1]:
   low,high=signed_envelope(np.array([a,b]),dict(ix=np.array([0,1]),ptr=np.array([0,2]),lens=np.array([2]),complete=np.array([True])),np.array([100]));assert low[0]==100*min(a,b) and high[0]==100*max(a,b)
 for info in infos.values():
  a,b=envelope(np.zeros(len(normal),bool),info,pop);assert not a.any() and not b.any()
  a,b=envelope(normal,info,pop);assert np.array_equal(a,info['complete']) and np.array_equal(b,info['complete'])
 anchor=pd.read_csv(TEMP/'reference_states.csv');anchor=anchor[(anchor.event_id=='E20240731')&anchor.selection_reasons.str.contains('H_noaction_closed_length_peak')];assert len(anchor)==1;frame=int(anchor.iloc[0].frame)
 protocol=dict(decision='MILI-D-20261020-020',parent_event='E20240731',stages=['H','L'],arms=['baseline','drainage'],
  prerequisites='Both original hydraulic cases must pass all unchanged original gates, independent pump checks and common-road postprocessing; held/failed cases stop this stage.',
  comparison='Same common-qualified roads, same109actual output clocks, frozen geometry/rain/masks. Unknown roads remain assumed available; this conditional branch is not verified safety.',
  residents='2020 official basic-unit resident stock,496345 residents in4746 blocks. Hypothetical at-residence scenario, not contemporaneous daytime occupancy or observed stranded people.',
  allocation='Original125/250m building-allocation intervals and final admission remain read-only by SHA; population is not assigned to centroids or evenly split among road nodes.',
  candidate_intervals='Existing0/10/25m candidate access radii;10m primary. All candidates lost => whole-block lower bound; any lost => whole-block upper bound; normal support incomplete => unresolved, not counted as loss.',
  temporal='109snapshots at5-minute spacing across9h. Report simultaneous conditional stocks and resident-hours; never add snapshot stocks to form unique people.',
  paired='Compute restored and new-loss candidate masks and signed net differences at each same time before block aggregation; do not subtract marginal interval endpoints. Physical drainage may worsen access.',
  access='Original turn-memory bidirectional reachability to any of16conditional outer road portals; outward/inward may use different portals. Not an assertion of safety or multimodal isolation.',
  detour='Frozen pre-treatment H/no-action closure-peak frame applied to BOTH stages and arms. Existing reference travel costs; slow depth0.1..0.3m =>2x, closed>=0.3m. Same baseline-and-treatment reachable endpoint cohort. No OD assignment, UE refit, dynamic congestion queues or real-time delays.',
  anchor_frame=frame,anchor_time_s=float(anchor.iloc[0].analysis_elapsed_s),depth_branches=['lower','upper'],
  budget={'maximum_new_hydraulics':0,'maximum_reachability_calls':872,'maximum_new_distance_states':8,'maximum_access_active_seconds':1800,'maximum_wait_seconds':7200,'automatic_retries':0},
  benchmark='No geotemporally admissible three-ward road observations in benchmark inventory v1. Numerical validity is not independent observed road accuracy.',
  stop='Respect parent/own PAUSE and STOP. Stop on parent hold, changed source hashes, graph mismatch, clock/support mismatch, interval/conservation error or budget exhaustion.',
  training=False,sealed_test=False,new_hydraulics=False,daytime_population_integrated=False)
 dump(P/'protocol.json',protocol);dump(P/'input_bindings.json',bindings())
 dump(R/'preflight_validation.json',dict(passed=True,normal_turn_graph_replayed=True,normal_population_stock_conserved=True,empty_and_total_loss_envelopes_checked=True,routing_chain_checked=True,anchor_selected_from_frozen_baseline=True,created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
 dump(R/'state.json',{'status':'prepared_waiting_for_parent','prerequisite':'Two hydraulic cases plus road transfer postprocessing must qualify','new_hydraulic_runs':0})
 report()

def report():
 state=json.loads((R/'state.json').read_text());proto=json.loads((P/'protocol.json').read_text());text='<h1>共享接收条件措施：交通与人口后处理</h1><p>仅沿用当前一场暴雨的高、低水位两组。两组水动力及原道路后处理都通过后才开始；没有追加水动力模拟。</p><p><b>状态：'+html.escape(state['status'])+'</b></p>'
 text+='<p>真实道路 benchmark 仍缺少可配对的三区观测。以下是情景结果，数值通过不等于历史浸水或实际受困人数已经验证。</p>'
 if (R/'summary.json').exists():
  s=json.loads((R/'summary.json').read_text());text+='<h2>同一时刻人口通达性影响</h2><p>2020年常住人口496,345人作为住地情景；不代表暴雨发生时白天实际人口。保留街区内接入位置区间。恢复与新增损失分列，不以净改善掩盖转移。</p>'+pd.DataFrame(s['main_population']).to_html(index=False,float_format=lambda x:f'{x:,.3f}')
  text+='<h2>固定时刻的车辆绕行与可达性</h2><p>沿用无措施高水位封路峰值时刻（第97帧，约8小时5分），两种水位均用同一时刻。共同可达端点群体的参考成本比，不是真实拥堵延误；不同深度分支分开，不把端点当作人口。</p>'+pd.DataFrame(s['anchor_access']).to_html(index=False,float_format=lambda x:f'{x:,.4f}')
  text+='<h2>封路与积水转移</h2><p>已准入的配对水动力结果与原道路汇总：</p>'+pd.DataFrame(s['road_effects']).to_html(index=False,float_format=lambda x:f'{x:,.6f}')
  text+='<p>目标控制单元外，水位确定升高至少0.10m的最大同时面积；零值不排除更小幅上升。规定接收水位没有模拟真实河道新增蓄水的下游影响。</p>'+pd.DataFrame(s['off_target']).to_html(index=False,float_format=lambda x:f'{x:,.6f}')
  text+='<p><b>扩展建议：</b>'+html.escape(s['next_step_judgment'])+'</p>'
 text+='<h2>解释范围</h2><p>0／10／25m候选接入范围，以10m为主；lower／upper保留原道路水深删失区间。桥隧与覆盖未知道路仍保持原“按可用假设”的条件；正常可达性支持不足的人口单列。未改楼宇分配、质量掩膜或任何数值准入阈值。</p><p>楼宇区间、白天实际人口和道路观测缺口不因本轮措施计算而消失。此处不训练、不计算实际受困人数，不做新的需求分配或拥堵优化。</p>'
 text+='<p><a href="../three_ward_shared_receiver_pilot_v1/review.html">Parent hydraulic / road review</a> · <a href="../../data-briefing/three_ward_road_flood_observations_v1/review.html">Observational benchmark audit</a> · <a href="../../../processed/three_ward_shared_receiver_access_v1/protocol.json">Protocol</a> · <a href="state.json">State</a></p>'
 if (R/'validation.json').exists():text+='<p><a href="validation.json">Postprocessing validation</a> · <a href="../../../processed/three_ward_shared_receiver_access_v1/paired_population_trajectories.csv">Paired population trajectories</a> · <a href="../../../processed/three_ward_shared_receiver_access_v1/anchor_endpoint_access.csv">Anchor endpoint access</a></p>'
 style='body{font:16px/1.7 system-ui;color:#23354a;background:#f3f6fa;max-width:1250px;margin:30px auto;padding:0 20px}h1{font-size:28px}table{display:block;max-width:100%;overflow:auto;border-collapse:collapse;background:white;font-size:12px}td,th{padding:8px;border:1px solid #d5dde7;text-align:left}a{color:#176c9b}p{max-width:1100px}'
 (R/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shared Receiver Access Effects</title><style>'+style+'</style>'+text+'</html>')

def run():
 assert json.loads((R/'preflight_validation.json').read_text())['passed'];assert same_sources()
 parent_state=json.loads((HR/'campaign_state.json').read_text());assert parent_state['status']=='qualified_complete','Parent not qualified; no access results admitted'
 assert json.loads((HR/'delivery_validation.json').read_text())['passed'] and json.loads((HR/'postprocessing_validation.json').read_text())['passed']
 manifest=json.loads((HP/'run_manifest.json').read_text())['runs'];assert len(manifest)==2
 for r in manifest:
  a=json.loads((HR/'execution'/r['case_id']/'audit.json').read_text());b=json.loads((HR/'execution'/r['case_id']/'pump_budget_summary.json').read_text());assert a['all_checks_passed'] and a['admission']['technical_reference'] and b['passed']
 started=time.monotonic();proto=json.loads((P/'protocol.json').read_text());budget=proto['budget']['maximum_access_active_seconds'];frame=proto['anchor_frame']
 def tick():gate();assert time.monotonic()-started<budget,'Access active budget';
 support=load(HP/'common_road_support.npz');q=support['qualified_roads'];qi=np.flatnonzero(q)
 blocks,pop,normal,infos=population_inputs();ward=blocks.ward.to_numpy(dtype=str);sels=[np.ones(len(pop),bool)]+[ward==w for w in ['13101','13102','13103']];scopes=['Three wards','Chiyoda','Chuo','Minato'];engine=Reachability();pr=engine.pr;cache={};calls=0;hits=0;road_records=[];population=[];paired=[];anchors=[];node_anchor={}
 def get(closed):
  nonlocal calls,hits
  tick();calls+=1;assert calls<=proto['budget']['maximum_reachability_calls'];key=engine.key(closed)
  if key in cache:hits+=1;return cache[key]
  local=P/'node_cache'/(key+'.npz');old=TEMP/'reachability_cache'/(key+'.npz')
  if local.exists():
   z=load(local);assert np.array_equal(z['closed_packed'],np.packbits(closed));ok=np.unpackbits(z['node_ok_packed'],count=len(normal)).astype(bool);hits+=1
  elif old.exists():
   z=load(old);assert np.array_equal(z['closed_packed'],np.packbits(closed));ok=np.unpackbits(z['node_ok_packed'],count=len(normal)).astype(bool);hits+=1
  else:ok=engine.direct(closed)
  np.savez_compressed(local,closed_packed=np.packbits(closed),node_ok_packed=np.packbits(ok));cache[key]=ok
  if len(cache)>160:cache.pop(next(iter(cache)))
  return ok
 # Bind only completed derived products, never mutable live hydraulic ledgers.
 source_files=[HP/'common_road_support.npz',HP/'paired_road_effects.csv',HP/'off_target_summary.csv',HR/'delivery_validation.json',HR/'effects_summary.json']
 for r in manifest:
  for arm in ['baseline','drainage']:source_files.append(HP/'exposure'/(r['case_id']+'__'+arm+'.npz'))
 new_binding={str(f.relative_to(ROOT)):sha(f) for f in source_files};dump(P/'qualified_output_bindings.json',new_binding)
 for run_index,r in enumerate(manifest):
  tick();case=r['case_id'];stage=r['stage'];ex={a:load(HP/'exposure'/(case+'__'+a+'.npz')) for a in ['baseline','drainage']}
  for a,z in ex.items():
   assert np.array_equal(z['road_indices'],qi) and z['road_state_codes'].shape==(109,2,len(qi))
   assert np.max(abs(z['actual_times']-np.arange(109)*300))<=.001
  assert np.array_equal(ex['baseline']['actual_times'],ex['drainage']['actual_times']);ts=ex['baseline']['analysis_times'];assert len(ts)==109 and ts[0]==0 and ts[-1]==32400
  keys=np.empty((109,2,2),dtype='U64');flags=np.zeros((109,2,2,3,len(pop),2),bool)
  for ti in range(109):
   tick()
   for bi,bound in enumerate(['lower','upper']):
    oks=[]
    for ai,arm in enumerate(['baseline','drainage']):
     closed=np.zeros(len(q),bool);closed[qi]=ex[arm]['road_state_codes'][ti,bi]==2;ok=get(closed);assert not(ok&~normal).any();keys[ti,bi,ai]=engine.key(closed);oks.append(ok);lost=normal&~ok
     for ui,(buf,info) in enumerate(infos.items()):
      low,high=envelope(lost,info,pop);flags[ti,bi,ai,ui,:,0]=low;flags[ti,bi,ai,ui,:,1]=high
      for sel,scope in zip(sels,scopes):population.append(dict(case=case,stage=stage,arm=arm,frame=ti,elapsed_s=float(ts[ti]),bound=bound,buffer_m=buf,scope=scope,total_stock=int(pop[sel].sum()),loss_lower=int(pop[sel&low].sum()),loss_upper=int(pop[sel&high].sum()),unresolved_stock=int(pop[sel&~info['complete']].sum())))
    delta=oks[0].astype('i1')-oks[1].astype('i1');delta[~normal]=0
    for buf,info in infos.items():
     low,high=signed_envelope(delta,info,pop);ra,rb=envelope(normal&~oks[0]&oks[1],info,pop);na,nb=envelope(normal&oks[0]&~oks[1],info,pop)
     for sel,scope in zip(sels,scopes):paired.append(dict(case=case,stage=stage,frame=ti,elapsed_s=float(ts[ti]),bound=bound,buffer_m=buf,scope=scope,net_avoided_loss_lower=int(-high[sel].sum()),net_avoided_loss_upper=int(-low[sel].sum()),restored_lower=int(pop[sel&ra].sum()),restored_upper=int(pop[sel&rb].sum()),new_loss_lower=int(pop[sel&na].sum()),new_loss_upper=int(pop[sel&nb].sum())))
   if ti%24==0:
    dump(R/'state.json',dict(status='population_access',current_stage=stage,completed_case_count=run_index,frame=ti,total_frames_per_case=109,elapsed_s=time.monotonic()-started,graph_calls=calls,cache_hits=hits));print('SHARED_ACCESS',stage,ti,'/108',flush=True)
  np.savez_compressed(P/'population'/(case+'.npz'),block_ids=blocks.block_id.to_numpy(dtype=str),flags=flags,node_cache_keys=keys,actual_times=ex['baseline']['actual_times'],analysis_times=ts)
  # Fixed reference-cost path comparison at the predeclared baseline anchor.
  ep=pd.read_csv(D/'three_ward_normal_vehicle_repair_v1/multi_network_endpoints.csv');gates=ep.loc[ep.kind.eq('conditional_outer_boundary_portal'),'component'].to_numpy();tref=load(D/'three_ward_normal_vehicle_choice_v1/choice_costs.npz')['travel_seconds']
  endpoints=load(D/'three_ward_normal_vehicle_repair_v1/multi_endpoint_components.npz')['native_node_index'];normal_ref=load(D/'three_ward_night_population_transport_v1/normal_node_access.npz')
  for bi,bound in enumerate(['lower','upper']):
   results={}
   for arm in ['baseline','drainage']:
    tick();codes=ex[arm]['road_state_codes'][frame,bi];cl=np.zeros(len(q),bool);sl=cl.copy();cl[qi]=codes==2;sl[qi]=codes==1;cost=tref.copy();cost[sl[pr['arc_to_road']]]*=2
    out,inn=distances(engine.rr,pr,cost,cl,gates);ok=np.isfinite(out)&np.isfinite(inn);cached=load(P/'node_cache'/(engine.key(cl)+'.npz'));expected=np.unpackbits(cached['node_ok_packed'],count=len(normal)).astype(bool);assert np.array_equal(ok,expected),'Shortest paths disagree with turn-state reachability'
    results[arm]=(out,inn,ok);np.savez_compressed(P/'anchor_nodes'/(case+'__'+bound+'__'+arm+'.npz'),outward_reference_seconds=out,inward_reference_seconds=inn,bidirectional=ok)
   base,treated=results['baseline'],results['drainage'];valid=normal[endpoints];cohort=valid&base[2][endpoints]&treated[2][endpoints];endpoint_rows=[]
   for arm,z in results.items():
    current=z[2][endpoints];seconds=np.maximum(z[0][endpoints],z[1][endpoints]);base_seconds=np.maximum(base[0][endpoints],base[1][endpoints]);rat=np.divide(seconds,base_seconds,out=np.full_like(seconds,np.nan),where=np.isfinite(base_seconds)&(base_seconds>0)&np.isfinite(seconds));used=cohort&np.isfinite(rat)
    for j in range(len(endpoints)):endpoint_rows.append(dict(case=case,stage=stage,bound=bound,arm=arm,frame=frame,endpoint_index=j,normal_bidirectional=bool(valid[j]),current_bidirectional=bool(current[j]),common_survivor=bool(cohort[j]),paired_cost_ratio=float(rat[j]) if used[j] else None))
    anchors.append(dict(stage=stage,bound=bound,arm=arm,frame=frame,normal_reachable_endpoints=int(valid.sum()),current_reachable_endpoints=int((valid&current).sum()),restored_endpoints=int((valid&~base[2][endpoints]&current).sum()),newly_lost_endpoints=int((valid&base[2][endpoints]&~current).sum()),common_survivor_endpoints=int(cohort.sum()),comparable_positive_cost_endpoints=int(used.sum()),median_same_endpoint_cost_ratio=float(np.median(rat[used])) if used.any() else None,ratio_p95=float(np.quantile(rat[used],.95)) if used.any() else None))
   road_records.extend(endpoint_rows)
  print('SHARED_ACCESS_CASE_DONE',stage,flush=True)
 pf=pd.DataFrame(population);pp=pd.DataFrame(paired);save(P/'population_trajectories.csv',pf);save(P/'paired_population_trajectories.csv',pp);save(P/'anchor_endpoint_access.csv',road_records);save(P/'anchor_access_summary.csv',anchors)
 summary=[]
 for key,g in pf.groupby(['case','stage','arm','bound','buffer_m','scope']):
  g=g.sort_values('frame');row=dict(zip(['case','stage','arm','bound','buffer_m','scope'],key));row.update(peak_loss_lower=int(g.loss_lower.max()),peak_loss_upper=int(g.loss_upper.max()),loss_lower_resident_hours=float(np.trapezoid(g.loss_lower,g.elapsed_s)/3600),loss_upper_resident_hours=float(np.trapezoid(g.loss_upper,g.elapsed_s)/3600),unresolved_stock=int(g.unresolved_stock.iloc[0]));summary.append(row)
 save(P/'population_summary.csv',summary);matched=[]
 for key,g in pp.groupby(['case','stage','bound','buffer_m','scope']):
  g=g.sort_values('frame');row=dict(zip(['case','stage','bound','buffer_m','scope'],key));anchor=g[g.frame==frame].iloc[0]
  for name in ['net_avoided_loss_lower','net_avoided_loss_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper']:row['anchor_'+name]=int(anchor[name]);row[name+'_resident_hours']=float(np.trapezoid(g[name],g.elapsed_s)/3600)
  row['maximum_new_loss_lower']=int(g.new_loss_lower.max());row['maximum_new_loss_upper']=int(g.new_loss_upper.max());matched.append(row)
 save(P/'paired_population_summary.csv',matched)
 assert len(pf)==2*109*2*2*3*4 and len(pp)==2*109*2*3*4
 assert pf[pf.scope=='Three wards'].total_stock.eq(496345).all() and (pf.loss_lower<=pf.loss_upper).all()
 assert (pp.net_avoided_loss_lower<=pp.net_avoided_loss_upper).all() and (pp.restored_lower<=pp.restored_upper).all() and (pp.new_loss_lower<=pp.new_loss_upper).all()
 assert same_sources() and all(sha(ROOT/n)==h for n,h in new_binding.items())
 roads=pd.read_csv(HP/'paired_road_effects.csv');shifts=pd.read_csv(HP/'off_target_summary.csv');main=pd.DataFrame(matched);main=main[(main.scope=='Three wards')&(main.bound=='lower')&(main.buffer_m==10)];roadmain=roads[(roads.Scope=='Three wards')&(roads['Depth Branch']=='lower')]
 improve=bool((roadmain['Closure Reduction km h']>0).all());major_transfer=bool((shifts.loc[shifts.Scope=='Three wards','Outside Target Cells Certain Head Rise Area']>0).any());newpop=bool(main.maximum_new_loss_upper.gt(0).any())
 judgment=('两种水位的封路累计暴露均有改善。' if improve else '两种水位的封路累计暴露并未都改善。')+('存在达到既定0.10m筛查阈值的三区非目标单元水位上升，扩展前需评估代价。' if major_transfer else '既定0.10m筛查未发现三区非目标单元确定水位上升；仍不排除更小变化或真实接收河道的下游影响。')+('保留了新增人口通达性损失的可能区间，不能只报告净收益。' if newpop else '未产生新增常住人口通达性损失区间；这一结论仍受未知道路按可用假设限制。')+'本轮只含一个历史父事件，尚不能证明措施跨暴雨普遍有效；不自动扩大水动力情景。'
 dump(R/'summary.json',dict(main_population=main.to_dict('records'),anchor_access=anchors,road_effects=roadmain.to_dict('records'),off_target=shifts.to_dict('records'),next_step_judgment=judgment,observational_benchmark_admitted=0,new_hydraulic_runs=0,congestion_queue_validated=False,population_total=496345))
 checks=dict(original_hydraulic_rules_passed=True,common_support_only=True,clocks_equal=True,normal_graph_exact=True,shortest_paths_match_reachability=True,population_stock_conserved=True,candidate_intervals_ordered=True,paired_signed_changes_not_marginal_subtraction=True,new_losses_retained=True,frozen_anchor_before_treatment=True,no_population_centroids=True,source_hashes_unchanged=True,no_new_hydraulics=True)
 dump(R/'validation.json',dict(passed=True,checks=checks,graph_calls=calls,cache_hits=hits,seconds=time.monotonic()-started,observed_flood_accuracy_verified=False))
 dump(R/'state.json',dict(status='qualified_complete',elapsed_s=time.monotonic()-started,completed_stages=['H','L'],new_hydraulic_runs=0));report();print('SHARED_RECEIVER_ACCESS_COMPLETE',flush=True)

def wait_and_run():
 prepare();started=time.monotonic();waitmax=json.loads((P/'protocol.json').read_text())['budget']['maximum_wait_seconds']
 while True:
  gate();assert same_sources();state=json.loads((HR/'campaign_state.json').read_text())
  if state['status']=='held':raise RuntimeError('Parent held: '+state.get('error','unknown'))
  if state['status']=='qualified_complete':break
  assert time.monotonic()-started<waitmax,'Prerequisite waiting budget'
  dump(R/'state.json',dict(status='waiting_for_parent',parent_status=state['status'],parent_case=state.get('current_case'),native_time_s=state.get('latest_native_time_s'),wait_seconds=time.monotonic()-started,new_hydraulic_runs=0));report();time.sleep(15)
 run()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','run','wait']);a=ap.parse_args()
 try:
  if a.phase=='prepare':prepare()
  elif a.phase=='run':run()
  else:wait_and_run()
 except BaseException as exc:
  if R.exists() and (P/'protocol.json').exists():dump(R/'state.json',dict(status='held',error=str(exc),automatic_retry=False));report()
  raise
