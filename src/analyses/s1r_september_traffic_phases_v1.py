#!/usr/bin/env python3
"""Bounded4/6/9hour September traffic supplement;no new hydraulics.
Original single-snapshot traffic implementation and completed files stay frozen.
"""
from pathlib import Path
import os,sys,json,time,datetime as dt,types,inspect,argparse,html,traceback,gc
for k in ['OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/udt01b-s1r-traffic-numba')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import numpy as np,pandas as pd
from shared_receiver_pilot_common_v1 import sha,dump
TAG='three_ward_s1r_september_traffic_phases_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG
ENGINE=ROOT/'src/analyses/s1r_september_traffic_v1.py';SCRIPT=Path(__file__).resolve();DECISION='MILI-D-20261024-001'
OLD=ROOT/'data/processed/three_ward_s1r_september_traffic_v1';OLDR=ROOT/'data/exp/estimation'/OLD.name
HYDRO=ROOT/'data/processed/three_ward_s1r_cross_event_validation_v1';FRAMES=[48,72,108];LABELS={48:'raised receiving stage',72:'after stage recession',108:'window end'}
MODULES=[];START=time.monotonic()

def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False)as z:return dict(z)
def dirs(frame):return P/'batches'/f'f{frame:03d}',R/'batches'/f'f{frame:03d}'
def active_seconds():return time.monotonic()-START-sum(m.PAUSED for m in MODULES)

def engine(frame):
 m=types.ModuleType(f's1r_phase_{frame}');m.__file__=str(ENGINE);exec(compile(ENGINE.read_text(),str(ENGINE),'exec'),m.__dict__)
 m.P,m.R=dirs(frame);m.SCRIPT=SCRIPT;m.FIXED_FRAME=frame;m.ENGINE_SOURCE=ENGINE;m.MASTER_PROTOCOL=P/'protocol.json'
 m.SNAPSHOT_JST=(dt.datetime(2025,9,11,13,15,tzinfo=dt.timezone(dt.timedelta(hours=9)))+dt.timedelta(seconds=frame*300)).isoformat()
 m.PHASE_SELECTION=f'Predeclared4/6/9hour phase coverage. This batch frame{frame},same H/L and all3portfolios. Not selected from treatment effects.'
 source=inspect.getsource(m.prepare);a=source.index(' refs=pd.read_csv(');b=source.index('\n sourcepaths=',a);source=source[:a]+' frame=FIXED_FRAME;assert frame in [48,72,108]'+source[b:]
 swaps={"dependency=[SCRIPT,":"dependency=[SCRIPT,ENGINE_SOURCE,MASTER_PROTOCOL,P/'context_adapter.py.txt',","'event_snapshot_jst':'2025-09-11T15:40:00+09:00'":"'event_snapshot_jst':SNAPSHOT_JST","'frozen17h survey-derived proxy,2xreference;controlled pressure case,not actual15:40demand'":"'frozen17h survey-derived proxy,2xreference;controlled pressure case,not observed event-hour demand'","'Frame29 chosen from frozen H/no-action road-closure peak,applied unchanged to both stages and all three portfolios. No treatment-selected peak.'":"PHASE_SELECTION","'same_anchor29_all6states'":"'same_declared_phase_time_all6states'"}
 for x,y in swaps.items():assert source.count(x)==1,x;source=source.replace(x,y)
 source=source.replace('frame29,15:40JST',f'frame{frame},{m.SNAPSHOT_JST}').replace('at frozen no-action Hpeak','at a predeclared phase')
 summary=inspect.getsource(m.summarize);assert summary.count("'frame':29")==1;summary=summary.replace("'frame':29","'frame':FIXED_FRAME")
 exec(compile(source+'\n'+summary,str(SCRIPT)+f'::phase{frame}','exec'),m.__dict__)
 adapter=source+'\n'+summary;m.render=render;raw=m.tick
 def gate():
  raw()
  if(P/'STOP').exists():raise RuntimeError('User STOP')
  if(P/'PAUSE').exists():
   begun=time.monotonic()
   while(P/'PAUSE').exists():
    if(P/'STOP').exists():raise RuntimeError('User STOP while paused')
    time.sleep(2)
   m.PAUSED+=time.monotonic()-begun
  assert active_seconds()<14400,'Frozen4hour supplementary traffic budget'
 m.tick=gate;MODULES.append(m)
 return m,adapter


def coverage():
 clip=load(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz');rows=[];diff=[]
 for q in ['Q2p5','Q5']:
  bp=HYDRO/'batches'/('E20250911_'+q);runs=read(bp/'run_manifest.json')['runs'];pairs={}
  for run in runs:
   stage=run['stage'];pairs[stage]={}
   for arm in ['baseline','drainage']:
    z=load(bp/'exposure'/(run['case_id']+'__'+arm+'.npz'));ix=z['road_indices'];pairs[stage][arm]=z
    if q=='Q5'and arm=='baseline':continue
    portfolio='none'if arm=='baseline'else q
    for ti in range(109):
     for bi,bound in enumerate(['lower','upper']):
      code=z['road_state_codes'][ti,bi];w=clip['union_length_m'][ix]/1000
      rows.append({'portfolio':portfolio,'stage':stage,'bound':bound,'frame':ti,'elapsed_s':float(z['analysis_times'][ti]),'closed_km':float(w[code==2].sum()),'slow_km':float(w[code==1].sum()),'assumed_stage_increment_m':float(np.interp(ti*300,run['stage_times_s'],run['stage_increment_m']))})
  for frame in [29]+FRAMES:
   for arm in ['baseline','drainage']:
    if q=='Q5'and arm=='baseline':continue
    a,b=[pairs[s][arm]for s in ['H','L']];assert np.array_equal(a['road_indices'],b['road_indices'])
    w=clip['union_length_m'][a['road_indices']]/1000
    for bi,bound in enumerate(['lower','upper']):diff.append({'frame':frame,'phase':'early reference'if frame==29 else LABELS[frame],'portfolio':'none'if arm=='baseline'else q,'bound':bound,'H_L_different_road_codes_km':float(w[a['road_state_codes'][frame,bi]!=b['road_state_codes'][frame,bi]].sum())})
 pd.DataFrame(rows).to_csv(P/'full_road_time_coverage.csv',index=False);pd.DataFrame(diff).to_csv(P/'H_L_phase_state_differences.csv',index=False)


def prepare():
 assert not(P/'protocol.json').exists(),'Preserve prepared supplement'
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 assert read(OLDR/'validation.json')['passed']and read(OLDR/'campaign_state.json')['status']=='qualified_complete'
 # Numerical criteria are inherited,not a demand that interventions show benefits.
 protocol={'decision':DECISION,'scope':'September traffic phase supplement only','frames':FRAMES,'elapsed_hours':[4,6,9],'selection':'Fixed receiving-stage plateau4h,recession+1h6h,and9hwindow end. No selection from favorable treatment outcomes. Original2h25snapshot retained.','new_traffic_states':18,'reused_traffic_states':6,'rainfall_parent_count':1,'new_hydraulics':0,'frozen':'Original ODx2pressure proxy,capacity,turn graph,reference costs,two-path UE,treatment locations,quality masks and numerical tolerances. 17hdemand is held fixed across snapshots,not observed time-varying traffic.','accounting':'Each time-stage compares all3portfolios on identical requested demand and common surviving OD weights. Add an all24states common cohort for cross-time cost comparisons without extra fitting. Restore and newly unreachable requests remain separate.','completion_conditions':['All18newfits and36path sweeps pass original source,request,flow,cost and gap checks.','All3fixedphases are represented for H/L and all3portfolios;do not require H/Loutcomes to differ.','Early reference and null/adverse/time-dependent results retained.','Final tables retain masks,unknown roads,proxy demand and cohort exclusions.','Close this September supplement after the declared list;no extra times,new rainfalls,locations,capacities,training or sealed tests automatically.'],'interpretation':'Numerical convergence is already achieved by previous fits. Phase completion closes a conditional September analysis,not independent observed-congestion validation or proof of general benefit. Mixed signs are reportable phase dependence,not a reason to add runs until favorable.','resource_budget':{'serial_threads':1,'new_fits':18,'new_path_sweeps':36,'total_supplement_seconds':14400,'per_sweep_seconds':600,'per_fit_seconds':600,'automatic_retries':0},'broader_research_gap':'July/August matching traffic fits are not yet run. Daytime population,actual queues and empirical road validation remain outside this batch.'}
 dump(P/'protocol.json',protocol);dump(R/'campaign_state.json',{'status':'preparing','completed_frames':[],'current_frame':None,'new_states_total':18});render()
 for frame in FRAMES:
  m,adapter=engine(frame);m.P.mkdir(parents=True,exist_ok=True);m.R.mkdir(parents=True,exist_ok=True);(m.P/'context_adapter.py.txt').write_text(adapter);m.prepare()
  assert read(m.R/'preflight_validation.json')['passed']
  meta=read(m.P/'state_manifest.json')['states'];assert len(meta)==6 and all(v['frame']==frame for v in meta)
  # Independently compare every staged state to its exact source frame;unknown roads stay0.
  for v in meta:
   q='Q2p5'if v['arm']=='none'else v['arm'];bp=HYDRO/'batches'/('E20250911_'+q);run=next(x for x in read(bp/'run_manifest.json')['runs']if x['stage']==v['stage']);arm='baseline'if v['arm']=='none'else'drainage';src=load(bp/'exposure'/(run['case_id']+'__'+arm+'.npz'));z=load(m.P/'states'/(v['case_id']+'.npz'))
   assert np.array_equal(z['codes'][src['road_indices']],src['road_state_codes'][frame,0])and not z['codes'][~z['qualified']].any()
   assert abs(v['actual_time_s']-300*frame)<.001 and v['event_snapshot_jst']==m.SNAPSHOT_JST
   if frame==48 and v['stage']=='H':assert np.interp(frame*300,run['stage_times_s'],run['stage_increment_m'])==1.5
 coverage()
 bound={str(f.relative_to(ROOT)):sha(f)for f in [SCRIPT,ENGINE,P/'protocol.json',OLD/'input_bindings.json',OLDR/'validation.json',OLDR/'campaign_state.json']}
 for n,h in read(OLD/'input_bindings.json')['sha256'].items():assert sha(ROOT/n)==h
 dump(P/'execution_bindings.json',bound)
 dump(R/'preflight_validation.json',{'passed':True,'new_states':18,'fixed_frames':[48,72,108],'all_staged_states_match_source_frames':True,'receiving_stage_plateau_captured':True,'existing_six_states_unchanged':True,'same_numerical_and_traffic_rules':True,'new_hydraulics':0})
 dump(R/'campaign_state.json',{'status':'prepared','completed_frames':[],'current_frame':None,'new_states_total':18})
 (R/'README.md').write_text('# September traffic phase closure\n\nDecision '+DECISION+'. Add exactly4/6/9hours x H/L x none,Q2.5,Q5,18traffic states. Reuse original2h25six states for context. No new hydraulic model. No treatment-selected peak or response-based stopping. Each phase uses the qualified previous traffic implementation with context-only frame/output binding and original numerical tolerances. Compare costs within each phase-stage on a common3portfolio cohort;an additional all24states common cohort keeps temporal comparison weights fixed. No means of different survivor cohorts are subtracted.\n\nRun physical-measures-py312/bin/python src/analyses/s1r_september_traffic_phases_v1.py prepare,then run once. Serial single thread,4hour total excluding explicit PAUSE,36new path sweeps and18new fits;per-sweep/per-fit600s. First failed source/conservation/gap/resource check holds;no auto retries. Source files and prior results are immutable. Whole-curve road coverage is descriptive,not extra traffic fits or nine-hour integrated traffic cost. Completion closes this declared September list regardless of beneficial,null or adverse effects. No automatic other-parent run,hydraulics,ML,sealed-test access,AnaSOP edit or figure generation. Fixed17h demand x2at all times is a controlled pressure scenario,not historical traffic replay.\n')
 render();print('THREE_PHASE_18_STATE_PREFLIGHT_PASSED',flush=True)


def finalize():
 sources=[(29,OLD,OLDR)]+[(f,*dirs(f))for f in FRAMES]
 for frame,bp,br in sources:assert read(br/'validation.json')['passed']
 for name in ['traffic_outcomes.csv','paired_traffic_effects.csv','road_load_summary.csv']:
  parts=[]
  for frame,bp,br in sources:
   d=pd.read_csv(bp/name)
   if'frame'in d:assert d.frame.eq(frame).all()
   else:d.insert(0,'frame',frame)
   d.insert(0,'elapsed_hours',frame/12);parts.append(d)
  out=pd.concat(parts,ignore_index=True);out.to_csv(P/('all_times_'+name),index=False)
  assert len(out)==(96 if name=='road_load_summary.csv'else 24)
 # A single positive-weight cohort across all24states prevents survivor-composition time effects.
 first=engine(FRAMES[0])[0];dem=load(ROOT/'data/processed/three_ward_normal_vehicle_repair_v1/multi_routing_demands.npz');q0=first.parent.original_request(dem);core=load(ROOT/'data/processed/three_ward_demand_protection_feedback_v1/protection_masks.npz')['core_endpoint_od']&dem['base'];cohort=core&(q0>0);fits={}
 for frame,bp,br in sources:
  for v in read(bp/'state_manifest.json')['states']:
   z=load(bp/'fits'/(first.parent.key(v['case_id'],'flood',0,'UE')+'.npz'));assert np.array_equal(z['original_requested'],q0);cohort&=z['reachable'];fits[(frame,v['stage'],v['arm'])]=z
 weight=float(q0[cohort].sum());rows=[]
 for(frame,stage,arm),z in fits.items():
  base=fits[(frame,stage,'none')];valid=weight>0;mean=float(q0[cohort]@z['mean_od_cost'][cohort]/weight)if valid else None;bmean=float(q0[cohort]@base['mean_od_cost'][cohort]/weight)if valid else None
  rows.append({'frame':frame,'elapsed_hours':frame/12,'stage':stage,'portfolio':arm,'all24states_common_request_proxy':weight,'common_cohort_mean_model_cost_min':mean/60 if valid else None,'same_time_noaction_cost_index':100*mean/bmean if valid and bmean>0 else None,'baseline_reachable_excluded_proxy':float(q0[core&base['reachable']&~cohort].sum()),'cohort_empty':not valid})
 pd.DataFrame(rows).to_csv(P/'fixed_all_times_cohort_costs.csv',index=False);pd.DataFrame({'expanded_od_index':np.flatnonzero(cohort),'original_request_weight':q0[cohort]}).to_csv(P/'fixed_all_times_cohort.csv',index=False)
 paired=pd.read_csv(P/'all_times_paired_traffic_effects.csv');ranges=[]
 for keys,g in paired.groupby(['stage','reference','treatment']):
  v=g.common_cohort_cost_change_percent.to_numpy();ranges.append(dict(zip(['stage','reference','treatment'],keys))|{'within_phase_paired_cost_percent_min':float(v.min()),'within_phase_paired_cost_percent_max':float(v.max()),'cost_change_sign_pattern':'mixed'if(v<0).any()and(v>0).any()else'nonpositive'if(v<=0).all()else'nonnegative','restored_request_proxy_max':float(g.restored_request_proxy.max()),'new_unreachable_request_proxy_max':float(g.newly_unreachable_request_proxy.max()),'meaning':'phase-specific paired contrasts;range not uncertainty interval or significance test'})
 pd.DataFrame(ranges).to_csv(P/'phase_effect_ranges.csv',index=False)
 assert all(sha(ROOT/n)==h for n,h in read(P/'execution_bindings.json').items())
 fitsval=[read(f)for frame in FRAMES for f in dirs(frame)[1].glob('*_fit_validation.json')];paths=[read(f)for frame in FRAMES for f in dirs(frame)[1].glob('*_path_validation.json')];assert len(fitsval)==18 and len(paths)==36 and all(v['passed']for v in fitsval+paths)
 dump(R/'validation.json',{'passed':True,'new_traffic_states':18,'reused_states':6,'new_path_sweeps':36,'max_gap':max(v['gap']for v in fitsval),'full_declared_phase_list_completed':True,'new_hydraulics':0,'prior_states_unchanged':True,'paired_within_phase_cohorts':True,'fixed_all_times_cohort_available':weight>0,'new_population_counts':False,'september_supplement_closed':True,'independent_congestion_validation':False,'automatic_additional_scenarios':False})
 dump(R/'summary.json',{'phase_effect_ranges':ranges,'fixed_all_times_common_request_proxy':weight,'scientific_stopping_rule':'Close this declared phase list whether effects are favorable,null,adverse or phase-dependent. No requirement to achieve a preferred ranking.','scope':'One inspected September rainfall parent;traffic estimates for July/August,real queues and daytime population remain unvalidated/unrun here.'})


def render():
 R.mkdir(parents=True,exist_ok=True)
 if not(R/'campaign_state.json').exists():return
 s=read(R/'campaign_state.json');rows=[];n=0
 for frame in FRAMES:
  bp,br=dirs(frame);st=read(br/'campaign_state.json')if(br/'campaign_state.json').exists()else{'status':'waiting','completed_cases':[]}
  n+=len(st['completed_cases']);rows.append({'Elapsed hours':frame/12,'Phase':LABELS[frame],'State':st['status'],'Qualified states':len(st['completed_cases']),'Current':st.get('current_case','')})
 body='<h1>9月交通时段补充与收束条件</h1><p><b>'+html.escape(s['status'])+'</b>；新增通过 '+str(n)+'/18，原6组保留。</p><aside>固定4、6、9小时，分别覆盖假设河道抬升期、退水后和9小时窗口末端。每个时刻H/L×无措施、Q2.5、Q5。只复用既有积水，不新增水动力。</aside>'+pd.DataFrame(rows).to_html(index=False)
 if s.get('error'):body+='<p class="warn">停止并保留结果：'+html.escape(s['error'])+'</p>'
 body+='<h2>何时结束这一阶段</h2><p>数值收敛已有原优化间隙≤0.001等准入门槛。此次只补时段覆盖。原18组全部完成并通过、保留原需求账本与质量掩膜、把无效和恶化一起报告后，收束这场雨的分析。结论可以是改善、没有明确改善或效果随阶段变化；不为追求正结果反复加时刻。</p><p>需求固定为17时驾驶出行代理×2，成本与容量原假设不调参。每个时刻/水位内用三方案共同可达OD做配对；另提供跨全部24状态的同一OD群体比较时间变化。不能把各时刻幸存者的平均成本相减，也不能把快照相加当成全时段独立车辆或动态排队。</p>'
 if(R/'summary.json').exists():
  z=read(R/'summary.json');body+='<h2>既定时段下的配对效果范围</h2>'+pd.DataFrame(z['phase_effect_ranges']).to_html(index=False,float_format=lambda x:f'{x:,.6f}')+'<p>范围是不同阶段的模型差异，不是置信区间，也不是显著性检验。</p>'
  body+='<h2>全部时段固定共同OD的成本</h2>'+pd.read_csv(P/'fixed_all_times_cohort_costs.csv').to_html(index=False,float_format=lambda x:f'{x:,.4f}')
 body+='<h2>固定边界</h2><p>本轮单线程串行，上限4小时，18次新优化、36次路径扫描；失败停止，不自动重试，不增加时刻、雨量或泵能力。严格已知道路与未知道路分开。水动力情景、夜间人口区间和原28＋8项水动力准入不变，交通原守恒/封闭零流/成本加载/优化间隙规则不变。</p><p>这一阶段完成不等于真实拥堵已验证。跨暴雨交通结论还需7月、8月同口径检验；白天人口和实测队列仍是缺口。本批不自动启动这些工作，不修改AnaSOP或生成新论文图。</p><p><a href="../../../processed/'+TAG+'/protocol.json">Protocol / completion conditions</a> · <a href="campaign_state.json">Live state</a> · <a href="preflight_validation.json">Preflight</a></p>'
 if(R/'validation.json').exists():body+='<p><a href="validation.json">Final validation</a> · <a href="summary.json">Summary</a></p>'
 style='body{font:16px/1.8 system-ui;max-width:1260px;margin:30px auto;padding:0 22px;color:#263e4c;background:#f7fafb}aside{padding:18px;background:#e6f1f5}table{display:block;max-width:100%;overflow:auto;background:white;border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #cddce5;text-align:left}.warn{background:#fff0d4;padding:15px}'
 refresh='<meta http-equiv="refresh" content="30">'if s['status']not in ['held','qualified_complete']else''
 tmp=R/'review.html.tmp';tmp.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+refresh+'<title>S1R Traffic Phase Coverage and Closure</title><style>'+style+'</style>'+body+'</html>');tmp.replace(R/'review.html')


def run():
 global START
 assert read(R/'preflight_validation.json')['passed'];s=read(R/'campaign_state.json');assert s['status']=='prepared'and not s['completed_frames'],'No automatic restart'
 assert all(sha(ROOT/n)==h for n,h in read(P/'execution_bindings.json').items());START=time.monotonic();s.update(status='running',pid=os.getpid(),started_utc=dt.datetime.now(dt.timezone.utc).isoformat());dump(R/'campaign_state.json',s);render()
 try:
  for frame in FRAMES:
   s.update(current_frame=frame);dump(R/'campaign_state.json',s);m,_=engine(frame);m.run();assert read(m.R/'validation.json')['passed'];s['completed_frames'].append(frame);s.update(elapsed_s=active_seconds());dump(R/'campaign_state.json',s);render();m.CACHE.clear();gc.collect();print('PHASE_QUALIFIED',frame,flush=True)
  s.update(status='summarizing',current_frame=None);dump(R/'campaign_state.json',s);render();finalize();s.update(status='qualified_complete',elapsed_s=active_seconds(),completed_utc=dt.datetime.now(dt.timezone.utc).isoformat());dump(R/'campaign_state.json',s);render();print('SEPTEMBER18_PHASE_STATES_COMPLETE',flush=True)
 except BaseException as ex:
  s.update(status='held',error=str(ex)or type(ex).__name__,elapsed_s=active_seconds());dump(R/'campaign_state.json',s);(R/'error.txt').write_text(traceback.format_exc());render();raise


def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','run','report']);a=ap.parse_args();{'prepare':prepare,'run':run,'report':render}[a.phase]()
if __name__=='__main__':main()
