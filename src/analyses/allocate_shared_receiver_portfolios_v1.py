#!/usr/bin/env python3
"""Allocate conditional static reservations from existing snapshots; no hydraulics."""
from pathlib import Path
import json,hashlib,html
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_shared_receiver_allocation_v1'
P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/research-design'/TAG
C=ROOT/'data/processed/three_ward_intake_receiver_candidates_v1'
EPS_VOLUME=1e-6;EPS_RATE=1e-9
POOL={17:'Sumida',71:'Kamejima',79:'Furukawa',111:'Furukawa',106:'Nihonbashi',110:'Kanda',51:'Unidentified'}
NAMES={'Sumida':'隅田川','Kamejima':'亀島川','Furukawa':'古川','Nihonbashi':'日本橋川','Kanda':'神田川','Unidentified':'未命名水域'}

def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as s:
  for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(f,x):f.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def allocate(request,allowed,distance,codes,grid,allowance,unit=.5):
 """Descending capped local reference demand;nearest outlet with free reservation.
 One outlet per source. Greedy reproducible reference,not claimed optimal.
 """
 pools=sorted(set(POOL.values()));remaining={k:(0. if k=='Unidentified' else allowance) for k in pools}
 rate=np.zeros(len(request));outlet=np.full(len(request),-1,int)
 for i in np.lexsort((grid,-request)):
  options=sorted(np.flatnonzero(allowed[:,i]),key=lambda j:(distance[j,i],codes[j]))
  for j in options:
   pool=POOL[int(codes[j])];q=min(unit,request[i],max(0.,remaining[pool]))
   if q<=EPS_RATE:continue
   rate[i]=q;outlet[i]=j;remaining[pool]=max(0.,remaining[pool]-q);break
 assert (rate<=np.minimum(unit,request)+1e-12).all() and ((outlet>=0)==(rate>0)).all()
 for pool in pools:
  members=np.array([POOL[int(codes[j])]==pool if j>=0 else False for j in outlet]);used=float(rate[members].sum())
  if np.isfinite(remaining[pool]):assert used<= (0. if pool=='Unidentified' else allowance)+1e-11
 return rate,outlet,remaining

def main():
 P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 paths=[C/'candidate_index.npz',C/'source_candidates_preprocessed.parquet',C/'intake_receiver_links_preprocessed.parquet',C/'receiver_candidates_preprocessed.parquet',C/'capability_scenarios_preprocessed.parquet',C/'manifest.json',ROOT/'docs/AnaSOP.md',ROOT/'data/processed/three_ward_barrier_drainage_v1/solver_binding.json',ROOT/'src/preprocessing/barrier_drainage_v1/pump_routines.inc']
 cases=['E20240731__H__U','E20240731__L__U'];paths += [C/(case+'_saved_states.npz') for case in cases]
 binding={str(f.relative_to(ROOT)):sha(f) for f in paths};up=json.loads((C/'manifest.json').read_text())
 for e in up['outputs']:
  if e['path'].endswith(('source_candidates_preprocessed.parquet','intake_receiver_links_preprocessed.parquet','receiver_candidates_preprocessed.parquet','capability_scenarios_preprocessed.parquet')):assert sha(ROOT/e['path'])==e['sha256']
 sources=pd.read_parquet(paths[1]);links=pd.read_parquet(paths[2]);receivers=pd.read_parquet(paths[3]);n=len(sources)
 with np.load(paths[0]) as z:grid=z['source_grid'];srcnative=z['source_native'];recnative=z['receiver_native'];si=z['link_source'];ri=z['link_receiver'];distance=z['link_distance'].reshape(7,n);target=z['target_storage'];sourceok=z['source_eligible']
 assert np.array_equal(si,np.tile(np.arange(n),7));codes=links['Receiver Group'].to_numpy().reshape(7,n)[:,0];assert set(codes)==set(POOL)
 core=sources['Ward Code'].to_numpy()>0;states={};max_excess=np.zeros(n);positive_any=np.zeros(n,bool);head_ok_all=np.ones((7,n),bool);partial_count=np.zeros((7,n),int)
 for case in cases:
  with np.load(C/(case+'_saved_states.npz')) as z:
   e=np.maximum(z['source_volume']-target,0.);e[:,~core]=0.;positive=e>EPS_VOLUME
   lift=np.maximum(z['receiver_stage'][:,ri]-z['source_stage'][:,si],0).reshape(109,7,n);trigger=(z['source_stage']-z['source_minimum'])>=.05
   passing=(lift+float(z['receiver_alignment_bound'])<=5.)&trigger[:,None,:]
   head_ok_all&=(~positive[:,None,:]|passing).all(axis=0);partial_count+=(positive[:,None,:]&passing).sum(axis=0)
   max_excess=np.maximum(max_excess,e.max(axis=0));positive_any|=positive.any(axis=0);states[case]={'excess':e,'actual':z['actual_time'].copy(),'analysis':z['analysis_time'].copy()}
 named=np.array([int(c)!=51 for c in codes]);base_ok=head_ok_all&sourceok[None,:]&positive_any[None,:]&named[:,None]
 pool_rows=[{'Receiving Pool':p,'Pool Name':NAMES[p],'Model Groups':','.join(str(c) for c in sorted(POOL) if POOL[c]==p),'Grouping Evidence':'same model river name pooled conservatively;physical acceptance unverified' if p!='Unidentified' else 'identity unresolved;withheld from this allocation','Capacity Evidence':'scenario assumption only;no measured allowance','Unidentified Receiver Held':p=='Unidentified'} for p in sorted(set(POOL.values()))]
 portfolios=[]
 for length in [500.,1000.]:
  for label,cap in [('Open Conditional',np.inf),('Finite 0.5',.5),('Finite 2.0',2.),('Blocked',0.)]:portfolios.append((f'R{int(length)}_T30_'+label.replace(' ','_').replace('.','p'),length,1800.,label,cap))
 for length in [500.,1000.]:portfolios.append((f'R{int(length)}_T60_Finite_2p0',length,3600.,'Finite 2.0',2.))
 selected=[];pool_alloc=[];overview=[];traces=[];summaries=[];gap_rows=[];artifact_bindings=[]
 for ident,length,horizon,label,cap in portfolios:
  allowed=base_ok&(distance<=length);haslink=allowed.any(axis=0);request_max=max_excess/horizon;request_max[~positive_any]=0.
  rate,outlet,remaining=allocate(request_max,allowed,distance,codes,grid,cap)
  chosen=np.flatnonzero(rate>0);assigned=np.array([POOL[int(codes[j])] if j>=0 else 'None' for j in outlet]);assert (sourceok[chosen]).all() and core[chosen].all() and not sources['Source Candidate'].iloc[chosen].duplicated().any()
  folder=P/'portfolios'/ident;folder.mkdir(parents=True,exist_ok=True)
  if len(chosen):
   pump=folder/'udt_pumps.dat'
   with pump.open('w') as f:
    f.write(str(len(chosen))+'\n')
    for i in chosen:
     j=outlet[i];linkix=j*n+i;r=int(ri[linkix]);assert int(codes[j])!=51
     f.write(f'{int(srcnative[i])+1} {int(recnative[r])+1} {rate[i]:.17e} 0.05 5.0\n')
     selected.append({'Portfolio':ident,'Source Candidate':str(sources['Source Candidate'].iloc[i]),'Grid Index':int(grid[i]),'Ward Code':int(sources['Ward Code'].iloc[i]),'Source Native Cell One Based':int(srcnative[i])+1,'Receiver Native Cell One Based':int(recnative[r])+1,'Receiver Candidate':str(receivers['Receiver Candidate'].iloc[r]),'Receiver Group':int(codes[j]),'Receiving Pool':assigned[i],'Reserved Rate':float(rate[i]),'Standalone Rate Ceiling':.5,'Straight Distance Lower Bound':float(distance[j,i]),'Reference Max Local Rate Request':float(request_max[i]),'Reference Max Local Excess':float(max_excess[i]),'Selected Location Meaning':'native modeled intake;not a physical pump machine','Field Deployment Verified':False})
   assert 0<len(chosen)<=10000;read=np.loadtxt(pump,skiprows=1,ndmin=2);assert len(read)==len(chosen) and np.array_equal(read[:,2],rate[chosen]);artifact_bindings.append({'path':str(pump.relative_to(ROOT)),'sha256':sha(pump),'rows':len(chosen),'execution_enabled':False})
  else:
   assert label=='Blocked';dump(folder/'reuse_baseline.json',{'mode':'baseline_reuse;do not create zero-count pump file','hydraulic_execution':False});assert not(folder/'udt_pumps.dat').exists()
  for pool in sorted(set(POOL.values())):
   used=float(rate[assigned==pool].sum());capvalue=0. if pool=='Unidentified' else cap
   pool_alloc.append({'Portfolio':ident,'Receiving Pool':pool,'Model Groups':','.join(str(c) for c in sorted(POOL) if POOL[c]==pool),'Scenario Shared Allowance':float(capvalue) if np.isfinite(capvalue) else None,'Allowance Mode':'unconstrained assumption' if not np.isfinite(capvalue) else ('withheld unresolved identity' if pool=='Unidentified' else 'finite hypothetical reservation ceiling'),'Reserved Rate':used,'Unused Scenario Allowance':float(max(0.,capvalue-used)) if np.isfinite(capvalue) else None,'Activated Model Intakes':int((assigned==pool).sum()),'Actual Discharge Known':False})
  for case,s in states.items():
   q=s['excess']/horizon;served=np.minimum(q,rate[None,:]);unassigned=q-served
   no_link=q[:,~haslink].sum(axis=1);unitgap=np.maximum(q[:,haslink]-.5,0.).sum(axis=1);allocationgap=(np.minimum(q[:,haslink],.5)-served[:,haslink]).sum(axis=1)
   assert np.allclose(unassigned.sum(axis=1),no_link+unitgap+allocationgap,rtol=0,atol=1e-10)
   local=[]
   for frame in range(109):
    row={'Portfolio':ident,'Case':case,'Frame':frame,'Actual Time':float(s['actual'][frame]),'Analysis Time':float(s['analysis'][frame]),'Snapshot Reference Requested Rate':float(q[frame].sum()),'Snapshot Reference Assigned Rate':float(served[frame].sum()),'Snapshot Reference Unassigned Rate':float(unassigned[frame].sum()),'No Robust Named Link Request':float(no_link[frame]),'Unit Ceiling Gap':float(unitgap[frame]),'Reservation Gap':float(allocationgap[frame]),'Snapshot Proxy Unassigned Volume':float(unassigned[frame].sum()*horizon),'Actual Pumped Volume':None,'Actual Unmet Drainage Requirement':None}
    local.append(row);traces.append(row)
   peak=int(np.argmax(q.sum(axis=1)));peakrow=local[peak]
   summaries.append({'Portfolio':ident,'Case':case,'Reference Horizon':horizon,'Receiving Scenario':label,'Straight Distance Screen':length,'Activated Model Intakes':len(chosen),'Reserved Rate':float(rate.sum()),'Peak Snapshot Reference Requested Rate':peakrow['Snapshot Reference Requested Rate'],'Assigned Rate At Same Peak':peakrow['Snapshot Reference Assigned Rate'],'Unassigned Rate At Same Peak':peakrow['Snapshot Reference Unassigned Rate'],'Proxy Unassigned Volume At Same Peak':peakrow['Snapshot Proxy Unassigned Volume'],'Reference Peak Analysis Time':peakrow['Analysis Time'],'Actual Road Closure Reduction':None,'Actual Unmet Drainage Requirement':None})
  for i in np.flatnonzero(positive_any&(~haslink|(rate<request_max-EPS_RATE))):
   reason='source evidence held' if not sourceok[i] else ('no named link passing every positive-demand saved frame within distance' if not haslink[i] else ('blocked receiving assumption' if cap==0 else 'partly or wholly unreserved under this greedy budget allocation'))
   gap_rows.append({'Portfolio':ident,'Source Candidate':str(sources['Source Candidate'].iloc[i]),'Grid Index':int(grid[i]),'Reason':reason,'Maximum Local Reference Excess':float(max_excess[i]),'Reference Max Local Rate Request':float(request_max[i]),'Unavoidable Unmet Drainage':None})
  overview.append({'Portfolio':ident,'Receiving Scenario':label,'Reference Clearance Horizon':horizon,'Straight Distance Screen':length,'Standalone Rate Ceiling':.5,'Static Lift Cutoff':5.,'Depth Trigger':.05,'Road Source Candidates':n,'Positive Local Reference Sources':int(positive_any.sum()),'Robust Named Link Sources':int(haslink.sum()),'Activated Model Intakes':len(chosen),'Reserved Rate':float(rate.sum()),'Median Reserved Intake Rate':float(np.median(rate[chosen])) if len(chosen) else None,'Maximum Reserved Intake Rate':float(rate.max()),'Machine Count':None,'Physical Deployment Verified':False,'New Hydraulic Runs':0})
  print('ALLOCATED',ident,len(chosen),float(rate.sum()),flush=True)
 allframes=pd.DataFrame(traces);ov=pd.DataFrame(overview);sel=pd.DataFrame(selected);pooldf=pd.DataFrame(pool_alloc);cs=pd.DataFrame(summaries)
 for ident,a in sel.groupby('Portfolio'):
  assert not a['Source Candidate'].duplicated().any();assert a['Reserved Rate'].gt(0).all() and a['Reserved Rate'].le(.5+1e-12).all();assert np.isclose(a['Reserved Rate'].sum(),ov.set_index('Portfolio').loc[ident,'Reserved Rate'],rtol=0,atol=1e-10)
 for ident in ov.loc[ov['Receiving Scenario']=='Blocked','Portfolio']:assert (allframes.loc[allframes.Portfolio==ident,'Snapshot Reference Assigned Rate']==0).all()
 # Same-name model groups 79 and 111 must share one budget throughout.
 f=pooldf[(pooldf['Receiving Pool']=='Furukawa')&pooldf['Scenario Shared Allowance'].notna()];assert (f['Reserved Rate']<=f['Scenario Shared Allowance']+1e-11).all() and f['Model Groups'].eq('79,111').all()
 tables={'receiving_pool_registry':pd.DataFrame(pool_rows),'portfolio_summary':ov,'selected_intakes':sel,'shared_receiver_reservations':pooldf,'snapshot_allocation_trajectories':allframes,'case_allocation_summary':cs,'unassigned_source_register':pd.DataFrame(gap_rows)}
 exports=[];dictionary=[]
 for name,a in tables.items():
  a=a.reset_index(drop=True);assert all(c.isascii() and c==c.strip() for c in a);f=P/(name+'_preprocessed.parquet');a.to_parquet(f,index=False);pd.testing.assert_frame_equal(a,pd.read_parquet(f));exports.append({'path':str(f.relative_to(ROOT)),'rows':len(a),'columns':len(a.columns),'sha256':sha(f)})
  for c in a:
   unit='identifier,category,count or boolean'
   if any(t in c for t in ['Rate','Allowance','Request','Gap']):unit='m3/s;offline reservation or proxy,not actual discharge'
   if any(t in c for t in ['Volume','Excess','Drainage']):unit='m3;proxy or unresolved'
   if any(t in c for t in ['Horizon','Analysis Time','Actual Time']):unit='s'
   if any(t in c for t in ['Distance','Lift','Trigger']):unit='m'
   dictionary.append({'Dataset':f.name,'Readable Field':c,'Unit':unit,'Final Variable':'no','Role':'conditional allocation reference;not a hydraulic result'})
 pd.DataFrame(dictionary).to_csv(R/'data_dictionary.csv',index=False)
 solver=json.loads((ROOT/'data/processed/three_ward_barrier_drainage_v1/solver_binding.json').read_text())['image_id']
 protocol={'status':'offline_allocated;execution_disabled','target':'Combined three-ward road-control candidates only;original masks and0.30m threshold','reference_demand':{'formula':'D_case,i,t=max(saved_volume_case,i,t-target_storage_i,0)/T;T=1800s with3600s sensitivity','meaning':'clear present local snapshot excess over assumed horizon without feedback;not true required drainage or accepted service guarantee','initial_selection':'max local rate request over both stages and all109saved maps;used to reserve static rate and rank sites,not summed as simultaneous demand','scope':'three-ward cells;buffer context retained upstream,no new buffer population inference'},'algorithm':{'order':'descending max local request,then ascending grid ID','outlet':'closest named receiving alternative with remaining shared reservation;one outlet per source','rate':'min(0.5m3/s,max local reference request,remaining shared allowance)','claim':'deterministic greedy reference,not an optimum or minimum-intake proof','duty':'link must pass static source,trigger and conservative lift screens at every saved frame with positive reference excess in both July stages;part-time alternatives held','static_reservations':'sum nominal site rates within each pool never exceeds its assumed shared cap;actual pump operator can only reduce site rate,so a new dynamic receiver allocator is not needed for this conservative static policy','unused_capacity':'idle reservations are not automatically reassigned;no minimum-yield guarantee','counts':'active native intake locations are derived;10m-local demand discretization affects counts;not physical machine count'},'receiving':{'pools':pool_rows,'allowances':'open conditional,0.5or2m3/s per named pool,blocked;all hypothetical','unidentified':'no allocation to model group51;its real capacity is unknown,not asserted zero','downstream':'different named rivers can connect;independent scenario caps do not prove downstream acceptance','Furukawa':'groups79and111 share ONE pool,not one allowance each'},'physical_run_inputs':{'image_id':solver,'files':'isolated review-only udt_pumps.dat files use one-based native IDs and original pump operator;no barrier files introduced','static_cap_enforcement':'sum fixed requested rates within shared pool provides an upper bound on actual simultaneous disposal under unchanged clipped pump operator','not_staged_into_runs':True,'execution_enabled':False},'limitations':['source-cell clearance proxy does not pool neighboring storage;counts may be large because of10m fragmentation','actual extraction changes source level,head,ground and river exchange;baseline curve allocation is not transport benefit','cannot sum unassigned volumes over repeated frames or call them actual remaining flood water','one parent with two stages informed selection;not independent historical validation','route distance remains Euclidean lower bound,field access and permission unknown','0.30m target is a closure boundary;strict h<0.30 service and safety margin require simulation checks','unselected or held cells may still benefit indirectly;reference gap is not unavoidable unmet drainage'],'new_hydraulic_runs':0,'training':0,'population_counts':False,'AnaSOP_changed':False,'next_step':'preflight a small paired-stage deployment batch with original numerical/source-sink gates,then hydraulically evaluate actual road and transfer impacts;do not treat offline proxy gaps as final outcomes'}
 dump(R/'protocol.json',protocol)
 dump(R/'decisions.json',{'human_basis':'requirements-led deployment and current authorization to execute offline allocation','new_agent_assumptions':['30min local snapshot clearance proxy;60min sensitivity','greedy static reservation rule','merge same-name Furukawa groups;withhold unidentified receiver'],'final_human_deployment_freeze':False,'reference_only':True,'counts_not_prespecified':True})
 assert all(sha(ROOT/f)==h for f,h in binding.items())
 dump(P/'manifest.json',{'input_sha256':binding,'candidate_source_manifest_sha256':sha(C/'manifest.json'),'upstream_physical_source_bindings':up['input_sha256'],'script_sha256':sha(Path(__file__)),'outputs':exports,'pump_inputs':artifact_bindings})
 validation={'passed':True,'portfolios':len(portfolios),'qualified_input_cases':cases,'new_hydraulic_runs':0,'same_named_Furukawa_groups_share_one_limit':True,'no_unidentified_receiver_allocated':True,'static_sum_within_shared_caps':True,'single_outlet_and_single_entry_per_source':True,'per_site_rate_within_cap_and_reference_request':True,'positive_frame_allocation_link_screens_checked':True,'snapshot_rate_gap_decomposition_closes':True,'blocked_portfolios_zero_and_reuse_baselines':True,'serialized_inputs_roundtrip_exact':True,'intake_count_below_existing_operator_limit':True,'parquet_roundtrip_exact':True,'source_hashes_and_AnaSOP_unchanged':True,'actual_benefits_and_true_unmet_drainage_unknown':True,'no_simulation_or_population_benefit_claim':True};dump(R/'validation.json',validation)
 # Standalone offline review;no new publication figure or formal variable promotion.
 display=ov[['Portfolio','Reference Clearance Horizon','Receiving Scenario','Activated Model Intakes','Reserved Rate','Median Reserved Intake Rate']].copy();view=cs[(cs['Case']=='E20240731__H__U')&(cs['Reference Horizon']==1800.)][['Portfolio','Peak Snapshot Reference Requested Rate','Assigned Rate At Same Peak','Unassigned Rate At Same Peak']]
 display=display.rename(columns={'Portfolio':'方案','Reference Clearance Horizon':'清除时限(s)','Receiving Scenario':'接收条件','Activated Model Intakes':'模型取水位置数','Reserved Rate':'预留总流量(m3/s)','Median Reserved Intake Rate':'单点预留中位数(m3/s)'})
 view=view.rename(columns={'Portfolio':'方案','Peak Snapshot Reference Requested Rate':'峰值参考请求(m3/s)','Assigned Rate At Same Peak':'同帧已分配(m3/s)','Unassigned Rate At Same Peak':'同帧未分配(m3/s)'})
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>共享接收额度与条件部署分配</title><style>body{{font:16px/1.8 system-ui;max-width:1340px;margin:30px auto;padding:0 24px;color:#233e4b}}aside{{background:#e9f3f5;border-left:4px solid #477e91;padding:18px}}.warn{{background:#fff1d8;padding:16px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #d1dce0;padding:8px;text-align:left}}</style><h1>共享接收额度与条件部署分配</h1><aside>已完成10套离线分配：500／1000m连接条件，各自比较开放、共享0.5、共享2m³/s与受阻，再增加两套60分钟需求代理对照。<b>位置数量从分配结果产生；没有运行新水动力模拟。</b></aside><h2>1. 本轮究竟在分配什么</h2><p>真实排水需求尚不能从既有基线直接确定。本轮明确采用条件代理：把某时刻道路单元超过原0.30m目标对应的储量，除以假设的30分钟清除时限，得到该快照的参考流量；另以60分钟对照。没有计入抽排后持续来水、自然出流及河道倒灌的反馈，也没有把局部峰值相加称作同时总需求。两种时限均为本轮提出的分析假设，不是已认可的公共服务标准。</p><p>每个位置按两组既有水位情景、全部109帧中的最大本地参考请求预留能力，上限0.5m³/s；优先最大请求较高的位置，优先选择仍有额度的最近具名接收连接。一个位置只分到一个接收端；位置必须在两组所有正需求保存帧通过触发及扬程条件，暂不利用间歇可用连接。该贪心方案可复现，未声称最优或最少位置。</p><h2>2. 接收额度没有重复分配</h2><p>古川模型组79与111合并为一个共享额度；其余具名接收池为隅田川、亀島川、日本橋川、神田川。每个池的0.5／2m³/s均是假设总额度。未命名组51身份不清，本轮暂不分配，不把未知真实容量记成零。不同具名河流可能汇入同一下游；这些情景额度不证明实际下游可接收。</p><p>采用静态预留后，池内所有位置设定流量之和不超过额度。原算子受储量、启动水深与扬程约束，只会降低实际流量；所以这一保守方案可通过固定位置流量上限表达共享约束，无须为本轮额外实现动态抢占分配器。某位置停泵时额度不会自动转给别的位置。</p><h2>3. 推导出的取水位置和预留流量</h2><div class="scroll">{display.to_html(index=False,float_format=lambda v:f'{v:,.5f}',na_rep='未确定')}</div><p>时限单位s，流量单位m³/s。开放条件没有接收额度上限，只作为条件参照，并非建议实施的无限资源方案。没有人为固定位置数量、点间距或全域密度。管线直线距离仍是长度下界。</p><div class="warn"><b>表内是10m模型中的激活取水位置数，不能当成设备台数或工程部署建议。</b>本地单元需求代理没有把相邻单元自动合并为共同集水池，因此可能产生很多小流量位置；中位预留流量用于暴露这种碎片化。真实机器可以服务多个进水口，也可能需要多机协同，这些组织关系尚未建模。</div><h2>4. 未分配的参考需求</h2><p>以下仅展示高水位、30分钟代理在同一全域需求峰值时刻的流量分解（m³/s）。完整两组109帧均保存在数据中。未分配部分分成：没有全时段合格具名连接、单点能力上限不足、共享额度／静态预留未分到。它们是参考缺口，不是实际无法排走的水量。</p><div class="scroll">{view.to_html(index=False,float_format=lambda v:f'{v:,.4f}')}</div><p>未分配参考体积仅为该帧差额乘时限，不能跨帧累加，也不能声称它是措施实施后剩余积水。邻近取水点可能间接排走未分配单元的水；实际封路、拥堵、人口影响和风险转移仍需水动力及交通对照。</p><h2>5. 已经具备哪些可执行输入</h2><p>每套非受阻方案都生成独立的udt_pumps.dat审阅文件，保留一基原生取水／接收单元、分配流量、原0.05m启动水深和5m静扬程截断。文件逐行回读核验；没有写入原运行目录，也没有增加挡水设施。受阻方案复用原基线，不生成算子不接受的零泵输入。</p><p>下一步可选择小批次方案做预运行检查，再用同样两组水位核验真实抽排量、守恒正性、封路改善及新增积水。只有实际新输出通过原规则后，才能接交通和人口影响；本轮不报告新增受影响人数。</p><h2>数据与核验</h2><ul>'''
 for e in exports:page+=f'<li><a href="../../../processed/{TAG}/{Path(e["path"]).name}">{html.escape(Path(e["path"]).name)}</a>（{e["rows"]:,}行）</li>'
 for e in artifact_bindings:page+=f'<li><a href="../../../processed/{TAG}/portfolios/{Path(e["path"]).parent.name}/udt_pumps.dat">{html.escape(Path(e["path"]).parent.name)}：泵算子审阅输入</a></li>'
 page+='</ul><p><a href="validation.json">核验</a> · <a href="protocol.json">规则及限制</a> · <a href="data_dictionary.csv">字段单位</a> · <a href="README.md">复现</a> · <a href="../three_ward_intake_receiver_candidates_v1/review.html">候选连接证据</a></p></html>'
 (R/'review.html').write_text(page)
 (R/'README.md').write_text('''# Shared receiving allocation reference\n\nRun `.venv/bin/python src/analyses/allocate_shared_receiver_portfolios_v1.py` from the project root. Reads only existing candidate/stage/storage arrays. No hydrodynamics,training,population analysis,sealed-test access or Git/DVC operations. All outputs are non-final reference variables;AnaSOP is unchanged.\n\nLocal snapshot request is positive stored volume above existing closure-boundary target divided by an assumed1800s clearance horizon,with3600s sensitivity. Source selection uses local maxima across JulyH/L and saved times for static reservation,not a simultaneous-volume claim. Source outlets must pass every positive-demand saved frame in both cases. Per-source request/cap,shared receiver capacity and closest available outlet drive a descending-demand greedy allocation,with deterministic ties. It is not an optimization or minimum-location proof.\n\nFurukawa79and111 share one pool. Unidentified51 is held. Named pools have hypothetical unconstrained,0.5or2m3/s,orblocked acceptance. Sum nominal fixed source rates respects each shared cap;existing physical operator only clips actual transfer downward. Idle reservations are not automatically reallocated. Physical downstream acceptance,field alignment,permissions and machine count are unresolved.10m local demand discretization can create many small native intakes. Reference unassigned rates/volumes are not actual unmet drainage or flood/transport benefits. Do not sum repeated snapshots.\n\nReview-only one-based native pump files are generated in portfolios/,never copied into existing solver runs. Blocked cases reuse baseline with no pump file. No new barriers. Original geometry,forcing,roof assignment,clocks,labels and masks are preserved. A new paired-stage hydraulic preflight/campaign must verify unchanged numerical gates plus explicit pump source/sink balance before claiming effects.\n''')
 print(ov.to_string(index=False));print('SHARED_RECEIVER_ALLOCATION_COMPLETE',len(sel),len(allframes),flush=True)
if __name__=='__main__':main()
