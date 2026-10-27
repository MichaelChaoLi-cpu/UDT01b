#!/usr/bin/env python3
"""Prepare reference-only service portfolios from existing qualified outputs; no solver."""
from pathlib import Path
import hashlib,json,html,subprocess,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
TAG='three_ward_protection_service_demand_plan_v1'
P=ROOT/'data/processed'/TAG; R=ROOT/'data/exp/research-design'/TAG
OLD=ROOT/'data/processed/three_ward_protection_requirements_register_v1'
FLUX=ROOT/'data/processed/three_ward_ground_flux_replay_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 P.mkdir(parents=True,exist_ok=True); R.mkdir(parents=True,exist_ok=True)
 sources=[OLD/n for n in ['road_target_groups_preprocessed.parquet','road_group_members_preprocessed.parquet','road_control_cell_group_links_preprocessed.parquet','control_cell_storage_preprocessed.parquet','group_zone_links_preprocessed.parquet','drainage_zone_register_preprocessed.parquet','zone_storage_trajectories_preprocessed.parquet']]
 sources += [FLUX/'zone_ground_budget_trajectories_preprocessed.parquet',ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/input_inventory.csv',ROOT/'docs/AnaSOP.md',ROOT/'data/exp/research-design/three_ward_requirements_led_measures_v2/plan.json']
 extra_inventory=pd.read_csv(sources[8])
 for row in extra_inventory.itertuples():
  if '20240731' in row.case_id:sources.extend([ROOT/row.map_path,OLD/'zones'/(row.case_id+'.npz')])
 sources.append(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz')
 binding={str(p.relative_to(ROOT)):sha(p) for p in sources}
 g=pd.read_parquet(sources[0]);members=pd.read_parquet(sources[1]);cl=pd.read_parquet(sources[2]);control=pd.read_parquet(sources[3]);links=pd.read_parquet(sources[4]);zones=pd.read_parquet(sources[5]);storage=pd.read_parquet(sources[6]);flux=pd.read_parquet(sources[7]);inv=pd.read_csv(sources[8])
 backbone=g['Backbone Class Candidate']; access=g['Nonincident Anchor Loss Under Group Probe']>0
 policies={'Backbone':set(g.loc[backbone,'Road Group']),'Area Access':set(g.loc[access,'Road Group'])}
 policies['Combined']=policies['Backbone']|policies['Area Access'];policies['Context']=set(g['Road Group'])-policies['Combined']
 assert len(policies['Backbone'])==213 and len(policies['Area Access'])==577 and len(policies['Combined'])==670
 grouping=g.copy()
 for name,ids in policies.items():grouping[name+' Candidate']=g['Road Group'].isin(ids)
 grouping['Target Decision']='proposed policy alternative;not a frozen deployment list'
 grouping['Network Meaning']='structural group-union probe;not simultaneous observed isolation or people'
 summaries=[];cells=[]
 for name,ids in policies.items():
  v=g[g['Road Group'].isin(ids)];m=members[members['Road Group'].isin(ids)]
  assert not m['Road Index'].duplicated().any()
  grids=np.sort(cl.loc[cl['Road Group'].isin(ids),'Grid Index'].unique())
  c=control[control['Grid Index'].isin(grids)].copy();assert len(c)==len(grids)
  c.insert(0,'Policy',name); c['Target Meaning']='minimum road elevation+0.30 across all linked groups;conservative shared-cell screening proxy,not policy-specific certified target'
  cells.append(c)
  summaries.append({'Policy':name,'Road Groups':len(ids),'Exposure Candidate Length':float(v['Exposure Candidate Length'].sum()),'Qualified Corridor Length':float(v['Qualified Group Length'].sum()),'Chiyoda Corridor Length':float(v['Chiyoda Length'].sum()),'Chuo Corridor Length':float(v['Chuo Length'].sum()),'Minato Corridor Length':float(v['Minato Length'].sum()),'Road Control Cells':len(c),'Frozen Protection List':False})
 policy=pd.DataFrame(summaries);cell=pd.concat(cells,ignore_index=True)
 cell.to_csv(P/'road_control_policy_bridge.csv',index=False)
 # Native helper only reads existing map arrays. It neither imports nor invokes SFINCS.
 native=r'''
from pathlib import Path
import sys,csv
import numpy as np
from netCDF4 import Dataset
root=Path(sys.argv[1]);out=Path(sys.argv[2]);old=root/'data/processed/three_ward_protection_requirements_register_v1'
with (out/'road_control_policy_bridge.csv').open() as f:rows=list(csv.DictReader(f))
policies=sorted(set(r['Policy'] for r in rows));contract={}
with np.load(root/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz') as z:ward=z['Ward Code'].ravel()
for p in policies:
 rr=[r for r in rows if r['Policy']==p];ix=np.array([int(r['Grid Index']) for r in rr]);v=np.array([float(r['Target Storage']) for r in rr]);assert np.isfinite(v).all()
 contract[p]=(ix,v,ward[ix]>0)
with (root/'data/processed/three_ward_five_event_temporal_analysis_v1/input_inventory.csv').open(encoding='utf-8-sig') as f:inventory=list(csv.DictReader(f))
fields=['Case','Policy','Frame','Actual Time','Analysis Time','Three Ward Control Excess Storage','Buffer Control Excess Storage','Selected Control Stored Volume','Selected Control Cells']
with (out/'policy_control_storage_native.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for row in inventory:
  case=row['case_id']
  if '20240731' not in case:continue
  with np.load(old/'zones'/f'{case}.npz') as z:actual=z['actual_times'];analysis=z['analysis_times']
  with Dataset(root/row['map_path']) as d:
   for frame in range(109):
    v=np.flipud(np.ma.asarray(d['subgrid_volume'][frame],dtype='float64').filled(np.nan)).ravel()
    for p,(ix,target,inside) in contract.items():
     vv=v[ix];assert np.isfinite(vv).all() and (vv>=0).all();e=np.maximum(vv-target,0.)
     w.writerow({'Case':case,'Policy':p,'Frame':frame,'Actual Time':float(actual[frame]),'Analysis Time':float(analysis[frame]),'Three Ward Control Excess Storage':float(e[inside].sum()),'Buffer Control Excess Storage':float(e[~inside].sum()),'Selected Control Stored Volume':float(vv.sum()),'Selected Control Cells':len(ix)})
  print('READ_EXISTING_MAP',case,flush=True)
'''
 subprocess.run([str(ROOT/'.frame-control/runtimes/physical-measures-py312/bin/python'),'-c',native,str(ROOT),str(P)],check=True)
 tr=pd.read_csv(P/'policy_control_storage_native.csv',float_precision='round_trip')
 assert len(tr)==872 and not tr.duplicated(['Case','Policy','Frame']).any()
 zlink=[];readiness=[];demand=[];bt=[]
 for name,ids in policies.items():
  zl=links[links['Road Group'].isin(ids)][['Case','Zone']].drop_duplicates();zl.insert(0,'Policy',name);zlink.append(zl)
  for case in inv.case_id:
   zz=set(zl.loc[zl.Case==case,'Zone']);a=flux[(flux.Case==case)&flux.Zone.isin(zz)]
   ready=len(a)>0;assert not ready or a.Zone.nunique()==len(zz)
   readiness.append({'Case':case,'Policy':name,'Linked Case Zones':len(zz),'Boundary Flux Observed':ready,'Road Control Map Screen Available':case in set(tr.Case),'Required Pump Capacity':None,'Readiness':'baseline exchange available;target,head and receiving allowance unresolved' if ready else 'ground exchange not replayed;do not copy July flux'})
   if not ready:continue
   cols=['Stored Volume','Reconstructed Rain Since First Map','Receiving Inflow Since First Map','Receiving Outflow Since First Map','Ground Inflow Since First Map','Ground Outflow Since First Map']
   b=a.groupby('Frame')[cols].sum().reset_index();clock=a.groupby('Frame')['Actual Time'].first();b['Actual Time']=b.Frame.map(clock);b.insert(0,'Policy',name);b.insert(0,'Case',case)
   dt=np.diff(b['Actual Time']);assert (dt>0).all();rain=np.diff(b[cols[1]]);ri=np.diff(b[cols[2]]);ro=np.diff(b[cols[3]]);gi=np.diff(b[cols[4]]);go=np.diff(b[cols[5]])
   net=rain+ri+gi-ro-go
   b['Five Minute Mean Net Loading']=np.r_[np.nan,net/dt]
   b['Meaning']='fixed baseline disjoint zone union;gross exchanges include internal transfers;five-minute mean is not instantaneous equipment sizing'
   bt.append(b)
   t=tr[(tr.Case==case)&(tr.Policy==name)];peak=t.loc[t['Three Ward Control Excess Storage'].idxmax()];last=b.iloc[-1]
   demand.append({'Case':case,'Policy':name,'Linked Case Zones':len(zz),'Maximum Simultaneous Three Ward Road Control Excess':float(peak['Three Ward Control Excess Storage']),'Road Control Excess Peak Analysis Time':float(peak['Analysis Time']),'Buffer Control Excess At Core Peak':float(peak['Buffer Control Excess Storage']),'Maximum Five Minute Mean Net Loading':float(np.max(net/dt)),'Reconstructed Rain On Linked Zones':float(last[cols[1]]),'Receiving Inflow On Linked Zones':float(last[cols[2]]),'Receiving Outflow On Linked Zones':float(last[cols[3]]),'Net Ground Inflow On Linked Zones':float(last[cols[4]]-last[cols[5]]),'Required Pump Capacity':None,'Interpretation':'storage and baseline loading screen only;no guaranteed capacity bound under changed flow'})
 # Combined/common control-cell minimum can be more restrictive than an isolated policy target.
 d=pd.DataFrame(demand);rdy=pd.DataFrame(readiness);zlink=pd.concat(zlink,ignore_index=True);bt=pd.concat(bt,ignore_index=True)
 for case in tr.Case.unique():
  a=tr[tr.Case==case].pivot(index='Frame',columns='Policy',values='Three Ward Control Excess Storage')
  assert np.all(a.Combined+1e-8>=a.Backbone) and np.all(a.Combined+1e-8>=a['Area Access'])
  combined=tr[(tr.Case==case)&(tr.Policy=='Combined')];context=tr[(tr.Case==case)&(tr.Policy=='Context')]
  # Shared coarse cells may be present in both Combined and Context. Never sum these policies.
  allscreen=storage[storage.Case==case].groupby('Frame')['Three Ward Control Excess Storage'].sum()
  assert np.max(combined['Three Ward Control Excess Storage'].to_numpy()-allscreen.to_numpy())<1e-5
 tables={'road_policy_membership':grouping,'policy_scope_summary':policy,'policy_control_cells':cell,'policy_zone_membership':zlink,'policy_control_storage_trajectories':tr,'policy_baseline_zone_budget_trajectories':bt,'policy_demand_screen':d,'policy_readiness':rdy}
 exports=[];dictionary=[]
 for n,a in tables.items():
  assert all(c.isascii() and c==c.strip() for c in a.columns)
  path=P/(n+'_preprocessed.parquet');a.to_parquet(path,index=False);pd.testing.assert_frame_equal(a,pd.read_parquet(path))
  exports.append({'path':str(path.relative_to(ROOT)),'rows':len(a),'columns':len(a.columns),'sha256':sha(path)})
  for c in a:
   unit='reference category,identifier,count or boolean'
   if 'Length' in c:unit='km'
   if any(x in c for x in ['Storage','Volume','Rain Since','Rain On','Inflow','Outflow']):unit='m3'
   if 'Loading' in c:unit='m3/s;five-minute interval mean'
   if 'Time' in c:unit='seconds since scenario start'
   if c=='Required Pump Capacity':unit='m3/s;unresolved,not zero'
   if 'Water Level' in c:unit='m in existing model vertical datum'
   if 'Threshold' in c:unit='m water depth'
   if c=='Target Above Native Minimum':unit='boolean'
   dictionary.append({'Dataset':path.name,'Readable Field':c,'Unit':unit,'Final Variable':'no','Role':'reference service-policy proposal or baseline screening diagnostic'})
 pd.DataFrame(dictionary).to_csv(R/'data_dictionary.csv',index=False)
 plan={'status':'proposal_ready;no deployment or service guarantee','scope':'Chiyoda,Chuo,Minato with original hydraulic and network buffer','policies':{'Backbone':'candidate continuous corridor service;213 groups','Area Access':'candidate bidirectional external-portal access;577 groups','Combined':'union of two policy candidates;670 groups','Context':'1482 other groups retained for transfer-of-risk review'},'overlap_groups':120,'service':{'primary':'strict h<0.30m on required paths;h==0.30 is closed','secondary':'h<0.10m only as a separate stricter scenario;not part of this storage screen','proposed_window_s':[0,32400],'window_meaning':'all saved times over the existing nine-hour scenario;proposal,not a newly accepted public service standard','temporal_limit':'109 maps at five-minute spacing;do not claim uninterrupted sub-five-minute passage or exact recovery','depth_branch':'original lower road-depth branch;upper and unknown structural branches remain separate','network_test':'continuous eligible backbone routes and bidirectional portal reachability must be evaluated on synchronized measure outputs;group membership is not proof of functioning routes'},'screening':{'cells':'unique admitted road-control cells per policy;shared global minimum target is a conservative proxy relative to policy-only road elevations','target':'road z+0.30 boundary,not strict safe-passage certificate;new numerical margin must be declared before deployment','zone_support':'unique case-specific observed-wet zone union linked to target groups;not a contributing catchment','rate':'five-minute mean baseline net loading,not peak step rate,required capacity or demand guarantee','not_additive':'policy alternatives overlap;do not sum lengths,volumes or gross boundary inflows across alternatives','demand_bounds':'No defensible installed-capacity lower/upper bound is inferred from unmodified baseline flows;receiving/backwater and target-stage feedback must be declared','coverage':'only July H/L have observed ground exchange;other eight cases remain missing'},'deployment_order':['review candidate target/service portfolios','use paired baseline screens to prioritize coupled source-intake-receiver candidates','declare hypothetical effective unit rate/head and open,limited,blocked receiver conditions','derive active locations from feasible allocation and leave unmet service explicit','freeze scenario matrix,budget and original numerical gates before any new measure runs'],'next_deliverable':'source-intake-receiver candidate register and explicit hypothetical capability/receiving envelopes;no predetermined pump count','hydraulic_simulations':0,'training':0,'population_counts':False,'AnaSOP_update':False,'final_variable_promotion':False}
 dump(R/'plan.json',plan)
 dump(R/'decisions.json',{'human_decision':'MILI-D-20261020-018 requirements-led deployment order retained','current_user_authorization':'execute next existing-result planning step','agent_proposals':['two overlapping protection policies and combined comparison','nine-hour saved-time service window'],'all_fields_reference_only':True,'no_new_human_policy_freeze':True,'missing_capacity':'left null;no zero imputation'})
 assert all(sha(ROOT/n)==s for n,s in binding.items())
 dump(P/'manifest.json',{'inputs':binding,'outputs':exports,'script_sha256':sha(Path(__file__)),'native_reader':'embedded read-only netCDF helper;no solver','policy_overlap':120})
 validation={'passed':True,'policies_overlap_explicit':True,'group_counts':{n:len(x) for n,x in policies.items()},'unique_road_ids_within_policy':True,'shared_control_cells_not_summed_across_policies':True,'same_frame_storage_aggregation':True,'road_storage_subset_bounded_by_all_road_storage':True,'ground_diagnostic_cases':sorted(flux.Case.unique().tolist()),'unreplayed_cases_not_imputed':True,'all_source_sha256_unchanged':True,'AnaSOP_unchanged':True,'new_hydraulic_runs':0,'new_training':0,'capacity_not_inferred':True,'parquet_roundtrip_exact':True,'outputs':exports}
 dump(R/'validation.json',validation)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>道路保护目标与排水需求筛查</title><style>body{font:16px/1.8 system-ui;max-width:1300px;margin:30px auto;padding:0 24px;color:#243c47}aside{background:#eaf3f5;border-left:4px solid #428395;padding:18px}.warn{background:#fff2db;padding:16px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #d2dce0;padding:8px;text-align:left}</style><h1>三区道路保护目标与需求筛查</h1><aside>已整理两类保护政策及其联合方案，接入7月31日高／低水位既有结果。<b>本次没有新模拟，也没有先定泵点数。</b>以下是待审阅的政策方案和基线筛查，不是已证实有效的部署。</aside><h2>1. 先保护什么</h2><p>“骨干路连续通行”候选213组；“片区对外连接”候选577组；交集120组，联合670组。其余1482组保留为背景和风险转移核查对象。候选来自五场暴雨、两种水位下曾封闭的道路与正常网络依赖诊断；各组可能跨多个源道路，不能把组数当作街道条数、同时封路数或受困片区数。</p>'''
 page+='<div class="scroll">'+policy.to_html(index=False,float_format=lambda x:f'{x:,.3f}')+'</div><p>长度单位km，Exposure Candidate Length为跨情景曾封闭道路的并集长度；Qualified Corridor Length包含同走廊的干路连接段。两类政策之间长度不可相加。主干道路类别用于提出候选，尚未证明所有候选都必须保护。</p><h2>2. 通行标准与服务时段</h2><p>建议先沿用原关闭标准：要求必要道路或替代路径水深严格小于0.30m；等于0.30m仍关闭。低于0.10m的“不进入减速区”作为更严格的独立方案。服务窗口建议覆盖现有0—9小时全过程的109个保存时刻，结束时仍积水的恢复时间保留右删失。这是本次提出的对照标准，尚未新增为用户已冻结的正式服务承诺。</p><p>骨干方案最终需检查连续路径；片区方案最终需检查到既有外部口的双向可达性。没有要求把577组每条道路全部抽干来替代可达性目标。原道路深度下界、严格道路层、步道补缺层及桥隧未知分支继续保留，未知不视为安全。五分钟地图无法证明两帧间持续可通行。</p><h2>3. 现有水量说明什么</h2><p>下表只在两组已补齐地表交换的7月31日情景计算。道路储量先按政策去重到控制单元，再同一帧求和后取峰值；三区内与缓冲区分开。目标沿用各共享单元所有道路采样高程的最小值加0.30m，因此对某一政策可能偏严格，只是储量筛查代理。其他8组缺少地面交换的部分继续留空。</p>'
 shown=d[d.Policy!='Context'][['Case','Policy','Maximum Simultaneous Three Ward Road Control Excess','Road Control Excess Peak Analysis Time','Maximum Five Minute Mean Net Loading','Required Pump Capacity']]
 combined_rows=d[d.Policy=='Combined'].sort_values('Case');parts=[]
 for row in combined_rows.to_dict('records'):
  parts.append(f"{html.escape(row['Case'])}：{row['Maximum Simultaneous Three Ward Road Control Excess']:,.0f}m³（峰值{row['Road Control Excess Peak Analysis Time']/3600:.2f}小时）")
 page+='<p><b>联合候选的三区道路单元筛查峰值：</b>'+'；'.join(parts)+'。两组峰值都在末帧，不能报告已完全恢复；这些是共享目标代理下的储量筛查结果，不是需要抽走的总水量或措施收益。</p>'
 page+='<div class="scroll">'+shown.to_html(index=False,float_format=lambda x:f'{x:,.3f}',na_rep='未确定')+'</div><p>储量m³，峰值时刻s，净来水强度m³/s（五分钟平均）。净来水是关联固定控制区内降雨＋地表净来水＋河道净交换的基线收支。关联控制区范围比道路控制单元大，可能包含缓冲区；不能把它的强度和道路储量直接配成泵能力。高／低水位控制区包络也不同，不是同一空间范围的纯水位因果差。</p><div class="warn"><b>这些数字不能直接换算成“需要多少泵”。</b>抽排和挡水会改变水位、河道倒灌、自然出流和地面来水；不能假定原通量保持不变。共享单元、政策和控制区不能重复计量；毛入流含内部转移，不能当外部总需求。当前没有给出可保证通行的容量上下界，也没有把峰值体积除以九小时。</div><h2>4. 接下来怎样形成部署</h2><ol><li>按两类目标分别提出源区—取水位置—接收水体的候选连接，并保留道路通行和人员进入条件。</li><li>声明单点有效能力及扬程范围、转输条件、启停规则；有证据用证据，缺少时明确作为情景假设。</li><li>接收端分别设置开放、有限共享排水额度及受阻／扬程不满足情景，不能把规定水位当作接收容量证据。</li><li>在这些条件下分配有效抽排量，再得出激活位置数量及未满足需求。实体泵台数还需设备组合定义。</li><li>部署与预算明确后再运行措施情景，按原数值规则核验封路、可达性、改善与新增积水。</li></ol><p>本轮没有新增人口影响统计、训练或封存测试访问；AnaSOP和已认可图件未修改。局部收支残差仍保留原审计解释，没有以其估算排水需求。</p><h2>数据与复现</h2><ul>'
 for e in exports:page+=f'<li><a href="../../../processed/{TAG}/{Path(e["path"]).name}">{html.escape(Path(e["path"]).name)}</a>（{e["rows"]}行）</li>'
 page+='</ul><p><a href="plan.json">方案合同</a> · <a href="validation.json">核验</a> · <a href="data_dictionary.csv">字段与单位</a> · <a href="README.md">复现说明</a> · <a href="../../estimation/three_ward_ground_flux_replay_v1/review.html">两组通量重放结果</a></p></html>'
 (R/'review.html').write_text(page)
 (R/'README.md').write_text('''# Protection service and demand screen proposal\n\nReference-only,requirements-led policy construction. Reproduce with `.venv/bin/python src/analyses/build_protection_service_demand_plan_v1.py` from the project root. Reads existing qualified maps through the existing physical-measures netCDF runtime; runs no hydrodynamics. All eight exported tables use readable English field names and remain non-final. No AnaSOP sections are updated.\n\nBackbone and external-access policy sets overlap. Combined and Context control-cell supports may also overlap at 10m. Never add policy results. Shared-cell target storage uses the existing minimum across all linked road groups and is a conservative screening proxy relative to policy-only targets. It is evaluated at the closure boundary,which does not certify strict passage.\n\nOnly the two July cases have fully observed ground-face exchange. Policy-zone supports are case-specific wet envelopes,not contributing catchments. Net budget trajectories sum distinct zones and synchronized times;gross exchanges still include internal transfers. Five-minute mean baseline loading and stored excess are not required pump capacity or physical bounds after deployment changes the flow field. No unit count is inferred,no missing diagnostics imputed. A nine-hour saved-time service window is a proposal,not a new human-frozen policy.\n\nNext: source-intake-receiver candidate register,declared hypothetical effective rate/head/receiving envelopes,and derived feasible deployment portfolios,then a separately declared simulation budget and gates.\n''')
 print(json.dumps({'status':'complete','groups':validation['group_counts'],'screen':shown.to_dict('records')},ensure_ascii=False,default=str),flush=True)
if __name__=='__main__':main()
