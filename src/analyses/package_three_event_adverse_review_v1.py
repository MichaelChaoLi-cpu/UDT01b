#!/usr/bin/env python3
"""Package completed diagnostic evidence and a proposed, unconfirmed plan refresh."""
from pathlib import Path
import json,html
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'data/processed/three_ward_local_adverse_diagnosis_v1';R=ROOT/'data/exp/estimation'/P.name;PLAN=ROOT/'data/exp/research-design/three_ward_transport_contract_refresh_r3'

def table(d):return '<div class="scroll">'+d.to_html(index=False,escape=True,float_format=lambda v:f'{v:,.4f}')+'</div>'
def page(title,body):return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>body{font:16px/1.8 system-ui;color:#203b4b;background:#f6f8fa;max-width:1250px;margin:30px auto;padding:0 22px}h1{font-size:28px}aside{background:#e4eff4;padding:18px}.warning{background:#fff0d9;padding:15px}.scroll{overflow:auto;max-width:100%}table{border-collapse:collapse;background:white;font-size:13px;width:100%}td,th{padding:8px;border:1px solid #d1dde5;text-align:left}a{color:#186b98}code{overflow-wrap:anywhere}</style><body><h1>'+title+'</h1>'+body+'</body></html>'

def selection_assessment(roads,towns,head):
 summary=json.loads((ROOT/'data/exp/estimation/three_ward_shared_receiver_controls_august_v1/three_event_synthesis.json').read_text());rows=[]
 for row in summary['rows']:
  event=row['event'];stage=row['stage'];record=dict(row);record.update(road_only_preference='S1',population_preference='unresolved_interval_spans_zero',actual_congestion_comparison='not_computed_for_frozen_portfolios')
  for arm in ['O1','S1']:
   g=roads[roads.event.eq(event)&roads.stage.eq(stage)&roads.reference.eq('No action')&roads.treatment.eq(arm)&roads.bound.eq('lower')].iloc[0]
   record[arm+'_new_closure_vs_noaction_km_h']=float(g.new_closed_km_h);record[arm+'_new_closure_small_shift_fraction']=float(g.new_road_closure_with_depth_increase_le_0p01_km_h/g.new_closed_km_h)if g.new_closed_km_h>0 else None
   h=head[head.event.eq(event)&head.stage.eq(stage)&head.treatment.eq(arm)];record[arm+'_certain_head_rise_max_simultaneous_m2']=int(h.groupby('frame').grid_index.nunique().max()*100)if len(h)else 0
  t=towns[towns.event.eq(event)&towns.stage.eq(stage)&towns.reference.eq('O1')&towns.bound.eq('lower')&towns.buffer_m.eq(10)&towns.net_upper_resident_hours.lt(-1e-8)]
  record['primary_negative_net_towns_S1_vs_O1']=';'.join(t.name.drop_duplicates());record['next_design_status']='candidate local access and road-group safeguards;no new layout selected';rows.append(record)
 pd.DataFrame(rows).to_csv(P/'measure_selection_assessment.csv',index=False)
 assert not (towns.groupby(['event','stage','reference','treatment','bound','buffer_m','town_id']).name.nunique()>1).any()

def main():
 assert json.loads((R/'validation.json').read_text())['passed'];PLAN.mkdir(parents=True,exist_ok=True)
 s=pd.read_csv(P/'road_pair_summary.csv');roads=pd.read_csv(P/'new_closure_roads.csv');towns=pd.read_csv(P/'town_population_effects.csv');groups=pd.read_csv(P/'road_group_effects.csv');heads=pd.read_csv(P/'native_head_rise_samples.csv')
 selection_assessment(s,towns,heads)
 primary=s[s.bound.eq('lower')&s.reference.eq('No action')];rn=roads[roads.bound.eq('lower')&roads.reference.eq('No action')&roads.length_km.gt(0)]
 assert len(rn)==4 and rn.max_depth_increase_at_new_closure_m.max()<.0013
 assert np.allclose(rn.new_closed_duration_h,1/12,atol=1e-7)
 main_towns=towns[towns.bound.eq('lower')&towns.buffer_m.eq(10)&towns.reference.eq('O1')]
 adverse_towns=main_towns[main_towns.new_loss_upper_resident_hours.gt(0)].sort_values('new_loss_upper_resident_hours',ascending=False)
 robust=[]
 for name in ['一番町','三番町']:
  t=towns[towns.event.eq('E20240830')&towns.reference.eq('O1')&towns.name.eq(name)]
  assert len(t)==12 and t.net_upper_resident_hours.lt(0).all()
  robust.append({'Town':name,'Event':'E20240830','Reference':'O1','Treatment':'S1','Sensitivity rows':len(t),'Net benefit direction':'negative across both depth branches,all three radii and both stages'})
 gt=groups[groups.bound.eq('lower')&groups.reference.eq('O1')&groups.new_closed_km_h.gt(0)].sort_values('new_closed_km_h',ascending=False)
 # Native-cell duration is a sampled indicator integral; union is explicitly not maximum simultaneous area.
 head_rows=[]
 for keys,g in heads.groupby(['event','stage','treatment','town_name']):
  head_rows.append(dict(zip(['Event','Stage','Treatment','Town'],keys),Unique_screened_cells=g.grid_index.nunique(),Any_time_union_m2=100*g.grid_index.nunique(),Maximum_simultaneous_m2=100*g.groupby('frame').grid_index.nunique().max(),First_sampled_hour=g.elapsed_s.min()/3600,Last_sampled_hour=g.elapsed_s.max()/3600,Maximum_certain_rise_m=g.certain_head_rise_m.max()))
 head=pd.DataFrame(head_rows);head.to_csv(P/'native_head_rise_town_summary.csv',index=False)
 # Reconstruct paired regional bounds by summing blocks, not subtracting marginal intervals.
 cols=[k for k in towns if k.endswith('_resident_hours')]
 population=towns[towns.bound.eq('lower')&towns.buffer_m.eq(10)].groupby(['event','stage','reference','treatment'],as_index=False)[cols].sum();population.to_csv(P/'primary_population_comparison.csv',index=False)
 summary=dict(passed=True,diagnostic_comparisons=30,new_hydraulic_runs=0,threshold_screen={'primary_noaction_positive_length_road_rows':len(rn),'maximum_depth_increment_m':float(rn.max_depth_increase_at_new_closure_m.max()),'sampled_exposure_duration_h':1/12,'exact_physical_duration_unknown':True,'readme':'All positive-length primary new closures versus no action occur at one saved time and with<1.3mm road-depth increase. No changed closure threshold or result filtering.'},robust_local_access_tradeoffs=robust,network_interpretation='Paired cached reachability confirms changed road configurations have neighborhood consequences. No single-edge causal attribution or new routing ablation was performed.',physical_transfer='Certain native head rise>=0.10m and road-threshold crossing are separate screens,with different locations and timing. Model-only evidence,not observed flooding.',future_constraints='Candidate protection safeguards for persistent losing neighborhoods and road groups;not a selected layout,capacity or authorized simulation.',Anasop='unchanged;targeted revision proposal awaits figure-plan confirmation')
 (R/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 coverage_note='人口主规则下仍有41,308名常住人口因正常接入不完整而暂留；277.920 km未知道路暴露单列，均未当作已验证安全。'
 b='<aside><b>副作用诊断已完成。</b> 三场暴雨、两种水位、五类直接配对，共30个比较；全部109帧、水深上下界和0／10／25 m人口候选规则保留。新增水动力模拟、调参和路由计算均为0。</aside>'
 b+='<h2>1. 相对无措施：新增封路主要是阈值敏感现象</h2><p>主分支中，8月有4条带正长度的道路记录在单个保存时刻跨过0.30 m封路阈值，水深增量为0.027—1.218 mm。表中5分钟为离散采样积分，不是实测或精确持续时长；真实跨阈值起止位于相邻输出之间。保留原结果，不能把这些毫米级变化当作稳健的新增道路灾害证据。另有跨区上下文道路长度在三区裁剪后为零，单列于明细，不计入三区封路长度。</p>'
 b+=table(rn[['event','stage','treatment','road_group','length_km','new_closed_duration_h','max_depth_increase_at_new_closure_m','first_sampled_new_closed_s']])
 b+='<h2>2. 相对O1：失去原保护的片区可以持续退步</h2><p>一番町和三番町在8月的12个敏感性组合（水位×水深分支×接入半径）中，S1相对O1的居民·小时净改善区间上界均小于零。这是对O1的相对退步，不表示比无措施更差。不能用全域封路收益或人口区间中点覆盖这些局部损失。</p>'+table(pd.DataFrame(robust))
 b+='<p>以下列出主规则下新损失上界较大的街区，作为总体审阅索引；完整表保留全部街区、比较方向及不确定性分支。上界不是实际受困人数。</p>'+table(adverse_towns[['event','stage','name','net_lower_resident_hours','net_upper_resident_hours','new_loss_lower_resident_hours','new_loss_upper_resident_hours']].head(15))
 b+='<h2>3. 道路组：相对保护变化与网络响应</h2><p>昭和通り等道路组在S1下失去O1原有的部分保护，其同刻封闭差异持续多个时刻，不能用上述毫米级短暂跨阈值概括。道路组记录与候选接入的可达性变化相符，但现有汇总没有唯一识别哪条边造成哪片区损失；不把空间邻近或相关性写成已证明的单路因果机制。</p>'+table(gt[['event','stage','road_group','road_name','all_registered_members_qualified','new_closed_km_h','net_avoided_km_h','any_new_closure_sampled_h']].head(16))
 b+='<h2>4. 原生水位抬升：独立保留水力副作用</h2><p>达到0.10 m确定水头抬升阈值的模型单元集中在若干街区，包括永田町、霞が関、日本橋等。下表区分任意时刻单元并集和最大同时面积；两者不能互换。人口新损失、道路新封闭和水头抬升未必同地同刻。单元通过原数值准入，不等于已获得现场浸水验证，接收河道新增储水与下游效应仍未模拟。</p>'+table(head)
 b+='<p>'+coverage_note+'</p>'
 b+='<h2>5. 措施选择判断</h2><ul><li>全域封路累计量：S1在三场、六个雨量—水位组合中均优于O1。</li><li>局部接入保护：O1可作为相对稳健的参照；S1需要对持续失去保护的街区和道路组设置候选保护约束。</li><li>阈值敏感记录：原数值与长度保留，发表时与较大水位抬升及持续网络变化分栏。</li><li>下一轮措施尚未选择新位置、额度或数量，本次诊断不启动改进模拟。</li></ul>'
 b+='<p><a href="../../../processed/'+P.name+'/measure_selection_assessment.csv">六个雨量—水位组合的措施选择判断表</a>：区分道路收益、人口未决区间、阈值敏感封路与原生水位抬升；尚未选择新部署。</p>'
 b+='<h2>核验与完整明细</h2><p><a href="validation.json">核验</a> · <a href="summary.json">结论与边界</a> · <a href="../../../../processed/'+P.name+'/road_pair_summary.csv">全部道路比较</a> · <a href="../../../../processed/'+P.name+'/town_population_effects.csv">街区人口配对</a> · <a href="../../../../processed/'+P.name+'/new_closure_roads.csv">新增封路明细</a> · <a href="../../research-design/'+PLAN.name+'/review.html">AnaSOP与图表修订建议</a></p>'
 # relative processed directory: report is data/exp/estimation/TAG, three parents to data.
 b=b.replace('../../../../processed/','../../../processed/')
 (R/'review.html').write_text(page('三区措施收益与局部副作用诊断',b))
 proposal=json.loads((PLAN/'proposal.json').read_text());fig=pd.DataFrame(proposal['figures']);tab=pd.DataFrame(proposal['tables'])
 pb='<aside><b>建议定点更新，不重写研究问题。</b> 状态：proposed_not_confirmed。AnaSOP尚未改动；既有3张done图保持原状，新模拟为0。以下完整清单需确认后才写入第8节。</aside>'
 pb+='<h2>可以继续制图，但先对齐证据范围</h2><p>新结果已支持三场暴雨的道路措施比较、居民·小时区间与局部副作用表达。旧AnaSOP主要描述两场、19个假设抽排单元的早期挡排试验；新冻结共享接收方案为Static／O1／S1、每池0.5 m³/s、总额度2.5 m³/s、62／319个原生取水位置。这些实验不能混成同一部署或同一工程成本。</p>'
 pb+='<h2>全部图：7张，其中3张已认可、4张待生成</h2>'+table(fig.rename(columns={'title':'Title','change':'Proposed change','purpose':'Evidence purpose','type':'Figure type','subpanels':'Panels','key_variables':'Confirmed concept names'}))
 pb+='<h2>全部表：5张待生成</h2>'+table(tab.rename(columns={'title':'Title','change':'Proposed change','rows':'Approximate rows','columns':'Approximate columns','row_meaning':'Row meaning','column_meaning':'Column meaning'}))
 pb+='<h2>AnaSOP定点修订范围与负责skill</h2>'+table(pd.DataFrame([{'Section':k,'Proposed update':v}for k,v in proposal['targeted_AnaSOP_updates'].items()]))
 pb+='<p>第1—2节保留研究问题与概念方向。第3—4节补齐新实验范围、措施配置及有明确范围的水头／水深差异；第5—7节说明跨事件直接配对、阈值敏感性、右删失和局部损失；第8节保留done行、修订pending范围。历史拥堵拟合结果保留原试验范围，O1／S1可达性不会被写成已经计算的拥堵效果。</p>'
 pb+='<h2>生成顺序与风格</h2><p>建议先生成 Cross-Event Allocation and Placement Effects，再做 Resident Access Loss and Restoration、Neighboring Flood Transfer；原试验的 Vehicle Access and Traffic Cost保留独立范围。每张图生成后逐张审阅。英文标注、450 dpi、经纬度网格、四边框及已认可布局继续沿用。</p>'
 pb+='<p class="warning">本页是完整修订提案，不是已确认计划。按figure-table-planning要求，确认后才写入AnaSOP第8节；此后由负责skill依次更新相关章节并核验变量口径，才开始正式新图。拟新增或扩展范围不静默标为done。</p><p><a href="proposal.json">完整JSON提案</a> · <a href="../../estimation/'+R.name+'/review.html">已完成副作用诊断</a></p>'
 (PLAN/'review.html').write_text(page('AnaSOP与图表计划：三场暴雨定点修订提案',pb))
 print('REPORTS_READY',R/'review.html',PLAN/'review.html')
if __name__=='__main__':main()
