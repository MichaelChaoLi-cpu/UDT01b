"""Whole-region strict/supplemented ground-path coverage and batch-readiness report."""
from pathlib import Path
import os,sys,json,collections,html,re
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_historical_surface_coverage_v2 import P,R,OLD,OLDR,H,CASES,TIMES,load,read,write,dump,sha
CODES=['13101','13102','13103','boundary_or_unassigned'];WARD={'13101':'千代田区','13102':'中央区','13103':'港区','boundary_or_unassigned':'边界／区属未定'}
STYLE={'robust_modeled_disconnection':('#bd3c3c','Robust modeled disconnection'),'conditional_sensitivity':('#d99b38','Assumption sensitive'),'insufficient_evidence':('#aab4bd','Insufficient evidence'),'stable_modeled_alternative':('#3f9283','Alternative in all branches')}
CASE={'radar_low':'假设低水位','radar_raised':'假设水位抬升'};KIND={'entrance':'站口关闭','station':'整站关闭，列车通过','section':'区段停运'}
def table(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def main():
 val=json.loads((R/'independent_validation.json').read_text());routes=json.loads((R/'route_witness_validation.json').read_text());assert val['all_passed'] and routes['all_passed'];checks=len(val['checks'])+len(routes['checks']);oldbase=read(OLDR/'normal_baseline_and_support.csv');base=read(R/'normal_baseline_and_support.csv');oldclass=read(OLDR/'probe_classification.csv');newclass=read(R/'probe_classification.csv');summary=read(R/'classification_summary.csv');prep=json.loads((R/'preparation_validation.json').read_text());reasons=read(R/'coverage_reason_summary.csv');meta=read(P/'road_segment_inventory.csv');oldmetrics=json.loads((OLDR/'analysis_summary.json').read_text());newmetrics=json.loads((R/'analysis_summary.json').read_text());oldA=load(OLD/'alternative_distances.npz');newA=load(P/'alternative_distances.npz');e=load(P/'surface_exposure.npz')
 eligible0={r['probe_id'] for r in oldbase if r['classification_evidence_eligible']=='True'};eligible1={r['probe_id'] for r in base if r['classification_evidence_eligible']=='True'};assert eligible0<=eligible1;added=eligible1-eligible0
 b0={r['probe_id']:r for r in oldbase if r['unknown_branch']=='optimistic'};b1={r['probe_id']:r for r in base if r['unknown_branch']=='optimistic'}
 cov=[]
 for ward in CODES:
  ids={p for p,r in b1.items() if r['ward']==ward};cov.append(dict(ward=ward,probes=len(ids),eligible_before=len(ids&eligible0),eligible_after=len(ids&eligible1),newly_eligible=len(ids&added),insufficient_before=len(ids-eligible0),insufficient_after=len(ids-eligible1)))
 write(R/'paired_ward_coverage_change.csv',cov)
 # Preserve exact comparison units; added support is not a change in rainfall or flooding.
 context={r['probe_id']:r['towns_intersecting_display_cell'] for r in read(ROOT/'data/exp/estimation/three_ward_historical_transport_islands_v1/probe_accessibility.csv')};index0={(r['case_id'],r['frame_id'],r['shutdown_type'],r['probe_id']):r for r in oldclass};index1={(r['case_id'],r['frame_id'],r['shutdown_type'],r['probe_id']):r for r in newclass}
 change=[]
 for k,r in index1.items():
  p=r['probe_id'];prior=index0[k];change.append(dict(**r,classification_before=prior['classification'],cohort='original_supported' if p in eligible0 else 'newly_supported' if p in added else 'still_insufficient',classification_changed=prior['classification']!=r['classification'],display_cell_town_context=context[p],context_not_whole_town=True))
 write(R/'probe_support_and_class_change.csv',change);write(R/'newly_supported_points_19jst.csv',[r for r in change if r['cohort']=='newly_supported' and r['frame_id']=='14400' and r['shutdown_type']=='section'])
 shared=[]
 for case in CASES:
  for kind in ['entrance','station','section']:
   for phase in ['14400','peak_envelope']:
    for cohort,ids in [('original_supported',eligible0),('newly_supported',added)]:
     for version,idx in [('strict',index0),('supplemented',index1)]:
      rr=[idx[case,phase,kind,p] for p in ids];count=collections.Counter(r['classification'] for r in rr);shared.append(dict(case_id=case,frame_id=phase,shutdown_type=kind,cohort=cohort,projection_version=version,cohort_size=len(ids),**{k:count[k] for k in STYLE}))
 write(R/'common_cohort_classification_comparison.csv',shared)
 # Show normal-path dependence on Tier B separately; existence is conditional, not surveyed truth.
 witnesses=[json.loads(x) for x in (P/'surface_exit_route_witnesses_19jst.jsonl').open()];depend=[]
 for case in CASES:
  rr=[r for r in witnesses if r['case_id']==case and r['reachable']];depend.append(dict(case_id=case,frame_id='14400',road_threshold_m=.3,unknown_branch='optimistic',reachable_surface_exit_paths=len(rr),paths_using_supplement=sum(r['supplemented_segments_on_path']>0 for r in rr),paths_using_remaining_unknown=sum(r['unknown_exposure_segments_on_path']>0 for r in rr),not_verified_safe_routes=True))
 write(R/'path_support_dependency.csv',depend)
 facts=dict(technical_qualification_complete=True,scoped_formal_scenario_batch_recommended=True,unqualified_full_region_absolute_inference_ready=False,formal_full_region_batch_ready=False,new_hydraulic_batch_started=False,
 rationale='The same two hydraulic outputs,12 actual clocks and624 network cases are reproduced.115 original supported probes are retained and29 added across all three wards. Strict and supplemented support versions must remain paired;65 probes remain insufficient, including unknown vertical and unmodeled support. This supports formal conditional scenario comparisons, not absolute city isolation or population inference.',
 required_scope=['Use supported cohorts and report full209-point coverage separately; separate original115 and newly supported29.','Keep strict classified-road and supplemented-ground projection as explicit support-model comparisons.','Preserve both station identities, unknown-availability branches, road thresholds and original per-branch normal baselines.','Keep bridge/tunnel/deck elevations unknown; do not label unknown or excluded regions safe.','Use synchronous states for access claims; keep peak envelopes separate.','No population estimates, observed-closure claims, travel-time or safe-refuge claims.'],
 next_experiment='Freeze this input/output version and a finite rainfall/stage/shutdown/urban-geometry comparison matrix before authorizing new hydraulic batches; active measures follow those baselines. No machine learning prerequisite.',
 new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False)
 dump(R/'batch_readiness.json',facts)
 # Whole-domain fixed19:00 comparisons. Newly eligible points have a black outline.
 s=load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');wc=s['Ward Code'];h=load(H/'paired_exposure_bounds.npz');extent=[-11.36,-3.66,-42.44,-32.32]
 def decorate(ax):
  for code,name in [(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
   ax.contour((wc==code).astype('f4'),levels=[.5],extent=extent,origin='upper',colors='#60717d',linewidths=.65);rr,cc=np.where(wc==code);ax.text((-11360+(cc.mean()+.5)*10)/1000,(-32320-(rr.mean()+.5)*10)/1000,name,fontsize=8,ha='center',bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
  ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xlabel='Easting (km), JGD2011 / IX',ylabel='Northing (km)')
 fig,axes=plt.subplots(2,2,figsize=(11,12),sharex=True,sharey=True,layout='constrained')
 for c,case in enumerate(CASES):
  for col,idx in enumerate([index0,index1]):
   ax=axes[c,col];ax.imshow(np.where(h['common_admitted']&(h['snapshot_19jst_lower'][c]>=.3),1.,np.nan),extent=extent,origin='upper',cmap='Blues',vmin=0,vmax=2,alpha=.25)
   rows=[idx[case,'14400','section',p] for p in b1]
   for label,(color,name) in STYLE.items():
    pp=[r for r in rows if r['classification']==label];ax.scatter([float(r['x_m'])/1000 for r in pp],[float(r['y_m'])/1000 for r in pp],s=17,c=color,edgecolors='white',linewidths=.3,zorder=3)
   if col:
    pp=[r for r in rows if r['probe_id'] in added];ax.scatter([float(r['x_m'])/1000 for r in pp],[float(r['y_m'])/1000 for r in pp],s=31,facecolors='none',edgecolors='#162d3a',linewidths=.7,zorder=4)
   ax.set_title(('Low stage' if c==0 else 'Raised stage')+' | '+('Strict classified roads' if col==0 else 'Ground-path supplement')+'\n19:00 JST | service-section suspension',fontsize=10);decorate(ax)
 handles=[Line2D([],[],marker='o',linestyle='',color=color,label=name) for color,name in STYLE.values()]+[Line2D([],[],marker='o',linestyle='',markerfacecolor='none',markeredgecolor='#162d3a',label='Newly supported probe')];fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=9);fig.suptitle('Coverage change under identical flood states, graph and scenario rules\nSupplement is a source-supported projection assumption; points are not island areas',fontsize=12);fig.savefig(R/'strict_supplemented_19jst.png',dpi=155);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');x=np.arange(3);ax.bar(x-.18,[r['eligible_before'] for r in cov[:3]],width=.36,label='Strict classified roads',color='#7f98a8');ax.bar(x+.18,[r['eligible_after'] for r in cov[:3]],width=.36,label='With ground-path supplement',color='#358b84');ax.set(xticks=x,xticklabels=['Chiyoda','Chuo','Minato'],ylabel='Eligible probes (not residents)',title='Paired baseline and local flood-support coverage');ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(R/'ward_support_gain.png',dpi=170);plt.close(fig)
 primary=[]
 for case in CASES:
  for kind in ['entrance','station','section']:
   r=next(r for r in summary if r['case_id']==case and r['frame_id']=='14400' and r['ward']=='all' and r['shutdown_type']==kind);primary.append([CASE[case],KIND[kind],*[r[k] for k in STYLE]])
 sharedrows=[]
 for case in CASES:
  for version in ['strict','supplemented']:
   r=next(r for r in shared if r['case_id']==case and r['frame_id']=='14400' and r['shutdown_type']=='section' and r['cohort']=='original_supported' and r['projection_version']==version);sharedrows.append([CASE[case],version,*[r[k] for k in STYLE]])
 body='<h1>批量补齐地面步道：覆盖改善与正式实验启动判断</h1><p class="lead">复用原两组历史雷达暴雨积水、相同12个实际时刻、原209个采样点与交通图。只补道路—地表对应，不增加或移动道路连接，不追加水动力模拟，也不计算人数。</p>'
 body+='<div class="notice"><b>建议启动保留质量掩膜与不确定性分支的正式条件情景对照批次。</b><p>可进入跨分支判断的采样点从115增至144，证据不足从94降至65；三区均增加。原严格道路版本继续并列，补缺层不等同于实测路面或桥隧高程。仍不支持无条件的三区绝对孤岛面积、实际停运或受困人数推断。</p></div>'
 body+='<h2>1. 补了什么，保留什么未知</h2><p>新增5,659条地面步行路段，来源长度合计约188.97 km；其中4,029条属于footway、path或pedestrian，其余主要为允许步行的service道路。原51,745条完整分类道路支持保持不变。新增范围限于原先对应不完整、来源层级为0|0|surface且无歧义的路径。</p>'
 body+='<p>沿原折线每0.50 m及所有来源顶点采样，整段必须具备有限细地形、开放地面／道路遮罩、两组共同准入格网支持，且不穿过映射水体。碰到建筑、结构不确定区、缺值、未准入或域外支持的路径继续未知；桥、隧道、室内、覆盖通道、台阶／电梯／走廊和原生垂直转换不通过本次补缺。0.50 m是地形检查间距，不是水动力分辨率或已知测量精度。</p>'
 body+=table(['判据的首个未通过项／支持类型','路段数','来源长度 km（非封路长度）'],[[r['reason'],r['segments'],f"{float(r['length_km']):.2f}"] for r in reasons])
 body+='<p>原13,236条相对层级或垂直连接未知路段全部保留未知。失败原因按判据顺序互斥统计，同一路段可以存在多个问题；表中长度包含原模型矩形及已有跨区连接，不等于三区受灾道路长度。</p>'
 body+='<h2>2. 三区覆盖改善</h2>'
 body+=table(['区属','采样点','原严格版本支持','补缺后支持','新增','证据不足：前→后'],[[WARD[r['ward']],r['probes'],r['eligible_before'],r['eligible_after'],r['newly_eligible'],str(r['insufficient_before'])+' → '+str(r['insufficient_after'])] for r in cov])
 body+='<img src="ward_support_gain.png" alt="Three ward support coverage"><p>全209点中新增29点，占原94个证据不足点的30.9%；其余65点继续保留未知。新增支持以来源地面路径及细地形投影为条件，没有调整原准入掩膜、补造桥隧高程或把缺值填成干燥。分母是采样点，不是人口或面积。</p>'
 body+='<h2>3. 同一19:00状态下的三区结果</h2>'
 body+=table(['水位假设','运营关闭假设','模型内稳健断连','条件敏感','证据不足','各分支有替代路径'],primary)
 body+='<p>保留原八分支：0.1／0.3 m道路情景阈值、未知连接保留／移除、原站口／来源规范化归属。未知连接两分支各自与相同可用性的正常基线比较。只有共同正常基线及邻接地面投影都有支持的点，才进入稳健性分类。稳健性仅相对这些模型与假设，不表示现实无法撤离。</p><img src="strict_supplemented_19jst.png" alt="Whole region strict versus supplemented classification"><p class="caption">两行是同雨场的两种水位假设，两列是严格／补缺支持方案，展示同一19:00的区段停运。黑圈标记新纳入支持的点；点位及显示单元不是孤岛边界。</p>'
 body+='<h2>4. 分开覆盖变化与风险判断变化</h2><p>以下固定原115个支持点比较，排除新增样本带来的分母变化：</p>'
 body+=table(['水位假设','道路支持版本','模型内稳健断连','条件敏感','证据不足','各分支有替代路径'],sharedrows)
 body+='<p>原115点中的稳健断连仍为低水位10点、抬升11点；两组各有12点由条件敏感转为所有声明分支均有替代路径。新增29点在19:00区段停运下，两组均有3点模型内稳健断连、14点条件敏感、12点各分支有替代。因此全样本稳健点变成13／14，不能解释为雨量或地表积水加重。</p>'
 body+='<p>新增地面路径采用“原状态水位减去沿线细地形”的最大采样水深，保留干浅水下／上界；原分类道路均值触发仍保留。两种支持度量不同，所以必须并列保留原严格对照，不能把补缺后结果称作已验证更准确。0.1／0.3 m为声明的分析参数，不是运营或人员安全标准。</p>'
 body+='<h2>5. 地面替代路径的依赖</h2>'
 body+=table(['情景','可达地面出口路径','其中用到新增投影','其中仍经过未知路段'],[[CASE[r['case_id']],r['reachable_surface_exit_paths'],r['paths_using_supplement'],r['paths_using_remaining_unknown']] for r in depend])
 body+='<p>上表仅固定19:00、0.3 m且未知连接保留的分支；路径端点是跨出三区边界的原生节点，不是经核验安全场所。未知路段不会隐藏为已核验可通行。步行距离、返向连接、混合路径最小步行距离和原生路径证据均已保存；没有时刻表、拥堵、运能、车辆或公交替代模型，也未推断可安全完成的动态旅程。</p>'
 body+='<h2>6. 是否启动正式批量实验</h2><p><b>可以启动有明确支持范围的正式条件情景对照，暂不启动无条件全域规模估计。</b>本阶段数据接口、分层、实际时钟、原掩膜、正常分母与双向路径均已核验；三区均有覆盖改善，原支持点没有被移除，剩余缺口有明确标记。正式场景实验不必等到每座桥或地下通道都完成工程测量，但必须让未知持续进入结果。</p>'
 body+=table(['正式批次的固定条件','要求'],[['空间样本','144个当前共同支持点为本轮主分析范围；原115点与新增29点分开；209点总覆盖和65个未知持续报告'],['投影对照','严格分类道路与补缺地面路径并列，不用补缺结果替换历史基线'],['情景与基线','保持道路阈值、站口归属、未知分支及各自正常基线；冻结有限降雨／水位／停运／城市几何对照矩阵'],['时序','沿用统一实际时刻，峰值包络另列；不从快照宣称阻断持续时间'],['推断边界','结果为条件性可达性和传播机制；不报告真实停运概率、绝对孤岛面积或受困人数'],['后续计算','本次只给启动建议，没有提交新水动力批次；主动措施在正式基线对照后衔接']])
 body+='<h2>7. 核验、来源与复现</h2>'
 body+=f'<p>完成624组网络对照，{checks}项独立及路径检查通过。新增投影逐段核验地面、无遮盖、非映射水体；全部补缺水深从原NetCDF独立重算；104组道路限制、624组出发／返回路径重新复算。原交通图、两组积水、12时刻、站口关闭选择、人口资料与原12＋4划分均保留；未打开封存测试。</p>'
 body+='<p>城市源数据不同年代、缺少实测河道时序，以及原水动力无管网、泵站和入渗等局限均不因补缺消失。源标签指向地面仍是一项可审查的投影假设，不是路面测量认证；三维城市信息的贡献需要后续同雨场、同求解器和网格的正式几何对照。Framelog: MILI-D-20261017-003。</p>'
 files=[P/'analysis_contract.json',P/'source_bindings.json',P/'road_segment_inventory.csv',P/'ground_path_supplement.npz',P/'scenario_index.csv',P/'alternative_distances.npz',P/'surface_exit_route_witnesses_19jst.jsonl',R/'paired_ward_coverage_change.csv',R/'coverage_reason_summary.csv',R/'probe_support_and_class_change.csv',R/'newly_supported_points_19jst.csv',R/'common_cohort_classification_comparison.csv',R/'path_support_dependency.csv',R/'classification_summary.csv',R/'scenario_accessibility_summary.csv',R/'batch_readiness.json',R/'preparation_validation.json',R/'independent_validation.json',R/'route_witness_validation.json',OLDR/'review.html']
 body+='<ul>'+''.join('<li><a href="'+os.path.relpath(p,R)+'">'+html.escape(p.name if p.name!='review.html' else '上一轮严格支持报告')+'</a></li>' for p in files)+'</ul><p>复现：prepare_historical_surface_coverage_v2.py → compare_historical_surface_coverage_v2.py → validate_historical_surface_coverage_v2.py → export_historical_surface_routes_v2.py → report_historical_surface_coverage_v2.py。仅进行既有数据的预处理和网络后处理。</p>'
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区地面步道覆盖补缺与批量实验验收</title><style>body{font:16px/1.8 system-ui;color:#253c49;background:#f3f6f9;max-width:1220px;margin:auto;padding:32px}h1,h2{line-height:1.4}h2{margin-top:36px}.lead{font-size:19px}.notice{background:#e8f2ed;border-left:5px solid #398071;padding:18px}img{width:100%;height:auto;margin:24px 0 8px}.caption{font-size:14px;color:#536b7c}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{border:1px solid #d4dfe7;padding:10px;text-align:left;vertical-align:top}th{background:#e5edf3}a{color:#17648b}</style><main>'+body+'</main></html>'
 (R/'review.html').write_text(page);broken=[p for p in re.findall(r'(?:src|href)="([^"]+)"',page) if not (R/p).resolve().is_file()];assert not broken
 # Delivery checks also protect old cohort and unchanged source outputs.
 delivery_checks=[dict(check='original_supported_cohort_is_subset_of_new',passed=eligible0<=eligible1),dict(check='same_network_and_hydraulic_exposure_bytes',passed=all(sha(OLD/n)==sha(P/n) for n in ['local_walking_graph.npz','surface_exposure.npz','surface_probe_inventory.csv'])),dict(check='report_links_resolve',passed=not broken),dict(check='all_case_clock_type_combinations_preserved',passed=len(read(P/'scenario_index.csv'))==624)]
 assert all(x['passed'] for x in delivery_checks);dump(R/'delivery_validation.json',dict(all_passed=True,checks=delivery_checks,independent_checks=checks,formal_batch_started=False))
 paths=[p for folder in [P,R] for p in folder.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']+[ROOT/'src/preprocessing/prepare_historical_surface_coverage_v2.py']+[ROOT/'src/analyses'/n for n in ['compare_historical_surface_coverage_v2.py','validate_historical_surface_coverage_v2.py','export_historical_surface_routes_v2.py','report_historical_surface_coverage_v2.py']]
 dump(R/'delivery_manifest.json',dict(files_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},source_bindings_sha256=sha(P/'source_bindings.json'),human_decision='MILI-D-20261017-003',new_hydraulics=False,people_counted=False))
 print(json.dumps(dict(support_before=len(eligible0),support_after=len(eligible1),new_support=len(added),insufficient_after=len(b1)-len(eligible1),independent_checks=checks,scoped_batch_recommended=True),indent=2),flush=True)
if __name__=='__main__':main()
