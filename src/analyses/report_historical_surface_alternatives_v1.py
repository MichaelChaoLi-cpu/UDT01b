"""Regional qualification report; population counts and new hydraulics excluded."""
from pathlib import Path
import os,sys,json,html,re,collections
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_historical_surface_access_v1 import P,R,N,T,H,CASES,TIMES,read,write,dump,load,sha
CN={'radar_low':'假设低水位','radar_raised':'假设水位抬升','entrance':'站口关闭','station':'整站关闭，列车通过','section':'区段停运','original_identity':'原归属','source_normalized_identity':'补充来源归属','13101':'千代田区','13102':'中央区','13103':'港区','boundary_or_unassigned':'边界／区属未定','all':'全部采样点'}
CLASSES={'robust_modeled_disconnection':('#bd3c3c','Robust modeled disconnection','模型内稳健断连'),'conditional_sensitivity':('#d99b38','Assumption sensitive','条件敏感'),'insufficient_evidence':('#aab4bd','Insufficient evidence','证据不足'),'stable_modeled_alternative':('#3f9283','Alternative in all branches','各分支均有替代路径')}
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def labeltime(t):return '峰值包络（非同时）' if str(t)=='peak_envelope' else f'{15+int(t)//3600:02d}:{(int(t)%3600)//60:02d}'
def main():
 validation=json.loads((R/'independent_validation.json').read_text());assert validation['all_passed'];witness_validation=json.loads((R/'route_witness_validation.json').read_text());assert witness_validation['all_passed'];total_checks=len(validation['checks'])+len(witness_validation['checks']);summary=json.loads((R/'analysis_summary.json').read_text());prep=json.loads((R/'preparation_validation.json').read_text());contract=json.loads((P/'analysis_contract.json').read_text());rows=read(R/'classification_summary.csv');details=read(R/'probe_classification.csv');base=read(R/'normal_baseline_and_support.csv');stat=read(R/'scenario_accessibility_summary.csv');prior=read(R/'prior_rail_candidates_surface_alternatives.csv');meta=read(P/'road_segment_inventory.csv');a=load(P/'alternative_distances.npz');probes=read(P/'surface_probe_inventory.csv');e=load(P/'surface_exposure.npz');w=load(P/'local_walking_graph.npz')
 oldcontext=read(ROOT/'data/exp/estimation/three_ward_historical_transport_islands_v1/probe_accessibility.csv');context={x['probe_id']:x['towns_intersecting_display_cell'] for x in oldcontext};selected=[r for r in details if r['frame_id']=='14400' and r['shutdown_type']=='section'];locations=[]
 for r in selected:
  if r['classification']=='stable_modeled_alternative':continue
  locations.append({**r,'display_cell_town_context':context[r['probe_id']],'not_whole_town_classification':True})
 write(R/'classification_location_inventory.csv',locations)
 # Quantify missing support by mutually exclusive source-evidence category.
 reasons=collections.Counter();lengths=collections.Counter()
 for i,r in enumerate(meta):
  if r['known_surface_grade']!='True':reason='relative_height_or_level_transition_unknown'
  elif r['ambiguous_parallel_geometry']=='True':reason='ambiguous_parallel_source_geometry'
  elif r['has_out_of_grid_sample']=='True':reason='partly_outside_hydraulic_grid'
  elif a['road_unknown'][i]:reason='ground_road_correspondence_or_admission_incomplete'
  else:reason='fully_supported_ground_road_segment'
  reasons[reason]+=1;lengths[reason]+=float(r['source_length_m'])
 coverage=[dict(reason=k,source_segments=v,source_segment_length_km=lengths[k]/1000,not_flooded_or_closed_length=True) for k,v in reasons.items()];write(R/'road_evidence_coverage.csv',coverage)
 # Stage completion and full-study readiness are distinct judgments.
 complete=validation['all_passed'];n=len(probes);supported=summary['classification_evidence_eligible'];unknown=n-supported
 readiness={'stage_technical_qualification_complete':complete,'formal_full_region_batch_ready':False,'reason':'The interface and declared scenario rules are reproducible, but only '+str(supported)+'/'+str(n)+' surface probes have a paired normal baseline under both unknown-link treatments and local supported ground exposure. The remaining '+str(unknown)+' are insufficient evidence; full-region absolute island claims would be dominated by evidence coverage.',
 'allowed_interpretation':'Conditional, unweighted scenario consequences on the supported cohort; no event accuracy claim, no whole-town isolation, no affected-person counts.',
 'next_bounded_task':'One regional evidence-coverage pass: distinguish unclassified at-grade footpaths from bridge/tunnel/vertical links, use existing0.5m terrain and explicit source geometry to build a separately flagged at-grade fallback, preserve unknown absolute deck/entry elevations, and reevaluate the SAME two outputs and12 clocks. No pointwise survey requirement and no new hydraulic cases.',
 'release_requirements':'Freeze auditable supported/fallback/unknown categories; demonstrate how exclusions affect each ward; recompute the same paired baselines and classifications before expanding hydraulic event batches. No requirement that all branches agree or that real-world exposure be certain.',
 'gates':[dict(gate=x,status='passed') for x in contract['batch_gates']], 'coverage_gate':'hold_for_full_region_absolute_inference','people_counted':False,'new_hydraulics':False,'training':False,'sealed_tests_opened':False}
 dump(R/'batch_readiness.json',readiness)
 # Fixed19:00 map, selected before outcomes; all three shutdown types retained.
 s=load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');wc=s['Ward Code'];h=load(H/'paired_exposure_bounds.npz');extent=[-11.36,-3.66,-42.44,-32.32]
 def decorate(ax):
  for code,name in [(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
   ax.contour((wc==code).astype('f4'),levels=[.5],extent=extent,origin='upper',colors='#647785',linewidths=.65);rr,cc=np.where(wc==code);ax.text((-11360+(cc.mean()+.5)*10)/1000,(-32320-(rr.mean()+.5)*10)/1000,name,fontsize=8,ha='center',bbox=dict(fc='white',ec='none',alpha=.8,pad=1))
  ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xlabel='Easting (km), JGD2011 / IX',ylabel='Northing (km)')
 fig,axes=plt.subplots(2,3,figsize=(15,12),sharex=True,sharey=True,layout='constrained');kinds=['entrance','station','section']
 for ci,case in enumerate(CASES):
  for j,kind in enumerate(kinds):
   ax=axes[ci,j];bg=np.where(h['common_admitted']&(h['snapshot_19jst_lower'][ci]>=.3),1.,np.nan);ax.imshow(bg,extent=extent,origin='upper',cmap='Blues',vmin=0,vmax=2,alpha=.25)
   rr=[r for r in details if r['case_id']==case and r['shutdown_type']==kind and r['frame_id']=='14400']
   for label,(color,en,cn) in CLASSES.items():
    pp=[p for p in rr if p['classification']==label];ax.scatter([float(p['x_m'])/1000 for p in pp],[float(p['y_m'])/1000 for p in pp],c=color,s=16,edgecolors='white',linewidths=.3,zorder=3)
   robust=sum(r['classification']=='robust_modeled_disconnection' for r in rr);ax.set_title(('Low stage' if ci==0 else 'Raised stage')+' | '+['A Entrance','B Station; trains pass','C Service section'][j]+f'\n19:00 JST | robust modeled loss: {robust}/{supported}',fontsize=10);decorate(ax)
 fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=en) for c,en,cn in CLASSES.values()],loc='outside lower center',ncol=2);fig.suptitle('Synchronized surface alternatives + rail accessibility | fixed19:00 JST\nEight branches: road depth0.1/0.3m x unknown availability x station identity\nPoints are unweighted probes, not isolated area or residents',fontsize=13);fig.savefig(R/'synchronized_19jst_maps.png',dpi=150);plt.close(fig)
 # Time course uses all twelve declared simultaneous states; peak has its own table.
 fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True,layout='constrained')
 for case,ax in zip(CASES,axes):
  rr=[next(r for r in rows if r['case_id']==case and r['shutdown_type']=='section' and r['ward']=='all' and r['frame_id']==str(t)) for t in TIMES]
  x=15+np.array(TIMES)/3600
  for key in ['robust_modeled_disconnection','conditional_sensitivity','stable_modeled_alternative']:
   color,en,cn=CLASSES[key];ax.plot(x,[int(r[key]) for r in rr],marker='o',markersize=3,label=en,color=color)
  ax.set(title='Low stage' if case==CASES[0] else 'Raised stage',xlabel='JST hour;24=next midnight',ylabel='Unweighted probe count',xticks=np.arange(15,25),ylim=(0,max(supported,1)+5));ax.grid(alpha=.2)
 axes[1].legend(fontsize=8);fig.suptitle('Section-suspension hypothesis | '+str(unknown)+' evidence-insufficient probes excluded from curves\nSnapshot connectivity is not a time-dependent traversable journey or obstruction duration');fig.savefig(R/'synchronized_time_course.png',dpi=170);plt.close(fig)
 # Save semantic point geometries without population attributes.
 features=[]
 for r in selected:features.append(dict(type='Feature',geometry=dict(type='Point',coordinates=[float(r['x_m']),float(r['y_m'])]),properties={k:v for k,v in r.items() if k not in ['x_m','y_m']}|dict(display_cell_town_context=context[r['probe_id']])) )
 dump(P/'synchronized_19jst_classification_6677.geojson',dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:6677')),features=features))
 mainrows=[]
 for case in CASES:
  for frame in ['14400','peak_envelope']:
   r=next(x for x in rows if x['case_id']==case and x['frame_id']==frame and x['shutdown_type']=='section' and x['ward']=='all');mainrows.append([CN[case],labeltime(frame),*[r[k] for k in CLASSES]])
 baserows=[]
 for ward in ['13101','13102','13103','boundary_or_unassigned']:
  b=[r for r in base if r['ward']==ward and r['unknown_branch']=='optimistic'];p=[r for r in base if r['ward']==ward and r['unknown_branch']=='pessimistic'];baserows.append([CN[ward],len(b),sum(x['baseline_has_modeled_alternative']=='True' for x in b),sum(x['baseline_has_modeled_alternative']=='True' for x in p),sum(x['classification_evidence_eligible']=='True' for x in b)])
 body='<h1>道路积水、地面替代路径与同步可达性</h1><p class="lead">复用两组已完成的历史雷达暴雨积水结果；接入原生分层步行网络及既有轨道网络，对照12个统一时刻、三类停运和八个分析分支。保持三区整体视角，没有新水动力模拟、训练或人数计算。</p>'
 body+=f'<div class="notice"><b>阶段技术核验已完成，但暂不建议启动面向三区绝对孤岛范围的全套批量实验。</b><p>624组网络对照已经完成并独立复算。当前{supported}/{n}个采样点具备两种未知连接处理下的共同正常基线与邻接地表投影，{unknown}个仍需归为证据不足。主要瓶颈是道路与积水支持的对应覆盖，而不是继续增加暴雨情景或逐点勘查。</p></div>'
 body+='<h2>1. 轨道断连不等于没有地面替代</h2><p>此前的77／83个点来自轨道主导、初末步行各≤600 m的峰值压力分析。本次允许沿真实分层步行图寻找更远的可用站口，以及跨出三区边界的道路连接。以下仍是峰值包络下、0.3 m道路限制及未知连接保留分支的条件结果，不能当作可安全步行或所有条件下均可达。</p>'
 body+=table(['水位情景','站口归属','原区段停运轨道候选点','仍有地面出口路径','其中步行≤2 km','可达出口的步行中位数'],[[CN[r['case_id']],CN[r['identity_policy']],r['old_peak_rail_only_candidate_points'],r['surface_exit_reachable'],r['surface_exit_within2km'],f"{float(r['surface_exit_median_m']):.0f} m" if r['surface_exit_median_m'] else '无'] for r in prior if r['shutdown_type']=='section'])
 body+='<p>跨区出口只表示离开研究边界，不是已核验安全场所，不能排除域外洪水、缺乏接驳或目的地服务不足。2 km仅用于显示绕行负担，非人员安全或可接受步行标准。轨道边只计零步行距离，因此“混合路径步行距离”不是旅行时间；不包含列车运行、等候、拥挤或体力负担。</p>'
 body+='<h2>2. 同一时刻的分类，和峰值包络分开</h2>'
 body+=table(['情景','时刻／包络','模型内稳健断连','条件敏感','证据不足','各分支均有替代路径'],mainrows)
 body+='<p>分类以每个时刻、同一停运类型的八分支为单位：道路均值阈值0.1／0.3 m × 未知连接保留／移除 × 原归属／来源规范化归属。只有共同正常基线与局部地表对应均有支持的点，才判断所有分支断连、所有分支仍可达或分支间不同。其余点列为证据不足，不记作安全或新增灾害断连。“稳健”仅限这套模型、交通方式与参数范围，不表示真实城市完全孤立。</p>'
 body+='<img src="synchronized_19jst_maps.png" alt="Whole three ward simultaneous19JST classification"><p class="caption">预先固定19:00 JST全域对照。红=模型内稳健断连，橙=条件敏感，灰=证据不足，绿=各分支有替代。浅蓝为该时刻模拟地表水深≥0.3 m背景，不是道路或站口实际观测。点符号不是孤岛面积。</p><img src="synchronized_time_course.png" alt="Twelve synchronized accessibility states"><p>15:00—24:00整点，加18:30与19:30，共12个实际输出状态；两组使用相同时间表。采样状态间的通行不能插值保证，未计算受阻持续时间，也未构建能随时间行走的旅程。峰值包络单列，不声称所有峰值同时发生。</p>'
 body+='<h2>3. 正常基线与证据覆盖</h2><p>未知连接的两种处理，各自与相同未知连接可用性的正常图比较。正常图不施加积水关闭，也不施加停运；这样不会将“资料不足所以暂时不连”算成暴雨新增断连。移除未知连接的正常图是证据支持下界，不是东京晴天交通状态的实测断路。</p>'
 body+=table(['区属','采样点','保留未知：正常有替代','移除未知：正常有替代','可进入稳健性分类'],baserows)
 body+=f'<p>道路图保留{prep["nodes"]:,}个节点、{prep["arcs"]:,}条有向连接和{prep["native_boundary_exit_nodes"]}个原生边界出口节点。没有按平面交叉造边，没有最近车站匹配，也未改变通行权限。原600 m接驳被显式原生道路路径替代，因此本次正常分母与上一轮141点不同，不能直接比较百分比。</p>'
 body+=table(['道路证据类型','来源路段数','来源路段长度 km（非封路长度）'],[[r['reason'],r['source_segments'],f"{r['source_segment_length_km']:.1f}"] for r in coverage])
 body+='<p>仅精确标记为0|0|surface的路段使用地表积水。高架、地下、垂直转换及其他不明层级均保留未知，不把地表水深投到桥面或隧道。地面段也要求采样格具有共同准入的分类道路地表；步道未落入分类道路、建筑或水域混合格、隔离区和域外部分保留未知，不作零水深。本次严格政策因此牺牲覆盖，尤其会使站口接驳在移除未知分支中缺失，不能把该分支解释为真实网络。</p>'
 body+='<h2>4. 距离与区域位置明细</h2><p>道路水深由实际水位在原0.50 m分类道路细地表重新积分为10 m道路地表均值，保留干浅水删失上下界。沿原道路折线每2.5 m及原始顶点映射到这些格子；任一格达到情景阈值时，假设关闭整段。2.5 m是匹配采样间距，不会把10 m水动力结果提升到2.5 m精度，也不能分辨每条人行道或路缘。</p>'
 robust=[r for r in locations if r['classification']=='robust_modeled_disconnection' and r['case_id']==CASES[0]]
 body+=table(['19:00低水位下的稳健候选点','区属','显示单元相交地名，非全街区判定'],[[r['probe_id'],CN[r['ward']],r['display_cell_town_context']] for r in robust])
 body+='<p>完整点位、三区汇总、出发／返回方向、地面出口距离、混合路径最小步行距离和相对各自正常基线的绕行增量均已导出。没有将500 m显示采样单元当作受影响面积，也未把街区人口分配到这些点。白天人口及楼宇分配的不确定性仍保持原状。</p>'
 body+='<h2>5. 完成条件与是否进入正式批量实验</h2>'
 body+=table(['本阶段完成条件','结果'],[['空间、实际时钟与分层匹配','通过：原质量掩膜保留；最大实际时刻误差小于0.001秒；桥隧不套地表水深'],['正常基线和未知缺口','通过：未知连接两分支各自成对；分母与资料缺口单列'],['地面与轨道替代路径','通过：出口及来源支持的站口路径，双向距离独立复算'],['同步状态与峰值包络分离','通过：相同12时刻，峰值另列；不作持续时间推断'],['规则冻结、可追溯与独立核验',f'通过：{total_checks}项独立检查，原输入哈希未变'],['三区完整绝对孤岛分布的证据覆盖',f'暂缓：仅{supported}/{n}点进入跨分支稳健性判断，且各区覆盖不均']])
 body+='<p><b>判断：</b>可以结束这一轮端到端技术验证；目前不建议直接铺开旨在发布三区绝对孤岛范围的全套暴雨批次。批量增加降雨或水位情景不会修复当前道路对应资料缺口。下一步应做一次限定范围的整体覆盖改进：批量区分未分类的地面步道与桥隧层级，用已有0.50 m地形及原生路径构建单独标记的地面备用投影；桥面、地下和入口真实高程继续保留未知，然后仅用这两组结果和相同12时刻复核。无需逐点核查，也不要求所有假设最终给出同一结论。</p>'
 body+='<h2>6. 局限与交付</h2><p>积水仍来自历史雷达代表强度驱动的情景，河道低水位和抬升过程均是假设；无管网、泵站或入渗，全部屋面雨水即时进入地表，城市几何和交通资料未完整还原到2024年。停运不代表真实运营行为；没有公交、机动车、时刻表、容量或拥堵，也未证明域外出口安全。“未找到模型内替代路径”不能等同于真实不可撤离或受困人数。</p><p>独立核验曾发现混合路径终点索引被邻域面积变量覆盖，已修复并加六目的地形状断言；最终624组结果全部重算，正常基线与所有路径重新独立核验。水动力输出未重跑或改写。Framelog: MILI-D-20261017-002。</p>'
 files=[P/'analysis_contract.json',P/'source_bindings.json',P/'local_walking_graph.npz',P/'surface_exposure.npz',P/'road_segment_inventory.csv',P/'station_closure_selection.csv',P/'road_context_index.csv',P/'scenario_index.csv',P/'alternative_distances.npz',P/'synchronized_19jst_classification_6677.geojson',R/'normal_baseline_and_support.csv',R/'scenario_accessibility_summary.csv',R/'classification_summary.csv',R/'probe_classification.csv',R/'classification_location_inventory.csv',R/'prior_rail_candidates_surface_alternatives.csv',R/'road_evidence_coverage.csv',R/'batch_readiness.json',R/'preparation_validation.json',R/'independent_validation.json',R/'route_witness_validation.json',P/'surface_exit_route_witnesses_19jst.jsonl']
 body+='<ul>'+''.join('<li><a href="'+os.path.relpath(f,R)+'">'+html.escape(f.name)+'</a></li>' for f in files)+'</ul><p>复现：prepare_historical_surface_access_v1.py（预处理）→ compare_historical_surface_alternatives_v1.py → validate_historical_surface_alternatives_v1.py → export_historical_surface_route_witnesses_v1.py → report_historical_surface_alternatives_v1.py。使用已有本地环境，不调用水动力求解器。</p>'
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区道路替代路径与同步可达性验收</title><style>body{font:16px/1.8 system-ui;color:#243b49;background:#f3f6f9;max-width:1230px;margin:auto;padding:32px}h1,h2{line-height:1.4}h2{margin-top:36px}.lead{font-size:19px}.notice{background:#fff2db;border-left:5px solid #c88d32;padding:18px}img{width:100%;height:auto;margin:24px 0 8px}.caption{font-size:14px;color:#536c7d}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{border:1px solid #d5e0e8;padding:10px;vertical-align:top;text-align:left}th{background:#e5edf4}a{color:#17648b}</style><main>'+body+'</main></html>'
 (R/'review.html').write_text(page);broken=[x for x in re.findall(r'(?:src|href)="([^"]+)"',page) if not (R/x).resolve().is_file()];assert not broken;dump(R/'report_link_validation.json',dict(all_passed=True,broken_links=broken))
 files=[f for folder in [P,R] for f in folder.rglob('*') if f.is_file() and f.name!='delivery_manifest.json']+[ROOT/'src/preprocessing/prepare_historical_surface_access_v1.py']+[ROOT/'src/analyses'/x for x in ['compare_historical_surface_alternatives_v1.py','validate_historical_surface_alternatives_v1.py','report_historical_surface_alternatives_v1.py','export_historical_surface_route_witnesses_v1.py']]
 dump(R/'delivery_manifest.json',dict(files_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},input_bindings_sha256=sha(P/'source_bindings.json'),human_decision='MILI-D-20261017-002',new_hydraulics=False,people_counted=False))
 print(json.dumps(readiness,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
