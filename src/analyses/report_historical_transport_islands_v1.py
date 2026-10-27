"""Whole-three-ward historical hypothetical transport-consequence delivery."""
from pathlib import Path
import os,json,csv,html,re,collections
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import shape
from shapely.ops import unary_union
from compare_historical_transport_islands_v1 import ROOT,P,R,N,H,HR,S,CASES,TYPES,POLICIES,BRANCHES,read,write,dump,sha
CN={'radar_low':'历史雷达雨场＋假设低水位','radar_raised':'同一雨场＋假设水位抬升','entrance':'A 站口关闭','station':'B 整站关闭（列车通过）','section':'C 区段停运','original_identity':'原站口归属','source_normalized_identity':'补充来源名称规范化归属','13101':'千代田区','13102':'中央区','13103':'港区','boundary_or_unassigned':'边界／区属未定','all':'全部采样点'}
EN={'radar_low':'Historical rain / low stage','radar_raised':'Same rain / raised stage','entrance':'A Entrance closure','station':'B Station closure; trains pass','section':'C Service-section suspension'}
BR=BRANCHES[1]
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def link(p):return '<a href="'+os.path.relpath(p,R)+'">'+html.escape(p.name)+'</a>'
def main():
 summary=json.loads((R/'summary.json').read_text());contract=json.loads((P/'comparison_contract.json').read_text());rows=read(R/'accessibility_summary.csv');detail=read(R/'probe_accessibility.csv');selection=read(P/'closure_selection.csv');portals=read(N/'portal_inventory.csv');source_screen={e['osm_node_id']:e for e in read(P/'entrance_exposure.csv') if e['case_id']==CASES[0]};stationmeta={s['station_id']:s for s in read(N/'station_units.csv')}
 pri=[r for r in rows if r['cohort']=='surface_grid' and r['ward']=='all' and r['depth_bound']=='lower' and r['network_branch']==BR]
 ds=[r for r in detail if r['cohort']=='surface_grid' and r['depth_bound']=='lower' and r['network_branch']==BR]
 def result(case,pol,kind,ward='all',cohort='surface_grid',branch=BR):return next(r for r in rows if r['case_id']==case and r['identity_policy']==pol and r['shutdown_type']==kind and r['ward']==ward and r['cohort']==cohort and r['depth_bound']=='lower' and r['network_branch']==branch)
 def points(case,pol,kind):return [r for r in ds if r['case_id']==case and r['identity_policy']==pol and r['shutdown_type']==kind]
 coverage=[]
 for sel in selection:
  if sel['depth_bound']!='lower':continue
  closed=set(sel['selected_portal_ids'].split(';'))-{''}
  for sid in sel['target_station_ids'].split(';'):
   if not sid:continue
   pp=[p for p in portals if sid in json.loads(p['station_ids'])];mapped=[p for p in pp if p['synthetic_unknown']=='False'];screened=[p for p in mapped if p['osm_node_id'] in source_screen];quality=[p for p in screened if source_screen[p['osm_node_id']]['quality_pass']=='True']
   coverage.append(dict(case_id=sel['case_id'],identity_policy=sel['identity_policy'],station_id=sid,mapped_portals=len(mapped),surface_screened_portals=len(screened),quality_screened_portals=len(quality),selected_closed_portals=sum(p['portal_id'] in closed for p in pp),unselected_mapped_portals_assumed_available=sum(p['portal_id'] not in closed for p in mapped),synthetic_unknown_portals=sum(p['synthetic_unknown']=='True' for p in pp),verified_ingress=False,verified_internal_circulation=False))
 write(R/'targeted_station_portal_coverage.csv',coverage)
 # Paired change inventory. A display-cell name is context, never a town-wide classification.
 changes=[]
 for pol in POLICIES:
  for kind in TYPES:
   low={p['probe_id']:p for p in points(CASES[0],pol,kind)}
   for r in points(CASES[1],pol,kind):
    b=low[r['probe_id']];delta=int(b['remaining_roundtrip_anchors'])-int(r['remaining_roundtrip_anchors'])
    if delta or b['new_all_roundtrip_loss']!=r['new_all_roundtrip_loss']:
     changes.append(dict(identity_policy=pol,shutdown_type=kind,probe_id=r['probe_id'],ward=r['ward'],x_m=r['x_m'],y_m=r['y_m'],low_remaining_roundtrip_anchors=b['remaining_roundtrip_anchors'],raised_remaining_roundtrip_anchors=r['remaining_roundtrip_anchors'],new_candidate_only_in_raised=r['new_all_roundtrip_loss']=='True' and b['new_all_roundtrip_loss']!='True',town_context=r['towns_intersecting_display_cell'],context_not_town_extent=True))
 write(R/'river_stage_accessibility_change.csv',changes)
 bounds=np.load(H/'paired_exposure_bounds.npz');support=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');wc=support['Ward Code'];extent=[-11.36,-3.66,-42.44,-32.32]
 def decorate(ax):
  for code,name in [(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
   ax.contour((wc==code).astype('f4'),levels=[.5],extent=extent,origin='upper',colors='#60717c',linewidths=.65)
   ri,ci=np.where(wc==code);ax.text((-11360+(ci.mean()+.5)*10)/1000,(-32320-(ri.mean()+.5)*10)/1000,name,fontsize=8,ha='center',bbox=dict(fc='white',ec='none',alpha=.75,pad=1))
  ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xlabel='Easting (km), JGD2011 / IX',ylabel='Northing (km)')
 styles=[('#aeb7be','Baseline gap / unknown'),('#4c9a87','All baseline round trips retained'),('#e6a84d','Some round trips lost'),('#bf3e3e','All declared round trips lost')]
 def state(r):return 0 if r['baseline_unknown']=='True' else 3 if r['new_all_roundtrip_loss']=='True' else 2 if int(r['lost_roundtrip_anchors']) else 1
 for pol in POLICIES:
  fig,axes=plt.subplots(2,3,figsize=(15,12),sharex=True,sharey=True,layout='constrained')
  for j,case in enumerate(CASES):
   background=np.where(bounds['common_admitted']&(bounds['ground_lower'][j]>=.3),1.,np.nan)
   for k,kind in enumerate(TYPES):
    ax=axes[j,k];ax.imshow(background,extent=extent,origin='upper',cmap='Blues',vmin=0,vmax=2,alpha=.3,interpolation='nearest');rr=points(case,pol,kind)
    for st,(color,label) in enumerate(styles):
     use=[r for r in rr if state(r)==st]
     if use:ax.scatter([float(r['x_m'])/1000 for r in use],[float(r['y_m'])/1000 for r in use],s=15,c=color,edgecolors='white',linewidths=.3,zorder=3)
    res=result(case,pol,kind);ax.set_title(EN[case]+'\n'+EN[kind]+f" | new complete loss: {res['new_all_roundtrip_loss_probes']}/141",fontsize=10);decorate(ax)
  handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l) for c,l in styles]
  fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=10)
  fig.suptitle('Whole-region static rail-access stress envelopes\n'+('Original frozen entrance identity' if pol==POLICIES[0] else 'Source-supported normalized identity sensitivity')+' | Light blue: simulated peak ground depth >=0.3m\nPoints are unweighted ground probes; no island area, timing or population estimate',fontsize=13)
  fig.savefig(R/(pol+'_maps.png'),dpi=150);plt.close(fig)
 # Baseline uses precisely the same sites. No discarded baseline holes.
 basepoints=points(CASES[0],POLICIES[0],'entrance');fig,ax=plt.subplots(figsize=(7,9),layout='constrained')
 for unknown,color,label in [(False,'#4c9a87','Normal round-trip covered (141)'),(True,'#aeb7be','Normal baseline unknown (68)')]:
  use=[p for p in basepoints if (p['baseline_unknown']=='True')==unknown];ax.scatter([float(p['x_m'])/1000 for p in use],[float(p['y_m'])/1000 for p in use],c=color,s=20,label=label,edgecolors='white',linewidths=.3)
 decorate(ax);ax.legend(loc='lower left',fontsize=9);ax.set_title('Normal surface-probe coverage\nSix declared external anchors; <=600m initial/final walks');fig.savefig(R/'baseline_coverage.png',dpi=160);plt.close(fig)
 # Strong sensitivity is presented alongside the original rather than as a preferred estimate.
 mainrows=[]
 for pol in POLICIES:
  for kind in TYPES:
   a,b=[result(c,pol,kind) for c in CASES]
   mainrows.append([CN[pol],CN[kind],a['new_all_roundtrip_loss_probes']+' / 141',b['new_all_roundtrip_loss_probes']+' / 141',a['lost_mutual_od']+' → '+b['lost_mutual_od']+' / 846'])
 wardrows=[]
 for pol in POLICIES:
  for w in ['13101','13102','13103','boundary_or_unassigned']:
   a,b=[result(c,pol,'section',w) for c in CASES];wardrows.append([CN[pol],CN[w],a['baseline_covered_probes'],a['baseline_unknown_probes'],a['new_all_roundtrip_loss_probes'],b['new_all_roundtrip_loss_probes']])
 evidence_rows=[[CN[s['case_id']],CN[s['identity_policy']],s['exposure_trigger_points'],s['selected_portals'],s['target_station_units'],s['unassigned_trigger_points']] for s in selection if s['depth_bound']=='lower']
 added=[x for x in changes if x['identity_policy']==POLICIES[1] and x['shutdown_type']=='section' and x['new_candidate_only_in_raised']]
 # Dynamic sensitivity diagnostics from matrices, not a priori claims.
 z=np.load(P/'paired_accessibility.npz');ind=read(P/'scenario_index.csv');lookup={(r['case_id'],r['identity_policy'],r['depth_bound'],r['shutdown_type'],r['network_branch']):int(r['scenario_index']) for r in ind}
 interval_diffs=0;surface_branch_diffs=0;gmask=np.array([r['cohort']=='surface_grid' for r in read(N/'access_probes.csv') if r['cohort'] in ['surface_grid','core_station']])
 for c in CASES:
  for p in POLICIES:
   for k in TYPES:
    for b in BRANCHES:interval_diffs+=int(np.count_nonzero(z['scenario_od'][lookup[(c,p,'lower',k,b)]]!=z['scenario_od'][lookup[(c,p,'upper',k,b)]]))
    surface_branch_diffs+=int(np.count_nonzero(z['scenario_od'][lookup[(c,p,'lower',k,BRANCHES[0])]][:,gmask]!=z['scenario_od'][lookup[(c,p,'lower',k,BRANCHES[1])]][:,gmask]))
 delivery=dict(comparisons=len(ind),interval_od_boolean_differences=interval_diffs,surface_transfer_branch_od_differences=surface_branch_diffs,raised_added_section_candidates=len(added),new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False)
 dump(R/'delivery_summary.json',delivery)
 body='<h1>历史暴雨积水接入交通网络：哪些片区可能失去轨道出行？</h1><p class="lead">同一份2024年8月21日雷达雨场，两组既有10 m积水结果；分别施加站口、整站与区段关闭假设，和同一正常交通网络成对比较。覆盖都心三区与已有必要跨区连接。</p>'
 body+='<div class="notice"><b>这是有条件的轨道可达性压力分析，不能认定真实停运或城市完全孤立。</b><p>积水是模型结果；低水位及＋1.5 m抬升过程均为边界假设；关闭阈值和停运范围也都是显式假设。不同格点的事件峰值不一定同时发生，结果是静态峰值包络下的连接压力测试，没有关闭时刻或受阻持续时间。</p></div>'
 body+='<h2>主要结果</h2><p>站口关闭下，其他入口通常仍提供模型内接驳；扩大到整站或区段后，轨道可达性损失明显增大。绝对数量对站口归属非常敏感，必须并列阅读原归属与来源规范化补充。两者不是统计置信区间，也没有独立验证哪一套更接近真实。</p>'
 body+=table(['站口归属','停运假设','低水位：丢失全部往返目的地点','抬升：丢失全部往返目的地点','往返OD损失：低→抬升'],mainrows)
 body+='<p>分母141是正常模型有覆盖的地表采样点，包含7个区界／区属未定点；严格属于三区的已覆盖点为134个。每点原本可往返六个目的地，共846个OD。其余68/209点基线就未覆盖，始终列为未知，不记入雨后断连。点数既不是人数，也不是面积。站区探针另计，不与地表点相加。</p>'
 body+='<h2>积水证据如何进入停运假设</h2><p>每个站口采用20 m格心邻域，要求共同准入开放地表≥100 m²且占该邻域有效地表≥80%；深度下界≥0.3 m的足迹达到100 m²时触发。0.3 m和100 m²沿用既有研究参数，不是运营商标准、入口门槛水深或人员安全标准。675个站口点中663个共同质量合格。</p>'
 body+=table(['积水情景','归属方案','触发站口邻域点','可进入关闭假设的站口','目标运营者／站名单元','触发但归属未落实的点'],evidence_rows)
 body+='<p>低水位108、抬升147个触发点已逐点复现上一轮积水报告。归属未知的触发点保留在证据台账，不就近强配、不视为干燥。补充归属只沿用已冻结的来源站区名／运营者规范化及同身份官方站体距离校验；不证明站内通道或2024年入口状态。原归属与补充归属不混合汇报。</p>'
 body+=table(['关闭类型','本次删除','保留与未知'],[[CN['entrance'],'仅选中站口的进出站内连接','其他站口、步行道路和列车运行保留；未筛选／未触发入口假设可用'],[CN['station'],'目标运营者／站名单元的进出站、上下车及相关换乘连接','保留列车通过服务；同名不同运营者不自动一同关闭'],[CN['section'],'整站关闭，另切断服务端点或同服务族参考站序经过目标站的运行边','不是实测隧道受淹，也未模拟折返、调度及整线可实施性']])
 body+='<p>站口关闭影响小不能证明入口安全或真实冗余：未测入口仍假设可用，站内通行是聚合假设。目标站的已测／未测入口覆盖已单独导出。3554条原运行边中201条缺少完整内部跨度，只能按端点处理，未补造轨道几何。</p>'
 body+='<h2>三区整体分布</h2><p>“轨道出行孤岛候选”定义为：正常时可往返至少一个指定域外目的地，假设关闭后失去全部这类往返连接的固定地表采样点。同时核查出发和返回方向；本次这些候选点的两方向连接均丢失，并非仅回程缺失。六个目的地为JR新宿、涩谷、上野、锦糸町、大井町和Metro丰洲；域外服务与目的地假设可用。</p>'
 body+='<img src="original_identity_maps.png" alt="Original identity paired regional accessibility"><p class="caption">原归属：两行是相同雨场的两种假设水位，三列是关闭范围。蓝底为模型峰值地表暴露，红点为新增失去全部指定目的地往返连接的采样点；灰点是正常基线缺覆盖。符号不是孤岛边界。</p>'
 body+='<img src="source_normalized_identity_maps.png" alt="Source normalized identity regional sensitivity"><p class="caption">仅补充来源支持的站口归属；雨场、网格、积水结果和正常网络相同。显示整个三区范围，不用个别站点核查替代区域分析。</p>'
 body+=table(['归属方案','区属','正常覆盖点','原本未知点','区段停运：低水位新增断连点','区段停运：抬升新增断连点'],wardrows)
 body+='<p>原归属下，较稳定的候选分布在青山、麻布及高轮—三田—芝／浜松町一带，另有胜哄附近和千代田区少数采样点。来源规范化分支扩大到赤坂—六本木—虎之门—新桥、麹町／永田町及银座—筑地等采样单元。以上是候选点所在或显示单元相交的地名，不代表这些片区全部孤立，也不是三区真实风险排名。</p>'
 body+='<h2>水位抬升增加什么影响？</h2><p>原归属分支下，新增触发点并未使地表采样点断连数量增加；这不代表水位抬升没有交通影响，只说明当前离散采样和二元拓扑指标没有增加损失。来源规范化分支下，整站关闭由41增至42点，区段停运由77增至83点；区段新增点为千代田区5个、中央区1个，港区仍54个。</p>'
 body+=table(['新增采样点','区属','显示单元相交的地名（不是全片区判定）'],[[x['probe_id'],CN[x['ward']],x['town_context']] for x in added])
 body+='<p>新增点集中在日比谷—有乐町—丸之内／大手町及银座相邻单元。补充归属下新增目标站单位为Metro有乐町、JR东京及Metro东京；其进入关闭假设不等于这些车站真实进水。铁路停运可通过网络传播到没有本地积水的片区，故红点无需与蓝色暴露足迹重合。</p>'
 body+='<h2>覆盖、不确定性与解释边界</h2><img src="baseline_coverage.png" alt="Normal network coverage and held gaps">'
 body+=f'<p>完成48次图可达性评估（2积水结果×2归属×2数值边界×3关闭类型×2换乘分支），全部复用既有水动力输出。水深下／上界分支的OD布尔差异为{interval_diffs}，说明本次既定阈值不受该数值区间影响；不代表地形、降雨、边界或真实淹没不确定性为零。两种换乘分支的地表OD差异为{surface_branch_diffs}，但站区探针仍依赖换乘与覆盖，详细表保留各自正常分母。</p>'
 body+='<p>正常图含7571节点、25392有向边；层级和权限不变，不按平面相交造边。地表点仅使用初末段各≤600 m的来源原生步行路径及中间轨道连接，不允许无限步行串联替代停运。道路积水封路、公交替代、车辆、时刻表、运能与拥堵均未加入，因此不能从本次判定完全无法撤离、出行时间或通勤瘫痪程度。保留未核验的站内通行和跨运营者换乘假设；这些图是正常运行候选网络，不是完整认证交通模型。</p>'
 body+='<p>水动力仍无管网、泵站和入渗，屋面降雨即时分配到地表，初始陆地干燥；这会影响暴露。城市几何与2025官方站线、2026 OSM快照未还原成完整2024现状。不能将雷达历史雨场驱动的结果当作已验证历史灾情。官方夜间街区人口基线、125/250 m楼宇分配及其不确定性均未改变，本次不加权、未计算人数，也未补造白天人口。</p>'
 body+='<p><b>下一步建议：</b>在同一整体网络上独立加入地面积水通行限制，并设未测连接可用／不可用分支，检查这些轨道孤岛候选是否仍有地面替代路径。完成后再决定是否引入白天出行与人口，当前不直接计算受困人数。</p>'
 files=[P/'comparison_contract.json',P/'source_bindings.json',P/'closure_selection.csv',P/'evidence_to_hypothesis_ledger.csv',P/'entrance_exposure.csv',P/'paired_accessibility.npz',P/'scenario_index.csv',P/'probe_accessibility_6677.geojson',R/'accessibility_summary.csv',R/'probe_accessibility.csv',R/'rail_access_island_candidates.csv',R/'river_stage_accessibility_change.csv',R/'targeted_station_portal_coverage.csv',R/'comparison_validation.json',R/'independent_validation.json',R/'delivery_summary.json',HR/'review.html']
 body+='<h2>核验与可复现文件</h2><p>正常基线逐矩阵复现；检查删边不产生新路径、A/B/C关闭集合嵌套、整站假设保留通过运行、地表接驳不变、已知与未知分母不混用。独立核验另行重建删边集和全部48组出发／返回路径，核对汇总与空间点位。输入、原12＋4划分和人口资料哈希保持不变，未打开封存测试。</p><ul>'+''.join('<li>'+link(p)+'</li>' for p in files)+'</ul><p>复现顺序：compare_historical_transport_islands_v1.py → report_historical_transport_islands_v1.py → validate_historical_transport_islands_v1.py；均位于src/analyses，仅做既有输出的网络后处理。Framelog: MILI-D-20261017-001。</p>'
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区历史暴雨与交通孤岛候选</title><style>body{font:16px/1.8 system-ui;color:#263c49;background:#f4f7fa;max-width:1230px;margin:auto;padding:32px}h1,h2{line-height:1.4}h2{margin-top:38px}.lead{font-size:19px}.notice{background:#fff3df;border-left:5px solid #c18b33;padding:18px}img{width:100%;height:auto;margin:24px 0 8px}.caption{font-size:14px;color:#586c7a}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:10px;border:1px solid #d7e1e8;text-align:left;vertical-align:top}th{background:#e5edf3}a{color:#17658c}</style><main>'+body+'</main></html>'
 (R/'review.html').write_text(page)
 print(json.dumps(delivery,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
