"""Reproducible whole-region report of explicitly hypothetical rail disruptions.
No additional hydraulic simulation, population allocation or sealed-test access.
"""
from pathlib import Path
import csv,json,hashlib,html,sys,collections,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Patch
from shapely.geometry import shape,box
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_transport_shutdown_scenarios_v1'
R=ROOT/'data/exp/estimation/three_ward_transport_shutdown_scenarios_v1'
S=ROOT/'data/processed/three_ward_station_surface_risk_v1'
BR='provider_transfers_assumed_bidir'
TYPES=['entrance','station','section']
LABEL={'entrance':'A · 站口关闭','station':'B · 整站关闭（通过运行保留）','section':'C · 区段停运'}
POL={'primary_0p3_area100':'原初筛归属 / 0.3 m + 100 m²','sensitivity_normalized_identity':'补充名称规范化归属','sensitivity_0p1_area100':'阈值 0.1 m + 100 m²','sensitivity_0p5_area100':'阈值 0.5 m + 100 m²','sensitivity_0p3_anyarea':'0.3 m / 任意支持面积','sensitivity_vicinity_fallback':'增加无站口归属的站区假设'}
WARD={'13101':'千代田区','13102':'中央区','13103':'港区','boundary_or_unassigned':'边界 / 区属未定','all':'全域'}
def rr(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def n(r,k):return int(r[k])
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def link(p,label=None):
 import os
 return '<a href="'+html.escape(os.path.relpath(p,R))+'">'+html.escape(label or p.name)+'</a>'
def polygon_list(g):
 if g.is_empty:return []
 if g.geom_type=='Polygon':return [g]
 if hasattr(g,'geoms'):return [p for q in g.geoms for p in polygon_list(q)]
 return []
def main():
 rows=rr(R/'scenario_accessibility_summary.csv');details=rr(R/'primary_probe_accessibility.csv');base=rr(R/'normal_baseline_coverage.csv');wards=rr(R/'ward_accessibility_summary.csv');select=rr(P/'closure_selection_by_parent.csv');bindings=rr(S/'scenario_bindings.csv');events={r['case_id']:r for r in bindings}
 build=json.loads((R/'network_build_summary.json').read_text());policy=json.loads((P/'scenario_policy.json').read_text())
 prior=rr(S/'station_entrance_identity_links.csv');supp=rr(P/'identity_attribution_sensitivity_links.csv');riskids={x['osm_node_id'] for x in rr(S/'entrance_screen_summary.csv')};new=[x for x in supp if x['osm_node_id'] in riskids]
 coverage={'exposure_points':len(riskids),'original_attributed_points':len({x['osm_node_id'] for x in prior}),'original_attributed_units':len({x['station_id'] for x in prior}),'normalized_attributed_points':len({x['osm_node_id'] for x in new}),'normalized_attributed_units':len({x['station_id'] for x in new})}
 metrics=[]
 for pol in POL:
  for kind in TYPES:
   q=[r for r in rows if r['exposure_policy']==pol and r['shutdown_type']==kind and r['network_branch']==BR]
   worst=max(q,key=lambda r:n(r,'surface_grid_lost_mutual_pairs'))
   metrics.append({'policy':pol,'shutdown_type':kind,'min_lost_surface_mutual_OD':min(n(r,'surface_grid_lost_mutual_pairs') for r in q),'max_lost_surface_mutual_OD':max(n(r,'surface_grid_lost_mutual_pairs') for r in q),'max_surface_origins_losing_all_six':max(n(r,'surface_grid_lost_all_declared_mutual_anchors') for r in q),'max_fraction_of_141_baseline_covered_samples':max(n(r,'surface_grid_lost_all_declared_mutual_anchors') for r in q)/141,'worst_case_by_OD':worst['case_id']})
 summary={'purpose':'Conditional transport accessibility under hypothetical shutdowns; not actual isolation or affected people.','human_decision':'MILI-D-20261016-024','parent_events':16,'graph_evaluations':len(rows),'normal_surface_samples':209,'normal_surface_covered_samples':141,'baseline_uncovered_samples':68,'fixed_destinations':6,'normal_mutual_surface_OD_pairs':846,'exposure_attribution_coverage':coverage,'metrics':metrics,'display_case':'train-04','display_case_selection':'Largest section-shutdown surface OD loss among the sixteen existing events under the frozen main attribution, descriptive illustration only.','identity_sensitivity_is_not_probability_or_confidence_interval':True,'temporal_status':'Per-event asynchronous peak stress envelope, no concurrent-failure timing claim.','population_uncertainty_preserved':True,'new_hydraulics':False,'people_counted':False,'sealed_tests_opened':False}
 dump(R/'summary.json',summary)
 # Main and attribution-sensitivity maps: same event, same extent, no population.
 core=shape(json.loads((ROOT/'data/processed/three_ward_transport_framework_v1/inventory_domains_6677.geojson').read_text())['features'][0]['geometry'])
 colors=['#d7dce0','#dcebdc','#edb65b','#b62c37'];names=['Baseline uncovered','All declared anchors retained','Some anchor pairs lost','All six declared anchors lost']
 for pol,file in [('primary_0p3_area100','closure_maps_main.png'),('sensitivity_normalized_identity','closure_maps_identity.png')]:
  fig,axes=plt.subplots(1,3,figsize=(13,8.8),sharex=True,sharey=True)
  for ax,kind,title in zip(axes,TYPES,['A  Entrance closure','B  Station closure','C  Section suspension']):
   q=[r for r in details if r['case_id']=='train-04' and r['exposure_policy']==pol and r['shutdown_type']==kind and r['cohort']=='surface_grid']
   for poly in polygon_list(core):ax.add_patch(Polygon(np.asarray(poly.exterior.coords)/1000,facecolor='#f9f9f6',edgecolor='#4b5563',lw=.7))
   counts=collections.Counter()
   for r in q:
    state=0 if r['baseline_uncovered']=='True' else 3 if r['lost_all_declared_mutual_anchors']=='True' else 2 if n(r,'lost_mutual_anchor_count') else 1
    counts[state]+=1;x=float(r['x_m']);y=float(r['y_m']);cell=box(math.floor(x/500)*500,math.floor(y/500)*500,(math.floor(x/500)+1)*500,(math.floor(y/500)+1)*500).intersection(core)
    for poly in polygon_list(cell):ax.add_patch(Polygon(np.asarray(poly.exterior.coords)/1000,facecolor=colors[state],edgecolor='white',lw=.4))
    ax.plot(x/1000,y/1000,'.',color='#525b65',ms=1.6)
   ax.set_title(title+'\nAll six lost: '+str(counts[3])+' / 141',fontsize=11);ax.set_aspect('equal');ax.grid(alpha=.15);ax.set_xlabel('JGD2011 / IX easting (km)')
  axes[0].set_ylabel('JGD2011 / IX northing (km)');b=core.bounds;axes[0].set_xlim(b[0]/1000-.3,b[2]/1000+.3);axes[0].set_ylim(b[1]/1000-.3,b[3]/1000+.3)
  fig.suptitle('train-04 | '+('Frozen exposure attribution' if pol.startswith('primary') else 'Additional normalized source identity attribution'),fontsize=14,y=.98)
  fig.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(colors,names)],loc='lower center',ncol=2,fontsize=9,frameon=False,bbox_to_anchor=(.5,.025))
  fig.text(.5,.008,'500 m sample display cells, not inundation/isolated area. Gray = normal-network data gap. Six selected destinations only.',ha='center',fontsize=8)
  fig.tight_layout(rect=[0,.11,1,.94]);fig.savefig(R/file,dpi=170);plt.close(fig)
 # All 16 events, paired main vs attribution sensitivity.
 fig,ax=plt.subplots(figsize=(11,5.7));xx=np.arange(3);cols=['#7294b4','#cc5c4a'];xs=[-.15,.15]
 for pol,c,dx,label in zip(['primary_0p3_area100','sensitivity_normalized_identity'],cols,xs,['Frozen exposure attribution','Normalized source attribution']):
  for i,kind in enumerate(TYPES):
   q=[r for r in rows if r['exposure_policy']==pol and r['shutdown_type']==kind and r['network_branch']==BR];vals=np.array([n(r,'surface_grid_lost_all_declared_mutual_anchors') for r in q]);jitter=np.linspace(-.065,.065,len(vals));ax.scatter(i+dx+jitter,vals,s=28,color=c,alpha=.7,label=label if i==0 else None);ax.plot([i+dx,i+dx],[min(vals),max(vals)],color=c,lw=2)
 ax.set_xticks(xx,['A  Entrance','B  Station','C  Section']);ax.set_ylabel('Samples losing all six declared round trips (of 141)');ax.set_ylim(-2,72);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False);ax.set_title('Sixteen existing parent scenarios — unweighted sample counts');fig.text(.5,.012,'Ranges reflect scenario/identity assumptions, not statistical confidence intervals or affected-person estimates.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(R/'scenario_comparison.png',dpi=170);plt.close(fig)
 # Paired coverage: sampled ground sites and station probes explicitly separate.
 btable=[]
 for r in base:
  if r['network_branch']==BR and r['cohort']=='surface_grid':btable.append([WARD[r['ward']],r['origin_probes'],r['origins_with_any_mutual_anchor'],r['origins_without_any_baseline_mutual_anchor'],r['mutual_pairs']])
 primarytable=[]
 for pol in ['primary_0p3_area100','sensitivity_normalized_identity']:
  for m in [m for m in metrics if m['policy']==pol]:primarytable.append([POL[pol],LABEL[m['shutdown_type']],f"{m['min_lost_surface_mutual_OD']}–{m['max_lost_surface_mutual_OD']} / 846",str(m['max_surface_origins_losing_all_six'])+' / 141',f"{100*m['max_fraction_of_141_baseline_covered_samples']:.1f}%",m['worst_case_by_OD']])
 senstable=[]
 for pol in POL:
  a=[m['max_surface_origins_losing_all_six'] for m in metrics if m['policy']==pol];senstable.append([POL[pol],*a])
 wtable=[]
 for pol in ['primary_0p3_area100','sensitivity_normalized_identity']:
  for r in wards:
   if r['case_id']=='train-04' and r['exposure_policy']==pol and r['shutdown_type']=='section' and r['network_branch']==BR and r['cohort']=='surface_grid':wtable.append([POL[pol],WARD[r['ward']],r['lost_all_declared_mutual_anchors'],f"{r['lost_mutual_pairs']} / {r['baseline_mutual_pairs']}",f"{float(r['lost_mutual_fraction'])*100:.1f}%"])
 # Explicit per-parent selections and effects, no cross-event pooled closure.
 eventtable=[]
 for e in bindings:
  row=[e['case_id'],e['parent_id'],e['split']]
  for pol in ['primary_0p3_area100','sensitivity_normalized_identity']:
   sel=next(x for x in select if x['case_id']==e['case_id'] and x['exposure_policy']==pol)
   q=[next(x for x in rows if x['case_id']==e['case_id'] and x['exposure_policy']==pol and x['shutdown_type']==kind and x['network_branch']==BR) for kind in TYPES]
   row.extend([sel['closed_mapped_portals']+' / '+sel['target_units_total'],' / '.join(x['surface_grid_lost_all_declared_mutual_anchors'] for x in q)])
  eventtable.append(row)
 # Coverage of targeted inventory; unknown unassessed portals remain available assumptions.
 portal=rr(P/'portal_inventory.csv');coverage_rows=[]
 pointlinks=collections.defaultdict(set)
 for x in prior:pointlinks[x['station_id']].add('osm:'+x['osm_node_id'])
 for sel in select:
  if sel['exposure_policy'] not in ['primary_0p3_area100','sensitivity_normalized_identity']:continue
  closed=set(sel['selected_portal_ids'].split(';'))-{''}
  for sid in (set(sel['entrance_evidence_station_ids'].split(';'))|set(sel['vicinity_assumed_station_ids'].split(';')))-{''}:
   ports=[x for x in portal if sid in json.loads(x['station_ids'])];actual={x['portal_id'] for x in ports if x['synthetic_unknown']!='True'};assessed={x for x in actual if x.replace('osm:','') in riskids};priorass=actual&pointlinks[sid]
   coverage_rows.append({'case_id':sel['case_id'],'exposure_policy':sel['exposure_policy'],'station_id':sid,'mapped_portals_in_normal_model':len(actual),'mapped_portals_with_surface_screen':len(assessed),'mapped_portals_with_original_risk_attribution':len(priorass),'selected_closed_mapped_portals':len(actual&closed),'mapped_portals_not_surface_screened':len(actual-assessed),'synthetic_unknown_access_ports':sum(x['synthetic_unknown']=='True' for x in ports),'unselected_or_unknown_ports_assumed_available_in_A':True,'internal_corridors_verified':False})
 if coverage_rows:
  with (R/'targeted_station_portal_coverage.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(coverage_rows[0]));w.writeheader();w.writerows(coverage_rows)
 files=[P/'scenario_policy.json',P/'closure_selection_by_parent.csv',P/'exposure_to_hypothesis_ledger.csv',P/'identity_attribution_sensitivity_links.csv',P/'identity_gaps.csv',P/'section_span_audit.csv',P/'transfer_evidence_and_assumptions.csv',P/'source_bindings.json',P/'paired_accessibility_matrices.npz',P/'scenario_index.csv',R/'scenario_accessibility_summary.csv',R/'ward_accessibility_summary.csv',R/'normal_baseline_coverage.csv',R/'primary_probe_accessibility.csv',R/'targeted_station_portal_coverage.csv',R/'validation.json',R/'summary.json']
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区交通可达性 · 停运假设对照</title>
<style>body{{margin:0;background:#f2f4f6;color:#243442;font:16px/1.75 -apple-system,BlinkMacSystemFont,"Noto Sans",Arial,sans-serif}}main{{max-width:1190px;margin:auto;padding:36px 24px 72px}}h1{{line-height:1.3;font-size:32px}}h2{{margin-top:36px;font-size:24px}}h3{{font-size:19px}}.tag{{color:#5e7586;font-size:13px;letter-spacing:1px}}section,.box{{background:white;border:1px solid #dde4e8;border-radius:12px;padding:22px 26px;margin:20px 0}}.notice{{border-left:5px solid #cf7950;background:#fff7ed}}.metrics{{display:flex;gap:16px;flex-wrap:wrap}}.metrics>div{{flex:1;background:#eaf1f5;padding:15px;min-width:160px;border-radius:8px}}strong.big{{display:block;font-size:29px;color:#244d63}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid #dde4e8;vertical-align:top}}th{{background:#eff3f6;white-space:nowrap}}.scroll{{overflow-x:auto}}img{{width:100%;height:auto;border:1px solid #e5e7eb}}a{{color:#196b96}}.small,figcaption{{font-size:13px;color:#526675}}li{{margin:8px 0}}code{{background:#eff2f4;padding:2px 4px}}details{{margin:18px 0}}summary{{cursor:pointer;font-weight:600}}.chain{{font-weight:600;background:#eaf1f5;padding:14px;border-radius:8px}}</style>
<main><p class="tag">UDT01b · 2026-10-16 · HYPOTHETICAL TRANSPORT ACCESSIBILITY</p><h1>站口、整站与区段关闭：<br>三区交通可达性会怎样变化？</h1>
<p>沿用已有16个开放父事件及其地表水深初筛，比较三类停运假设。只计算正常基线下原本可达的采样点—目的地往返连接；全程没有追加水动力模拟、训练、封存测试或受困人数计算。</p>
<div class="box notice"><b>主要判断：停运范围的假设会放大连接损失，但绝对范围对站口归属仍很敏感。</b><p>在同一0.3 m与100 m²筛选参数下，原初筛归属的区段停运最多影响22/141个已覆盖地表采样点；补充来源明确的名称规范化归属后，最多影响67/141个。二者是归属假设敏感性，不是影响范围的统计上下界。当前结果支持比较假设，不支持宣布真实交通孤岛范围。</p></div>
<div class="metrics"><div><strong class="big">16</strong>既有父事件（12训练＋4验证）</div><div><strong class="big">141 / 209</strong>正常模型具备往返连接的地表点</div><div><strong class="big">68</strong>基线未覆盖，单列为未知</div><div><strong class="big">576</strong>图可达性评估，非新增水动力情景</div></div>
<section><h2>1. 证据怎样进入停运假设</h2><p class="chain">既有模拟的地表峰值足迹 → 声明的筛选参数 → 来源支持的站口归属 → 假设删除通行边 → 与同一正常网络成对比较</p>
{table(['层次','本次使用','解释边界'],[['积水证据','站口20 m邻域；有效地表面积≥100 m²、有效比例≥80%；水深下界≥0.3 m的峰值地表足迹≥100 m²','网格开放地表平均水深的峰值包络；不是站口门槛水深或地下进水证据'],['关闭参数','0.3 m、100 m²及质量门槛在比较前保存；另测0.1/0.5 m和面积敏感性','研究者声明的情景参数，不是运营商停运或人员安全阈值'],['时序假设','每个父事件内，联合关闭其峰值触发设施；不跨事件合并','不同网格峰值不一定同时发生；属于静态压力包络，无关闭先后、持续时间和发生概率'],['归属关系',f"原初筛：{coverage['original_attributed_points']}/675个点、{coverage['original_attributed_units']}/114个站单位；补充：{coverage['normalized_attributed_points']}/675个点、{coverage['normalized_attributed_units']}/114个站单位",'补充仅做来源站区名称/运营者规范化，并校验同身份官方站体≤500 m；不做最近站归属，不证明站内通道']])}
<p>每个“站单位”是运营者＋站名，不等于已核验的完整综合车站。同名的另一运营者不自动跟随关闭。归属补充与原初筛结果并列，原始积水曲线、谱系、划分和掩膜未改变。</p>
{table(['假设','删除什么','保留什么'],[[LABEL['entrance'],'仅删除选中站口与站内枢纽之间的双向连接','其他入口、地面道路、列车运行均保留；未测入口按可用处理'],[LABEL['station'],'在触发的运营者/站名单元删除进出站、上下车及相关换乘连接','本次采用列车通过但不停靠假设，保留列车贯通服务'],[LABEL['section'],'在B基础上，再中断经过目标站单位的服务连接；含同服务族参考站序中的越站区间','其余服务假设可用；不模拟折返、调度或整线运营可实施性']])}
<p class="small">201/3554条服务边缺少完整内部站序跨度，只能用端点检查；C不是经过实测的同一隧道或股道受淹。对应审计文件保留未知项。</p></section>
<section><h2>2. 正常基线覆盖多少，比较的是哪种可达性</h2><p>固定500 m采样单元内选择真实地面步行节点，形成209个地表采样点。以JR新宿、涩谷、上野、锦糸町、大井町及Metro丰洲为六个跨区目的地，分别检查出发与返回；六个目的地及域外线路假设可用，未核验其雨中安全性。</p>
<p>初末段分别使用来源原生道路图中<strong>不超过600 m的步行路径</strong>，中间使用轨道连接；起点只能出发、终点只能到达，防止无限步行串联掩盖停运。未加入道路积水封路、公交替代、机动车、时刻表、拥堵或运能。本次衡量的是这一交通方式组合的连接存在性，不能等同于出行时间、通勤完成率或城市全面孤立。</p>
{table(['范围','地表采样点','正常有往返目的地','基线未覆盖','正常往返OD'],btable)}
<p>正常有覆盖的141个点在当前图中均可往返六个目的地，共846个OD对；其余68个点从未作为“雨后断连”计入。采样点不是人口，500 m显示单元不是孤岛面积。另设114个站区探针：不加跨运营者换乘时99个正常有往返，加入33条来源支持但双向通行仍属假设的换乘关系后104个有往返。来源列车序列覆盖106/114个核心站单位，缺失8个不补造。</p>
<p class="small">同运营者站内通行是聚合假设，非逐条核验的站内几何。513个区域站单位缺少映射入口而保留未知入口假设，仅供站区探针；它们不会凭坐标生成地表采样点的接驳。跨运营者有7条关系无法明确解析，未强行连通。</p></section>
<section><h2>3. 三类停运的成对比较</h2><p>下表使用“来源支持的跨运营者换乘按双向可用”分支，范围来自16个既有事件。损失只相对同分支正常基线计算。最大值可能出自不同事件；不是全部设施同时关闭的综合情景。</p>
{table(['归属方案','停运类型','往返OD损失范围','最多失去全部六目的地的地表点','占141个基线覆盖点','OD损失最大事件'],primarytable)}
<figure><img src="scenario_comparison.png" alt="16 events and attribution sensitivity"><figcaption>点代表既有父事件，竖线仅显示这些事件的最小/最大值。蓝色为原初筛归属，红色为归属补充；不是概率分布或置信区间。</figcaption></figure>
<p><strong>A类影响小不能证明真实入口冗余。</strong>未触发、未测或尚未归属的入口按可用保留；站内通行也采用聚合假设。B类失去本地上下车后仍可有列车穿越，C类进一步切断运行连接，因此在这套嵌套假设中损失不减，这是设定的结果。</p>
<p>两种正常换乘分支在地表点的损失计数相同；并不说明换乘证据无关。归属补充后的C类最大站区探针断连，在不加跨运营者连接/加入假设连接时为34/30个，说明站区结果仍依赖换乘假设。</p></section>
<section><h2>4. 同一事件的三区空间对照</h2><p>展示train-04（P0004），因为它在原初筛归属下具有最大的C类地表OD损失，仅作描述性展示，不用于重选训练/验证事件。A/B/C使用同一事件、同一地图范围和同一正常基线。</p>
<figure><img src="closure_maps_main.png" alt="Whole-region comparison under frozen attribution"><figcaption>原初筛归属。灰色=基线未覆盖，浅绿=六目的地均保留，橙色=部分往返OD丢失，红色=全部六个目的地的往返OD丢失；色块是采样显示单元，不是淹水/交通孤岛面积。</figcaption></figure>
<figure><img src="closure_maps_identity.png" alt="Whole-region comparison with normalized source identity"><figcaption>只补充来源站口归属，积水证据、筛选门槛及正常网络不变。空间范围明显扩大，说明应先稳定归属覆盖再作绝对风险排序。</figcaption></figure>
{table(['归属方案','区属','C类失去六目的地的点','C类往返OD损失','同区基线OD损失比例'],wtable)}
<p>在这一事件中，港区的采样点连接损失更突出；但该区正常地表采样覆盖仅58/105，且采样密度、入口资料与运营连接完整度不同，不能据此排序三区真实受困风险。</p></section>
<section><h2>5. 参数与未知连接的敏感性</h2>{table(['筛选/归属方案','A最多丢失全部六目的地点','B最多丢失全部六目的地点','C最多丢失全部六目的地点'],senstable)}
<p>各项仅改变所列筛选或归属假设；均保持16事件和正常图不变。站区回退只对原本没有站口归属的站单位增加B/C类关闭假设，采用50 m邻域、0.3 m足迹≥100 m²且占有效地表≥5%，不生成虚假站口。它不是进水证据。</p>
<p>这轮最值得优先处理的是<strong>批量核对站口—运营者/车站归属与未测入口覆盖</strong>，随后将道路积水通行限制作为独立一层接入，检验步行接驳是否保留。两步均应保留“未知可用/不可用”假设分支。在此之前不计算受困人数；正常基线缺连接的68个地表点继续保持未知。</p>
<details><summary>查看16个父事件：触发设施与地表损失</summary>{table(['事件','父事件','原划分','原归属：站口/站单位','原归属：A/B/C丢失点','补充归属：站口/站单位','补充归属：A/B/C丢失点'],eventtable)}</details></section>
<section><h2>6. 人口、核验与复现</h2><p>官方夜间人口基线496,345人与4,746个基本单位区、125/250 m楼宇分配差异和未分配/不确定性均保持原状，没有按站、采样点或损失比例加权。白天人口没有在本次补造。全部结果不包含“受困人数”。</p>
<p>已核验：A/B/C删边集合嵌套；删边不产生新可达路径；整站关闭保留通过服务；初末段步行路径不变且均≤600 m；来源绑定、人口文件及暴雨输入未变；原训练/验证事件不变。下方交付核验进一步逐矩阵重算OD损失、核对情景索引与基线分母。</p>
<p><b>复现顺序：</b><code>.venv/bin/python src/preprocessing/build_transport_shutdown_scenarios_v1.py</code> → <code>.venv/bin/python src/analyses/compare_transport_shutdown_scenarios_v1.py</code> → <code>.venv/bin/python src/analyses/report_transport_shutdown_scenarios_v1.py</code> → <code>.venv/bin/python src/analyses/validate_transport_shutdown_delivery_v1.py</code>。</p>
<ul>{''.join('<li>'+link(p)+'</li>' for p in files)}</ul>
<p class="small">来源引用与哈希保存在source_bindings.json；全部输入来自现有本地版本，未调用新水动力模拟、训练、封存测试或真实停运资料。Framelog：MILI-D-20261016-024。</p></section></main></html>'''
 (R/'review.html').write_text(body)
 artifactfiles=[p for p in sorted(P.iterdir()) if p.is_file()]+[p for p in sorted(R.iterdir()) if p.is_file() and p.name not in ['delivery_manifest.json','delivery_validation.json']]
 scripts=[ROOT/'src/preprocessing/build_transport_shutdown_scenarios_v1.py',ROOT/'src/analyses/compare_transport_shutdown_scenarios_v1.py',Path(__file__).resolve(),ROOT/'src/analyses/validate_transport_shutdown_delivery_v1.py']
 dump(R/'delivery_manifest.json',{'human_decision':'MILI-D-20261016-024','artifacts':[{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in artifactfiles+scripts if p.exists()],'policy_sha256':sha(P/'scenario_policy.json'),'source_bindings_sha256':sha(P/'source_bindings.json'),'new_hydraulics':False,'people_counted':False})
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
