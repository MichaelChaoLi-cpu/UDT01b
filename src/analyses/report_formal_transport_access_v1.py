"""Regional offline review of the complete matched transport matrix."""
from pathlib import Path
import os,sys,json,collections,html
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import shape
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_formal_transport_access_v1 import P,R,OLD,STRICT,CASES,PROJECTIONS,KINDS,D,F,read,write,dump,load,sha
from compare_urban_geometry_pair_controls_v1 import table
CODES={'insufficient_evidence':0,'robust_modeled_disconnection':1,'conditional_sensitivity':2,'stable_modeled_alternative':3}
COLORS=['#919aa2','#b74646','#d89832','#39837a'];WARD={'13101':'千代田区','13102':'中央区','13103':'港区','boundary_or_unassigned':'边界／区属未定'}
CN={'no_assumed_rail_shutdown':'无轨道停运假设','entrance':'站口关闭','station':'整站关闭、列车通过','section':'区段停运'}

def main():
 audit=json.loads((R/'independent_validation.json').read_text());routes=json.loads((R/'route_witness_validation.json').read_text());assert audit['all_passed']and routes['all_passed']
 summary=json.loads((R/'analysis_summary.json').read_text());classes=read(R/'probe_classification.csv');agg=read(R/'classification_summary.csv');changes=read(R/'geometry_classification_changes.csv');metrics=read(R/'scenario_accessibility_summary.csv');probes=read(OLD/'surface_probe_inventory.csv');support=load(P/'projection_support.npz');contract=json.loads((P/'analysis_contract.json').read_text());frames=contract['frame_ids']
 contexts={}
 for row in read(ROOT/'data/exp/estimation/three_ward_historical_transport_islands_v1/probe_accessibility.csv'):contexts[row['probe_id']]=row['towns_intersecting_display_cell']
 index={(x['case_id'],x['frame_id'],x['projection'],x['shutdown_type'],x['probe_id']):x for x in classes}
 candidates=[]
 for row in classes:
  if row['frame_id']=='14400'and row['shutdown_type']=='section'and row['projection']=='ground_path_supplement'and row['outward_classification']in ['robust_modeled_disconnection','conditional_sensitivity']:
   candidates.append(dict(**row,display_cell_town_context=contexts.get(row['probe_id'],''),context_not_whole_town=True))
 write(R/'candidate_location_context_19jst.csv',candidates)
 # Output GIS points: point classifications do not label complete neighborhoods.
 features=[]
 for p in probes:
  key=('R1__H__U','14400','ground_path_supplement','section',p['probe_id']);r=index[key];strong=index['R125__H__U','14400','ground_path_supplement','section',p['probe_id']]
  features.append(dict(type='Feature',geometry=dict(type='Point',coordinates=[float(p['x_m']),float(p['y_m'])]),properties=dict(probe_id=p['probe_id'],ward=p['ward'],display_cell_town_context=contexts.get(p['probe_id'],''),R1_H_U_outward=r['outward_classification'],R125_H_U_outward=strong['outward_classification'],original_or_added_cohort=r['cohort'],time_s=14400,shutdown_assumed='section',projection='ground_path_supplement',not_whole_town_or_population=True)))
 dump(P/'probe_access_19jst_6677.geojson',dict(type='FeatureCollection',name='Conditional access probe classifications at19JST',crs=dict(type='name',properties=dict(name='EPSG:6677')),features=features))
 # Compact interactive codes cover all2496 directional states without a huge raw table.
 states=collections.defaultdict(lambda:{'outward':[],'two_direction':[]})
 for row in classes:
  key='|'.join(row[x]for x in ['case_id','frame_id','projection','shutdown_type'])
  for direction in ['outward','two_direction']:states[key][direction].append(str(CODES[row[direction+'_classification']]))
 states={k:{d:''.join(vals)for d,vals in v.items()}for k,v in states.items()};assert all(len(x)==209 for v in states.values()for x in v.values())
 payload=dict(states=states,points=[dict(id=p['probe_id'],ward=WARD.get(p['ward'],p['ward']),context=contexts.get(p['probe_id'],''),x=(float(p['x_m'])+11360)/10,y=(-32320-float(p['y_m']))/10,cohort='原115点'if support['original_supported'][j]else'补缺新增29点'if support['added_supported'][j]else'原证据不足点')for j,p in enumerate(probes)])
 dump(R/'map_states.json',payload)
 wdata=np.load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');wardgrid=wdata['Ward Code'];background=''
 for geo,value in shapes(wardgrid.astype('uint8'),mask=wardgrid>0,transform=from_origin(-11360,-32320,10,10)):
  g=shape(geo).simplify(35,preserve_topology=True);polys=list(g.geoms)if g.geom_type=='MultiPolygon'else[g]
  for poly in polys:
   path=''
   for ring in [poly.exterior]+list(poly.interiors):
    coords=list(ring.coords);path+='M'+' L'.join(f'{(x+11360)/10:.1f},{(-32320-y)/10:.1f}'for x,y in coords)+'Z'
   background+='<path d="'+path+'" fill="#e8eef0" fill-rule="evenodd" stroke="#a4b5be" stroke-width="1"/>'
 # Fixed-time comparison map; every panel contains the same209 probes.
 fig,axs=plt.subplots(2,3,figsize=(13,10),layout='constrained');extent=[-11.36,-3.66,-42.44,-32.32]
 for ai,arm in enumerate(['T','U']):
  for ri,rain in enumerate(['R0','R1','R125']):
   ax=axs[ai,ri];ax.imshow(np.where(wardgrid>0,wardgrid,np.nan),extent=extent,origin='upper',cmap='Greys',vmin=0,vmax=15,alpha=.35)
   rows=[index[f'{rain}__H__{arm}','14400','ground_path_supplement','section',p['probe_id']]for p in probes]
   for code in CODES:
    rr=[x for x in rows if x['outward_classification']==code];ax.scatter([float(x['x_m'])/1000 for x in rr],[float(x['y_m'])/1000 for x in rr],s=22,color=COLORS[CODES[code]],edgecolor='white',linewidth=.25)
   robust=sum(r['outward_classification']=='robust_modeled_disconnection'for r in rows);ax.set_title(f'{rain} | {arm} | robust outward loss: {robust}',fontsize=11);ax.set_xlim(-11.36,-3.66);ax.set_ylim(-42.44,-32.32);ax.set_aspect('equal');ax.set_xlabel('Easting (km)')
   if ri==0:ax.set_ylabel('Northing (km)')
 handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,label=l,markersize=7)for c,l in zip(COLORS,['Insufficient evidence','Robust modeled outward loss','Assumption-sensitive','Retained alternative'])]
 fig.legend(handles=handles,loc='outside lower center',ncol=2);fig.suptitle('Three wards |19:00 JST | raised-stage assumption | section suspension | supplemented paths')
 fig.savefig(R/'three_ward_geometry_access_19jst.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for si,stage in enumerate(['L','H']):
  for rain,arm,color,style in [('R1','T','#6686a1','--'),('R1','U','#246479','-'),('R125','T','#bc9a58','--'),('R125','U','#ae553c','-')]:
   rr=[r for r in agg if r['case_id']==f'{rain}__{stage}__{arm}'and r['frame_id']!='peak_envelope'and r['ward']=='all'and r['projection']=='ground_path_supplement'and r['shutdown_type']=='section'and r['direction']=='outward'];rr.sort(key=lambda r:float(r['frame_id']));axs[si].plot([15+float(r['frame_id'])/3600 for r in rr],[int(r['robust_modeled_disconnection'])for r in rr],style,marker='o',markersize=3,color=color,label=rain+' '+arm)
  axs[si].set_title(stage+' stage assumption');axs[si].set_xlabel('Hour JST');axs[si].set_ylabel('Supported probes with robust outward loss');axs[si].grid(alpha=.2);axs[si].legend()
 fig.suptitle('Fixed synchronous states; connecting lines do not establish disruption duration');fig.savefig(R/'synchronous_access_comparison.png',dpi=150);plt.close(fig)
 keyrows=[r for r in agg if r['frame_id']=='14400'and r['ward']=='all'and r['projection']=='ground_path_supplement'and r['direction']=='outward']
 geometry19=[r for r in changes if r['frame_id']=='14400'and r['projection']=='ground_path_supplement'and r['direction']=='outward'and r['cohort']=='all'and r['shutdown_type']=='section']
 headline=dict(**summary,fixed19jst_supplemented_outward=keyrows,geometry19jst_section=geometry19,route_witnesses=routes['records'],route_witnesses_reachable=routes['reachable_routes'],independent_checks=len(audit['checks']),not_observed_shutdown=True,not_entire_town_or_population=True)
 dump(R/'delivery_summary.json',headline)
 body='<h1>三区交通孤岛：12组积水 × 四类轨道停运假设</h1><p class="lead">已完成9,984组网络评估，核查地面替代、出向与返回联系，并在同一交通图上比较仅地形T和城市约束U。</p>'
 body+='<aside><b>这是模型内、假设条件下的交通联系损失。</b><p>209个既有分析点中，严格道路层115点有共同支持，步道补缺层144点；另外65点继续保留为证据不足。点数不是人口、街区数量或受困人数。水位抬升和轨道关闭均是假设，不能称为实际灾情或真实停运。</p></aside>'
 body+='<h2>主要发现与几何对照</h2><p>保持原19:00同步时刻、步道补缺和区段停运假设，原雨量低／高水位的稳健出向断连点由T的11／11变为U的13／14；增强雨量时由12／12变为20／21。城市信息会改变部分交通断连判断；这仍不是其真实准确性优于T的验证。</p><p>原雨量抬升水位的U组，即便不假设轨道停运，也有13个稳健出向断连点；区段停运假设下为14。这个指标显示道路限制已造成多数完全出向联系损失，不能据此说轨道停运影响很小：这里允许到达任一既定目标，未评价每对OD、班次、容量和出行时间。</p>'
 body+=table(['雨量','水位','T稳健出向断连','U稳健出向断连','U新增稳健点','U移出稳健点','分类变化点'],[[r['rainfall'],r['stage'],r['T_robust'],r['U_robust'],r['new_robust_in_U'],r['removed_robust_in_U'],r['classification_changes']]for r in geometry19])
 body+='<img src="three_ward_geometry_access_19jst.png" alt="Whole three-ward matched access probe maps"><h2>切换情景与时刻</h2><p>地图显示既有分析点。灰色表示证据不足；绿色表示在所列分支中保留模型内替代，不能称为安全或交通正常。点击点可看区属、位置上下文与证据队列。</p>'
 body+='<div class="controls"><label>物理情景<select id="case">'+''.join('<option value="'+c+'"'+(' selected'if c=='R1__H__U'else'')+'>'+c+'</option>'for c in CASES)+'</select></label><label>时刻<select id="frame">'+''.join('<option value="'+f+'"'+(' selected'if f=='14400'else'')+'>'+('非同时峰值包络'if f=='peak_envelope'else f'{15+int(f)//3600:02d}:{int(f)%3600//60:02d} JST')+'</option>'for f in frames)+'</select></label>'
 body+='<label>道路支持<select id="projection"><option value="strict_classified_road">严格分类道路</option><option value="ground_path_supplement" selected>地面步道补缺</option></select></label><label>轨道假设<select id="shutdown">'+''.join('<option value="'+k+'"'+(' selected'if k=='section'else'')+'>'+CN[k]+'</option>'for k in KINDS)+'</select></label><label>方向<select id="direction"><option value="outward">出向替代</option><option value="two_direction">双向目标集联系</option></select></label></div>'
 body+='<p id="state-note"></p><p id="counts"></p><div class="mapwrap"><svg id="map" viewBox="0 0 770 1012" role="img" aria-label="Conditional transport-access probe map">'+background+'<g id="points"></g></svg><div><p><span style="color:#b74646">●</span>模型内稳健联系损失</p><p><span style="color:#d89832">●</span>条件敏感</p><p><span style="color:#919aa2">●</span>证据不足</p><p><span style="color:#39837a">●</span>保留模型内替代</p><p id="detail">点击分析点查看位置上下文。</p><p>黑色外圈标记补缺新增29点。</p></div></div>'
 body+='<h2>固定19:00的四类停运对照</h2>'+table(['算例','轨道假设','稳健出向断连点','条件敏感点','证据不足点','保留替代点'],[[r['case_id'],CN[r['shutdown_type']],r['robust_modeled_disconnection'],r['conditional_sensitivity'],r['insufficient_evidence'],r['stable_modeled_alternative']]for r in keyrows])
 body+='<p>每一行在同一个道路支持方案和停运类型内，跨两档道路阈值（0.1／0.3 m）、两种站口归属、两个未知链接分支共8种设置分类。严格道路与步道补缺各自单列，不能把补缺条件下的稳健性说成跨所有证据模型均已证实。</p><img src="synchronous_access_comparison.png" alt="Synchronous sampled-state access comparison">'
 body+='<h2>哪些位置需要关注</h2><p>以下为原雨量或增强雨量、抬升水位U组在19:00区段停运假设下的稳健出向断连分析点。片区名称来自既有展示单元相交上下文，只帮助定位，不能解释为整个町丁目断连。</p>'
 candidate_table=[]
 for p in probes:
  a=index['R1__H__U','14400','ground_path_supplement','section',p['probe_id']];b=index['R125__H__U','14400','ground_path_supplement','section',p['probe_id']]
  if a['outward_classification']=='robust_modeled_disconnection'or b['outward_classification']=='robust_modeled_disconnection':candidate_table.append([WARD.get(p['ward'],p['ward']),contexts.get(p['probe_id'],''),p['probe_id'],a['cohort'],'稳健断连'if a['outward_classification']=='robust_modeled_disconnection'else'条件敏感或保留','稳健断连'if b['outward_classification']=='robust_modeled_disconnection'else'条件敏感或保留'])
 body+=table(['区属','位置上下文','分析点','证据队列','原雨量','增强雨量'],candidate_table)
 body+='<h2>替代路径和绕行</h2><p>已对12组物理情景、两种道路支持和两个未知分支，在固定19:00及0.3 m参数下，为全209点导出出向和返回的原生步行路径证据，共'+f"{routes['records']:,}"+'条记录，其中'+f"{routes['reachable_routes']:,}"+'条可达。每条路径的方向、未关闭原生弧、长度和目标均核验；未知高程路段及补缺路段在路径中明确计数。到达的是三区分析边界外节点，不是已核验的安全避难点。</p><p>共享最短路径路段表记录不同分析点选用同一路段的次数；不代表实际客流、交通容量或唯一瓶颈。混合步行／轨道结果的距离是步行部分，轨道边计零步行长度，不能当成总行程距离或旅行时间。</p>'
 body+='<h2>双向与停运假设的含义</h2><ul><li>出向替代：能够走到任一原生跨区边界目标，或通过步行＋轨道到达任一既定外部锚点。</li><li>双向目标集联系：同时存在至少一个出向和至少一个返回替代，两方向可使用不同目标和方式；未证明往返同一车站或实际出行时序可行。</li><li>无轨道停运：保留所有轨道弧，道路积水限制仍适用。</li><li>站口关闭：只关闭假设触发入口相关连接；整站关闭进一步关闭上下车与相应换乘，保留列车通过；区段停运进一步切断关联运行边。</li></ul>'
 body+='<p>桥隧、高架与地下分层和原通行权限保持原样。未知分支各有对应正常基线，原本无支持或无正常联系的点不会被计成暴雨新增断连。站口触发依据20 m邻域的地表积水，不等于入口实际进水；未建模地下传播、地铁运营调度、公交和车辆容量。</p>'
 body+='<h2>验证与可解释范围</h2><p>另以原始NetCDF状态和分布在三区的细地形单元，直接核验了3,348项同步水深对照；离线地图的情景切换、209点覆盖及点击交互也已通过浏览器检查。</p><p>完成'+str(len(audit['checks']))+'项独立校验及'+str(len(routes['checks']))+'项路径证据检查，包括全部9,984组关闭语义与双向最短路径复算、1,248组道路限制复算，以及完整复现先前严格／补缺层的1,248组历史U网络结果。原115点与新增29点保持各自队列；人口不确定性未改变，未计算受困人数。</p><p>全部雨量缩放、水位和几何后代仍属于同一历史父事件。结论限于既定图、目的地集合与情景，未验证实际停运、真实城市瘫痪范围或城市几何的真实精度优势。12个时刻是同步切片，不能给出精确发生时刻或连续断连时长；峰值包络始终单列。</p><p>下一步可在这套基线上比较站口防护和地面替代连接等主动措施，并在计算受影响人口前补充白天人口与出行目的地信息。</p>'
 body+='<h2>复现文件</h2><ul>'
 files=[P/'analysis_contract.json',R/'analysis_summary.json',R/'delivery_summary.json',R/'normal_baseline_and_support.csv',R/'classification_summary.csv',R/'probe_classification.csv',R/'geometry_classification_changes.csv',R/'scenario_accessibility_summary.csv',R/'candidate_location_context_19jst.csv',R/'shared_shortest_path_segments_19jst.csv',P/'probe_access_19jst_6677.geojson',P/'surface_route_witnesses_19jst.jsonl',P/'scenario_index.csv',P/'road_context_index.csv',P/'station_closure_selection.csv',R/'independent_validation.json',R/'route_witness_validation.json',R/'preparation_validation.json',R/'projection_direct_validation.json',R/'browser_validation.json',P/'source_bindings.json',R/'delivery_validation.json',R/'delivery_manifest.json']
 for f in files:body+='<li><a href="'+os.path.relpath(f,R)+'">'+html.escape(f.name)+'</a></li>'
 body+='</ul><p>Framelog: MILI-D-20261017-007。</p>'
 js='''const DATA=__DATA__;const colors=['#919aa2','#b74646','#d89832','#39837a'];const labels=['证据不足','模型内稳健联系损失','条件敏感','保留模型内替代'];const ids=['case','frame','projection','shutdown','direction'];function draw(){const vals=ids.map(id=>document.getElementById(id).value);const codes=DATA.states[vals.slice(0,4).join('|')][vals[4]];const group=document.getElementById('points');group.replaceChildren();let counts=[0,0,0,0];DATA.points.forEach((p,i)=>{const n=Number(codes[i]);counts[n]++;const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',7);c.setAttribute('fill',colors[n]);c.setAttribute('stroke',p.cohort==='补缺新增29点'?'#273b47':'white');c.setAttribute('stroke-width',1.4);c.style.cursor='pointer';const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=p.ward+' · '+p.context+' · '+p.id+' · '+labels[n];c.append(title);c.addEventListener('click',()=>{document.getElementById('detail').textContent=p.ward+'；位置上下文：'+p.context+'；分析点：'+p.id+'；队列：'+p.cohort+'；分类：'+labels[n]+'。此标签不代表整个町丁目。'});group.append(c);});document.getElementById('counts').textContent=labels.map((v,i)=>v+' '+counts[i]+'点').join(' ｜ ');document.getElementById('state-note').textContent=(vals[1]==='peak_envelope'?'当前为非同时峰值包络，不能用于同步孤岛或时长判断。':'当前为统一实际时刻切片。')+(vals[4]==='two_direction'?' 双向联系损失表示至少一个方向缺少替代，不表示两个方向都不通。':' 当前检查出向替代。');}ids.forEach(id=>document.getElementById(id).addEventListener('change',draw));draw();'''
 js=js.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区正式交通孤岛情景对照</title><style>body{font:16px/1.8 system-ui;max-width:1260px;margin:auto;padding:30px;background:#f4f7f9;color:#263f4d}h1,h2{line-height:1.4}h2{margin-top:36px}.lead{font-size:19px}aside{background:#e5f0ed;border-left:5px solid #3b8372;padding:20px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #d5e0e7;padding:9px;text-align:left}th{background:#e5edf2}.scroll{overflow:auto}img{max-width:100%}.controls{display:flex;gap:14px;flex-wrap:wrap}.controls label{display:grid;gap:4px}select{padding:8px;font:inherit}.mapwrap{display:grid;grid-template-columns:minmax(260px,650px) minmax(180px,1fr);gap:25px;background:white;padding:20px}svg{max-height:720px;width:100%;background:#fafcfd}a{color:#17658e}@media(max-width:700px){.mapwrap{grid-template-columns:1fr}body{padding:16px}}</style>'+body+'<script>'+js+'</script></html>'
 (R/'review.html').write_text(page)
 # Registration page links the scientific result; original frozen protocol and earlier reports stay intact.
 path=D/'review.html';text=path.read_text();marker='<!-- FORMAL_TRANSPORT_RESULTS -->'
 if marker not in text:text=text.replace('<h2>首批与完整水动力矩阵</h2>',marker+'<p><a href="'+os.path.relpath(R/'review.html',D)+'">已完成：三区正式交通可达性与几何对照结果</a></p><h2>首批与完整水动力矩阵</h2>');path.write_text(text)
 print('REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
