"""Regional review of a fixed assumed protection comparison, with spatial and branch support."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,gzip,collections,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from shapely.geometry import shape,mapping
from shapely.ops import transform
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import from_origin
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/analyses'),str(ROOT/'src/preprocessing')]
from compare_motor_road_protection_v1 import P,R,SOURCE,N,ARMS,read,write,dump,load,sha
from compare_urban_geometry_pair_controls_v1 import table
LABEL={'none':'不干预','corridor':'关键通道保障','distributed':'分散道路保障'}
COLORS=['#969fa7','#b94a48','#db9d35','#378a79'];CODES={'insufficient_evidence':0,'robust_modeled_loss':1,'conditional_sensitivity':2,'retained_access':3}

def main():
 assert json.loads((R/'validation.json').read_text())['all_passed'];protocol=json.loads((P/'protocol.json').read_text());agg=read(R/'classification_summary.csv');pc=read(R/'probe_classification.csv');burden=read(R/'directional_benefits.csv');probes=read(SOURCE/'probe_inventory.csv');selected=read(P/'protected_road_inventory.csv');masks=load(P/'protected_roads.npz')['protected'];si=read(SOURCE/'scenario_index.csv');a=load(P/'accessibility.npz')['road_distance_m'];support=load(SOURCE/'evidence_support.npz');index={(r['arm'],r['case_id'],r['frame_id'],r['direction']):r for r in agg}
 grid=load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')['Ward Code'];extent=[-11.36,-3.66,-42.44,-32.32]
 wanted={r['source_edge_id']for r in selected};geom={}
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as f:
  for line in f:
   e=json.loads(line)
   if e['edge_id']in wanted:geom[e['edge_id']]=e['geometry']
 assert len(geom)==len(wanted)
 proj=Transformer.from_crs(6677,4326,always_xy=True);features=[]
 for r in selected:features.append(dict(type='Feature',properties=r,geometry=mapping(transform(proj.transform,shape(geom[r['source_edge_id']])))))
 dump(P/'protected_roads_4326.geojson',dict(type='FeatureCollection',features=features))
 fig,axs=plt.subplots(1,2,figsize=(11,8),layout='constrained')
 roadstats=[]
 for i,arm in enumerate(ARMS[1:]):
  ax=axs[i];ax.imshow(np.where(grid>0,grid,np.nan),origin='upper',extent=extent,cmap='Greys',vmin=0,vmax=15,alpha=.35)
  rows=[r for r in selected if r['arm']==arm];lines=[np.array(geom[r['source_edge_id']]['coordinates'])/1000 for r in rows];ax.add_collection(LineCollection(lines,colors=['#326fa8'if arm=='corridor'else'#d58b28'],linewidths=2));ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xlabel='Easting (km)',ylabel='Northing (km)',title=f'{arm.title()} | {len(rows)} segments | {protocol["actual_length_m"][arm]/1000:.3f} km')
  bins={(int(float(r['midpoint_x_m'])//500),int(float(r['midpoint_y_m'])//500))for r in rows};roadstats.append(dict(arm=arm,source_segments=len(rows),source_length_m=protocol['actual_length_m'][arm],geographic_500m_midpoint_strata=len(bins)))
 fig.suptitle('Fixed assumed passability protection | source-road length matched');fig.savefig(R/'protected_road_plans.png',dpi=150);plt.close(fig);write(R/'plan_inventory_summary.csv',roadstats)
 # Fixed19:00 whole-region probe comparison, not area/population estimates.
 fig,axs=plt.subplots(2,3,figsize=(12,10),layout='constrained')
 pi={(r['arm'],r['case_id'],r['frame_id'],r['probe_id']):r for r in pc}
 for row,rain in enumerate(['R1','R125']):
  case=rain+'__H__U'
  for col,arm in enumerate(ARMS):
   ax=axs[row,col];ax.imshow(np.where(grid>0,grid,np.nan),origin='upper',extent=extent,cmap='Greys',vmin=0,vmax=15,alpha=.35)
   vals=[pi[arm,case,'14400',pr['source_display_cell_id']]['outward_classification']for pr in probes]
   for k,color in zip(CODES,COLORS):
    pp=[pr for pr,v in zip(probes,vals)if v==k];ax.scatter([float(pr['x_m']or pr['cell_center_x_m'])/1000 for pr in pp],[float(pr['y_m']or pr['cell_center_y_m'])/1000 for pr in pp],s=23,c=color,edgecolor='white',linewidth=.25)
   robust=index[arm,case,'14400','outward']['robust_modeled_loss'];ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',title=f'{rain} | {arm} | robust exit loss: {robust}',xlabel='Easting (km)')
 fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,label=l)for c,l in zip(COLORS,['Insufficient evidence','Robust modeled loss','Assumption-sensitive','Retained alternative'])],loc='outside lower center',ncol=2);fig.suptitle('19:00 JST | raised stage | urban constraints | same93 outward probes');fig.savefig(R/'access_protection_19jst.png',dpi=150);plt.close(fig)
 fixed=[]
 for case in protocol['main_transfer_cases']+['R1__L__U']:
  for arm in ARMS:
   r=index[arm,case,'14400','outward'];base=index['none',case,'14400','outward'];fixed.append(dict(**r,robust_loss_reduction=int(base['robust_modeled_loss'])-int(r['robust_modeled_loss'])))
 write(R/'fixed19jst_summary.csv',fixed)
 timecmp=[]
 for case in sorted({r['case_id']for r in agg}):
  for arm in ARMS[1:]:
   for direction in ['outward','two_direction']:
    rows=[r for r in agg if r['case_id']==case and r['arm']==arm and r['direction']==direction and r['simultaneous']=='True'];diff=[int(index['none',case,r['frame_id'],direction]['robust_modeled_loss'])-int(r['robust_modeled_loss'])for r in rows];timecmp.append(dict(case_id=case,arm=arm,direction=direction,synchronized_slices=len(rows),slices_with_lower_robust_loss=sum(d>0 for d in diff),max_robust_loss_reduction=max(diff),not_duration=True))
 write(R/'synchronized_slice_comparison.csv',timecmp)
 robust_keys={(r['case_id'],r['frame_id'],r['direction']):int(r['robust_modeled_loss'])for r in agg if r['arm']=='none'}
 domination=all(int(r['robust_modeled_loss'])<=robust_keys[r['case_id'],r['frame_id'],r['direction']]for r in agg)
 high=[r for r in burden if r['case_id']in ['R1__H__U','R125__H__U']and r['frame_id']=='14400'and float(r['threshold_m'])==.3 and r['direction']=='outward']
 summary=dict(status='qualified_complete',new_intervention_evaluations=1248,reused_no_intervention=624,physical_cases=12,synchronized_slices=12,separate_non_simultaneous_peak=True,actual_length_m=protocol['actual_length_m'],length_difference_m=abs(protocol['actual_length_m']['corridor']-protocol['actual_length_m']['distributed']),outward_common_probes=93,two_direction_common_probes=89,insufficient_outward_probes=116,common_outward_direction_pairs=117,fixed19jst=fixed,paired_high_scenarios=high,no_new_robust_loss=domination,new_hydraulics=False,people_counted=False,engineering_feasibility_demonstrated=False)
 dump(R/'summary.json',summary)
 bg=''
 for geo,value in shapes(grid.astype('uint8'),mask=grid>0,transform=from_origin(-11360,-32320,10,10)):
  g=shape(geo).simplify(35,preserve_topology=True)
  for poly in list(g.geoms)if g.geom_type=='MultiPolygon'else[g]:
   path=''
   for ring in [poly.exterior]+list(poly.interiors):path+='M'+' L'.join(f'{(x+11360)/10:.1f},{(-32320-y)/10:.1f}'for x,y in ring.coords)+'Z'
   bg+='<path d="'+path+'" fill="#e8edf1" stroke="#b2c0c8" stroke-width="1" fill-rule="evenodd"/>'
 groups=collections.defaultdict(list)
 for i,s in enumerate(si):groups[s['case_id']+'|'+s['frame_id']].append(i)
 states={}
 for key,ids in groups.items():
  states[key]={}
  for ai,arm in enumerate(ARMS):
   finite=np.isfinite(a[ai,ids]);outs=finite[:,0].any(axis=1);two=finite.any(axis=2).all(axis=1);states[key][arm]={}
   for d,values,eligible in [('outward',outs,support['eligible_outward']),('two_direction',two,support['eligible_two_direction'])]:
    states[key][arm][d]=''.join('0'if not eligible[j]else'1'if not values[:,j].any()else'3'if values[:,j].all()else'2'for j in range(209))
 payload=dict(states=states,points=[dict(id=p['source_display_cell_id'],x=(float(p['x_m']or p['cell_center_x_m'])+11360)/10,y=(-32320-float(p['y_m']or p['cell_center_y_m']))/10)for p in probes]);dump(R/'map_states.json',payload)
 def n(x):return '—'if x in [None,'']else f'{float(x):.1f}'
 mainrows=[r for r in fixed if r['case_id']in ['R1__H__U','R125__H__U']]
 body='<h1>三区道路主动措施：有限保障长度对照</h1><p class="lead">固定约4.77 km的道路保障，比较关键通道方案、分散方案与不干预。</p><aside><b>本轮已完成1248组干预网络计算，复用624组不干预结果。</b><p>保障仅表示假设所选地面道路保持可通行；水深、水位、地形与其他道路不变。不能据此认定挡水、抽排或抬路工程已可行，也未模拟干预造成的积水转移、车流、排队和受困人数。</p></aside>'
 original=index['none','R1__H__U','14400','outward'];key=index['corridor','R1__H__U','14400','outward'];strong=index['none','R125__H__U','14400','outward'];strongkey=index['corridor','R125__H__U','14400','outward']
 body+=f'<p><b>固定方案在抬升水位下仍有模型内改善：</b>原雨量的稳健出向断连从{original["robust_modeled_loss"]}降至{key["robust_modeled_loss"]}，增强雨量从{strong["robust_modeled_loss"]}降至{strongkey["robust_modeled_loss"]}。本次分散方案在这两组的稳健断连数未减少。原雨量下四分支均有替代的点仅由{original["retained_access"]}增至{key["retained_access"]}，增强雨量由{strong["retained_access"]}增至{strongkey["retained_access"]}，主要改善是从稳健断连转为条件敏感，不能称为全面恢复。</p>'
 body+='<h2>固定19:00的跨区出向联系</h2>'+table(['情景','方案','稳健断连点','比不干预减少','条件敏感点','四分支均有替代','证据不足'],[[r['case_id'],LABEL[r['arm']],r['robust_modeled_loss'],r['robust_loss_reduction'],r['conditional_sensitivity'],r['retained_access'],r['insufficient_evidence']]for r in mainrows])
 body+='<p>所有方案沿用93个出向、89个双向共同支持点。116个出向证据不足点始终单列；点数不代表街区数、面积或人口。稳健断连是0.1／0.3 m阈值与两档未知道路分支下均失去出向联系；减少稳健断连可能只转为条件敏感，不等于四分支全部恢复。</p><img src="access_protection_19jst.png" alt="Whole-region fixed19JST protection comparison">'
 body+='<h2>道路方案怎样选</h2><p>两方案只用原雨量、低水位、城市约束U组的19:00、0.3 m及未知道路可用分支选路。候选为2855条有积水证据的已知地面车行源路段。关键通道方案按正常最短路修复包的方向恢复数量／新增长度逐次选择；分散方案按500 m网格空间分散轮选，不按可达性效果排名。两者均为一次固定的启发式方案，没有求全局最优，也没有寻找有利的随机种子。两方案同时改变选路依据与空间分布，差异不能单独归因于空间集中度。</p>'
 body+=table(['方案','源路段数','整段中心线长度m','覆盖500m中点网格数'],[[LABEL[r['arm']],r['source_segments'],n(r['source_length_m']),r['geographic_500m_midpoint_strata']]for r in roadstats])
 body+=f'<p>实际长度差 {summary["length_difference_m"]:.3f} m。长度按与三区相交的完整源路段计一次，不是裁剪后的区内长度、车道公里或等工程成本。5 km上限内的完整修复包实际使用约4.77 km，分散方案匹配其实际用量。</p><img src="protected_road_plans.png" alt="Frozen road-protection plans">'
 body+='<h2>固定方案能否迁移到更强雨量和抬升水位</h2><p>选路后固定保障道路，在12组已有积水、12个统一实际时刻、两档阈值和两档未知分支上检验。非同时峰值包络单列，不计算持续时间。R125是原雷达雨场×1.25，H是既定假设抬升水位；这些仍属同一历史父事件，不能当作独立暴雨验证。选路情景的改善具有设计内偏向，不能据此宣称所有关键通道方案普遍优于所有分散方案。</p>'
 body+='<div class="controls"><label>情景<select id="case">'+''.join('<option'+(' selected'if c=='R125__H__U'else'')+'>'+c+'</option>'for c in sorted({s['case_id']for s in si}))+'</select></label><label>时刻<select id="frame">'+''.join('<option value="'+f+'"'+(' selected'if f=='14400'else'')+'>'+('非同时峰值包络'if f=='peak_envelope'else f'{15+int(f)//3600:02d}:{int(f)%3600//60:02d} JST')+'</option>'for f in dict.fromkeys(s['frame_id']for s in si))+'</select></label><label>联系方向<select id="direction"><option value="outward">车辆出向</option><option value="two_direction">出向及返回均有替代</option></select></label></div><p id="note"></p><div class="maps">'+''.join('<section><h3>'+LABEL[arm]+'</h3><p id="counts-'+arm+'"></p><svg id="map-'+arm+'" viewBox="0 0 770 1012">'+bg+'<g id="points-'+arm+'"></g></svg></section>'for arm in ARMS)+'</div><p id="detail">点击分析点查看其编号与分支分类；不代表整个街区。</p>'
 body+='<h2>恢复了哪些方向；原本能走的路少绕多少</h2><p>以下为19:00、0.3 m假设，正常两分支共同支持的117个“分析点→边界方向”对。新恢复方向单列；距离节省仅在不干预时仍能走的同一批方向对上比较。原本断连的方向不混入距离节省，也不把方向对当成交通流量。</p>'
 body+=table(['情景','未知分支','方案','失去方向对','新恢复方向对','原可达对数','节省距离中位数m','节省距离P90m'],[[r['case_id'],r['unknown_branch'],LABEL[r['arm']],r['lost_direction_pairs'],r['restored_direction_pairs'],r['baseline_reachable_pairs'],n(r['paired_distance_saving_median_m']),n(r['paired_distance_saving_p90_m'])]for r in high])
 body+='<h2>证据与边界</h2><ul><li>只解除被选道路的积水关闭门，保留原车行权限、单行、转向及分层。桥隧、域外或高程不明道路的原未知分支保持不变。</li><li>保障全时段有效且不设工程失效水深，是理想化可通行假设；真实措施的有效水深、费用、流量容量和邻区积水影响尚未评估。</li><li>仅判定到研究边界方向的模型内联系，不代表到达安全地点或完成完整城市行程；双向替代也不保证同一出口、同一时刻往返。</li><li>这是同一套城市数据和历史雨场上的条件措施比较。没有实测交通断连／措施验证，不能把模型内改善写成实际防灾效果。</li><li>原人口基线和楼宇分配区间未改变，也未用于本轮选路或人数计算。</li></ul>'
 body+='<h2>可复核输出</h2><ul>'
 for f in [P/'protocol.json',P/'protected_road_inventory.csv',P/'protected_roads_4326.geojson',P/'selection_steps.json',P/'planning_route_packages.json',R/'summary.json',R/'classification_summary.csv',R/'probe_classification.csv',R/'directional_benefits.csv',R/'synchronized_slice_comparison.csv',R/'validation.json',R/'path_validation.json',R/'browser_validation.json',R/'delivery_validation.json',R/'delivery_manifest.json']:
  body+='<li><a href="'+os.path.relpath(f,R)+'">'+f.name+'</a></li>'
 body+='</ul><p>研究决定：MILI-D-20261017-010。分散方案最初的坐标顺序偏差在查看效果前已修正；初版选路及停止记录保留在 processed 下的 superseded_selection_before_outcome_review。</p>'
 js='''const DATA=__DATA__;const arms=['none','corridor','distributed'];const colors=['#969fa7','#b94a48','#db9d35','#378a79'];const labels=['证据不足','稳健断连','条件敏感','保留替代'];function draw(){const key=document.getElementById('case').value+'|'+document.getElementById('frame').value;const d=document.getElementById('direction').value;for(const arm of arms){const values=DATA.states[key][arm][d],g=document.getElementById('points-'+arm);g.replaceChildren();let counts=[0,0,0,0];DATA.points.forEach((p,i)=>{let v=+values[i];counts[v]++;let c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',7);c.setAttribute('fill',colors[v]);c.setAttribute('stroke','white');c.setAttribute('stroke-width',1);c.onclick=()=>document.getElementById('detail').textContent=p.id+' ｜ '+labels[v]+'。不代表整个街区；不是受困人数。';let t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=p.id+' '+labels[v];c.append(t);g.append(c);});document.getElementById('counts-'+arm).textContent=labels.map((x,i)=>x+counts[i]).join(' ｜ ');}document.getElementById('note').textContent=document.getElementById('frame').value==='peak_envelope'?'非同时峰值包络，仅为压力包络，不代表统一时刻或持续时间。':'统一实际时刻比较；三方案保持同一支持集合。';}['case','frame','direction'].forEach(x=>document.getElementById(x).onchange=draw);draw();'''.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
 (R/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区道路保障措施对照</title><style>body{font:16px/1.8 system-ui;max-width:1300px;margin:auto;padding:28px;background:#f4f7f8;color:#263e4b}h1,h2{line-height:1.4}h2{margin-top:34px}.lead{font-size:20px}aside{padding:18px;background:#e3eeea;border-left:5px solid #378a79}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{border:1px solid #d1dce2;padding:8px;text-align:left}th{background:#e7eef2}.scroll{overflow:auto}img{max-width:100%}.controls{display:flex;gap:16px;flex-wrap:wrap}.controls label{display:grid}select{padding:8px;font:inherit}.maps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.maps section{padding:12px;background:white}.maps p{font-size:13px}svg{width:100%;max-height:600px}a{color:#17648c}@media(max-width:750px){.maps{grid-template-columns:1fr}}</style>'+body+'<script>'+js+'</script></html>')
 print('PROTECTION REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
