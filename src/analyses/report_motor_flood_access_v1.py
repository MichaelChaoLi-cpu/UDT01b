"""Regional vehicle-access result review, retaining uncertainty and mode boundaries."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,collections,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import shape
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_motor_flood_access_v1 import P,R,F,CASES,read,write,dump,load,sha
from compare_urban_geometry_pair_controls_v1 import table
CODES={'insufficient_evidence':0,'robust_modeled_loss':1,'conditional_sensitivity':2,'retained_access':3};COLORS=['#929ba3','#b64747','#da9a31','#39877a']


def main():
 val=json.loads((R/'validation.json').read_text());assert val['all_passed'];summary=json.loads((R/'analysis_summary.json').read_text());agg=read(R/'classification_summary.csv');classes=read(R/'probe_classification.csv');probes=read(P/'probe_inventory.csv');policy=json.loads((P/'analysis_contract.json').read_text());frames=policy['frame_ids'];a=load(P/'accessibility.npz');support=load(P/'evidence_support.npz');scenarios=read(P/'scenario_index.csv');context={}
 for row in read(ROOT/'data/exp/estimation/three_ward_historical_transport_islands_v1/probe_accessibility.csv'):context[row['probe_id']]=row['towns_intersecting_display_cell']
 ix={(r['case_id'],r['frame_id'],r['probe_id']):r for r in classes};geometry=[]
 for rain in ['R0','R1','R125']:
  for stage in ['L','H']:
   for frame in frames:
    pairs=[(ix[f'{rain}__{stage}__T',frame,p['source_display_cell_id']],ix[f'{rain}__{stage}__U',frame,p['source_display_cell_id']])for p in probes]
    for direction in ['outward','two_direction']:
     key=direction+'_classification';good=[(t,u)for t,u in pairs if t[key]!='insufficient_evidence'and u[key]!='insufficient_evidence'];geometry.append(dict(rainfall=rain,stage=stage,frame_id=frame,direction=direction,common_supported_cells=len(good),T_robust=sum(t[key]=='robust_modeled_loss'for t,u in good),U_robust=sum(u[key]=='robust_modeled_loss'for t,u in good),classification_changes=sum(t[key]!=u[key]for t,u in good)))
 write(R/'matched_geometry_access_changes.csv',geometry)
 # Directional OD here means geographic cell to boundary-exit sector, not observed journeys.
 normal=a['normal_road_distance_m'];od=[]
 for i,row in enumerate(scenarios):
  bi=0 if row['unknown_branch']=='unknown_available'else 1
  for direction in range(2):
   eligible=support['eligible_outward']if direction==0 else support['eligible_two_direction'];common=np.isfinite(normal[:,direction]).all(axis=0)&eligible[None,:];before=normal[bi,direction];after=a['road_distance_m'][i,direction];retained=common&np.isfinite(after);lost=common&~np.isfinite(after);extra=after[retained]-before[retained];positive=retained&(before>0);ratio=after[positive]/before[positive]
   od.append(dict(**row,direction=['outward','return'][direction],normal_common_direction_pairs=int(common.sum()),lost_direction_pairs=int(lost.sum()),retained_direction_pairs=int(retained.sum()),lost_fraction=float(lost.sum()/common.sum())if common.any()else None,median_extra_distance_m=float(np.median(extra))if len(extra)else None,p90_extra_distance_m=float(np.quantile(extra,.9))if len(extra)else None,median_detour_ratio=float(np.median(ratio))if len(ratio)else None))
 write(R/'common_support_directional_burden.csv',od)
 candidates=[]
 for r in classes:
  if r['frame_id']=='14400'and r['outward_classification']in ['robust_modeled_loss','conditional_sensitivity']:candidates.append(dict(**r,display_cell_town_context=context.get(r['probe_id'],''),not_whole_town=True))
 write(R/'candidate_locations_19jst.csv',candidates)
 states=collections.defaultdict(lambda:{'outward':[],'two_direction':[]})
 for r in classes:
  for direction in ['outward','two_direction']:states[r['case_id']+'|'+r['frame_id']][direction].append(str(CODES[r[direction+'_classification']]))
 payload=dict(states={k:{d:''.join(v)for d,v in values.items()}for k,values in states.items()},points=[dict(id=p['source_display_cell_id'],x=(float(p['x_m']or p['cell_center_x_m'])+11360)/10,y=(-32320-float(p['y_m']or p['cell_center_y_m']))/10,context=context.get(p['source_display_cell_id'],''),source_node=p['car_source_node'],has_car_node=p['candidate_available']=='True')for p in probes]);assert len(payload['states'])==156 and all(len(v)==209 for x in payload['states'].values()for v in x.values());dump(R/'map_states.json',payload)
 # Coarse regional background only; no neighborhood polygon is classified from one point.
 grid=load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')['Ward Code'];background=''
 for geo,value in shapes(grid.astype('uint8'),mask=grid>0,transform=from_origin(-11360,-32320,10,10)):
  g=shape(geo).simplify(35,preserve_topology=True)
  for poly in list(g.geoms)if g.geom_type=='MultiPolygon'else[g]:
   path=''
   for ring in [poly.exterior]+list(poly.interiors):path+='M'+' L'.join(f'{(x+11360)/10:.1f},{(-32320-y)/10:.1f}'for x,y in ring.coords)+'Z'
   background+='<path d="'+path+'" fill="#e7edf0" stroke="#a7b7bf" stroke-width="1" fill-rule="evenodd"/>'
 fig,axs=plt.subplots(2,3,figsize=(13,10),layout='constrained');extent=[-11.36,-3.66,-42.44,-32.32]
 for ai,arm in enumerate(['T','U']):
  for ri,rain in enumerate(['R0','R1','R125']):
   ax=axs[ai,ri];ax.imshow(np.where(grid>0,grid,np.nan),origin='upper',extent=extent,cmap='Greys',vmin=0,vmax=15,alpha=.35);rows=[ix[f'{rain}__H__{arm}','14400',p['source_display_cell_id']]for p in probes]
   for label,code in CODES.items():
    chosen=[(p,r)for p,r in zip(probes,rows)if r['outward_classification']==label];ax.scatter([float(p['x_m']or p['cell_center_x_m'])/1000 for p,r in chosen],[float(p['y_m']or p['cell_center_y_m'])/1000 for p,r in chosen],s=22,c=COLORS[code],edgecolor='white',linewidth=.25)
   count=sum(x['outward_classification']=='robust_modeled_loss'for x in rows);ax.set_title(f'{rain} | {arm} | robust vehicle exit loss: {count}',fontsize=10);ax.set_xlim(-11.36,-3.66);ax.set_ylim(-42.44,-32.32);ax.set_aspect('equal');ax.set_xlabel('Easting (km)')
   if ri==0:ax.set_ylabel('Northing (km)')
 fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,label=l,markersize=7)for c,l in zip(COLORS,['Insufficient evidence','Robust modeled exit loss','Assumption-sensitive','Retained alternative'])],loc='outside lower center',ncol=2);fig.suptitle('Motorcar access |19:00 JST | raised-stage scenario | same93 supported outward probes');fig.savefig(R/'motor_access_geometry_19jst.png',dpi=150);plt.close(fig)
 key=[r for r in agg if r['frame_id']=='14400'and r['direction']=='outward'];same19=[r for r in geometry if r['frame_id']=='14400'and r['direction']=='outward'];selected_od=[r for r in od if r['frame_id']=='14400'and r['direction']=='outward'and float(r['threshold_m'])==.3]
 dump(R/'delivery_summary.json',dict(**summary,fixed19jst_outward=key,geometry19jst=same19,common_directional_burden19jst=selected_od,not_traffic_volume_or_queue=True,not_entire_neighborhood_isolation=True))
 body='<h1>机动车道路交通：首轮暴雨情景结果</h1><p class="lead">直接复用现有车行网络与规则，已完成12组积水情景对应的624组道路可达性对照。</p><aside><b>本轮回答车辆能否进出、哪些方向失去联系，以及绕行增加多少。</b><p>原209个地理单元中有206个车行候选点；同时满足正常基线与积水对应条件的出向比较点为93个，双向比较点89个。其余单元保留证据不足，不计为暴雨新增断连。没有追加水动力模拟，也未模拟车流、排队或受困人数。</p></aside>'
 body+='<h2>三区整体结果</h2><p>固定19:00，原降雨＋水位抬升的城市约束U组中，93个共同支持点有13个在四个阈值／未知组合下均无法驶出；增强降雨时为17个。对应仅地形T组均为13个。这个指标只表示到既定跨区边界出口的模型内联系，不能直接称为整片街区或整个城市瘫痪。</p>'
 body+=table(['雨量','水位','T稳健出向断连','U稳健出向断连','共同支持单元','分类变化单元'],[[x['rainfall'],x['stage'],x['T_robust'],x['U_robust'],x['common_supported_cells'],x['classification_changes']]for x in same19])
 body+='<img src="motor_access_geometry_19jst.png" alt="Three-ward vehicle access scenario comparison"><h2>切换情景与时刻</h2><p>红色为模型内稳健联系损失，橙色为条件敏感，灰色为证据不足，绿色为保留模型内替代。地图是分析点，不代表整个町丁目。</p><div class="controls"><label>情景<select id="case">'+''.join('<option value="'+c+'"'+(' selected'if c=='R1__H__U'else'')+'>'+c+'</option>'for c in CASES)+'</select></label><label>时刻<select id="frame">'+''.join('<option value="'+f+'"'+(' selected'if f=='14400'else'')+'>'+('非同时峰值包络'if f=='peak_envelope'else f'{15+int(f)//3600:02d}:{int(f)%3600//60:02d} JST')+'</option>'for f in frames)+'</select></label><label>方向<select id="direction"><option value="outward">车辆出向</option><option value="two_direction">出向与返回均有替代</option></select></label></div><p id="counts"></p><p id="note"></p><div class="mapwrap"><svg id="map" viewBox="0 0 770 1012" aria-label="Motor-vehicle access probe map">'+background+'<g id="points"></g></svg><div><p>● 红：稳健联系损失</p><p>● 橙：条件敏感</p><p>● 灰：证据不足</p><p>● 绿：保留替代</p><p id="detail">点击分析点查看位置上下文。</p></div></div>'
 body+='<h2>固定19:00的结果明细</h2>'+table(['算例','稳健断连','条件敏感','证据不足','保留替代'],[[x['case_id'],x['robust_modeled_loss'],x['conditional_sensitivity'],x['insufficient_evidence'],x['retained_access']]for x in key])
 body+='<p>R0=无雨，R1=原历史雷达雨场，R125=同雨场×1.25；L/H为两档假设河道水位，T/U为仅地形／城市约束。所有后代仍属同一个历史父事件。</p><h2>方向损失和绕行</h2><p>出口按北、东、南、西四个地理方向分组，保持原生道路与单行方向。下表采用0.3 m中断假设，只比较两个正常分支共同支持的“单元→出口方向”对；距离仅对仍可达的方向对统计，失去联系的方向对另列，避免把无法到达当成无限绕行或从统计中悄悄删除。</p>'
 def number(x):return'—'if x in [None,'']else f'{float(x):.1f}'
 body+=table(['情景','未知路段分支','正常共同方向对','失去方向对','仍可达方向对','新增距离中位数 m','新增距离P90 m','绕行倍率中位数'],[[x['case_id'],x['unknown_branch'],x['normal_common_direction_pairs'],x['lost_direction_pairs'],x['retained_direction_pairs'],number(x['median_extra_distance_m']),number(x['p90_extra_distance_m']),number(x['median_detour_ratio'])]for x in selected_od if x['case_id']in ['R1__H__T','R1__H__U','R125__H__T','R125__H__U']])
 body+='<h2>怎样读这些结果</h2><ul><li>已有机动车权限、单行、转向和分层设置原样复用。路由复现了此前18条正常道路距离，没有另开逐路交通细则调查。</li><li>使用0.1／0.3 m两档道路中断情景，并保留高架、桥隧、域外和地面对应不足路段的可用／移除分支。阈值是研究假设，不是车型涉水能力或真实关闭标准。</li><li>已知地面道路按车辆源中心线与0.50 m地形取样，结合10 m模拟水位；没有把桥面或隧道内部当作地面DEM，也没有把步行专用连接赋给车辆。</li><li>正常时就缺资料或缺联系的单元不算暴雨新增断连。两个方向可使用不同出口，尚未证明往返同一地点或真实行程时序。</li><li>本轮是道路通行和可达性，不是拥堵预测；没有OD流量、道路容量和信号排队，不能推断车辆数、堵车时长或道路实际瘫痪程度。</li></ul>'
 body+='<p>机动车与此前步行／轨道的支持单元、节点和目的地不同，原始断连点数不能直接比较谁受影响更大。道路源为2026年快照，历史降雨作用于这套城市表达，不是2024年交通系统完整复原。</p><h2>路径证据与数据</h2><p>已导出原雨量抬升水位U组在19:00、0.3 m及未知路段可用假设下，所有单元到四个方向的出向／返回路径记录。每条可达路径均检查原生有向道路、原有转向规则、未关闭路段和距离；未知路段在路径中明确计数。到达的是研究边界出口，不是已确认安全目的地。</p><ul>'
 files=[R/'analysis_summary.json',R/'delivery_summary.json',R/'classification_summary.csv',R/'probe_classification.csv',R/'matched_geometry_access_changes.csv',R/'directional_access_and_detours.csv',R/'common_support_directional_burden.csv',R/'candidate_locations_19jst.csv',R/'shared_car_route_segments_19jst.csv',R/'projection_coverage.csv',P/'car_route_witnesses_19jst.jsonl',P/'analysis_contract.json',P/'scenario_index.csv',P/'road_inventory.csv',R/'normal_route_reuse_check.csv',R/'validation.json',R/'browser_validation.json',R/'delivery_validation.json',R/'delivery_manifest.json']
 for f in files:body+='<li><a href="'+os.path.relpath(f,R)+'">'+html.escape(f.name)+'</a></li>'
 body+='</ul><p>沿用简化研究范围：MILI-D-20261017-009；原步行、轨道结果与人口不确定性保持不变。</p>'
 js='''const DATA=__DATA__;const colors=['#929ba3','#b64747','#da9a31','#39877a'];const labels=['证据不足','稳健联系损失','条件敏感','保留替代'];function draw(){const cs=document.getElementById('case').value;const frame=document.getElementById('frame').value;const direction=document.getElementById('direction').value;const values=DATA.states[cs+'|'+frame][direction];const group=document.getElementById('points');group.replaceChildren();let counts=[0,0,0,0];DATA.points.forEach((p,i)=>{const n=Number(values[i]);counts[n]++;const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',7);c.setAttribute('fill',colors[n]);c.setAttribute('stroke','white');c.setAttribute('stroke-width',1);c.style.cursor='pointer';const t=document.createElementNS('http://www.w3.org/2000/svg','title');t.textContent=p.id+' '+p.context+' '+labels[n];c.append(t);c.addEventListener('click',()=>document.getElementById('detail').textContent=p.id+'；位置上下文：'+p.context+'；'+labels[n]+'；'+(p.has_car_node?'原生车行节点：'+p.source_node:'本单元未选出车行节点')+'。不代表整个町丁目。');group.append(c);});document.getElementById('counts').textContent=labels.map((x,i)=>x+' '+counts[i]+'个单元').join(' ｜ ');document.getElementById('note').textContent=(frame==='peak_envelope'?'非同时峰值包络，不能当作同一时刻或断连时长。':'统一实际时刻切片。')+(direction==='two_direction'?' 至少一个方向缺少替代即构成双向联系损失。':'');}['case','frame','direction'].forEach(x=>document.getElementById(x).addEventListener('change',draw));draw();'''.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
 (R/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区机动车道路积水通行对照</title><style>body{font:16px/1.8 system-ui;max-width:1220px;margin:auto;padding:30px;color:#263e4c;background:#f3f6f8}h1,h2{line-height:1.4}h2{margin-top:34px}.lead{font-size:19px}aside{background:#e5efec;border-left:5px solid #3d8470;padding:18px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border:1px solid #d6e0e6;padding:9px;text-align:left}th{background:#e6edf2}.scroll{overflow:auto}img{max-width:100%}.controls{display:flex;gap:16px;flex-wrap:wrap}.controls label{display:grid}select{padding:8px;font:inherit}.mapwrap{display:grid;grid-template-columns:minmax(250px,620px) 1fr;gap:20px;background:white;padding:20px}svg{width:100%;max-height:720px}a{color:#17658e}@media(max-width:700px){.mapwrap{grid-template-columns:1fr}}</style>'+body+'<script>'+js+'</script></html>')
 # Link current output while retaining the earlier source-readiness snapshot.
 d=ROOT/'data/exp/research-design/three_ward_motor_vehicle_plan_v1';f=d/'review.html';s=f.read_text();marker='<!-- MOTOR_FLOOD_RESULT -->'
 if marker not in s:s=s.replace('<h2>1. 直接复用现有车行网络</h2>',marker+'<p><b>简化方案首轮已完成：</b><a href="'+os.path.relpath(R/'review.html',d)+'">机动车积水通行、方向损失与绕行结果</a>。上方为首步资料核查的历史快照。</p><h2>1. 直接复用现有车行网络</h2>');f.write_text(s)
 # Keep the earlier source-readiness manifest consistent with its updated navigation link.
 mfile=ROOT/'data/exp/feasibility-check/three_ward_motor_vehicle_v1/delivery_manifest.json';m=json.loads(mfile.read_text());m['files'][str(f.relative_to(ROOT))]=sha(f);dump(mfile,m)
 print('MOTOR REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
