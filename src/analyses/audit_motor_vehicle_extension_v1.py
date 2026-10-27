"""Vehicle-priority execution plan and local source/probe readiness audit.
Preserves all original graphs; no flood routing, new hydraulics or population work.
"""
from pathlib import Path
import os,sys,json,csv,gzip,collections,html
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_historical_surface_access_v1 import read,write,dump,load,sha
from normal_transport_rules_v1 import selfcheck
N=ROOT/'data/processed/three_ward_normal_transport_network_v1'
NR=ROOT/'data/exp/data-preprocessing/three_ward_normal_transport_network_v1'
P=ROOT/'data/processed/three_ward_motor_vehicle_inputs_v1'
R=ROOT/'data/exp/feasibility-check/three_ward_motor_vehicle_v1'
D=ROOT/'data/exp/research-design/three_ward_motor_vehicle_plan_v1'

def main():
 for p in [P,R,D]:p.mkdir(parents=True,exist_ok=True)
 conservative=load(N/'drive_conservative_graph.npz');possible=load(N/'drive_unknown_open_graph.npz');restr=json.loads((N/'supported_motorcar_restrictions.json').read_text());conditional=json.loads((N/'conditional_motorcar_restrictions.json').read_text());held=set(json.loads((N/'held_turn_way_ids.json').read_text()));checks=[]
 def check(name,v):
  checks.append(dict(check=name,passed=bool(v)))
  if not v:raise AssertionError(name)
 check('21_current_access_direction_and_turn_function_checks',len(selfcheck())==21 and all(x['passed']for x in selfcheck()))
 check('24_saved_full_normal_network_rule_checks',all(x['passed']for x in json.loads((NR/'rule_selfchecks.json').read_text())))
 check('unresolved_turn_ways_absent_in_conservative_graph',not held.intersection(map(int,conservative['osm_way_id'])))
 for name,z in [('conservative',conservative),('unknown_open',possible)]:
  check(name+'_graph_indices_weights',np.min(z['from_index'])>=0 and np.max(z['from_index'])<len(z['node_ids'])and np.max(z['to_index'])<len(z['node_ids'])and np.isfinite(z['length_m']).all()and np.all(z['length_m']>=0))
 oldroutes=[x for x in read(NR/'routing_probe_checks.csv')if x['mode']=='drive'];check('existing36_turn_aware_normal_probes_found',len(oldroutes)==36 and all(x['status']=='found'and int(x['turn_rules_applied'])==len(restr)for x in oldroutes))
 # Select actual car vertices in the same geographic source cells as the209 old probes.
 # No edge is added, and pedestrian vertices are never relabeled as car vertices.
 want={str(k)for k,inside in zip(conservative['node_ids'],conservative['inside_core'])if inside and str(k).endswith(':0|0|surface')};coord={}
 with(N/'surface_nodes.csv').open(encoding='utf-8-sig')as f:
  for row in csv.DictReader(f):
   if row['node_id']in want:coord[row['node_id']]=(float(row['x_m']),float(row['y_m']))
 bycell=collections.defaultdict(list)
 for key,(x,y)in coord.items():bycell[int(np.floor(x/500)),int(np.floor(y/500))].append((key,x,y))
 oldprobes=read(ROOT/'data/processed/three_ward_surface_coverage_v2/surface_probe_inventory.csv');probes=[]
 for p in oldprobes:
  _,i,j=p['probe_id'].split(':');i=int(i);j=int(j);xx=(i+.5)*500;yy=(j+.5)*500;candidates=bycell[i,j]
  chosen=min(candidates,key=lambda r:(r[1]-xx)**2+(r[2]-yy)**2)if candidates else None
  probes.append(dict(source_display_cell_id=p['probe_id'],source_cell_context_ward=p['ward'],cell_center_x_m=xx,cell_center_y_m=yy,car_source_node=chosen[0]if chosen else'',x_m=chosen[1]if chosen else'',y_m=chosen[2]if chosen else'',car_candidates_in_cell=len(candidates),candidate_available=bool(chosen),role='Fixed native motorcar probe in same500m geographic cell; no assumed building/parking access and no pedestrian transfer connector.'))
 check('all209_source_cells_retained',len(probes)==209 and len({p['source_display_cell_id']for p in probes})==209)
 check('selected_nodes_are_native_core_ground_car_vertices',all(p['car_source_node']in want for p in probes if p['candidate_available']))
 write(P/'motor_probe_inventory.csv',probes)
 groups=collections.defaultdict(lambda:dict(physical_source_segments=0,whole_intersecting_segment_length_km=0.,with_lanes_tag=0,with_maxspeed_tag=0,motor_only_segments=0,intersects_hydraulic_rectangle=0))
 motor_core=[];road_kinds=collections.Counter();xmin,xmax=-11360,-3660;ymin,ymax=-42440,-32320
 with gzip.open(N/'surface_physical_edges.jsonl.gz','rt')as f:
  for line in f:
   e=json.loads(line)
   if not e['inside_core']or e['drive_state']=='excluded':continue
   kind='ground_exact_grade'if e['grade_key']=='0|0|surface'else e['structure']if e['structure']in ['bridge','tunnel']else'other_relative_level';g=groups[kind];g['physical_source_segments']+=1;g['whole_intersecting_segment_length_km']+=e['length_m']/1000;g['with_lanes_tag']+=int('lanes'in e['tags']);g['with_maxspeed_tag']+=int('maxspeed'in e['tags']);g['motor_only_segments']+=int(e['walk_state']=='excluded');xy=np.asarray(e['geometry']['coordinates']);inbox=bool(np.max(xy[:,0])>=xmin and np.min(xy[:,0])<=xmax and np.max(xy[:,1])>=ymin and np.min(xy[:,1])<=ymax);g['intersects_hydraulic_rectangle']+=int(inbox);road_kinds[e['highway']]+=1
   motor_core.append(dict(source_edge_id=e['edge_id'],osm_way_id=e['osm_way_id'],highway=e['highway'],name=e['tags'].get('name',''),grade_key=e['grade_key'],structure=e['structure'],drive_state=e['drive_state'],drive_direction_state=e['drive_direction_state'],source_length_m=e['length_m'],has_lanes_tag='lanes'in e['tags'],has_maxspeed_tag='maxspeed'in e['tags'],walk_excluded=e['walk_state']=='excluded',intersects_hydraulic_rectangle_bbox=inbox,flood_road_height_known=False if kind!='ground_exact_grade'else None))
 coverage=[dict(surface_group=k,**v,length_not_clipped_to_ward=True,bbox_intersection_not_exact_coverage=True)for k,v in sorted(groups.items())]
 write(R/'core_motor_surface_inventory.csv',motor_core);write(R/'road_grade_coverage.csv',coverage);write(R/'road_highway_counts.csv',[dict(highway=k,segments=v)for k,v in road_kinds.most_common()])
 source_paths=[N/n for n in ['drive_conservative_graph.npz','drive_unknown_open_graph.npz','supported_motorcar_restrictions.json','conditional_motorcar_restrictions.json','held_turn_way_ids.json','surface_physical_edges.jsonl.gz','surface_nodes.csv','source_acquisition_manifest.json']]+[NR/n for n in ['routing_probe_checks.csv','delivery_validation.json','validation.json']]+[ROOT/'src/preprocessing/normal_transport_rules_v1.py',ROOT/'data/processed/three_ward_surface_coverage_v2/surface_probe_inventory.csv']
 binding={str(p.relative_to(ROOT)):sha(p)for p in source_paths};dump(P/'source_bindings.json',binding)
 policy=dict(human_decision='MILI-D-20261017-008',priority='Motor-vehicle road accessibility is the primary next transport extension. Relative impact ranking versus other modes is not assumed proven.',vehicle='Generic ordinary motorcar access as encoded by source tags; do not apply to trucks, emergency vehicles, buses or all vehicle wading capacities.',baseline_reuse=dict(conservative_graph=str((N/'drive_conservative_graph.npz').relative_to(ROOT)),unknown_permission_graph=str((N/'drive_unknown_open_graph.npz').relative_to(ROOT)),unconditional_turn_rules=len(restr),conditional_turn_rules=len(conditional),unresolved_turn_from_ways=len(held)),routing='Use turn-memory-aware node/way-history states; ordinary node adjacency/SCC alone is not a qualified motorcar router. Preserve one-way, node barriers, native level portals and via-node/via-way restrictions.',uncertainty='Separate permission/calendar/turn uncertainty from unknown flood exposure. Every branch has its own normal baseline; known legal exclusions and known one-way directions are never relaxed.',origin_selection='Retain209 geographic500m source cells; choose native conservative motorcar vertices within each cell once, without creating edges. No candidate is a baseline evidence gap, not a storm-created road island.',destinations='Use source-native cross-core road exits plus multiple external destination directions. Evaluate each direction rather than just any target. Do not use rail boarding nodes as car destinations or invent parking access.',flood_projection='Reuse12 certified hydraulic cases and12 synchronized actual output times plus a separate peak envelope. Bind road centerline/road-ground support independently of prior pedestrian supplements. Do not assign surface DEM levels to bridge decks or tunnel interiors.',closure_policy='Predeclare separate analyst depth sensitivity hypotheses for motorcars before formal flood routing; do not relabel pedestrian thresholds as certified vehicle limits. No empirical depth-speed or queue model is currently validated.',initial_scenarios='Dry-low identity control first; then original radar rainfall x two stages x T/U at the same normal network. Enhanced rainfall follows after source/turn/projection gates. No automatic queue is launched by this plan.',metrics=['Directional reachable destination fraction relative to the same normal branch','Supported car probes losing all outward paths; return access separately','Lost road length by class and known height support','Extra shortest road distance and detour ratio for still-reachable matched OD','Alternative route dependence and shared corridors; not observed flows or capacities'],whole_city_scope='Chiyoda/Chuo/Minato and existing10km road halo. Hydraulic-domain exterior and unknown road elevations remain explicit uncertainty; out-of-domain is not automatically dry.',pass_gates=['Preserve native permissions, one-way and turn-memory routing; explicitly retain138 conditional rules and37 held source ways.','Projection coverage tabulated by road length/class/ward, with unknown bridge/tunnel heights unchanged.','Each direction and uncertainty branch compared against its own normal baseline; baseline gaps excluded from storm losses.','Original and urban geometry compare matched cases, source support and origins; all historical derivatives share one parent.','Only synchronous states used for time-specific access; event envelope remains separate.'],later_congestion='Traffic assignment/dynamic queues require OD demand, mode shares, lane capacities, speed controls and signal timing. Lanes/maxspeed tags alone are not observed flow or road capacity. Do not claim congestion extent from connectivity.',later_measures='After car baseline qualification compare protected road/underpass approaches, retained corridors, traffic diversions and demand-management assumptions. Hydraulic protection changes require a separately authorized physical input and rerun plan.',existing_walking_and_rail_results='Retained as separate modes; not promoted to vehicle results.',new_hydraulics=False,people_counted=False,training=False,sealed_tests_opened=False,motor_flood_cases_executed=0,full_vehicle_flood_qualification=False)
 dump(D/'execution_plan.json',policy)
 summary=dict(conservative_nodes=len(conservative['node_ids']),conservative_directed_arcs=len(conservative['from_index']),uncertain_nodes=len(possible['node_ids']),uncertain_directed_arcs=len(possible['from_index']),supported_turn_rules=len(restr),conditional_turn_rules=len(conditional),held_turn_source_ways=len(held),existing_normal_probe_routes_found=len(oldroutes),source_display_cells=len(probes),cells_with_ground_native_car_candidate=sum(p['candidate_available']for p in probes),cells_without_candidate=sum(not p['candidate_available']for p in probes),road_grade_coverage=coverage,normal_candidate_reused=True,normal_routing_already_smoke_checked=True,motor_flood_cases_executed=0,new_hydraulics=False,people_counted=False)
 dump(R/'summary.json',summary);check('bound_original_sources_unchanged',all(sha(ROOT/k)==v for k,v in binding.items()));dump(R/'validation.json',dict(all_passed=True,checks=checks))
 body='<h1>机动车道路交通优先：执行方案与首步核查</h1><p>研究主线转为暴雨下车辆通行、跨区联系和绕行负担；步行与轨道保留为独立比较层。机动车是否受影响最大，由相同基线口径的后果指标检验。</p>'
 body+='<aside><b>已完成本步：复核候选机动车数据并建立车行分析点。</b><p>已复用保守车行图'+f"{summary['conservative_nodes']:,}"+'节点、'+f"{summary['conservative_directed_arcs']:,}"+'条有向弧；'+str(len(restr))+'条基础转向规则。209个原地理单元中，'+str(summary['cells_with_ground_native_car_candidate'])+'个找到同单元内原生地面车行点，'+str(summary['cells_without_candidate'])+'个保持正常基线证据缺口。尚未启动机动车积水批次，没有新增水动力模拟。</p></aside>'
 body+='<h2>1. 正常车行网络与规则</h2><p>复用既有约10 km跨区道路范围，保留机动车权限、单行、交叉口转向、桥隧分层和原生节点。已有36条转向感知正常路由探针均找到路径；这不是所有道路均已核验。138条条件转向和37个未解决转向来源道路继续单列，不按缺规则就自由转弯处理。</p><h2>2. 接入积水与道路高程</h2><p>使用已有12组水动力结果和12个统一实际时刻，峰值另列。按车辆可通行源路段独立重建积水对应，不能直接把步行补缺赋给车辆。地面路段可与地表高程对接；桥面、下穿和隧道缺绝对高程时保留未知，入口暴露与内部淹水分开。</p>'
 body+='<h2>3. 首批与指标</h2><p>先做无雨低水位恒等控制，再做原降雨×两水位×T/U四组配对；通过正常路由和积水覆盖闸门后，再接增强降雨。机动车关闭阈值作为独立情景参数事先冻结，不引用步行阈值充当车型能力，也不凭水深编造速度曲线。</p><ul><li>每个方向还剩多少正常可达目的地；哪些车行分析点失去全部出向或返回联系。</li><li>仍可达OD增加多少行驶距离，绕行倍率多大。</li><li>哪些道路、桥隧接近段和跨区通道反复影响替代路径；未调查高度和域外洪水继续保留未知。</li><li>按原有地理单元和各自正常基线比较机动车、步行与轨道，不直接比较不同模式的原始断连点数。</li></ul>'
 body+='<h2>4. 后续拥堵与主动措施</h2><p>第一轮给出通行与可达性后果。拥堵、排队和道路容量另需OD流量、方式分担、车道容量及信号等数据；已有车道数和限速标签不等于流量观测。之后再评估保通走廊、入口防护、绕行和交通管制等措施，明确每项能力假设；涉及改变积水的措施另行确定物理输入。</p>'
 body+='<h2>现有道路资料覆盖</h2><table><tr><th>分层</th><th>与三区相交的源路段</th><th>其中步行禁止路段</th><th>带车道数标签</th><th>带限速标签</th></tr>'
 for x in coverage:body+='<tr>'+''.join('<td>'+html.escape(str(x[k]))+'</td>'for k in ['surface_group','physical_source_segments','motor_only_segments','with_lanes_tag','with_maxspeed_tag'])+'</tr>'
 body+='</table><p>表按与三区相交的完整源路段计数，未按区界裁成道路长度，也不是车辆数量；矩形覆盖检查仅用于预筛，不能代替精确积水投影。</p><h2>资料与可复现结果</h2><ul>'
 for path in [D/'execution_plan.json',R/'summary.json',R/'validation.json',R/'road_grade_coverage.csv',R/'core_motor_surface_inventory.csv',R/'road_highway_counts.csv',P/'motor_probe_inventory.csv',P/'source_bindings.json']:
  body+='<li><a href="'+os.path.relpath(path,D)+'">'+path.name+'</a></li>'
 body+='</ul><p>现有道路源为2026-10-15 OSM快照，并非逐路测量日期；历史雨场情景不等于完整复现2024年的道路与交通。原人口分配不确定性保留，不计算受困人数、不训练、不打开封存测试。Framelog: MILI-D-20261017-008。</p>'
 (D/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>机动车道路交通优先方案</title><style>body{font:16px/1.8 system-ui;max-width:1100px;margin:auto;padding:32px;background:#f3f6f8;color:#263e4c}h1,h2{line-height:1.4}h2{margin-top:32px}aside{padding:18px;background:#e4efec;border-left:5px solid #3c806d}table{width:100%;border-collapse:collapse;background:white}td,th{border:1px solid #d5e0e6;padding:9px;text-align:left}a{color:#17648b}</style>'+body+'</html>')
 # Append-only decision log is separate; original multimodal outputs are untouched.
 files=[p for p in P.iterdir()if p.is_file()]+[p for p in R.iterdir()if p.is_file()and p.name!='delivery_manifest.json']+[D/'execution_plan.json',D/'review.html',Path(__file__)]
 dump(R/'delivery_manifest.json',dict(files={str(p.relative_to(ROOT)):sha(p)for p in files},new_hydraulics=False,motor_flood_cases_executed=0));print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
