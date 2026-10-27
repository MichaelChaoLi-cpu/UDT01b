"""Create the input inventory and five-stage SFINCS project specification.

This prepares contracts and a synthetic installation test, not a Tokyo model.
Run with the existing project Python; no additional Python packages are needed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import platform
import subprocess
from datetime import datetime, timezone

TAG = 'v2.4.0_Galibier_release'
COMMIT = '1f1d8286520a8709b2e513854dff18a3616583db'
IMAGE = 'deltares/sfincs-cpu@sha256:4301bffc74f3cb3f334be5d21aa75dfe5742a16c9838942cdfc2f870859fb404'


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main(root):
    proc = root / 'data/processed/three_ward_sfincs_framework'
    report = root / 'data/exp/data-briefing/three_ward_sfincs_framework'
    proc.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    # File presence and scientific readiness are deliberately separate.
    rows = []

    def item(key, priority, label, path, state, gap, action, target):
        p = root / path if path else None
        rows.append(dict(id=key, priority=priority, input=label, source_path=path,
                         local_present=bool(p and p.exists()), readiness=state,
                         gap=gap, next_action=action, solver_target=target))

    item('domain','P0','三区及直接跨区汇水域','data/processed/plateau_water/central_three_ward_boundaries.geojson','partial',
         '三区区界已备；现有1 km缓冲不是完整汇水域。河川全流域不必全部纳入高分辨率二维域。',
         '以连续地表汇水划分模型域；河川上游优先用实测H(t)/Q(t)或外接一维模型。','active mask + boundary segments')
    item('terrain','P0','裸地地形与元数据','data/processed/hibiya_water_corridor/official_dem_0_50m_mosaic.tif','partial',
         '0.50 m仅日比谷；三区现有PLATEAU约5 m TIN。测量日期和高程成果版本未核实；屋顶不能当裸地。',
         '按直接汇水域批量补0.50 m DEM；保留源瓦片、测绘时期、垂直基准及拼接差记录。','sfincs_subgrid.nc (stage-volume and face conveyance)')
    item('terrain_context','P1','现有三区粗地形','data/processed/three_ward_surface_trial/terrain_20m.tif','diagnostic_only',
         '20 m中心采样不能作为高精度道路地形；非水域仍有缺值。',
         '仅用于域初划与低成本对照；不向上采样伪造0.50 m细节。','domain preview / coarse control')
    item('vertical','P0','统一垂直基准和成果时期','data/exp/data-briefing/three_ward_road_connectivity/boundary_source_manifest.json','partial',
         'T.P.、A.P.、水尺零点、晴海DL及高程成果版本未逐源对齐；不能使用统一假定偏移。',
         '为每源登记基准、零点、转换证据、生效日期；统一到已核实的T.P.版本。','DEM / initial stage / boundary stage / bank crest')
    item('urban_geometry','P1','建筑与地面道路','data/processed/three_ward_surface_trial/building_roof_proxy_5m.tif','partial',
         '现有建筑是RoofEdge代理，5 m栅格是几何采样分辨率；桥梁、地下层与冲突像元未完整分层。',
         '批量建地面障碍/道路层；热点检查层级。不可将所有RoofEdge抬成实体挡水墙；屋面降雨计入经核实的径流分配。','subgrid elevation/roughness + explicit barrier lines')
    item('water_geometry','P1','水域与候选连接','data/processed/central_three_water_network/water_inventory.geojson','partial',
         'W05和wtr提供几何线索；没有证实水力连接、河床、岸顶或水位。',
         '按河段划界、核实主要岸顶/开口；池塘和濠按有限储水体处理，不能默认无限出口。','river boundary segmentation / weir crest / storage')
    item('rain','P0','2024-08-21分布暴雨','data/processed/tokyo_rain_events/2024-08-21/jma_rain_10min.csv','partial',
         '已有4站10分钟雨量及149站汇总；缺港区核心事件多站曲线/数值雷达场，JMA东京单站不足以代表三区。',
         '获取核心站1–10分钟曲线或XRAIN数值场；守恒转换累计雨量、核对缺测及时间区间。','netamprfile or amprfile (mm/hr)')
    item('river_stage','P0','河段水位/来流','data/exp/data-briefing/three_ward_road_connectivity/upstream_and_stage_data_needs.csv','missing',
         '渋谷川—古川、神田川—日本橋川—亀島川、隅田川缺事件H(t)/Q(t)、零点和主要设施运行。',
         '先补区内/上下游边界站1–10分钟数据；外接流域来水不与域内降雨重复计入。','bndfile+bzsfile / srcfile+disfile')
    item('sea_stage','P1','晴海实测潮位','data/processed/three_ward_road_connectivity/tokyo_harumi_tide_20240821_22.csv','partial',
         '48条小时平滑实测已备；2024 DL标高−1.884 m。滤波数据不能代表分钟响应或所有河段水位。',
         '补细时间实测潮位与河段梯度，核对高程成果版本；当前只作海侧背景约束。','sea boundary bzsfile')
    item('banks','P0','主要岸顶/防潮设施/实际地表开口','','missing',
         '大范围最低岸顶和开口尚缺，wtr Z不能替代。保护岸边若直接设水位边界可能虚构排水。',
         '优先关键河段和模型热点，有限岸顶用可越流weir，岸侧水位按对应河段设置。','weirfile + verified water-side boundary')
    item('roughness','P1','粗糙度、入渗和初始湿度','','missing',
         '无经标定的分区参数；此前统一15 mm/h损失是试算假设。',
         '基于不透水率/土地覆盖建立参数范围及敏感性组。子网格模式粗糙度写入子网格表。','subgrid roughness / infiltration file')
    item('drainage','P1','现有排水与抽排等效作用','','missing',
         '未获取大范围有效排水能力；不能把入渗项冒充管网，也不能默认为完全无排水。',
         '用有证据的分区等效抽排及受纳水位约束；资料不足先给无抽排与能力范围情景，清楚标记。','drnfile or evidence-based source/sink coupling')
    item('initial_state','P1','初始河池水位与地表湿度','','missing',
         '实际事件初始状态和足够预热时段未建立。',
         '海河边界先预热，地表干湿/土壤状态作敏感性；初始池水需有限储水与出入口证据。','zsini / ini / restart + infiltration state')
    item('validation','P0','事件积水证据','data/processed/tokyo_rain_events/2024-08-21/matched_town_records.csv','partial',
         '町级记录无深度、时刻或精确坐标；静态风险图不能当2024实测，缺记录不等于无积水。',
         '整理有时间地点的道路淹水/水痕，统一统计口径；按证据等级验证，不虚报深度RMSE。','event validation / calibration holdout')
    item('measures','P2','第五步主动措施空间、容量和运行','','missing',
         '尚未选具体工程或控制策略；措施量值不能编造。',
         '先建情景登记，待前四步完成后填选址、容量、设施约束并做同条件对照。','volfile / weirfile / drnfile / infiltration variants')
    with (proc/'input_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

    stages=[
        dict(step=1,name='统一数据与模型域',status='in_progress',deliverables=['直接地表汇水域','统一高程/时间元数据','DEM与地面几何目录','分布降雨与河段边界目录'],exit='模型域及边界来源明确；关键输入齐备或将不确定性写成明确情景'),
        dict(step=2,name='成熟二维求解器基线',status='framework_ready_engine_check_pending',deliverables=['SFINCS固定版本','10/20 m网格敏感性','0.50 m子网格储水与过流表','河岸边界和排水假设组'],exit='运行完成且水量闭合、时间步/网格敏感性与关键地形连通性检查通过；不是仅有图片'),
        dict(step=3,name='真实暴雨复现与验证',status='awaiting_forcing_and_observations',deliverables=['2024-08-21事件回放','降雨/水位对齐','积水范围与时程验证','不确定性和留出验证'],exit='根据可得实测给出能力范围；没有深度观测时不宣称深度验证完成'),
        dict(step=4,name='三区机制、影响与三维表达',status='planned',deliverables=['雨量分布/河水位效应分离','积水深度/持续时间/道路影响','研究问题与可信区间','PLATEAU三维叠加'],exit='三区基线与解释完成，清楚区分粗网格水动力结果和高分辨率显示'),
        dict(step=5,name='主动措施模拟',status='after_steps_1_to_4',deliverables=['单措施及组合情景','相同强迫条件的基线对照','减淹/风险转移/失效条件','实施与运行约束'],exit='设施数据齐备且对比覆盖目标区、邻区和受纳水体；不改变暴雨以制造效果')]
    save_json(proc/'workflow.json',{'steps':stages,'user_requested_order':'第五步在前四步完成之后','updated_at':datetime.now(timezone.utc).isoformat()})
    spec={
        'schema_version':1,'model_id':'tokyo_central_three_wards_sfincs','scope':['千代田区','中央区','港区'],
        'purpose':'三区暴雨地表积水与河渠/潮位影响；之后比较主动措施',
        'solver':{'name':'SFINCS','release':TAG,'commit':COMMIT,'docker_image':IMAGE,'platform':'linux/amd64','host_architecture':platform.machine(),'arm_note':'官方此标签仅amd64；Apple Silicon用Docker仿真，性能待实际三区规模测试。大批量情景建议原生x86_64 Linux。'},
        'builder':{'recommended':'HydroMT-SFINCS','candidate_stable_version':'1.2.2','status':'not_installed_not_compatibility_tested','reason':'优先稳定版正式子网格表生成，必须在独立环境验证SFINCS 2.4输入输出；latest文档可能对应2.0预发布API，不混用。'},
        'coordinates':{'horizontal':'EPSG:6677 / JGD2011 plane rectangular IX','vertical_target':'T.P. (height-result epoch must be verified)','source_alignment_complete':False,'time_storage':'UTC ISO8601 with JST preserved','solver_time':'seconds from UTC tref; explicit conversion, never silently drop +09:00'},
        'grid':{'initial_hydraulic_cell_m':20,'sensitivity_hydraulic_cell_m':10,'subgrid_target_pixel_m':0.5,'choice_status':'candidate_to_validate','note':'20 m起步是成本试验；关键狭路/岸边视连通与收敛加密。0.50 m地形不是0.50 m水动力求解、不是精度保证。'},
        'event':{'candidate':'2024-08-21','data_window_jst':['2024-08-21T00:00:00+09:00','2024-08-22T06:00:00+09:00'],'data_window_is_sufficient_recession':False,'extend_recession_if_needed':True},
        'water_strategy':'优先河段H(t)驱动地表边界，保持岸顶/开口；预测或措施改变河道过程时升级为1D河道/流域耦合。',
        'excluded_as_ground_truth':['wtr Z作为河水位/河床','W05交点作为已证实水力连接','单一晴海潮位套所有河段','单一东京雨量代表三区','RoofEdge当真实地面FootPrint','自编试算图作为验证','1 km缓冲当完整汇水域'],
        'baseline_model_ready':False,'official_solver_bundle_created':False,
        'numerical_checks':['闭合水量及非负水深','网格10/20 m敏感性','时间步控制与湿干阈值敏感性','细地形储水与岸边过流检查','边界距离/域范围敏感性'],
        'reporting':'只有达到实测支持与数值检验范围的结果才作为研究输出；速度与风险指标定义另行核实。'}
    save_json(proc/'model_spec.json',spec)
    save_json(proc/'input_contract.json',{
        'raster':{'horizontal_crs':'EPSG:6677','elevation_unit':'m_positive_up','no_data':'explicit, never filled to 0 automatically','vertical_metadata':['datum','height_result_epoch','survey_date','conversion_evidence'],'source_resolution_distinct_from_accuracy':True},
        'rain':{'model_unit':'mm/hr','observed_interval_metadata_required':True,'cumulative_to_rate':'preserve interval volume; document interval end/start; verify model interpolation integral','missing_is_zero':False,'forcing_extents_cover_domain':True},
        'water':{'stage_unit':'m_same_reference_as_ground','discharge_unit':'m3/s','metadata':['gauge_zero','datum','height_result_epoch','site_coordinates','valid_dates','sampling_and_filter'],'boundary_segments_are_hydraulic_not_only_geometric':True},
        'mass_accounting':['gross_rainfall','roof_runoff_routing','infiltration','equivalent_drainage','boundary_inflow_outflow','surface_storage_change','measure_storage_change','pump_transfer'],
        'measures':'M1-M4 are registered options, not implemented engineering claims; M5 automatic controls require a verified coupling implementation.',
        'outputs':{'native':['sfincs_map.nc','sfincs_his.nc','sfincs_log.txt'],'derived':['h_max = max_t(h(t))','first_threshold_time','duration_above_declared_depth_threshold','maximum_velocity_with_wet_depth_threshold','paired_measure_minus_baseline','neighboring_risk_transfer','water_balance_and_quality_flags'],'depth_thresholds_m':[0.1,0.3,0.5],'thresholds_are_analysis_choices_not_official_safety_standards':True,'duration_requires_sufficient_output_frequency':'candidate 60 s maps /10 s selected observations, verify sampling sensitivity','fine_map':'downscaled coarse water surface minus fine terrain, not independent fine-grid flow'}})
    measures=[
        dict(id='M1',name='分散蓄水/公园调蓄',interface='subgrid volfile or surveyed depression geometry',needs=['选址','新增有效容量m³','前期占用','入口/溢流标高','排空规则'],limits='volfile是简化前期储水表达；有水位控制出流的池塘需真实几何/连接，不能重复计入地形库容。'),
        dict(id='M2',name='透水铺装/雨水花园',interface='infiltration variant + optional finite storage',needs=['实施面积','入渗参数','初始湿度','可用储水','堵塞/饱和条件'],limits='先核实v2.4选定入渗方法及产流范围；不得把管网能力算作土壤入渗，不得无限吸水。'),
        dict(id='M3',name='地表导流/有限高度挡水',interface='weirfile + geometry variant',needs=['措施线/面','绝对岸顶或挡水标高','通道宽度','入口和溢流方向'],limits='可越流防护用有限高度weir；thin dam无限高只作明确极限对照；必须检查邻街风险转移。'),
        dict(id='M4',name='临时抽排/增设泵',interface='v2.4 legacy drnfile type=1 fixed capacity',needs=['进水点','出水点','实测有效流量m³/s','受纳水体','运行/失效时段'],limits='固定泵能力原生可用；按事件/分段运行需经过验证的重启动或外部耦合。同域出入水严格守恒，不能消失到未知出口。'),
        dict(id='M5',name='自动启停/闸泵联合控制',interface='future verified control coupling, not claimed native v2.4 rule support',needs=['触发阈值','时滞','控制逻辑','泵曲线','闸门约束','河道响应'],limits='latest网页包含晚于固定发布版的接口；v2.4不使用新TOML rule语法，必要时另评估正式新版本。')]
    save_json(proc/'measure_catalog.json',{'activation':'after_stages_1_to_4','options':measures,'paired_design':{'fixed':['domain','base_geometry_except_measure','rain','upstream_forcing','initial_conditions','background_drainage','solver_settings'],'vary':['documented_measure_geometry_or_capacity_or_operation'],'metrics':['目标区深度/持续时间变化','道路受淹长度和持续时间','邻区增加积水','受纳水位/排放变化','容量/触发失效条件'],'independent_boundary_warning':'固定实测河段水位的地表模型无法评估显著改变河道水位的工程反馈；这些措施需要河道耦合。'}})
    save_json(proc/'scenario_registry.json',{'scenarios':[{'id':'B0','kind':'event_replay','status':'awaiting_inputs','bundle':None,'event':'2024-08-21','measures':[]},{'id':'N10','kind':'grid_sensitivity','status':'after_B0_build','bundle':None,'hydraulic_grid_m':10},{'id':'M1_to_M5','kind':'intervention','status':'after_steps_1_to_4','bundle':None,'requires':'validated_baseline_and_measure_inputs'}],'historical_prototype':'retained as diagnostics; not executable through this framework'})
    save_json(proc/'readiness.json',{'baseline_run_ready':False,'event_validation_ready':False,'intervention_run_ready':False,'critical_missing_inputs':[r['id'] for r in rows if r['priority']=='P0' and r['readiness']!='ready'],'source_file_presence':{r['id']:r['local_present'] for r in rows}})

    smoke=proc/'engine_smoke_test'
    smoke.mkdir(exist_ok=True)
    if not (smoke/'sfincs.inp').exists():
        (smoke/'sfincs.dep').write_text((' '.join(['0.0']*10)+'\n')*10)
        (smoke/'sfincs.msk').write_text((' '.join(['1']*10)+'\n')*10)
        (smoke/'sfincs.prcp').write_text('0 10\n3600 10\n')
        (smoke/'sfincs.obs').write_text('55 55 center\n')
        (smoke/'sfincs.inp').write_text('''x0 = 0
y0 = 0
dx = 10
dy = 10
mmax = 10
nmax = 10
rotation = 0
epsg = 6677
tref = 20240821 000000
tstart = 20240821 000000
tstop = 20240821 010000
depfile = sfincs.dep
mskfile = sfincs.msk
inputformat = asc
outputformat = net
precipfile = sfincs.prcp
obsfile = sfincs.obs
zsini = 0
qinf = 0
manning = 0.03
dtout = 600
dthisout = 600
dtmaxout = 3600
advection = 1
alpha = 0.5
theta = 1.0
viscosity = 1
storecumprcp = 1
storevel = 1
timestep_analysis = 1
''')
    save_json(smoke/'test_contract.json',{'synthetic':True,'tokyo_coordinates':False,'test':'closed flat 100m x100m, 10mm/hr rain for3600s, no loss','expected_final_depth_m':0.01,'expected_volume_m3':100,'does_not_test':['subgrid','urban_flow_accuracy','Tokyo_inputs','river_boundary','intervention_interfaces'],'engine_image':IMAGE})
    assessment='''已选主求解器：SFINCS 2.4.0 Galibier（2026-06-15正式发布），以官方固定镜像和源代码标签复现。
选择依据：三区地表暴雨、外部河川来水与潮位的复合影响；子网格储水/过流关系；批量情景与后续地表措施。简化物理模型仍需网格、时间步与现场证据验证，不能默认解决全部道路拓扑。
对照候选：LISFLOOD-FP适合二维积水及不同数值方案；HEC-RAS 7.0.1支持Linux计算引擎，Windows建模生态较重，可后续用于一维河道/复杂构筑物对照。选择SFINCS是本项目判断，不是无条件精度排名。
统一路线：1数据/域 → 2成熟二维基线 → 3真实事件验证 → 4三区机制与三维影响表达 → 5主动措施。
数据策略：PLATEAU负责城市几何和三维表达；裸地DEM负责地表，河岸与开口决定实际交换，河段H(t)/Q(t)和潮位控制边界。不把水域几何高度当水位。
初拟20m水动力网格/0.50m子网格地形，10m作敏感性；未通过验证之前不冻结。局部细节核查只针对影响总体结果的热点。
优先缺口：完整直接汇水域与裸地DEM元数据、核心分布暴雨、对应河段水位/来水及主要岸顶与开口、事件积水证据。粗糙度、入渗、等效排水及初始状态需参数范围，不沿用试算假设当实测。
本次建立的是可执行输入检查、运行入口和情景合同；尚未生成三区正式子网格表、暴雨场和河岸边界，也没有产生经过验证的三区积水结果。
主动措施在前四步之后启动：容量/选址/运行记录齐备，同暴雨和来水做基线对照，报告减淹、邻区风险转移与失效。大到改变河道水位的措施需要河道耦合，固定实测H(t)不能回答该反馈。
'''
    (report/'assessment.txt').write_text(assessment)
    sources=[('固定正式发布','https://github.com/Deltares/SFINCS/releases/tag/'+TAG),('固定输入文档','https://github.com/Deltares/SFINCS/blob/'+COMMIT+'/docs/input.rst'),('固定措施接口','https://github.com/Deltares/SFINCS/blob/'+COMMIT+'/docs/input_structures.rst'),('子网格论文','https://gmd.copernicus.org/articles/18/843/2025/'),('HydroMT稳定版','https://github.com/Deltares/hydromt_sfincs/releases/tag/v1.2.2'),('HEC-RAS当前下载','https://www.hec.usace.army.mil/software/hec-ras/download.aspx'),('LISFLOOD-FP官方','https://www.bristol.ac.uk/geography/research/hydrology/models/lisflood/')]
    save_json(report/'solver_selection.json',{'selected':'SFINCS','selection_is_project_judgment':True,'alternative_summary':{'LISFLOOD-FP':'二维洪水强候选；本轮未安装或进行相同事件性能比较','HEC-RAS':'7.0.1支持Linux计算；后续一维河道/结构对照候选'},'sources':sources,'release_docs_cached':'data/raw/solver_reference/sfincs_v2.4.0','version_warning':'latest手册已含后续Hautacam控制接口，实施以固定标签文档为准。'})
    esc=html.escape
    table=''.join('<tr>'+''.join('<td>'+esc(str(r[k]))+'</td>' for k in ['priority','input','readiness','gap','next_action'])+'</tr>' for r in rows)
    cards=''.join(f'<section><b>{s["step"]}. {esc(s["name"])}</b><p>{esc("；".join(s["deliverables"]))}</p><small>{esc(s["exit"])}</small></section>' for s in stages)
    measure_html=''.join(f'<section><b>{m["id"]} {esc(m["name"])}</b><p>需要：{esc("、".join(m["needs"]))}</p><small>{esc(m["limits"])}</small></section>' for m in measures)
    (report/'review.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区二维水动力框架</title><style>body{{font:16px/1.6 -apple-system,sans-serif;color:#173144;background:#f4f7f8;margin:0}}main{{max-width:1200px;margin:auto;padding:30px}}h1{{font-size:30px}}.note{{background:#fff1d5;padding:18px;border-left:5px solid #db9a27}}section{{background:white;padding:18px;margin:12px 0;border-radius:8px}}small{{display:block;color:#536573}}.table{{overflow:auto}}table{{border-collapse:collapse;min-width:850px;width:100%;background:white}}td,th{{padding:12px;text-align:left;vertical-align:top;border:1px solid #dce3e7}}th{{background:#e6edf1}}a{{color:#12677e}}code{{overflow-wrap:anywhere}}footer{{margin-top:25px;color:#536573}}</style><main><h1>都心三区：二维水动力基线与主动措施框架</h1><p>千代田区 · 中央区 · 港区，以及必要的直接跨区地表汇水区</p><div class="note"><b>状态：框架已建，三区正式基线仍待关键数据。</b> 本页没有把早期自编试算当作经过验证的水动力模型。运行环境与小型合成测试的实际结果见 engine_validation.json。</div><h2>求解器选择</h2><section><b>SFINCS 2.4.0 Galibier</b><p>适合地表降雨、外部河川来水和潮位的联合情景分析；子网格保留细地形的储水和过流关系。20 m水动力 / 0.50 m子网格为待验证起点，10 m作敏感性。</p><small>固定官方镜像仅linux/amd64；本机arm64用Docker仿真。HydroMT-SFINCS正式数据转换尚未安装、执行或兼容性验证。</small></section><h2>五步路线</h2>{cards}<h2>输入与缺口</h2><p>partial表示有部分资料；missing表示缺少可用模型输入；diagnostic_only只用于诊断。文件存在不等于可用于复现。</p><div class="table"><table><thead><tr><th>优先级</th><th>输入</th><th>状态</th><th>关键缺口</th><th>高效补齐方式</th></tr></thead><tbody>{table}</tbody></table></div><h2>第五步：主动措施</h2>{measure_html}<h2>已有运行入口</h2><section><code>.venv/bin/python src/exp/sfincs_workflow.py audit</code><p>检查已登记来源存在性和基线关键缺口；运行命令、五步流程和对比指标均保存为JSON。正式模型需另行构建输入包并完成数值/事件验证。</p></section><h2>官方来源</h2><ul>{''.join(f'<li><a href="{esc(u)}">{esc(t)}</a></li>' for t,u in sources)}</ul><footer>此框架保留未完成项；合成安装测试不能证明东京模拟的准确性。更新时间：{datetime.now(timezone.utc).isoformat()}</footer></main></html>''')
    print(json.dumps({'processed':str(proc),'report':str(report),'inputs':len(rows),'critical_missing':json.loads((proc/'readiness.json').read_text())['critical_missing_inputs']},ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); args=ap.parse_args(); main(args.root.resolve())
