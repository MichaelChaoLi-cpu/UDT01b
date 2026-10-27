#!/usr/bin/env python3
"""Physical Measure Road Effects.
AnaSOP Sections 5-7: matched two-parent/two-stage physical-arm trajectories,
common qualified roads, original lower-depth branch, synchronized actual clocks.
Revised display: simultaneous baseline-minus-treatment net closed-length changes.
Presentation only: no simulation, training, new statistical model or source edits.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/udt01b-article-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TITLE = 'Physical Measure Road Effects'
AUDIT = ROOT/'data/exp/figure-table-generation/three_ward_transport_v1'
BASE = ROOT/'data/processed/three_ward_analysis_contract_v1'
PHYSICAL = ROOT/'data/processed/three_ward_barrier_drainage_v1'
OUTPUT = ROOT/'data/results/figures/Figure_physical_measure_road_effects.png'
EVENTS = ['E20240618', 'E20250911']
STAGES = ['L', 'H']
ARMS = ['none', 'barrier', 'drainage', 'combined']
LABELS = ['No measure', 'Barrier', 'Drainage', 'Combined']
COLORS = ['#555555', '#ee7733', '#4477aa', '#228833']
STYLES = ['-', (0,(5,2)), (0,(1,1.5)), (0,(5,2,1,2))]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def render_explanation(protocol, protected_km, corridor_share, receipt):
    caption = f"""Physical Measure Road Effects under a fixed local deployment. Net simultaneous closed-road length reduction, D_m(t) = L_0(t) - L_m(t), for 2024-06-18 (a,b) and 2025-09-11 (c,d), under low (a,c) and raised (b,d) prescribed receiving-water stages. Orange dashed, blue dotted and green dash-dot curves denote barrier, drainage and combined measures. Zero denotes no net change; positive values indicate fewer closed road kilometers and negative values more. Shading below zero highlights net worsening. This aggregate difference does not identify individually restored roads: restored and newly closed links can coexist. The maximum of D_m(t) is not the difference between each arm's own peak. Its integral matches the previously defined avoided closure exposure. All panels share limits and use original lower-depth road bounds, 0.30 m closure rules, native clocks, 109 frames and common qualified support of 836.958 km. Unknown exposure of 277.920 km is separate. The frozen target corridor includes 118 source roads totaling {protected_km:.3f} km ({100*corridor_share:.2f}% of qualified length); this is not an exact hydraulic benefit footprint. Barriers occupy 696 ten-meter faces (6.96 km), stand 0.5 m above sampled static face ground and permit overtopping (coefficient 0.6). Road-crossing exclusions preserve 610 candidate face openings. Nineteen hypothetical pump intakes, each rated at most 0.5 cubic meters per second, activate at source depth 0.05 m above the native minimum, respect prior donor storage and a 5 m lift cutoff, and discharge to identified prescribed-stage water receivers through assumed straight links no longer than 500 m. The total 9.5 cubic meters per second is a capacity ceiling, not realized discharge. Combined runs apply both structures and pumps without adding their separate outcome changes. Parameters and placement are hypothetical, frozen before the new intervention runs; the 2025 parent supplied earlier selection context and 2024 is a limited transfer check. SFINCS computes two-dimensional surface flow on the existing 10 m grid using terrain and three-dimensional-city-derived building, road and connection constraints. River storage/receiver capacity, detailed pipes and surveyed construction feasibility are not represented. These two-parent, sixteen-state results describe one local deployment rather than the effectiveness of a three-ward or Tokyo-wide mitigation portfolio. Absolute closure histories remain archived; no sources, numerical gates or cases were changed. All arms retain closure at nine hours, so post-window recovery is unknown."""
    (AUDIT/'physical_measure_road_effects_caption.md').write_text(caption+'\n')
    methods=f"""# Physical Measure Road Effects: implementation and scope

The current experiment evaluates one frozen local deployment under two historical rainfall parents and two prescribed receiving-water stages. It does not optimize or comprehensively deploy mitigation across the three wards or Tokyo.

## Model chain

Terrain and building/road/connectivity information derived from PLATEAU constrain a two-dimensional SFINCS surface-water model on the existing full-domain 10 m grid. Historical radar rain and conservative roof redistribution supply water; receiving-water surfaces are prescribed low/high scenarios. The setup preserves the original dry start, actual step clocks, positivity limiter and source/sink budget. All four arms retain identical forcing, solver, road support and approved output times within each event-stage.

No measure retains the urban baseline. Barrier runs add overtoppable face structures. Drainage runs add finite-capacity source-cell removal and explicit receiving-side accounting. Combined runs apply both operators during the physical simulation; their outcomes are not a sum of two separately simulated effects. Twelve treatment runs plus four reused baselines provide sixteen states, with 109 full-field times per state. Existing numerical admission establishes computational consistency, not empirical impact validation.

At each simultaneous output, the original ground-to-road mapping supplies lower and upper road depth bounds. The main view retains the lower branch and closes qualified road length at depth at least 0.30 m. It reports the unchanged sum of unique ward-clipped lengths, with unknown structural exposure kept separate. No protected road is forced dry or manually reopened. The pointwise display subtracts treatment closed length from no-measure closed length at matching approved frame/time. This is a net aggregate: it does not show which individual links reopen or close. Its integral is avoided closure exposure; its maximum is not peak-to-peak avoided closed length.

## Placement

An earlier 2025-event road-selection exercise froze 118 target source roads; their three-ward clipped length is {protected_km:.6f} km, {100*corridor_share:.4f} percent of qualified length. The hydraulic corridor expands their source cells by one 10 m cell. This length share describes target-road support rather than the exact protected area or the share of floodwater removed. Placement used this frozen corridor and static terrain, without optimizing against the new intervention outcomes. The 2025 storm remains selection context and 2024 a limited transfer check.

Barrier candidates lie on corridor-edge faces, excluding previously blocked faces and all mapped motor-road centerline crossings with a 1 m buffer. Unsupported crest-elevation candidates are also excluded. There are 696 retained faces, each 10 m, totaling 6.96 km of structure, with 610 candidate faces excluded to retain road openings. Structure length is not protected road length. Crest elevation is sampled maximum static face ground plus 0.5 m; the weir coefficient is 0.6. Structures restrict face exchange but allow overtopping rather than forming sealed enclosures. Opening widths are unresolved at the 10 m model scale.

Pump candidate cells lie within the expanded corridor. Static depression score, then low ground and stable index order rank them. Candidate intakes require an eligible prescribed-water receiver within 500 m straight-line distance and separation of at least 100 m from other chosen intakes. The procedure permits up to 20 units; 19 are present in the actual saved inputs. Link lines are assumptions, not surveyed pipe/hose routes.

## Pump operator

Each unit has maximum capacity 0.5 cubic meters per second, so all 19 units together have a ceiling of 9.5. Pumping is allowed when source depth above the native minimum reaches 0.05 m and receiving-surface minus source-surface positive lift does not exceed 5 m. Each step transfers at most capacity times step length and available prior donor volume, retaining the existing roundoff reserve. Previously reserved pump outflow and outgoing face demand cannot spend the same stored water twice. Rainfall and simultaneous incoming transfers are not pre-spent.

The operator pairs internal source and sink transfers. In this saved deployment, recipients are prescribed-stage water cells: discharged volume leaves the modeled dynamic storage and is explicitly recorded as an external sink. Changing receiving stage affects the lift check, but discharged water does not raise a modeled river hydrograph. River storage, outlet capacity, hose routes, actual pump curves, energy and sewer hydraulics are unmodeled. Maximum nominal capacity is not actual pumped discharge or proof of sufficient receiving capacity.

## Scale and future scenario contrasts

The current four arms vary measure type. Two rain parents and two stages vary forcing, but all use one fixed deployment scale, layout and operating rule. Small whole-area effects here cannot establish that barriers or drainage are ineffective in general. Conversely, scaling up the nominal capacity does not guarantee proportional benefits.

A follow-on design should separate: (1) capacity changes at identical locations and barrier geometry; (2) spatial coverage or placement alternatives at a fixed total pump-capacity and barrier-length budget; (3) alternative operation rules at fixed geometry and capacity; and (4) barrier, drainage and combined arms under the same forcing. Expansion beyond current corridors needs a new eligible-site/receiver inventory. Compare local target and all-other-road outcomes and retain adverse depth, newly closed roads and new resident access loss. Receive-system capacity must be explicit. The exact scenario counts, parameter levels and compute budget remain a proposed next design, not a newly registered experiment or engineering recommendation. No new hydraulics were run for this revision.

## Provenance

All parameter statements were checked against the saved protocol, barrier and pump tables, placement script, pump operator and solver binding. File hashes and twelve pointwise difference-integral checks are in physical_measure_road_effects_validation.json. Original absolute-trajectory output is preserved under revisions/physical_before_relative_effects_20261020/. English manuscript caption is provided separately.
"""
    (AUDIT/'physical_measure_road_effects_methods.md').write_text(methods)
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>局部固定部署：措施如何改变封路</title><style>body{{font:16px/1.75 system-ui;max-width:1250px;margin:30px auto;padding:0 24px;color:#223845}}img{{width:100%;height:auto}}aside{{background:#edf4f7;padding:16px;border-left:4px solid #47788f}}a{{color:#176992}}h1{{font-size:27px}}h2{{font-size:22px;margin-top:32px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d6e0e5;padding:10px;text-align:left}}.chain{{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}}.chain span{{padding:12px;background:#edf4f7;border-radius:6px;flex:1;min-width:140px}}.scope{{background:#fff6e7;padding:16px}}small{{color:#596c78}}</style>
<h1>局部固定部署：挡水、排水如何改变三区封路</h1><p>第三张图按反馈修订，等待图面确认。前两张图已认可。使用原16组结果，本次没有追加水动力模拟。</p>
<aside><b>这张图回答：同一时刻，相对于不采取措施，净减少了多少封路？</b><br>纵轴＝无措施封路长度－措施下封路长度。正值表示净减少，负值表示净增加，零表示总长度不变。负值区域浅红底色突出恶化。它是三区汇总的净差，同一时刻可以既有恢复道路、又有新封道路；曲线最大值也不是两个方案各自峰值之差。</aside>
<img src="../../../results/figures/Figure_physical_measure_road_effects.png?v={receipt['output_sha256'][:12]}" alt="Four panels of net closed-road length reduction relative to no measure">
<p>上排2024-06-18，下排2025-09-11；左列低水位，右列抬升水位。三条线比较挡水、排水与挡排结合，四幅图共用坐标。保留450 dpi、原质量掩膜、0.30 m封路阈值与水深下界主分支。曲线积分已逐一核对，等于原累计封路影响的减少量。</p>
<h2>这轮部署有多大？</h2>
<div class="scope"><b>它是一套局部走廊部署，不能代表三区整体或全东京的措施潜力。</b><p>预先选定118条源道路，三区内裁剪长度约{protected_km:.3f} km，占共同准入道路836.958 km的<b>{100*corridor_share:.2f}%</b>。这个比例描述目标道路，不等于实际保护面积或水动力影响范围。19个泵点最大合计9.5 m³/s，实际排量还受有水可抽、启动与扬程条件约束。</p><p>挡水结构共6.96 km，与目标道路长度是不同对象，不能当作“保护了6.96 km道路”。当前没有系统比较部署规模、位置或运行规则；现有结果只能说明这套固定局部方案的响应。</p></div>
<h2>措施怎么布设、如何进入模型？</h2>
<table><tr><th>方案</th><th>布设与参数</th><th>计算中发生什么</th></tr>
<tr><td>无措施</td><td>保留原城市地形、建筑、道路与地表连通约束。</td><td>提供每个同雨同水位情景的基线。</td></tr>
<tr><td>挡水</td><td>沿冻结走廊外缘选择696个10 m格面，总长6.96 km；堰顶为采样地面最高值＋0.5 m，系数0.6。排除已阻断格面及道路穿越位置，保留610个候选格面开口。</td><td>将结构作为可漫顶堰加入格面过流计算，改变水的流向和通量。不是把道路强制设成干燥，也没有直接把挡水位置设为交通封闭。</td></tr>
<tr><td>排水</td><td>按走廊内静态低洼程度、地面高程排序，泵点间隔至少100 m，并要求500 m直线范围内有合格接收水域。最多选20点，实际保存19点。每点最大0.5 m³/s；启动水深0.05 m，最大扬程5 m。</td><td>每一步仅抽取已有可用储水，最大为额定排量×步长；检查接收水位和扬程。实际抽水进入显式水量预算，不借用同一步尚未到达的雨水或入流。</td></tr>
<tr><td>挡排结合</td><td>使用相同挡水结构和相同泵点。</td><td>在同一水动力模拟中同时启用两类措施，结果允许相互作用，不把两组独立效果直接相加。</td></tr></table>
<p>位置来源于此前2025场景选出的固定道路走廊及静态地形；没有看完本轮措施结果再选点。2025场仍属于位置选择背景，2024场只是有限的跨事件检查。</p>
<details><summary>查看既有部署图：挡水格面、抽水点及假设接收连线</summary><img src="../../estimation/three_ward_barrier_drainage_v1/deployment.png" alt="Existing hypothetical deployment map"><p><small>既有布设示意图使用EPSG:6677投影坐标。连线是假设的直线转输联系，不是已核验的管线；本图作为方法说明，未新增部署。</small></p></details>
<h2>从城市三维信息到封路结果</h2>
<div class="chain"><span>地形＋PLATEAU建筑／道路／连通约束</span><span>历史雷达降雨＋屋面保守分配＋指定河道水位</span><span>SFINCS二维地表水动力<br>10 m网格＋挡水／抽水算子</span><span>每5分钟道路积水深度<br>原阈值判定封路</span><span>同一时刻长度配对相减<br>正负效果均保留</span></div>
<p>三维城市信息约束地表蓄水与过流；求解器计算二维地表积水过程，并非完整三维河道、管网或地下空间流动。两场降雨×两档指定水位×四种措施共有16组，含4组复用基线和12组既有措施模拟，每组9小时、109个输出时刻。各组通过原时钟、水量预算、正性与质量检查。</p>
<p><b>排出的水去了哪里？</b>进入指定水位的接收水域，在动态储量预算中作为外排量记录。接收水位影响扬程开关，但这些抽排水不会让模拟河道自动涨水。当前没有河道储量／接纳能力、实际软管路线、完整泵曲线或下水道耦合。因此9.5 m³/s是方案上限，不能称为实测设施能力或工程可行性。</p>
<h2>措施也应有多情景：下一轮应怎样扩展</h2>
<p>你指出的缺口成立：当前已比较雨型、水位和措施类型，尚未比较部署规模、布设方式与运行规则。以下是扩展设计方向，<b>尚未运行、没有新结果</b>：</p>
<table><tr><th>要区分的问题</th><th>只改变什么</th><th>必须固定／核查什么</th></tr>
<tr><td>容量是否不足</td><td>在相同位置设置低、中、高抽排容量。</td><td>保持布点、挡水与启动规则不变；计算保护范围降雨汇流量、实际抽排量及容量受限时段，不能只数泵点。</td></tr>
<tr><td>布点是否合理</td><td>静态低洼布点与交通关键连接优先布点；局部与更分散的覆盖。</td><td>同总抽排容量、同挡水长度预算比较；扩展范围先核验候选地与接收水域。等额物理预算不等于等工程成本。</td></tr>
<tr><td>运行是否关键</td><td>预设的启动／停止规则。</td><td>同位置、同容量；规则先冻结，避免逐场看结果调参。</td></tr>
<tr><td>组合是否互补</td><td>挡水、排水、结合，以及无措施基线。</td><td>相同雨、水位、网格与质量支持；不假定线性增益。保留保护走廊之外新增积水、封路和人口通达性损失。</td></tr></table>
<p>先用已有两场降雨、两档水位检查上述机制，再决定扩展至其他历史雨型。具体参数档位、可用部署范围和计算预算需在下一轮方案中确定；这次只修图并补齐方法及范围说明。</p>
<p>共同准入道路836.958 km，未知暴露277.920 km单列；所有方案在9小时末仍有封路，完整恢复时间未知。图中的净长度改善不能替代交通成本、人口通达性或邻区风险评估。</p>
<p><a href="../../../results/figures/Figure_physical_measure_road_effects.png">450 dpi PNG</a> · <a href="physical_measure_road_effects_caption.md">English caption</a> · <a href="physical_measure_road_effects_methods.md">English methods and scope</a> · <a href="physical_measure_road_effects_validation.json">核验记录</a> · <a href="revisions/physical_before_relative_effects_20261020/Figure_physical_measure_road_effects.png">修改前绝对长度图</a></p></html>"""
    (AUDIT/'physical_measure_road_effects_review.html').write_text(html)

def main():
    doc=ROOT/'docs/AnaSOP.md'; doc_sha=sha(doc)
    planpath=ROOT/'data/exp/research-design/three_ward_figure_table_plan_v1/confirmed_plan.json'
    plan=json.loads(planpath.read_text());item=next(x for x in plan['figures'] if x['title']==TITLE)
    assert plan['status']=='confirmed' and item['panels']==4 and item['type']=='line'
    assert f'| {TITLE} |' in doc.read_text()
    sources=[BASE/'road_trajectories_preprocessed.parquet',BASE/'road_summaries_preprocessed.parquet',BASE/'export_manifest.json',PHYSICAL/'road_trajectories.csv',PHYSICAL/'road_summary.csv',planpath]
    for rel in ['data/exp/data-preprocessing/three_ward_analysis_contract_v1/validation.json','data/exp/research-design/three_ward_estimation_framework_v1/validation.json','data/exp/estimation/three_ward_barrier_drainage_v1/delivery_validation.json','data/exp/estimation/three_ward_barrier_drainage_v1/postprocessing_validation.json']:
        p=ROOT/rel;assert json.loads(p.read_text())['passed'];sources.append(p)
    sources += [PHYSICAL/'protocol.json', PHYSICAL/'measures/barrier_faces.csv',
                PHYSICAL/'measures/pump_sites.csv', PHYSICAL/'measures/barrier_exclusions.csv',
                ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz',
                ROOT/'src/preprocessing/prepare_barrier_drainage_inputs_v1.py',
                ROOT/'src/preprocessing/barrier_drainage_v1/pump_routines.inc',
                PHYSICAL/'solver_binding.json']
    bindings={str(p.relative_to(ROOT)):sha(p) for p in sources}
    protocol=json.loads((PHYSICAL/'protocol.json').read_text())
    barriers=pd.read_csv(PHYSICAL/'measures/barrier_faces.csv')
    pumps=pd.read_csv(PHYSICAL/'measures/pump_sites.csv')
    exclusions=pd.read_csv(PHYSICAL/'measures/barrier_exclusions.csv')
    with np.load(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/clipped_road_lengths.npz') as z:
        protected=z['protected']; qualified=z['qualified']; weights=z['union_length_m']
        protected_km=float(weights[protected].sum()/1000)
        corridor_share=protected_km/float(weights[qualified].sum()/1000)
        assert protected.sum()==118 and not np.any(protected & ~qualified)
    assert len(barriers)==protocol['barrier_faces']==696
    assert len(pumps)==protocol['pumps']==19
    assert np.allclose(barriers.crest_m-barriers.ground_max_m,.5,atol=1e-8)
    assert np.allclose(barriers.Cd,.6) and np.allclose(pumps.capacity_m3_s,.5)
    assert pumps.capacity_m3_s.sum()==protocol['total_capacity_m3_s']==9.5
    assert np.allclose(pumps.trigger_depth_above_native_min_m,.05)
    assert np.allclose(pumps.maximum_lift_m,5) and pumps.straight_link_length_m.max()<=500
    assert (exclusions.reason=='mapped_road_opening_retained').sum()==610
    assert protocol['engineering_parameters_observed'] is False

    manifest=json.loads((BASE/'export_manifest.json').read_text())
    for e in manifest['outputs']:
        if e['label'] in ['Road trajectories','Road summaries']:
            assert sha(ROOT/e['source'])==e['source_sha256']
            assert sha(ROOT/e['output'])==e['output_sha256'] and e['exact_values_and_nulls']
    df=pd.read_parquet(sources[0]); summary=pd.read_parquet(sources[1])
    data=df[(df['Road Depth Bound']=='lower')&(df['Reporting Scope']=='Three wards')&(df['Road Stratum']=='all')]
    sums=summary[(summary['Road Depth Bound']=='lower')&(summary['Reporting Scope']=='Three wards')&(summary['Road Stratum']=='all')]
    assert len(data)==1744 and len(sums)==16
    known=836.9583262602582;unknown=277.9200922406389
    assert np.allclose(data['Qualified Road Length'],known,rtol=0,atol=1e-8)
    assert np.allclose(data['Unknown Road Length'],unknown,rtol=0,atol=1e-8)
    assert np.allclose(data['Open Road Length']+data['Slow Road Length']+data['Closed Road Length'],known,rtol=0,atol=1e-7)
    curves={};checks=[];clocks={}
    for event in EVENTS:
        for stage in STAGES:
            for arm in ARMS:
                case=f'{event}__{stage}__U__{arm}'
                g=data[data.Case==case].sort_values('Frame')
                assert len(g)==109 and np.array_equal(g.Frame,np.arange(109))
                # Read only the tiny clock members, never the native-depth cube.
                with np.load(PHYSICAL/'exposure'/f'{case}.npz') as z:
                    actual=z['actual_times'];analysis=z['analysis_times']
                times=g['Elapsed Time'].to_numpy();values=g['Closed Road Length'].to_numpy()
                # CSV-to-Parquet export preserves source decimals; binary clock roundtrip differs by at most 4e-12 s.
                assert np.allclose(times,analysis,rtol=0,atol=1e-10) and np.all(np.diff(actual)>0)
                assert np.max(abs(actual-times))<=.001 and times[0]==0 and times[-1]==32400
                assert np.all(values>=0) and np.isfinite(values).all()
                row=sums[sums.Case==case];assert len(row)==1;row=row.iloc[0]
                integral=float(np.trapezoid(values,times)/3600)
                assert np.isclose(integral,row['Closed Road Duration'],rtol=0,atol=1e-8)
                assert np.isclose(values.max(),row['Peak Closed Road Length'],rtol=0,atol=1e-8)
                assert int(np.argmax(values))==int(row['Peak Closure Frame'])
                assert np.isclose(values[-1],row['Final Closed Road Length'],rtol=0,atol=1e-8)
                curves[event,stage,arm]=(actual/3600,values)
                clocks[case]={'file':str((PHYSICAL/'exposure'/f'{case}.npz').relative_to(ROOT)),'actual_clock_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'analysis_clock_sha256':hashlib.sha256(analysis.tobytes()).hexdigest(),'max_actual_analysis_difference_s':float(np.max(abs(actual-times))),'table_clock_roundtrip_error_s':float(np.max(abs(times-analysis)))}
                checks.append({'case':case,'frames':len(g),'peak_closed_km':float(values.max()),'peak_frame':int(np.argmax(values)),'closure_exposure_km_h':integral,'closed_at_9h_km':float(values[-1]),'end_window_right_censored':bool(values[-1]>0)})
    contrasts=[]
    for event in EVENTS:
        for stage in STAGES:
            baseline=next(x for x in checks if x['case']==f'{event}__{stage}__U__none')
            for arm in ARMS[1:]:
                treatment=next(x for x in checks if x['case']==f'{event}__{stage}__U__{arm}')
                contrasts.append({'event':event,'stage':stage,'measure':arm,'avoided_peak_closed_km':baseline['peak_closed_km']-treatment['peak_closed_km'],'avoided_closure_exposure_km_h':baseline['closure_exposure_km_h']-treatment['closure_exposure_km_h']})
    differences={}; difference_checks=[]
    for event in EVENTS:
        for stage in STAGES:
            base_t,base_y=curves[event,stage,'none']
            for arm in ARMS[1:]:
                t,y=curves[event,stage,arm]
                assert np.max(abs(t-base_t))*3600<=.001
                # Join by the unchanged frame and approved analysis time, not nearest timestamps.
                case=f'{event}__{stage}__U__{arm}'
                g=data[data.Case==case].sort_values('Frame')
                analysis=g['Elapsed Time'].to_numpy()
                bg=data[data.Case==f'{event}__{stage}__U__none'].sort_values('Frame')
                assert np.allclose(analysis,bg['Elapsed Time'],rtol=0,atol=.001)
                delta=base_y-y
                integral=float(np.trapezoid(delta,analysis)/3600)
                original=next(x for x in contrasts if x['event']==event and x['stage']==stage and x['measure']==arm)
                assert np.isclose(integral,original['avoided_closure_exposure_km_h'],rtol=0,atol=1e-8)
                differences[event,stage,arm]=(base_t,delta)
                difference_checks.append({'event':event,'stage':stage,'measure':arm,'minimum_net_closed_length_reduction_km':float(delta.min()),'maximum_net_closed_length_reduction_km':float(delta.max()),'integrated_net_reduction_km_h':integral,'original_exposure_difference_matched':True})
    sns.set_theme(style='ticks',context='paper',font='DejaVu Sans',rc={'font.size':11,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.linewidth':.8})
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.6),sharex=True,sharey=True)
    fig.subplots_adjust(left=.085,right=.98,bottom=.14,top=.94,hspace=.34,wspace=.20)
    for ri,event in enumerate(EVENTS):
        for ci,stage in enumerate(STAGES):
            ax=axes[ri,ci]
            ax.axhspan(-.75,0,color='#fcf0ee',zorder=0)
            ax.axhline(0,color='#555555',lw=1.0,zorder=2)
            for ai,arm in enumerate(ARMS[1:],1):
                t,y=differences[event,stage,arm]
                ax.plot(t,y,color=COLORS[ai],linestyle=STYLES[ai],lw=1.9,label=LABELS[ai],zorder=2+ai)
            ax.set_xlim(0,9);ax.set_ylim(-.75,4);ax.set_xticks([0,3,6,9]);ax.set_yticks([-.5,0,1,2,3,4])
            ax.grid(axis='y',color='#dce3e8',lw=.7);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            ax.set_xlabel('Elapsed model time (h)');ax.set_ylabel('Net closed-road length reduction (km)')
            ax.tick_params(labelbottom=True,labelleft=True)
            day=f'{event[1:5]}-{event[5:7]}-{event[7:9]}'
            letter='abcd'[ri*2+ci];stage_label='Low stage' if stage=='L' else 'Raised stage'
            ax.text(-.10,1.065,f'{letter}: {day}, {stage_label.lower()}',transform=ax.transAxes,fontsize=12,fontweight='bold',va='bottom')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.005),ncol=3,frameon=False,fontsize=11,handlelength=3.5,columnspacing=2.2)
    fig.text(.54,.055,'Relative to no measure: positive = fewer closed km; negative = more closed km',ha='center',fontsize=10,color='#444444')
    positions={k:list(a.get_position().bounds) for k,a in zip('abcd',axes.flat)}
    errors={'a_c_left':abs(positions['a'][0]-positions['c'][0]),'b_d_left':abs(positions['b'][0]-positions['d'][0]),'a_b_top':abs(sum(positions['a'][1::2])-sum(positions['b'][1::2])),'c_d_bottom':abs(positions['c'][1]-positions['d'][1])}
    assert max(errors.values())<1e-12
    OUTPUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUTPUT,dpi=450,bbox_inches='tight',facecolor='white');plt.close(fig)
    with Image.open(OUTPUT) as im:
        metadata={'width_px':im.width,'height_px':im.height,'dpi':list(im.info['dpi'])}
        assert abs(im.info['dpi'][0]-450)<.1
    assert sha(doc)==doc_sha
    assert all(sha(ROOT/k)==v for k,v in bindings.items())
    receipt={'title':TITLE,'passed':True,'status':'generated_awaiting_user_confirmation','revision_decision':'MILI-D-20261020-015','scale_context_decision':'MILI-D-20261020-016','display':'simultaneous no-measure minus treatment net closed-road length','difference_checks':difference_checks,'protected_corridor_km':protected_km,'protected_corridor_share_of_qualified_length':corridor_share,'deployment_context':'one frozen local-corridor deployment;not a ward-wide or Tokyo-wide mitigation portfolio','research_question':'RQ4','framework':'AnaSOP Sections 5-7: matched physical road trajectories, original threshold and same-branch peak/exposure contrasts','rain_parents':2,'event_stage_cases':4,'arms_per_case':4,'total_states':16,'trajectory_rows':1744,'depth_branch':'lower','qualified_road_length_km':known,'unknown_road_length_km':unknown,'closure_threshold_m':.30,'actual_clocks_checked':clocks,'case_checks':checks,'same_branch_contrasts':contrasts,'panel_positions':positions,'alignment_error_figure_fraction':errors,'common_axes':{'elapsed_h':[0,9],'net_closed_length_reduction_km':[-.75,4]},'dpi':450,'png_metadata':metadata,'source_sha256':bindings,'script_sha256':sha(Path(__file__)),'output_sha256':sha(OUTPUT),'output':str(OUTPUT.relative_to(ROOT)),'source_data_unchanged':True,'AnaSOP_unchanged_during_generation':True,'new_simulations':0,'new_training':0,'new_traffic_fits':0,'style_decision':'MILI-D-20261020-008'}
    AUDIT.mkdir(parents=True,exist_ok=True);write_json(AUDIT/'physical_measure_road_effects_validation.json',receipt)
    render_explanation(protocol, protected_km, corridor_share, receipt)
    print(f'Saved: {OUTPUT.relative_to(ROOT)}');print('Validated:16 states x109 frames, original clocks/support, all peak and exposure summaries, unchanged sources,450 dpi.');print(json.dumps(contrasts,indent=2))

if __name__=='__main__':main()
