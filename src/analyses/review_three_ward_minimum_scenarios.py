# Regenerate local forcing-QC review and narrow skill-routed AnaSOP status updates.
from pathlib import Path
import sys,json,csv,hashlib,re,html
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'))
import generate_three_ward_minimum_scenarios as gen
import load_three_ward_minimum_scenarios as api

def main(root=ROOT):
 root=Path(root);report=root/gen.REPORT;out=root/gen.LIB
 summary=json.loads((report/'build_summary.json').read_text());v=json.loads((report/'validation.json').read_text());cat=api.catalog(root)
 if not v['all_passed'] or not (out/'freeze.json').exists():raise ValueError('Validated, frozen forcing catalog required')
 # Dependency checks for skill-guided estimation updates: same applied question,
 # candidate variables and pending evidence outputs; no new model claim is added.
 sop=root/'docs/AnaSOP.md';before=sop.read_text();backup=report/'previous-AnaSOP.md'
 if not backup.exists():backup.write_text(before)
 parts=re.split(r'(?=^## [1-8]\. )',before,flags=re.M);oldparts=list(parts)
 def replace_section(number,fn):
  for i,s in enumerate(parts):
   if s.startswith(f'## {number}. '):parts[i]=fn(s);return
  raise ValueError('Missing AnaSOP section '+str(number))
 def four(s):
  s=s.replace('Constructed uniform physical sequences and roof-redistributed surface fields on a common forcing-knot clock. Preserve one-second ramps and total volume; valid no-rain zeros are distinguished from missing data.',
   'Constructed uniform physical sequences and fixed conservative roof redistribution for 48 synthetic parents and two new controls. Preserve exact variable-length knot unions, original one-second control ramps and rainfall volume; valid no-rain zeros are distinguished from missing data.')
  s=s.replace('Constructed original point levels, boundary-group increments and point-specific low-stage offsets. Same-name groups remain distinct; receiver links do not create ground-to-ground paths across water.',
   'Constructed original point levels and point-specific low-stage offsets, with coherent additive triangular pulses for new parents. Preserve all seven boundary groups and the original pulse for controls. Working heights are inherited, not epoch-certified; receiver links do not create ground-to-ground paths across water.')
  marker='### Versioned minimum-library forcing construction'
  if marker not in s:
   s=s.rstrip()+'''\n\n### Versioned minimum-library forcing construction

Forty-eight new normalized synthetic rainfall realizations are generated: 24 training, 8 validation, 8 sealed main test and 8 separate challenges. Two new deterministic controls are also generated; the four existing controls are referenced unchanged. All six controls belong to one legacy control family and do not increase independent-parent counts. Parent-event identifiers, deterministic seeds, accepted attempts, native control points, forcing hashes, family membership and partition bindings are saved. Grid, tile, time-window, intensity/stage and later measure descendants must inherit their family partition.

The generator uses the versioned envelopes in Section 5. Physical rainfall is spatially uniform and linearly interpolated between native control points. Receiving-water pulses share a coherent increment over the seven distinct boundary groups, added to their point-specific low levels. The curve time knots are forcing interpolation nodes, not hydraulic response-output times. Exact-knot padded batches carry a time-valid mask; padding is not dry weather.

Frozen near-duplicate screening removes rainfall start, duration and peak scaling, then samples each normalized curve at 257 relative times. Flag pairs with normalized root mean squared difference at most 0.02, or at most 0.04 with correlation at least 0.995, regardless of stage differences. No cross-family or cross-partition near duplicates are found under this engineering rule. The screen does not establish statistical independence, observed-event diversity or sufficient training sample size.

Rainfall volume is checked against 38.9768 square kilometres of contributing area, consisting of 25.936197 square kilometres of open ground and 13.040603 square kilometres of roof area. Every fixed roof recipient is conserved under 5 m to 10 m aggregation and native adaptive allocation. The rule sends all roof rain immediately to the surface without delay, loss or underground capture; this is a runoff scenario rather than measured drainage. File-reader-equivalent precipitation and boundary checks cover six representative exports, preserve valid zero rain and reproduce total input volume within the declared rounding gate. These checks do not execute or validate the hydraulic solver.

The forcing catalog is frozen for audit; the label-quality policy and formal reference release remain pending. Fifty regular and eight adaptive descendants are registered as unrun. No new hydraulic responses, learned normalization, training or released test responses exist. Forcing-only quality checks can inspect sealed inputs; response access remains closed. Local custody is an application-level protocol rather than cryptographic isolation.

'''
  return s
 def five(s):
  s=s.replace('Counts are proposed allocations, not acquired events, statistical power, permission to simulate or proof that deep models will generalize.',
   'These allocations now bind 48 generated synthetic forcing realizations and two new controls, with source-linked seeds, normalized-shape checks and frozen partitions. They are not observed events, statistical power, permission to simulate or proof that deep models will generalize.')
  s=s.replace('Before any later execution, reserve parent lineages and partitions.', 'Parent lineages and partitions are now bound to generated forcings before any new hydraulic execution.')
  s=s.replace('Filling reserved IDs with numerical forcings is a later authorized task; lineage and near-duplicate audits can reduce the effective parent count.',
   'The forcing-only construction and normalized near-duplicate audit are complete. All 58 new hydraulic descendants remain unrun; later reference admission and response checks can still reduce the eligible learning pool.')
  s=s.replace('Use the proposed 60/20/20 split of the 40 main parents.', 'The frozen forcing partition uses the 60/20/20 split of the 40 main synthetic parents.')
  return s
 def seven(s):
  s=s.replace('The present update plans methods and samples; it does not execute training or additional hydraulic cases.',
   'Methods remain planned; the minimum-library forcing generator, volume checks, lineage audit and forcing partitions are now implemented. This update does not execute training or additional hydraulic cases.')
  s=s.replace('| 3. Reserve and later assemble the minimum library |','| 3. Construct forcings and later assemble reference responses |')
  s=s.replace('Diversity, grouping and leakage control: planned | Separate run authorization; forcing and roof-volume validation; lineage/near-duplicate review; no sealed-response inspection.',
   'Forcings and family partitions: constructed and checked; response library: absent | Keep frozen input hashes and family inheritance; separately authorize reference runs after numerical admission and output checks; no sealed-response inspection during setup.')
  s=s.replace('Current readiness remains planning-only: Section 4 candidate variables are not automatically finalized by the method decision. Input tensor/graph construction, reference admission, scenario generation, event splits and test custody require their own checks.',
   'Current readiness includes constructed static tensors and graph, 50 new forcing inputs, and audited family partitions. Reference responses for the new forcings remain absent. Section 4 variables retain candidate status; reference admission, target-output qualification, training-only scaling and response-release rules must be resolved before learning.')
  return s
 replace_section(4,four);replace_section(5,five);replace_section(7,seven)
 after=''.join(parts);changed=[i for i,(a,b) in enumerate(zip(oldparts,parts)) if a!=b]
 # Prevent inadvertent edits outside routed responsibilities.
 assert all(parts[i].startswith(('## 4. ','## 5. ','## 7. ')) for i in changed)
 if after!=before:sop.write_text(after)
 actual_changed=[re.match(r'## (\d)',parts[i]).group(1) for i in changed]
 gen.dump(report/'anasop-update.json',{'skills':{'4':'data-preprocessing','5':'estimation-framework-planning','7':'estimation-framework-planning'},
 'sections_changed_in_this_execution':actual_changed,'evidence':'validated frozen forcing catalog; MILI-D-20261015-005',
 'preimage_sha256':gen.sha(backup),'after_sha256':gen.sha(sop),'other_sections_unchanged':True,'language':'English','final_variable_status':'no','new_estimation_or_training':False})
 rows=[
 {'priority':'P0','item':'Rain and stage generator','status':'completed_forcing_only','next':'48 parent curves plus2 new controls frozen; preserve hashes and protocol.'},
 {'priority':'P0','item':'Roof-volume and export preflight','status':'passed_without_solver_execution','next':'Physical allocation conserved at10m/5m/adaptive; reader-equivalent export checks are not engine validation.'},
 {'priority':'P0','item':'Family lineage and split custody','status':'implemented','next':'24 train,8 validation,8 sealed test,8 separate challenges; no response release or normalization.'},
 {'priority':'P0','item':'Reference and quality-policy admission','status':'pending','next':'Review whole-domain numerical quarantine and freeze development-side admission policy before new sealed responses; retain current bounds and exclusions.'},
 {'priority':'P0','item':'Hydraulic output contract and first setup tranche','status':'pending_authorization','next':'Check solver/input version, water budget, synchronized states, native levels, exact/interval bounds and barrier diagnostics using designated8 setup cases and8 companions when later authorized.'},
 {'priority':'P1','item':'Output cadence','status':'not_changed','next':'No blanket denser-output demand for peak-only labels. Any time-series/path/flux claim needs separately qualified temporal outputs.'},
 {'priority':'P1','item':'Road length and critical-region rule','status':'pending','next':'Verify centerline/length-to-cell mapping and freeze region definitions on development evidence.'},
 {'priority':'P1','item':'Training-only preprocessing','status':'pending_no_training','next':'Fit scalers only on train inputs after separate authorization; preserve feature/time validity and typed graph links.'},
 {'priority':'P2','item':'Real-event and river-channel validity','status':'unestablished','next':'Source epochs, upstream coverage, measured events and bathymetry still limit physical claims; synthetic levels are boundary scenarios.'},
 {'priority':'P2','item':'Active measures','status':'later_milestone_5','next':'Require verified asset capacities, receivers, schedules and paired no-action/action responses; descendants inherit event split.'}]
 gen.write_csv(report/'remaining_prerequisites.csv',rows)
 readme='''# 三区最小情景库：强迫输入实现与核查

本轮仅生成输入，未训练、未搜索、未下载、未执行任何新的水动力模拟。沿用三区及现有相邻汇水范围，不逐点核查。

- 48 个新合成父事件：训练24、验证8、封存测试8、挑战8。
- 另生成2个开放控制组；4个已有控制组保持原样。6个控制组共用CONTROL-LEGACY谱系。
- 50份新强迫输入，未来50个10m运行及8个自适应伴随运行均登记为unrun；不是58个独立暴雨。
- 生成器记录种子、控制点、参数、接受尝试、语义哈希、谱系和划分。2次候选因进入主样本禁止的复合角落而重抽；没有调整去重阈值。
- 所有新曲线为合成试验设置，未赋予实测事件或重现期含义。48种归一雨型通过版本化近重复筛查，不等同于统计独立或足够样本量。

雨型：前峰、中峰、后峰、多峰；主样本峰值20–100 mm/h，历时30–90分钟。附加2×4种挑战：复合、雨强、历时和水位错峰。新父事件采用30分钟上升和30分钟下降的共同三角水位增量；原控制组水位曲线保持原样。7个边界组、10488个点的低水位偏移和点序保留。没有新增河道流量或河床断面。

雨量核查：物理雨深乘38.9768 km²汇水面积，分成25.936197 km²地面直降雨和13.040603 km²屋面分配。5m→10m及原生自适应分配完全嵌套。屋面即时全量入地表、无延迟无损失、不入地下管网，是固定情景假设。有效雨量使用名义水动力格面积作分母，不能再用开敞地面积重复放大。

数据格式：共享几何加逐事件NPZ曲线，避免复制50套时空栅格。JSON/CSV提供语义、单位、来源和谱系，不把空间数组转成Excel/parquet。原始数据和旧标签未修改。变长时间用Time Valid补齐，保留原控制组一秒斜坡，未拟合归一化器。

封存：输入QC允许查看测试/挑战强迫曲线；响应读取在默认接口中关闭。所有后代必须继承父事件划分。该机制为应用层约束，不是文件加密。正式标签准入、训练释放和测试响应开启均未完成。

## 本地复现

在项目根目录使用具备numpy、rasterio、scipy、netCDF4的项目Python运行：

```sh
python src/analyses/validate_three_ward_minimum_scenarios.py --freeze
python src/analyses/review_three_ward_minimum_scenarios.py
```

生成入口为`src/preprocessing/generate_three_ward_minimum_scenarios.py`。当前版本已经冻结，入口拒绝覆盖；需要修改设计时另建版本，不能覆盖本版。验证会复用已有的6组文件导出并核查其内容。加载与纯文件导出入口为`src/preprocessing/load_three_ward_minimum_scenarios.py`，不包含求解器执行接口。

## 结果入口

- `review.html`：曲线选择、划分、体积与缺口。
- `validation.json`：27项独立检查及6组文件回读；官方引擎未运行。
- `rainfall_volume_audit.csv`：所有54个已有/新增条件的体积预算。
- `shape_pair_audit.csv`：全部1431对的归一雨型检查，干控制无形状距离。
- `remaining_prerequisites.csv`：完成项与正式建响应库前的缺口。
- 处理目录`data/processed/three_ward_minimum_scenarios_v1`：曲线、共享几何、schema、目录、谱系/划分/运行清单和冻结哈希。

建议下一步先冻结参考响应的数值准入与输出合同，再在另外授权后使用预定开放/训练/验证设置组做运行检查。现在尚无新的积水图、学习标签或代理模型效果。

用户研究决策记录：MILI-D-20261015-005。AnaSOP仅同步第4、5、7节，保持候选变量、物理限制与后续主动措施阶段。
'''
 (report/'README.md').write_text(readme)
 curves=[];mass={r['case_id']:r for r in csv.DictReader((report/'rainfall_volume_audit.csv').open())}
 for c in cat['cases']:
  d=api.load_forcing(c['case_id'],root)
  curves.append({'id':c['case_id'],'split':c['split'],'parent':c['parent_id'],'family':c['family_id'],
    'time':d['Event Time'].tolist(),'rain':d['Physical Rainfall Rate'].tolist(),'stage':d['River Stage Increment By Group'][:,0].tolist(),
    'parameters':c['parameters'],'mass':mass[c['case_id']]})
 export_check=next(c['evidence'] for c in v['checks'] if c['check']=='six_Galibier_file_preflights_no_engine_execution')
 template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区最小情景库 · 强迫输入</title>
<style>body{font:16px system-ui;background:#f3f6f8;color:#1f3342;max-width:1100px;margin:auto;padding:32px}h1{font-size:30px}h2{font-size:21px;margin-top:30px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}.card,section{background:white;border:1px solid #dce5eb;border-radius:12px;padding:20px;margin:12px 0}.value{font-size:28px;font-weight:650}small{color:#536774}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #e5ebef;padding:10px}select{font:inherit;padding:8px;max-width:100%}svg{width:100%;height:auto;display:block}pre{font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}a{color:#1d698c}.tag{color:#246d51;font-weight:600}</style>
<h1>三区最小情景库：强迫输入已冻结</h1><p>三区及现有跨区汇水范围 · 合成雨型与边界水位 · 无新水动力模拟、无训练、无文献搜索</p>
<div class="cards"><div class="card"><div class="value">48</div>新父事件雨型<small> · 24 / 8 / 8 + 8</small></div><div class="card"><div class="value">50</div>新增强迫输入<small> · 含2个开放控制</small></div><div class="card"><div class="value">58</div>未来运行登记<small> · 全部unrun</small></div><div class="card"><div class="value">27 / 27</div>检查通过<small> · 引擎未执行</small></div></div>
<section><h2>先看曲线</h2><p>此处是输入一致性审查。封存测试／挑战的输入可检查，响应尚不存在，读取接口关闭。水位图仅示共同增量，实际各点还需加上原低水位偏移。</p><select id="case"></select><div id="identity"></div><svg id="plot" viewBox="0 0 1000 420" role="img" aria-label="Rainfall and prescribed stage increment"></svg><div id="budget"></div><details><summary>该事件的种子、控制点与参数</summary><pre id="meta"></pre></details></section>
<section><h2>划分与谱系</h2><table><tr><th>组别</th><th>数量</th><th>约束</th></tr><tr><td>Training</td><td>24</td><td>4种雨型各6；未来归一化只从训练父事件拟合</td></tr><tr><td>Validation</td><td>8</td><td>4种雨型各2；用于开发选择</td></tr><tr><td>Sealed test</td><td>8</td><td>4种雨型各2；响应封存，模型与指标冻结后再开启</td></tr><tr><td>Challenge</td><td>8</td><td>复合、雨强、历时、错峰各2；不参与模型选择</td></tr><tr><td>Open controls</td><td>6（4旧+2新）</td><td>同一CONTROL-LEGACY谱系，不算独立暴雨</td></tr><tr><td>Adaptive companions</td><td>8（未来）</td><td>2新控制+前4训练+前2验证；继承原划分</td></tr></table><p>按归一雨型比较全部1431对。去掉起始时刻、雨强和历时缩放后，以RMSE≤0.02或RMSE≤0.04且相关系数≥0.995标记近重复；跨谱系、跨划分命中均为0。这是工程筛查，不证明统计独立或样本充分性。</p></section>
<section><h2>雨量体积与文件检查</h2><p>汇水面积38.9768 km² = 地面25.936197 km² + 屋面13.040603 km²。10 m、5 m和原生自适应分配保守一致。直接预算最大舍入差约4.7×10⁻¹⁰ m³。</p><p>6组代表性文件已按固定Galibier读取约定回读：坐标半格偏移、时间、有效零雨量、边界点序与原生网格采样均通过。最大体积相对误差 <strong>3.81×10⁻⁹</strong>。这是文件级检查，不是官方引擎运行验证。</p><p>屋面规则为全部即时进入地表，无延迟、无损失、不入地下管网；该假设不等于真实屋面排水。7组、10488个边界点仍为规定水位，不包含真实河床、流量或独立河段洪水演进。</p></section>
<section><h2>正式建立响应库前还缺什么</h2><ol><li>在开发证据上冻结参考标签质量政策，处理全域数值隔离区的准入与覆盖报告。</li><li>固定求解器、输出和预算合同；以后获授权再运行预定设置批次。峰值任务不自动要求全域补密时序；动态路径任务需要另行合格的时序／通量。</li><li>核定道路长度映射和关键区域定义；后续只用训练父事件拟合归一化。</li><li>地形时期、跨区上游覆盖、实测事件与河床信息仍限制物理准确性。主动措施保留至第5阶段。</li></ol><p>本轮未产生新的积水范围、训练标签或代理模型精度结论。</p><p><a href="README.md">复现说明</a> · <a href="validation.json">检查明细</a> · <a href="rainfall_volume_audit.csv">雨量预算</a> · <a href="shape_pair_audit.csv">去重审计</a> · <a href="remaining_prerequisites.csv">剩余条件</a></p></section>
<script>const DATA=__DATA__;const select=document.getElementById('case');for(const c of DATA){const op=document.createElement('option');op.value=c.id;op.textContent=c.id+' · '+c.split;select.appendChild(op)}
const fmt=(n,d=2)=>Number(n).toLocaleString('en-US',{maximumFractionDigits:d});function render(){const c=DATA.find(x=>x.id===select.value),svg=document.getElementById('plot');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';function el(tag,attrs,txt){const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(txt!==undefined)e.textContent=txt;svg.appendChild(e);return e}const X=t=>70+t/14400*850;for(const [i,title,key,color,max] of [[0,'Physical rainfall (mm/h)','rain','#2878a0',120],[1,'Prescribed stage increment (m)','stage','#b37532',1.5]]){const y0=35+i*200,H=125,Y=v=>y0+H-H*v/max;el('text',{x:70,y:y0-12,fill:color,'font-size':15},title);for(let h=0;h<=4;h++){el('line',{x1:X(h*3600),y1:y0,x2:X(h*3600),y2:y0+H,stroke:'#e4ebef'});el('text',{x:X(h*3600),y:y0+H+23,fill:'#526672','font-size':13},h+' h')}for(const val of [0,max/2,max]){el('line',{x1:70,y1:Y(val),x2:920,y2:Y(val),stroke:'#e4ebef'});el('text',{x:12,y:Y(val)+4,fill:'#526672','font-size':13},String(val))}el('polyline',{points:c.time.map((t,j)=>X(t)+','+Y(c[key][j])).join(' '),fill:'none',stroke:color,'stroke-width':2.5})}document.getElementById('identity').textContent='Parent: '+c.parent+' | Family: '+c.family+' | Split: '+c.split;document.getElementById('budget').textContent='Integrated physical depth: '+fmt(c.mass.physical_depth_mm,3)+' mm | Direct ground: '+fmt(c.mass.direct_ground_volume_m3)+' m³ | Roof: '+fmt(c.mass.roof_volume_m3)+' m³ | Total: '+fmt(c.mass.expected_volume_m3)+' m³';document.getElementById('meta').textContent=JSON.stringify(c.parameters,null,2)}select.addEventListener('change',render);render();</script></html>'''
 (report/'review.html').write_text(template.replace('__DATA__',json.dumps(curves,ensure_ascii=True).replace('</','<\\/')))
 manifest={str(p.relative_to(root)):gen.sha(p) for p in sorted(out.iterdir()) if p.is_file()}
 gen.dump(report/'product_manifest.json',{'products_sha256':manifest,'files':len(manifest),'bytes':sum(p.stat().st_size for p in out.iterdir() if p.is_file()),'human_decision':'MILI-D-20261015-005','new_hydrodynamic_runs':0,'training_runs':0})
 print(json.dumps({'review':str(report/'review.html'),'ana_sop_sections_changed':actual_changed,'checks_passed':len(v['checks']),'frozen_input_files':len(manifest)},ensure_ascii=False))
if __name__=='__main__':main()
