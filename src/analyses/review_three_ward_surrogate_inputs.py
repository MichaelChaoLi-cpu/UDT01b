#!/usr/bin/env python3
"""Write a whole-area technical handover for the existing input package."""
from pathlib import Path
import csv,html,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 root=ROOT;dest=root/'data/processed/three_ward_surrogate_inputs';out=root/'data/exp/data-preprocessing/three_ward_surrogate_inputs'
 schema=json.loads((dest/'input_schema.json').read_text());v=json.loads((out/'validation.json').read_text());assert v['valid'] and all(v['independent_checks'].values())
 with np.load(dest/'static_tensor_10m.npz') as d:area=d['Ground Area'];valid=d['Eligible Ground Mask'];ward=d['Ward Code']
 scopes={'Whole domain':valid,'Three wards':valid&(ward>0),'Chiyoda':valid&(ward==1),'Chuo':valid&(ward==2),'Minato':valid&(ward==3),'Cross-ward contributing area':valid&(ward==0)}
 coverage=[]
 for case in schema['cases']:
  with np.load(dest/(case+'_label_admission.npz')) as d:l=d['Label Lower'].astype('float64');u=d['Label Upper'].astype('float64');q=d['Quarantine Mask'];exact=d['Exact Label Mask'];interval=d['Interval Label Mask']
  for name,scope in scopes.items():
   wet=scope&(l>.1);wetq=wet&q
   coverage.append(dict(case=case,scope=name,open_area_km2=float(area[scope].sum()/1e6),exact_admitted_km2=float(area[scope&exact].sum()/1e6),interval_admitted_km2=float(area[scope&interval].sum()/1e6),quarantine_km2=float(area[scope&q].sum()/1e6),quarantine_open_fraction=float(area[scope&q].sum()/area[scope].sum()),reference_certain_wet_area_km2=float(area[wet].sum()/1e6),quarantine_certain_wet_km2=float(area[wetq].sum()/1e6),quarantine_certain_wet_fraction=float(area[wetq].sum()/max(area[wet].sum(),1))))
 csvwrite(out/'mask_coverage_at_10cm.csv',coverage)
 gaps=[
  dict(priority='P0',item='Future forcing generator and physical mass checks',current='Four existing controls prepared; no new forcing generated',required='Implement48 reserved parent realizations plus2 controls; preserve physical/effective rain, canonical time knots and coherent stages; reject bad support or failed roof-volume budgets.',blocks='formal scenario execution'),
  dict(priority='P0',item='Actual parent lineage and sealed custody',current='Design reserves24/8/8 plus8, not realized independent events',required='Bind normalized shape seeds, amplitudes, all descendants and near-duplicate groups before assignment; keep test/challenge response access separate.',blocks='formal sample-library claims'),
  dict(priority='P0',item='Reference and admission release',current='Version1 masks constructed; numerical flags quarantined',required='Review the whole-area and wet-area effects of the conservative mask and freeze it before new test responses. Retain quarantined outputs; decide whether further batch numerical review is required.',blocks='full-coverage supervised-reference claims'),
  dict(priority='P1',item='Generalized time/raster/graph consumption',current='NumPy loader and14 independent input checks passed for4 controls',required='Freeze variable-length knot padding or lossless resampling protocol for future events; preserve masks and typed receiver links. Fit normalization on training parents only after those events exist.',blocks='network fitting; not a reason to train now'),
  dict(priority='P1',item='Road length and critical-region definitions',current='Road ground area and paired road-face width exist',required='Verify a road centerline/length-to-cell mapping and freeze critical-region/threshold definitions on development support. Area/face width is not road length.',blocks='road-length and critical-region performance claims'),
  dict(priority='P1',item='Full-domain input/source interpretation',current='Reference-compatible graph and fallback fractions present',required='Keep cross-ward coverage, source epoch, ambiguous intra-cell geometry and height-reference limits explicit. Connectivity graph alone does not repair missing geometry.',blocks='real-world accuracy or complete-upstream-coverage claims'),
  dict(priority='P2',item='Independent event validation and river geometry',current='Synthetic boundary stages; no event-calibrated full riverbed dynamics',required='Pair rainfall/stages/observed inundation and obtain required channel data before real-event accuracy claims. Surface graph is not a river-channel network.',blocks='physical event validation, not conditional simulation emulation'),
  dict(priority='P2',item='Active measures and dynamic operation',current='No-action only; no action-response library',required='After milestones1-4, verify assets/capacity/receiver/schedule and paired reference responses; dynamic control additionally requires time-resolved states, face fluxes and source/sink accounting.',blocks='measure screening or closed-loop control')]
 csvwrite(out/'remaining_prerequisites.csv',gaps)
 # Extend the focused geospatial variable inventory with exact source-family mappings.
 rows=list(csv.DictReader((out/'variable_list.csv').open()))
 for i,row in enumerate(rows):
  if i<4:source='data/processed/three_ward_sfincs_scenarios/terrain_sampling_0_50m.tif';original='band1 on original eligible fine ground'
  elif i==4 or i==13:source='data/processed/three_ward_common_depth_labels/common_support_10m.npz';original='Ground Area / Eligible Ground Mask'
  elif i in [5,6,7]:source='data/processed/three_ward_urban_constraints/constraints_0_50m.tif';original='band1 classes1/2/3'
  elif i==8:source='data/processed/three_ward_urban_constraints/manning_0_50m.tif';original='band1'
  elif i==9:source='data/processed/three_ward_sfincs_scenarios/terrain_source_id_0_50m.tif';original='band1 code2'
  elif i==10:source='data/processed/three_ward_fixed_runoff/contributing_area_10m.tif';original='band1 /100m2'
  elif i==11:source='data/processed/three_ward_urban_constraints/subcell_connectivity_flags_10m.tif';original='band1 >0'
  elif i==12:source='data/processed/three_ward_adaptive_connectivity/automatic_refinement_10m.tif';original='band1'
  elif i==14:source='data/processed/three_ward_sfincs_framework/adaptive_connectivity_comparison/rain60_low_corrected_10m/sfincs_map.nc';original='msk==2'
  elif i==15:source='data/processed/three_ward_sfincs_scenarios/active_mask_20m.tif';original='band1>0, nested2x'
  elif i<28:source='data/processed/three_ward_adaptive_connectivity/regular_face_geometry.npz; data/processed/three_ward_sfincs_framework/adaptive_connectivity_comparison/rain60_low_corrected_10m/sfincs_subgrid.nc';original=['native_open_width_m','native_road_width_m','uv_zmin'][(i-16)%3]
  elif i==28:source='existing scenario_contract.json and roof_surface_precip.nc time';original='rain_mm_h and original one-second ramp times'
  elif i==29:source='existing roof_surface_precip.nc';original='Precipitation read raw, solver corner convention'
  elif i==30:source='existing sfincs.bnd / sfincs.bzs / boundary_points.csv';original='native prescribed point stages and group identities'
  else:source='existing sfincs_map.nc first frame';original='h at time0; verified dry on eligible interior'
  row.update(source_dataset=source,original_name=original,is_final_variable='no',construction='see reproducible preprocessing script and input_schema.json')
 csvwrite(out/'variable_list.csv',rows)
 whole=[r for r in coverage if r['scope']=='Whole domain'];g=schema['graph']
 report=f'''# 三区统一10m输入、连通图与标签准入

现有四组开发控制已整理为可用本地NumPy读取的输入与监督包。它们对应4种强迫，不能作为独立训练／测试库。保持三区及既有跨区候选汇水范围，未训练、未补跑、未生成新强迫、未下载或搜索文献。原始地形和已有标签保持不变。

## 已完成的接口

- 静态基础张量：28×1012×770，另带同形状的逐通道有效性掩膜。包括开敞地面高程统计、地面／建筑／道路面积、粗糙度、PLATEAU补缺比例、几何歧义与加密标志、四向开口／道路面宽和岸面／地面接口高程。有效性为假的数值0仅用于数组存储，不能当作观测地形或无水。没有拟合归一化参数。
- 共同开敞地面：25.936197km²、358384个10m单元。建筑全覆盖单元可作为CNN几何上下文，但没有地面预测标签；规定水位边界也仅作上下文。
- 输入时间轴：0、3600、3601、5400、7200、7201、10800、14400秒，保留一秒降雨坡段与全部水位节点。这是强迫时间，不是300秒响应输出。现有物理降雨为空间均匀序列，按10m网格精确广播；有效地表降雨另存8×1012×770张量，含屋面重分配。
- 水位：每个时刻保留10488个原边界点，另存7个组的增量与逐点低水位偏移。同名古川的不同组没有合并，未命名水体保留原组标识。原NetCDF把0兼作填充值，现按求解器约定保留真实无雨0值。四组内部初态均已核验为干地面。

## 连通图

图含{g['nodes']}个地面节点、{g['undirected_surface_faces']}条无向开放地表接口，对应{g['directed_surface_edges']}条有向边；道路子图保留{g['directed_road_edges']}条有向边。所有40310个已核验的封闭面都排除在消息边之外，并保留封闭面记录。节点特征与栅格特征逐项一致。

另存{g['receiver_links']}条地表到规定水位边界的连接，采用独立的边界点命名空间；它们不会被当作普通地表邻接，也没有添加水域两岸之间的道路捷径。共有{g['boundary_points_with_receiver_links']}个边界点具有直接开放地表连接，其余点仍按原边界配置保留，不强行接通。

地表图保留{g['surface_components']}个连通分量，其中{g['isolated_ground_nodes']}个节点无地表邻边；没有为让图“连成一片”补边。这些是当前水动力参考网格的拓扑，不是实测下水管网或河流河网。10m单元内部仍可能合并分离空地，约1.349396km²地面带有既有几何歧义标志，保留为信息标志，没有一律剔除。图模型不能自动消除这些几何局限。

## 标签技术准入规则 v1

1. 精确标签：有效开敞地面、上下界有限且一致、区间宽≤1e-7m，并且没有隔离标志。
2. 区间标签：相同质量条件，区间宽在1e-7m以上、1cm以内；另留1e-6m浮点容差。原上下界不变，不补中点、不补零。精确监督使用第一类；区间监督使用两类的并集。
3. 隔离：既有300秒抽样发现负记账水量，或原10m假设建筑障碍出现超越，或标签上下界无效。这是保守数值隔离，不能把负记账水量直接解释为负的真实水深。隔离单元保留原标签、原因和位置，暂不作合格监督。
4. 信息标志：单元内连通歧义、跨区外缘、PLATEAU补缺、结构不确定性继续保留；不因这些标志自动删除整个区域。
5. 掩膜仅在监督侧。由模拟响应得到的负水量、障碍超越和标签准入标志没有混入预测输入，避免目标信息泄漏。评价必须同时报告已准入范围、隔离面积及其积水区占比，不能通过遮掉困难区域提高“全域精度”。

这是现有控制的技术政策。未来正式试验前需冻结研究准入政策和参考版本；所有对照使用同一政策，并在看到封存测试响应前固定。当前文件不授权训练或宣称物理模型已经验证。

## 覆盖与核验

四组均满足技术格式、输入哈希与已有水量平衡检查。三个有雨组的隔离地面面积约0.0695–0.0721km²，即0.27%–0.28%的开敞地面。无雨抬升组约0.001635km²隔离，其余大部分浅水区域作为区间标签保留。阈值比较统一提升为float64后使用严格大于0.10m，避免float32标量提升规则造成不同报告的边界不一致；浮点舍入不是物理精度。还单独计算了隔离地面在确定积水区内的比例，见 mask_coverage_at_10cm.csv，不用干地面占比掩盖积水区的缺口。

通过14项独立检查，包括面方向算例、原始细地面直接聚合、节点与栅格特征一致性、开放面／道路子图、封闭面排除、边界隔离、水位散布往返、强迫体积、标签原值保持、掩膜互斥和无响应泄漏。当前降雨体积相对误差最大约{max(x['rain_volume_relative_error'] for x in v['checks'].values()):.2e}；河段增量加逐点低水位的重构误差在存储精度容差内。

## 正式建情景库前还缺什么

优先完成P0：新父事件时间雨型／水位生成器及雨量积分、屋面分配和输入检查；实际谱系、近重复审查及封存响应管理；数值隔离对全域和积水区影响的评审，并冻结参考与准入政策。现有48个名额只是设计分配，不是已实现样本。

P1：未来不同长度强迫的时间填充／编码协议、仅在训练父事件上拟合归一化、道路真实长度映射和重点区规则。已有NumPy加载器能验证四个控制，尚未选择或运行神经网络架构。跨区覆盖、测量年代、高程历元和内部通道歧义限制要持续报告。

P2：若要声称真实事件精度，需要独立暴雨—水位—积水响应配对及所需河道几何。主动措施继续放在第5阶段，另需容量、接收连接、时序和配对物理响应；动态调度需状态／面流量／源汇数据。完整缺口见 remaining_prerequisites.csv。

## 文件与使用

数据在 data/processed/three_ward_surrogate_inputs/：static_tensor_10m.npz、surface_connectivity_graph.npz、四份输入、四份准入掩膜，以及input_schema.json和quality_policy.json。使用NPZ保留多维栅格和类型化图结构，通用表格技能的parquet／Excel转换不适用于此任务；没有安装新依赖。

构建脚本：src/preprocessing/prepare_three_ward_surrogate_inputs.py。独立校验：src/analyses/validate_three_ward_surrogate_inputs.py。加载接口：src/preprocessing/load_three_ward_surrogate_inputs.py，分别返回规定输入和监督标签；forcing_rasters可把空间均匀降雨和边界水位转为带有效性标志的栅格视图。

复现使用已有 /private/tmp/udt01b-sfincs-runner/bin/python 顺序执行构建、校验与 src/analyses/review_three_ward_surrogate_inputs.py。本轮研究决定为 MILI-D-20261015-004。AnaSOP仅第4节更新为本轮已构建接口与技术准入规则，其他节保持不变；17组研究候选变量均未自动定为最终变量，训练授权仍分开管理。
'''
 (out/'README.md').write_text(report)
 def tbl(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
 h='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区10m输入张量、连通图与标签准入</title><style>body{max-width:1100px;margin:32px auto;padding:0 24px;font:16px/1.85 system-ui;color:#203442}p{white-space:pre-line}.note{background:#eaf3f6;padding:18px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d3dfe6;text-align:left;vertical-align:top}.scroll{overflow:auto}a{color:#096598}</style><h1>三区10m输入张量、连通图与标签准入</h1><p class="note">28个静态基础通道；358384个地面节点；641284条无向开放接口；6185条单独编码的水域边界连接。四个开发控制已打包，训练和新增情景均为零。</p>'
 h+='<h2>全域监督覆盖</h2>'+tbl(['情景','精确准入/km²','区间准入/km²','隔离/km²','确定积水区被隔离比例'],[[r['case'],f"{r['exact_admitted_km2']:.6f}",f"{r['interval_admitted_km2']:.6f}",f"{r['quarantine_km2']:.6f}",f"{r['quarantine_certain_wet_fraction']:.2%}"] for r in whole])
 for block in report.split('\n\n')[1:]:h+=('<h2>'+html.escape(block[3:])+'</h2>') if block.startswith('## ') else '<p>'+html.escape(block)+'</p>'
 h+='<h2>建库前缺口</h2>'+tbl(['优先级','项目','当前','需要补齐'],[[x['priority'],x['item'],x['current'],x['required']] for x in gaps])
 files=['README.md','variable_list.csv','label_admission_by_scope.csv','mask_coverage_at_10cm.csv','remaining_prerequisites.csv','validation.json','product_manifest.json','decisions.json']
 h+='<p>'+' · '.join('<a href="'+f+'">'+f+'</a>' for f in files)+'</p></html>'
 (out/'review.html').write_text(h)
 dump(out/'readiness.json',dict(existing_controls_technically_packaged=True,independent_checks_passed=14,formal_scenario_library=False,training_release=False,new_scenarios=0,source_physics_validated=False,graph=schema['graph'],static_shape=schema['static']['shape'],case_coverage=whole,prerequisites=gaps))
 print(json.dumps(dict(review=str(out/'review.html'),whole_domain_coverage=whole,remaining_gaps=len(gaps)),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
