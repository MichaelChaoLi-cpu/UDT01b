#!/usr/bin/env python3
"""Validate reconstructed reference labels and render the whole-area audit report."""
from pathlib import Path
import csv, hashlib, html, importlib.util, json
import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parents[2]

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 root=ROOT;p=root/'data/processed/three_ward_common_depth_labels';out=root/'data/exp/data-preprocessing/three_ward_common_depth_labels'
 s=load(out/'summary.json');contract=load(p/'label_contract.json')
 spec=importlib.util.spec_from_file_location('depth_preprocess',root/'src/preprocessing/reconstruct_three_ward_common_depth.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 z=np.zeros((20,20),np.float32);cl=np.zeros((20,20),np.uint8)
 zz,pre,count,_,_=module.ground_tables(z,cl,np.ones((1,1),bool))
 def check_case(owners,peak,states,expected_lower,expected_upper):
  a,n=module.bounds(zz,pre,count,np.array([owners],np.int32),np.array([.2],np.float32),np.array([0.],np.float32),np.array(peak,np.float32),np.zeros(len(peak),np.float32),np.array(states,np.float32))
  assert abs(a[2,0]-expected_lower)<1e-7 and abs(a[3,0]-expected_upper)<1e-7,(a,expected_lower,expected_upper)
  assert a[4,0]<=a[5,0]+1e-7 and a[5,0]<=a[3,0]+1e-7
  return True
 states=np.zeros((49,4),np.float32);states[0,:2]=.2;states[1,2:]=.2
 synthetic={
  'asynchronous_peaks_not_averaged_as_true_peak':check_case([0,1,2,3],[.2]*4,states,.1,.2),
  'single_native_owner_recovers_peak_without_dense_times':check_case([0,0,0,0],[.2],np.zeros((49,1)),.2,.2),
  'censored_states_remain_intervals':check_case([0,1,2,3],[np.nan]*4,np.full((49,4),np.nan),0.,.01),
  'known_peak_tightens_censored_frame_upper':check_case([0,0,0,0],[.004],np.full((49,1),np.nan),.004,.004),
  'native_peak_proves_missed_common_cell_peak':check_case([0,1,2,3],[.8,0,0,0],np.zeros((49,4)),.2,.2)}
 # Compare the optimized storage calculation with a direct sum over irregular fine ground.
 z=np.linspace(-.1,.7,400,dtype='float32').reshape(20,20);cl[3:6,4:9]=1;z[0,0]=np.nan
 zz,pre,count,_,_=module.ground_tables(z,cl,np.ones((1,1),bool));eta=.37
 a,_=module.bounds(zz,pre,count,np.zeros((1,4),np.int32),np.array([eta]),np.array([-.1]),np.array([eta]),np.array([-.1]),np.full((49,1),eta))
 eligible=(cl!=1)&np.isfinite(z);expected=float(np.maximum(eta-z[eligible].astype('float64'),0).mean())
 assert abs(a[0,0]-expected)<1e-12 and abs(a[2,0]-expected)<1e-12
 synthetic['exact_direct_fine_ground_sum_with_buildings_and_unknowns']=True
 with np.load(p/'common_support_10m.npz') as f:area=f['Ground Area'];valid=f['Eligible Ground Mask'];owners=f['Native State Count']
 assert area.sum()==25936197.0 and np.all(area[valid]>0)
 product_checks=[]
 for case in s['cases']:
  for grid in ['corrected_10m','adaptive_10_5m']:
   path=p/(case+'_'+grid+'.npz')
   with np.load(path) as f:
    lo=f['Peak Mean Depth Lower'];up=f['Peak Mean Depth Upper'];point=f['Peak Mean Depth'];width=f['Peak Interval Width']
    assert lo.shape==up.shape==area.shape and np.all(np.isnan(lo[~valid]))
    assert np.all(lo[valid]>=0) and np.all(up[valid]+1e-6>=lo[valid])
    assert np.allclose(width[valid],(up-lo)[valid],atol=1e-6)
    assert np.all(np.isnan(point[valid&(width>1e-7)]))
    assert np.allclose(point[valid&(width<=1e-7)],lo[valid&(width<=1e-7)],atol=1e-6)
    if grid=='adaptive_10_5m':
     sl=f['Sample Peak Lower 300s'];su=f['Sample Peak Upper 300s'];c=f['Confirmed Temporal Miss Lower']
     assert np.all(sl[valid]<=su[valid]+1e-6) and np.all(su[valid]<=up[valid]+1e-6)
     assert np.allclose(c[valid],np.maximum(lo-su,0)[valid],atol=1e-6)
    product_checks.append(dict(case=case,grid=grid,eligible_cells=int(valid.sum()),point_label_cells=int(np.isfinite(point[valid]).sum()),interval_cells=int((width[valid]>1e-7).sum())))
   with rasterio.open(path.with_suffix('.tif')) as d:
    assert d.crs.to_epsg()==6677 and d.shape==area.shape and d.count==3
    assert d.descriptions==('Peak Mean Depth Lower (m)','Peak Mean Depth Upper (m)','Peak Interval Width (m)')
    assert np.allclose(d.read(1),lo,equal_nan=True)
 assert sha(root/'docs/AnaSOP.md')==contract['ana_sop_sha256_unchanged']
 manifest=[dict(file=str(f.relative_to(root)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(p.iterdir()) if f.is_file()]
 dump(out/'product_manifest.json',manifest)
 v=load(out/'validation.json');v.update(analytic_cases=synthetic,product_checks=product_checks,geotiff_crs_and_bands_valid=True,review_script_sha256=sha(root/'src/analyses/review_three_ward_common_depth_labels.py'))
 dump(out/'validation.json',v)
 # Derive reproducible screening metrics, not a new formal acceptance rule.
 decisions=[]
 labels={'rain60_low':'60 mm/h / Low stage','rain100_low':'100 mm/h / Low stage','rain60_raised':'60 mm/h / +1.5 m pulse','dry_raised':'No rain / +1.5 m pulse'}
 for case in s['cases']:
  q=next(r for r in s['domain_rows'] if r['case']==case and r['subset']=='multiple native states')
  t=next(r for r in s['domain_thresholds'] if r['case']==case)
  decisions.append(dict(case=case,multi_native_open_area_km2=q['open_area_km2'],weighted_p95_interval_width_mm=q['peak_interval_p95_m']*1000,
   area_width_gt_1cm_m2=q['area_interval_gt_0.01m_km2']*1e6,
   fraction_multi_native_area_width_le_1cm=1-q['area_interval_gt_0.01m_km2']/q['open_area_km2'],
   fraction_domain_area_width_le_1cm=1-q['area_interval_gt_0.01m_km2']/s['open_ground_km2'],
   area_width_gt_5cm_m2=q['area_interval_gt_0.05m_km2']*1e6,
   area_confirmed_miss_gt_1cm_multi_native_m2=q['area_confirmed_temporal_miss_gt_0.01m_km2']*1e6,
   threshold_10cm_ambiguous_area_m2=t['adaptive_ambiguous_area_km2']*1e6,
   grid_definite_class_disagreement_area_m2=t['regular_adaptive_definite_class_disagreement_km2']*1e6,
   suggested_action='Keep native10m peak reconstruction; retain adaptive intervals and unresolved masks. Whole-domain denser output is not yet justified for peak-only emulation; dynamic targets need separate cadence qualification.'))
 csvwrite(out/'temporal_sampling_decision.csv',decisions)
 methods='''# Candidate Machine-Learning Methods for the Three-Ward Surrogate

Status: method options for discussion, using current local design and reconstructed labels. No literature search, training, architecture selection, sample generation or new hydraulic execution. These options do not change AnaSOP or finalize its candidate variables.

## A compact comparison with distinct purposes

| Candidate | Input and output | Purpose | Prerequisite and limitation |
|---|---|---|---|
| Training-only nearest-forcing field and low-complexity response baseline | Rainfall/stage descriptors to a stored training response or regional depths | Establish whether a learned field model improves on simple interpolation | No test-response lookup; four inspected controls cannot establish generalization. |
| Conditional U-Net | Static ground/building/road channels and encoded full prescribed rain/stage sequence to Peak Mean Depth and optional wetness | Candidate full-field surrogate on a fixed common mesh | Boundary time series require boundary-group spatial association; optional wetness head thresholds the same cell-mean target. Evaluate separate fine-ground footprint only with a separate defined target. |
| Connectivity-aware graph or gated spatial model | Native verified open connections, blocked faces and geometric features, with forcing descriptors to a common-grid field | Test whether explicit road/building connectivity reduces false cross-obstacle transfer or misses | Adjacency alone is insufficient; multiple disconnected open components may need separate representation. Conditioning a graph does not guarantee correct hydraulic routing or conservation. Treat as an extension after a baseline comparison. |

## Label uncertainty is a useful methodological addition

For exact reference depths, use a predeclared event-balanced regression error with wet/boundary strata. For a qualified reference interval [L,U], a possible interval-consistency loss is
`loss(yhat;L,U)=max(L-yhat,0)^2+max(yhat-U,0)^2`.

A zero interval loss means compatibility with that interval, not accurate identification of the unknown peak. Wide intervals can hide poor point predictions. Keep their width and sample weights visible; train/evaluate on exact or sufficiently narrow qualified labels, or report interval compatibility separately. Do not replace intervals with zero, their midpoint, or a fabricated reference truth. For a proposed point prediction, best-case absolute error is distance to [L,U], while worst-case absolute error is max(abs(yhat-L),abs(yhat-U)); these are conditional label-error bounds, not calibrated confidence probabilities.

The current regular10m configuration has reconstructible peak labels and is a practical candidate primary reference. Adaptive bounds quantify numerical configuration uncertainty and remain development diagnostics; choosing the formal reference version is a separate design decision. Do not mix two configurations as interchangeable truth, and do not count their grid variants as independent storms. Small aggregation intervals do not eliminate the observed grid-configuration disagreement or establish real-event accuracy.

## Prediction uncertainty and efficient new samples are later options

A small ensemble or quantile model could assess model uncertainty after enough parent events exist and calibration is evaluated on held-out validation families. Distinguish model disagreement, forcing uncertainty and reference-label intervals. If later authorized, active learning can prioritize additional hydraulic cases from a training-only candidate pool, using model disagreement plus coverage of prescribed rain/stage combinations. It must not select on sealed test responses or silently replace required compound extremes and negative controls. No active sample selection or model fitting occurs now.

## Connection to active measures

Maintain the first four milestones. Afterwards, extend the input with verified action geometry/capacity, connected receiver and schedule, and obtain matched no-action/action reference pairs. Screen candidates with a trained action-aware surrogate only within its response library; physically rerun shortlisted and some rejected candidates and assess risk transfer across the whole domain. Peak-only labels cannot support feedback control, time-varying storage operation, flow-path trajectories or a claim of dynamic conservation. Those require synchronized states, internal face fluxes, source/sink accounting and an independently qualified cadence.

Suggested sequence: qualify the present labels and a single reference version; establish simple baseline versus conditional U-Net; compare interval-aware treatment and, if warranted, connectivity encoding under the same event-family split. Keep the number of compared methods proportionate to the future event library. No method is claimed novel or superior before evaluation.
'''
 (out/'ml-method-options.md').write_text(methods,encoding='utf-8')
 maxp95=max(d['weighted_p95_interval_width_mm'] for d in decisions)
 maxamb=max(d['threshold_10cm_ambiguous_area_m2'] for d in decisions)
 report=f'''# 三区统一10m水深标签与加密峰值区间

已重建四种强迫情景在两套网格上的参考标签与区间，覆盖既有25.936197 km²开敞地面。共同网格为EPSG:6677、10m、1012×770，358384个有开敞面积的单元。标签积分使用同一0.50m地面与建筑排除规则。新文件写入专用 processed 目录，既有地形、模型结果和AnaSOP保持不变。

## 主要发现

1. 规则10m：三个有雨情景的峰值平均水深可由原生峰值水位直接重建；无雨抬升组保留浅水区间，最大不超过1cm，没有将删失值补为零。点标签仅在区间退化时保留，其余为NaN，另存上下界和有效支撑。
2. 加密结果：共同10m内需要合并多个原生状态的开敞地面为{s['multiple_native_state_area_km2']:.6f} km²。四种情景在这部分地面上，面积加权区间宽度第95百分位均不超过{maxp95:.3f}mm；每个情景超过1cm的面积为{min(d['area_width_gt_1cm_m2'] for d in decisions):.0f}–{max(d['area_width_gt_1cm_m2'] for d in decisions):.0f}m²。少量尾部区间仍较宽，已保留原区间和空间位置，未做逐点核验。
3. 10cm阈值下，峰值区间导致的积水分类不确定面积为{min(d['threshold_10cm_ambiguous_area_m2'] for d in decisions):.0f}–{maxamb:.0f}m²；规则与加密网格之间已经确定的分类分歧为{min(d['grid_definite_class_disagreement_area_m2'] for d in decisions):.0f}–{max(d['grid_definite_class_disagreement_area_m2'] for d in decisions):.0f}m²。后者更大，不能靠补密输出时间就视为解决。

## 是否需要补密时序

针对当前整体峰值代理任务，暂不建议把现有全域情景全部补密重跑。规则10m峰值利用原生峰值输出恢复，不依赖5分钟帧恰好捕捉峰值；加密结果大部分区间已很窄，剩余宽区间可标为不确定参考，避免强行变成精确标签。

这不是认定300秒足够描述所有动态过程。现有快照确实漏掉部分峰值，尤其单一原生状态的峰值可以由zsmax找回；多原生状态单元内已确认漏采超过1cm的面积，各情景约{min(d['area_confirmed_miss_gt_1cm_multi_native_m2'] for d in decisions):.0f}–{max(d['area_confirmed_miss_gt_1cm_multi_native_m2'] for d in decisions):.0f}m²。若目标改为峰值时刻、积水历时、排水路径演化或主动调度，仍需单独核定更密的同步状态、源汇和内部面流量。没有据此指定一个未经验证的60秒或其他新间隔，也没有补跑。

1cm/2cm/5cm是本轮展示的不确定性尺度，不是用户已批准的训练准入阈值。是否剔除宽区间、采用区间监督或补导出，需要在正式训练变量和参考配置确定时固定。四种强迫仍只是开发控制，尚无足够的独立训练/验证/封存测试样本。

## 上下界的含义

- 下界同时利用49个同步输出时刻的最大开敞地面平均水深，以及各原生单元峰值对共同单元的最低必然贡献；属于同一原生单元的多个子区域先合并贡献，因此未加密单元可恢复原生峰值。
- 上界为各原生单元独立峰值贡献之和，保留浅水删失上界；瞬时浅水帧还由已知原生峰值收紧上界。
- 下界与上界包住当前离散参考模型的目标，并不包住真实世界误差。区间宽度包含异步峰值和删失的不确定性，不能全部称为实际时间采样误差。
- 另存“已确认漏采下限”：峰值下界减现存快照峰值上界的正部分。只有这一差为正，才说明快照确实漏掉至少这么大的共同单元峰值。
- 对300/600/1200秒嵌套快照的对照仅是已有输出的抽样检查，不能证明向更细时间间隔收敛。

## 解释边界

本轮阈值面积按共同10m单元的峰值平均水深判断，再以开敞面积加权。它不同于之前把每个0.50m地面位置的局部峰值超阈面积相加，不能直接把两次面积数值当同一指标对照。行政区归属按10m单元中心近似，跨区汇水外缘单列；全域开敞面积与旧支撑逐块精确一致。

负记账水量标志随标签保留，既有建筑假设障碍超越、未完成的高程历元统一、跨区完整性、河床与真实事件验证问题仍有效。小的聚合区间只说明标签聚合歧义小，不证明水动力参考模型准确。

## 可增加的机器学习方法

可以加入三个层次的候选比较：简单强迫匹配/低复杂度基线、条件U-Net场预测、利用既有道路与建筑开闭面的连通图模型。先以基线与U-Net回答加速与保真问题，再评估连通编码是否值得增加复杂度。

与这次结果直接相关的方法是区间监督和不确定性分解：对精确标签做常规回归，对合格区间做区间一致性约束，分别报告参考标签区间、模型预测误差与强迫情景不确定性。宽区间不能因训练损失小就被当作预测准确。后续主动学习可减少新增水动力样本，但需在已有足够事件、完成分组并获准新模拟后开展。方法选项详见 ml-method-options.md；本轮均未训练或定为最终方法。

## 文件与复现

- 处理文件：`data/processed/three_ward_common_depth_labels/`，8份NPZ标签/区间、8份三波段GeoTIFF、10m共同支撑及label_contract.json。地理栅格保留空间支撑，采用NPZ/GeoTIFF；通用技能中的表格parquet输出不适用于这次栅格构建，未安装新的表格依赖。
- 变量表、构建决定、全域/各区汇总、阈值面积上下界、产品SHA256与校验均在本目录。原始字段与路径只存于这些溯源文件。
- AnaSOP第4节保留原候选变量状态，本轮未覆盖；本轮结果不自动把所有候选变量定为最终训练变量。机器学习候选讨论另存，没有更改第5–7节。
- 人工研究决定：MILI-D-20261015-002。其范围是标签构建与不训练/不补跑；未替用户批准架构或精度门槛。
- 复现：用具备numpy、numba、rasterio、netCDF4、shapely、pyproj的本地Python运行 `src/preprocessing/reconstruct_three_ward_common_depth.py`，随后运行 `src/analyses/review_three_ward_common_depth_labels.py`。本次使用已有 `/private/tmp/udt01b-sfincs-runner/bin/python`，没有新增依赖。
- 校验：6个独立算例检验异步峰值、单原生状态恢复、浅水区间、已知峰值收紧、确认漏采和不规则地面直接求和；8份产品核对上下界、点标签缺失规则、CRS、栅格波段与支撑一致性。未进行训练或新情景执行。
'''
 (out/'README.md').write_text(report,encoding='utf-8')
 def table(rows,cols,names=None):
  if names is None:names=cols
  text='<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in names)+'</tr></thead><tbody>'
  for r in rows:
   text+='<tr>'
   for c in cols:
    v=r[c];v=(f'{v:.6f}' if 0<abs(v)<.001 else f'{v:.3f}') if isinstance(v,float) else str(v)
    text+='<td>'+html.escape(v)+'</td>'
   text+='</tr>'
  return text+'</tbody></table></div>'
 h='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区统一10m标签与峰值上下界</title><style>body{max-width:1160px;margin:32px auto;padding:0 24px;font:16px/1.85 system-ui;color:#203442}h1{line-height:1.4}p{white-space:pre-line}.note{padding:18px;background:#eaf4f6;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #d3dfe5;text-align:left}.scroll{overflow:auto}a{color:#09689a}</style><h1>三区统一10m水深标签与加密峰值上下界</h1><p class="note">已完成四种情景 × 两套网格的标签重建。当前峰值任务暂不建议全域补密重跑；保留小范围宽区间。动态历时和调度另需核定时序。未训练、未新增水动力情景、未搜索文献。</p>'
 h+='<h2>全域加密聚合不确定性</h2>'+table(decisions,['case','weighted_p95_interval_width_mm','area_width_gt_1cm_m2','threshold_10cm_ambiguous_area_m2','grid_definite_class_disagreement_area_m2'],['Scenario','加密合并区P95宽度/mm','区间宽>1cm面积/m²','10cm分类不确定面积/m²','两网格确定分类分歧/m²'])
 h+='<p>面积均以开敞地面加权；当前分类基于10m单元平均峰值，不能与过去细地面峰值超阈面积混用。区间宽度不是地形误差或95%统计置信区间。</p>'
 h+='<h2>各区可能积水支撑内的区间</h2>'+table(s['ward_rows'],['case','scope','peak_interval_p95_m','area_interval_gt_0.01m_km2'],['Scenario','Ward','P95宽度/m','宽>1cm面积/km²'])
 for block in report.split('\n\n')[1:]:
  if block.startswith('## '):h+='<h2>'+html.escape(block[3:])+'</h2>'
  else:h+='<p>'+html.escape(block)+'</p>'
 h+='<h2>机器学习候选方法说明</h2><details><summary>展开英文方法选项</summary><pre style="white-space:pre-wrap;font:14px/1.7 system-ui">'+html.escape(methods)+'</pre></details>'
 files=['README.md','adaptive_peak_bounds_summary.csv','regular_label_summary.csv','threshold_area_bounds.csv','temporal_sampling_decision.csv','ml-method-options.md','decisions.json','validation.json','product_manifest.json']
 h+='<p>'+' · '.join('<a href="'+f+'">'+f+'</a>' for f in files)+'</p></html>'
 (out/'review.html').write_text(h,encoding='utf-8')
 print(json.dumps(dict(valid=v['valid'],analytic_checks=len(synthetic),products_checked=len(product_checks),temporal_decisions=decisions,total_product_bytes=sum(x['bytes'] for x in manifest)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
