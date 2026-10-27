#!/usr/bin/env python3
"""Package qualified paired observational replays;do not infer equipment sizing."""
from pathlib import Path
import json,hashlib,html
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_ground_flux_replay_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 ana=sha(ROOT/'docs/AnaSOP.md');manifest=json.loads((P/'run_manifest.json').read_text());allrows=[];valid=[]
 for run in manifest['runs']:
  a=R/'execution'/run['case_id'];v=json.loads((a/'replay_validation.json').read_text());assert v['passed'] and v['all_actual_clocks_and_step_budgets_bitwise_identical'] and v['all_original_B_q_bitwise_identical'];assert json.loads((a/'audit.json').read_text())['all_checks_passed'];valid.append(v)
  t=pd.read_csv(a/'zone_ground_budget_trajectories.csv',float_precision='round_trip');assert t.groupby('Zone').size().eq(109).all() and t['Required Pump Capacity'].isna().all();allrows.append(t)
 tr=pd.concat(allrows,ignore_index=True);assert not tr.duplicated(['Case','Zone','Frame']).any();tr.to_parquet(P/'zone_ground_budget_trajectories_preprocessed.parquet',index=False);pd.testing.assert_frame_equal(tr,pd.read_parquet(P/'zone_ground_budget_trajectories_preprocessed.parquet'))
 # No summation across road-to-zone many-to-many links.
 summaries=[];peaks=pd.read_csv(ROOT/'data/exp/data-preprocessing/three_ward_protection_requirements_register_v1/zone_register_native.csv')
 for (case,zone),g in tr.groupby(['Case','Zone']):
  last=g.loc[g.Frame.idxmax()];maxrow=g.loc[g['Local Closure Difference'].abs().idxmax()];peakframe=int(peaks.loc[(peaks.Case==case)&(peaks.Zone==zone),'Control Excess Peak Frame'].iloc[0]);atpeak=g[g.Frame==peakframe].iloc[0]
  summaries.append({'Case':case,'Zone':zone,'Ground Inflow Since First Map':float(last['Ground Inflow Since First Map']),'Ground Outflow Since First Map':float(last['Ground Outflow Since First Map']),'Receiving Inflow Since First Map':float(last['Receiving Inflow Since First Map']),'Receiving Outflow Since First Map':float(last['Receiving Outflow Since First Map']),'Reconstructed Rain Since First Map':float(last['Reconstructed Rain Since First Map']),'Maximum Absolute Local Closure Difference':float(abs(maxrow['Local Closure Difference'])),'Map Quantization Bound At Maximum Difference':float(maxrow['Map Storage Change Quantization Bound']),'Saved Face Arithmetic Difference At Maximum Difference':float(maxrow['Saved Face Product Arithmetic Difference']),'Road Control Peak Frame':peakframe,'Road Control Peak Actual Time':float(atpeak['Actual Time']),'Ground Net Inflow At Road Control Peak':float(atpeak['Ground Inflow Since First Map']-atpeak['Ground Outflow Since First Map']),'Independent Cell Grouped Roundoff Saved':False,'Required Pump Capacity':None,'Budget Status':'All boundary-face exchange observed;independent cell-grouped source/roundoff attribution unavailable;not certified residual closure'})
 zs=pd.DataFrame(summaries);zs.to_parquet(P/'zone_ground_budget_summary_preprocessed.parquet',index=False);pd.testing.assert_frame_equal(zs,pd.read_parquet(P/'zone_ground_budget_summary_preprocessed.parquet'))
 cs=[]
 for v in valid:
  case=v['case'];g=zs[zs.Case==case]
  cs.append({'Case':case,'Zones':len(g),'New Ground Faces':v['new_ground_face_count'],'Ground Inflow Membership Sum':float(g['Ground Inflow Since First Map'].sum()),'Ground Outflow Membership Sum':float(g['Ground Outflow Since First Map'].sum()),'Selected Zone Net Ground Inflow':float(g['Ground Inflow Since First Map'].sum()-g['Ground Outflow Since First Map'].sum()),'Receiving Inflow':float(g['Receiving Inflow Since First Map'].sum()),'Receiving Outflow':float(g['Receiving Outflow Since First Map'].sum()),'Reconstructed Rain':float(g['Reconstructed Rain Since First Map'].sum()),'Maximum Local Closure Difference':v['maximum_absolute_local_closure_difference_m3'],'Maximum Map Quantization Bound':v['maximum_map_storage_change_quantization_bound_m3'],'Maximum Saved Face Arithmetic Difference':v['maximum_saved_face_product_arithmetic_difference_m3'],'Original Numerical Gates Passed':True,'Physical Map Fields Bitwise Identical':True,'Maximum Positivity Reporting Difference':v.get('positivity_reporting_comparison',{}).get('max_abs_held_reporting_difference_m3',0.0),'Independent Local Residual Closed':False})
 cases=pd.DataFrame(cs);cases.to_parquet(P/'case_ground_budget_summary_preprocessed.parquet',index=False);pd.testing.assert_frame_equal(cases,pd.read_parquet(P/'case_ground_budget_summary_preprocessed.parquet'))
 # Preserve unknown capacities/targets and receiving constraints.
 dictionary=[];units={'Actual Time':'s','Road Control Peak Actual Time':'s','Required Pump Capacity':'m3/s,unresolved'}
 exports=[]
 for path in sorted(P.glob('*_preprocessed.parquet')):
  d=pd.read_parquet(path);assert all(c.isascii() and c==c.strip() for c in d)
  for c in d:
   num=pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]);quantity=num and any(x in c for x in ['Inflow','Outflow','Rain','Storage','Bound','Difference','Arithmetic'])
   dictionary.append({'Dataset':path.name,'Readable Field':c,'Unit':units.get(c,'m3' if quantity else 'identifier/category/count/boolean'),'Role':'reference construction diagnostic;no final analysis variable promoted','Missing Policy':'unresolved capacity stays missing;no zero imputation'})
  exports.append({'path':str(path.relative_to(ROOT)),'rows':len(d),'sha256':sha(path)})
 pd.DataFrame(dictionary).to_csv(R/'data_dictionary.csv',index=False)
 dump(P/'export_manifest.json',{'outputs':exports,'script_sha256':{n:sha(ROOT/'src/analyses'/n) for n in ['audit_ground_flux_replay_v1.py','run_ground_flux_replay_v1.py','package_ground_flux_replay_v1.py']},'mapping_probe':str((R/'mapping_validation.json').relative_to(ROOT)),'reporting_correction_sha256':sha(R/'reporting_comparison_correction.json') if (R/'reporting_comparison_correction.json').exists() else None,'run_manifest_sha256':sha(P/'run_manifest.json'),'original_inputs_binding_sha256':sha(P/'protected_sources.json')})
 sources=''.join(f'<li>{html.escape(v["case"])}：<a href="execution/{v["case"]}/replay_validation.json">重放对照</a> · <a href="execution/{v["case"]}/audit.json">原数值准入</a> · <a href="execution/{v["case"]}/zone_ground_budget_trajectories.csv">控制区时序</a></li>' for v in valid)
 files=''.join(f'<li><a href="../../../processed/{TAG}/{p.name}">{p.name}</a></li>' for p in P.glob('*_preprocessed.parquet'))
 style='body{font:16px/1.8 system-ui;max-width:1260px;margin:30px auto;padding:0 24px;color:#203b49}aside{background:#edf4f7;padding:18px;border-left:4px solid #477e91}.warn{background:#fff3df;padding:16px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border:1px solid #d4e0e6;text-align:left}'
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>地表通量诊断重放结果</title><style>{style}</style><h1>地表通量诊断重放结果</h1><aside>7月31日高、低水位两组已完成诊断重放并通过核验。新增观测没有改变原109帧物理地图、实际时钟、逐步水量预算或原B断面流量。<b>只重放两组，没有扩展其余8组；未运行新的措施情景。</b></aside><h2>1. 已经补到哪些证据</h2><p>两组分别补采20,466和20,009个唯一地面格面的逐步流量；实际运行映射、端点、方向和封闭状态均已由原求解器库核验。所有配置封闭格面流量逐步为零，原负体积、雨量、守恒及参考标签规则均通过。新增G断面和原6763条B河道断面分开核验，没有放宽门槛。</p><div class="scroll">{cases.to_html(index=False,float_format=lambda v:f'{v:,.6f}',formatters={'Maximum Positivity Reporting Difference':lambda v:f'{v:.2e}'})}</div><p>体积单位m³，区间为首张实际地图至末张实际地图。地面流入／流出的“成员和”包括相邻控制区间的内部转移，<b>不能当作整个三区的新增来水或总排水需求</b>。共享格面两侧使用相反外向符号，净地面来水列按入水减出水求和，相邻区内部交换在符号上抵消；负值表示向控制区集合之外净排出。控制区净收支按唯一格面处理；三区与模型缓冲区的固定控制体范围继续沿用原定义。</p><p>计算耗时total_runtime单列，不要求与旧运行相同；源码将其写为墙钟结束时间减开始时间。并行统计held的微小归约差按线程数与float64舍入界核验，其他限制器计数、最小因子和时钟保持一致。原数值准入规则未改，首次过严比较及修正均保留在<a href="reporting_comparison_correction.json">审计修正记录</a>中；没有为此重跑高水位水动力。</p><h2>2. 部分水量收支推进到了哪一步</h2><p>每个控制区已有同一实际步号对应的储量变化、重建屋面分配降雨、河道双向交换，以及本次真正补采的地表双向交换。完整109帧时序可直接追踪地面来水是否集中在道路控制峰值之前；没有拿不同时间的局部峰值相加，也没有按道路组重复累加同一控制区。</p><div class="warn">仍有“局部闭合差额”，其中包含逐单元分组连续方程算术、雨量加到dvol后的舍入，以及地图float32储量写出误差。报告已分别量化可观测的格面乘积算术差与地图写出量化界；没有保存的逐单元组合舍入仍未独立归因。<b>不能把全部差额称作数值错误，也不能把它当作所需抽排量。</b></div><p>本轮保持所有Required Pump Capacity为空。控制区仍是原情景内湿区连接包络，不是固定汇水分区；完整目标水位、严格通行余量、设备扬程能力和共享接收限额还需与保护目标一起明确。</p><h2>3. 对下一步部署设计的意义</h2><p>现在可依据道路保护目标，区分本地降雨、地面持续来水和河道倒灌，再核查可布点位置与接收条件。其他8组未补采的地表通量继续保留缺失，不能把本次两组的数值套用过去。若要独立认证局部残差或细化抽排运行规则，再针对被动分区源项／舍入日志作最小接口补充；本轮未改求解器或建立新镜像。</p><p>未计算新增受影响人口、未训练、未打开封存测试；AnaSOP和三张已认可图保持不变。</p><h2>核验及数据</h2><ul>{sources}</ul><ul>{files}</ul><p><a href="validation.json">总核验</a> · <a href="mapping_validation.json">运行时映射</a> · <a href="campaign_state.json">运行状态</a> · <a href="data_dictionary.csv">字段及单位</a> · <a href="README.md">复现方法</a></p></html>'''
 (R/'review.html').write_text(page)
 (R/'README.md').write_text('''# Ground face observational replay

Two original2024-07-31urban-geometry,10m scenarios are replayed with additional observational crosssections. All original physical input bytes,image,quality masks,rainfall,parent lineage and split remain unchanged. Only sfincs.crs and the run identity differ in isolated directories. No new forcing/measure scenario,training,people computation or sealed-test access.

Scripts:

1. `.venv/bin/python src/preprocessing/prepare_ground_flux_replay_v1.py`
2. `.venv/bin/python src/preprocessing/qualify_ground_flux_mapping_v1.py`
3. `.venv/bin/python src/analyses/run_ground_flux_replay_v1.py`

Staging and execution refuse existing attempts. Do not rerun them into completed directories. PAUSE/STOP flags reside in this processed-data folder. Mapping is tested without advancing time,using a helper linked against the exact existing image library. This creates no new solver image. The high-stage run must pass before low-stage starts;no other cases start automatically.

The unchanged original numerical auditor receives a read-only view of the original6763B section columns and their identical original geometry. All original gate functions and thresholds execute unchanged against the new run identity/input hashes. Extra G columns are audited independently. All physical map variables,actual-clock and water-budget diagnostics,and original B discharges must be bitwise identical. The map total_runtime is a wallclock cost and is recorded separately. The limiter held reporting scalar uses an OpenMP positive-sum reduction;its comparison uses4*thread_count*float64_epsilon times the larger magnitude,while counts,minimum factors and clocks stay exact. Original numerical admission is unchanged. The initial overstrict reporting comparison,source snapshots and correction receipt are preserved;high-stage hydraulics are not retried. A held campaign may continue with --resume-qualified only after saved H outputs pass re-audit and L remains unlaunched. Runtime native face/endpoints/sign/closed flags must match the prepared requests. Every saved closed-face flow must be exactly zero.

New ground exchanges use actual step dt and the original float32 applied-volume convention. Zone-oriented signed incidence cancels shared internal faces. Gross zone membership sums include transfers between selected zones and are not whole-domain inflow. The saved-face-product arithmetic difference compares float32(q_saved*dt)with double arithmetic on the saved float32Q;it does not recover width-product or cell-grouped rounding. Map-volume quantization bounds are computed per zone from half-ULP storage bounds at both endpoints. Unobserved cell-grouped/source-increment arithmetic is not called zero. Local closure differences remain explicitly unattributed;they are not independent numerical residual certification or pump demand.

All capacities stay missing. Final targets,service margins,unit capability and receiver allowances remain unresolved. No numerical results are transferred to the other eight un-replayed cases.
''')
 assert sha(ROOT/'docs/AnaSOP.md')==ana
 validation={'passed':True,'cases':[v['case'] for v in valid],'hydraulic_replays':2,'new_forcing_or_measure_scenarios':0,'all_original_numerical_gates_passed':True,'all_original_physical_map_fields_and_actual_clocks_bitwise_identical':True,'all_original_B_discharge_bitwise_identical':True,'ground_exchange_collected':True,'native_ground_faces':sum(v['new_ground_face_count'] for v in valid),'trajectory_rows':len(tr),'closed_face_q_exact_zero':True,'independent_local_grouped_roundoff_saved':False,'local_residual_independently_closed':False,'required_capacity_all_missing':True,'AnaSOP_unchanged':True,'new_training':0,'new_people_counts':False,'case_summaries':cs};dump(R/'validation.json',validation)
 state=json.loads((R/'campaign_state.json').read_text());state.update(status='qualified_complete',delivery_validation='validation.json');dump(R/'campaign_state.json',state)
 print('GROUND_FLUX_REPLAY_PACKAGE_COMPLETE',len(tr),flush=True)
if __name__=='__main__':main()
