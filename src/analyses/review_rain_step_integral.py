"""Unchanged-rule control admission and whole-area effects of actual-step rain integration."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,csv,re,html
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_rain_step_integral import OUT,REPORT,PRIOR,CASES
from prepare_sfincs_cpu_build import dump,sha
from reference_label_policy import event_admission,integrate_rate_at,CONTRACT
from audit_adaptive_positivity_controls import structured
from audit_three_ward_negative_volumes import readnp,csvwrite
from compare_adaptive_positivity import wquantile

def table(rows):
 if not rows:return ''
 keys=list(rows[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>'for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>'for k in keys)+'</tr>'for r in rows)+'</table></div>'
def main():
 out=ROOT/OUT;report=ROOT/REPORT;old=ROOT/PRIOR;checks=[];assess={};cases={};rows=[];supportrows=[];rainproof={};plots={}
 m=json.loads((out/'run_manifest.json').read_text());aud=json.loads((report/'audit.json').read_text());boundary=json.loads((report/'boundary_comparison.json').read_text());geom=json.loads((report/'matched_boundary_geometry.json').read_text())
 support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');area=support['Ground Area'];ward=support['Ward Code'];valid=support['Eligible Ground Mask']
 scopes={'Whole domain':valid,'Three wards':valid&(ward>0),'Chiyoda':valid&(ward==1),'Chuo':valid&(ward==2),'Minato':valid&(ward==3)}
 for run in m['runs']:
  case=run['case_id'];p=ROOT/run['folder'];q=ROOT/run['source_folder'];d=structured(p/'udt_step_budget.csv');prev=structured(q/'udt_step_budget.csv');trace=structured(p/'udt_rain_integral.csv');t=d['actual_state_time'];dt=d['actual_step_dt'];knots=np.array([0.,3600,3601,7200,7201,14400]);peak=0. if case==CASES[0]else 60.;ar=38976800.;rates=np.array([0,0,peak,peak,0,0])*ar/3.6e6
  exact=integrate_rate_at(knots,rates,np.clip(t,0,14400));applied=d['rain_applied_cumulative'];raw=np.r_[0,np.cumsum(trace['integrated_native_rate_volume_m3'])];oldexact=integrate_rate_at(knots,rates,np.clip(prev['actual_state_time'],0,14400))
  traceok=len(trace)==len(t)-1 and np.array_equal(trace['actual_end_time_s'],t[1:])and np.array_equal(trace['dt_s'],dt[1:])and np.max(abs(trace['sum_weight_seconds']-dt[1:]))<1e-9
  checks.append({'case':case,'check':'integral_diagnostic_binds_every_actual_step','pass':bool(traceok)})
  encoded=float(next(c['details']['encoded_m3']for c in aud['checks']if c['case']==case and c['check']=='whole_domain_rain_volume_matches_original_control'));tol=max(1.,encoded)*1e-6
  checks.append({'case':case,'check':'original_1e6_relative_rain_gate_not_relaxed','pass':bool(abs(float(applied[-1]-exact[-1]))<=tol)})
  checks.append({'case':case,'check':'all_step_integrated_rain_consistent_with_original_curve','pass':bool(np.max(abs(applied-exact))<=tol and np.max(abs(raw-exact))<=tol)})
  drysame=all(np.array_equal(d[k],prev[k])for k in d.dtype.names)if case==CASES[0]else None
  if drysame is not None:checks.append({'case':case,'check':'dry_control_same_step_ledger_byte_identical','pass':bool(drysame and(p/'udt_step_budget.csv').read_bytes()==(q/'udt_step_budget.csv').read_bytes())})
  rainproof[case]={'original_physical_depth_mm':0 if peak==0 else 60,'original_contributing_area_m2':ar,'original_curve_integral_m3':float(exact[-1]),'encoded_original_volume_m3':encoded,'final_encoded_difference_m3':float(applied[-1]-encoded),'native_applied_m3':float(applied[-1]),'final_error_m3':float(applied[-1]-exact[-1]),'max_prefix_error_m3':float(np.max(abs(applied-exact))),'original_final_tolerance_m3':tol,'previous_endpoint_final_error_m3':float(prev['rain_applied_cumulative'][-1]-oldexact[-1]),'previous_endpoint_max_prefix_error_m3':float(np.max(abs(prev['rain_applied_cumulative']-oldexact))),'integrated_native_rate_minus_original_m3':float(raw[-1]-exact[-1]),'native_applied_minus_integrated_rate_m3':float(applied[-1]-raw[-1]),'dry_control_identical':drysame}
  a=readnp(old/(case+'_adaptive_candidate_reference.npz'));b=readnp(out/(case+'_adaptive_candidate_reference.npz'));al=a['Peak Mean Depth Lower'];au=a['Peak Mean Depth Upper'];bl=b['Peak Mean Depth Lower'];bu=b['Peak Mean Depth Upper'];gap=np.maximum(np.maximum(bl-au,al-bu),0.)
  for name,sel in scopes.items():
   wet=sel&((au>.1)|(bu>.1));i=area[sel&(al>.1)&(bl>.1)].sum();u=area[sel&((al>.1)|(bl>.1))].sum();possible_union=area[sel&((au>.1)|(bu>.1))].sum();possible_intersection=area[sel&(au>.1)&(bu>.1)].sum()
   rows.append({'case':case,'scope':name,'old_certain_wet_km2':float(area[sel&(al>.1)].sum()/1e6),'corrected_certain_wet_km2':float(area[sel&(bl>.1)].sum()/1e6),'certain_extent_iou':float(i/u)if u else None,'interval_iou_lower':float(i/possible_union)if possible_union else None,'interval_iou_upper':float(min(1.,possible_intersection/u))if u else None,'confirmed_depth_change_p95_m':wquantile(gap[wet],area[wet],.95),'confirmed_depth_change_max_m':float(np.max(gap[wet]))if wet.any()else None,'confirmed_change_gt_0_1m_km2':float(area[wet&(gap>.1)].sum()/1e6)})
   supportrows.append({'case':case,'scope':name,**{k.replace(' Mask','').replace(' ','_')+'_km2':float(area[sel&b[k]].sum()/1e6)for k in ['Exact Label Mask','Interval Label Mask','Wide Interval Mask','Quarantine Mask','Admitted Mask']},'peak_interval_p95_m':wquantile((bu-bl)[sel],area[sel],.95),'peak_interval_max_m':float(np.max((bu-bl)[sel]))})
  if case==CASES[0]:checks.append({'case':case,'check':'dry_labels_and_masks_unchanged','pass':all(np.array_equal(a[k],b[k],equal_nan=True)for k in a)})
  failed={c['check']for c in aud['checks']if c['case']==case and not c['pass']}
  # Same mapping of frozen checks to the ten ORIGINAL event gates as the prior review.
  gates={'completion':'completed_full_existing_control'not in failed,'identity':'bound_input_hashes_and_identity'not in failed,'geometry':geom['all_physical_open_receiving_segments_equal']and'physical_topology_matches_frozen_geometry'not in failed and'all_closed_receiving_faces_have_zero_actual_flux'not in failed,'fields':'finite_native_state_fields'not in failed and'strict_nonnegative_at_every_applied_step'not in failed,'time':'adaptive_map_geometry_and_complete_horizon'not in failed and'one_complete_online_peak_window'not in failed,'clock_alignment':'actual_step_clock_complete'not in failed and'output_actual_clock_binds_accounted_states'not in failed,'boundary':boundary['all_checks_passed'],'rain':'whole_domain_rain_volume_matches_original_control'not in failed and'roof_donor_assignment_conserved_in_native_sampling'not in failed and'one_second_linear_meteo_brackets'not in failed and'seconds_preserved_from_original_minute_coordinates'not in failed,'budget':'unchanged_formal_endpoint_and_prefix_balance_gates'not in failed,'censoring':'known_shallow_censoring_and_online_peak_envelope'not in failed}
  assess[case]={'required_gates':gates,'preview':event_admission(gates,b),'failed_checks':sorted(failed),'formal_registry_changed':False,'training_release':False}
  cases[case]={'numerics':aud['cases'][case],'rain':rainproof[case],'three_ward_extent':next(r for r in rows if r['case']==case and r['scope']=='Three wards'),'three_ward_support':next(r for r in supportrows if r['case']==case and r['scope']=='Three wards'),'admission':assess[case]['preview']};plots[case]=(t,applied-exact,prev['actual_state_time'],prev['rain_applied_cumulative']-oldexact,bl)
 csvwrite(report/'whole_area_effects.csv',rows);csvwrite(report/'reference_support.csv',supportrows);dump(report/'rain_integral_audit.json',{'cases':rainproof,'checks':checks,'all_checks_passed':all(c['pass']for c in checks),'input_normalization':False,'tolerance_relaxation':False})
 dump(report/'reference_admission_assessment.json',assess);dump(report/'summary.json',{'cases':cases,'training':0,'batch58':0,'new_parents':0,'formal_reference_release':False,'real_event_accuracy_verified':False})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(2,2,figsize=(12,10),layout='constrained')
 for col,case in enumerate(CASES):
  t,err,ot,oe,depth=plots[case];ax=axs[0,col];ax.plot(ot/3600,oe,label='Previous endpoint integration',color='#ba6f35');ax.plot(t/3600,err,label='Corrected step integral',color='#197184');ax.axhline(0,color='#7d8b90',lw=.5);ax.set_xlabel('Actual simulation time (hours)');ax.set_ylabel('Applied minus original rain volume (m3)');ax.set_title('Dry control'if col==0 else'Rain control');ax.legend(fontsize=8)
  ax=axs[1,col];extent=[-11.36,-3.66,-42.44,-32.32];ax.imshow(np.where(valid,1,np.nan),extent=extent,cmap=ListedColormap(['#e7ecee']),interpolation='none');p=ax.imshow(np.where(valid&(depth>.01),depth,np.nan),extent=extent,cmap='Blues',vmin=0,vmax=1,interpolation='none')
  for code in [1,2,3]:ax.contour(-11.36+(np.arange(770)+.5)*.01,-32.32-(np.arange(1012)+.5)*.01,(ward==code).astype('f4'),levels=[.5],colors='#5c737d',linewidths=.5)
  ax.set_xlabel('EPSG:6677 easting (km)');ax.set_ylabel('EPSG:6677 northing (km)');ax.set_title('Corrected peak depth lower bound');ax.set_aspect('equal')
 fig.colorbar(p,ax=list(axs[1]),label='Common10m eligible-ground peak mean depth (m)',shrink=.8,extend='max');fig.savefig(report/'rain_integral_controls.png',dpi=150);plt.close(fig)
 display=[{'控制组':c,'原雨量 m³':round(cases[c]['rain']['original_curve_integral_m3'],3),'实际施加 m³':round(cases[c]['rain']['native_applied_m3'],3),'最终偏差 m³':round(cases[c]['rain']['final_error_m3'],6),'最大过程偏差 m³':round(cases[c]['rain']['max_prefix_error_m3'],6),'严格负体积格':cases[c]['numerics']['negative_native_cells'],'最大预算残差 m³':round(cases[c]['numerics']['max_abs_budget_residual_m3'],6),'原政策预览':cases[c]['admission']['state']}for c in CASES]
 intro='保持原雨型、屋面分配、输入字节、谱系和划分，仅修正实际步长内的降雨积分。只重放现有干控制组和降雨控制组的全域10/5 m配置，保留旧10 m与旧加密结果；未训练，未运行58个待建库情景。'
 notes='逐段积分原始分段线性雨强，步均雨强用于正性限制可用雨量、连续方程和同一步预算；累计降雨使用同一量，瞬时雨强输出保留原语义。不强迫时间步落在雨量结点，不缩放雨型或屋面权重，不裁零储量、不放宽原雨量及水量平衡阈值。最终单精度施加舍入与步积分截断误差分开报告；正式雨量门槛仍使用原文件编码后的体积，名义物理曲线的体积另列。'
 limits='通过数值事件门槛仅支持模型条件下的部分参考标签。宽峰值区间、原建筑屏障接触和其他原隔离规则继续生效；未取区间中点、未零填补。本次重建到统一10 m支持的加密响应仍是诊断标签，不替代原政策指定的规则10 m主标签。默认注册表与加载器保持关闭，未释放训练标签。峰值范围不是同时积水面或动态排水路径；两组合成控制不验证实测精度、河道动力学或网格收敛。'
 doc='# 降雨实际步积分修正与原准入规则核验\n\n'+intro+'\n\n'+notes+'\n\n'+json.dumps(display,ensure_ascii=False,indent=2)+'\n\n'+limits+'\n\nFramelog: MILI-D-20261015-016.\n\n范围影响表比较修正前后同一10/5 m配置；边界接口对账另外复用冻结10 m结果，字段中明确区分两种网格。\n'
 doc+='\n初始接口测试中的瞬时雨强断言曾要求浮点逐位相等，改为两倍单精度舍入容差后通过；修正前后的测试和镜像构建证据保留在预核验目录，均未启动水动力控制。最终镜像的全部生产回归已重新核验。该测试容差与原标签雨量准入阈值无关。\n'
 (report/'README.md').write_text(doc)
 links=['summary.json','audit.json','rain_integral_audit.json','reference_admission_assessment.json','whole_area_effects.csv','reference_support.csv','boundary_group_comparison.csv','build_validation.json','runtime_links.txt','frozen_forcing_library_check.json','source_binding.json','solver_binding.json','README.md']
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区降雨积分修正</title><style>body{font:16px/1.8 system-ui;background:#edf2f4;color:#203c48;margin:0}main{max-width:1200px;margin:auto;padding:28px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;text-align:left;border-bottom:1px solid #dde4e8}img{width:100%}a{color:#146c81}</style><main><p>Mili · whole-three-ward controls · original admission policy</p><h1>降雨步积分修正：能否通过原标签准入？</h1><p>'+intro+'</p><section><h2>实际雨量与同一步预算</h2>'+table(display)+'<p>'+notes+'</p><img src="rain_integral_controls.png" alt="实际施加雨量偏差及全域峰值平均水深"></section><section><h2>与修正前加密结果的整体对照</h2>'+table(rows)+'<p>阈值0.10 m，所有地面支持均保留；确认水深变化为区间分离下界，区间重叠不等于结果相同。干组逐步账本与标签单独检验一致性。</p></section><section><h2>原政策标签支持</h2>'+table(supportrows)+'<p>'+limits+'</p></section><section><h2>复核材料</h2><p>'+' · '.join('<a href="'+x+'">'+x+'</a>'for x in links)+'</p><p>Framelog MILI-D-20261015-016</p></section></main></html>'
 (report/'review.html').write_text(page)
 dump(out/'qualification.json',{'image_id':json.loads((out/'solver_binding.json').read_text())['image_id'],'build_qualified':True,'two_existing_controls_completed':True,'qualified_hydraulic_grid_profile':'existing adaptive10/5m controls only; no new uniform10m replay','all_original_numerical_checks_passed':aud['all_checks_passed'],'rain_integral_checks_passed':all(c['pass']for c in checks),'all_original_event_gates_passed':all(all(x['required_gates'].values())for x in assess.values()),'formal_reference_release':False,'training':0,'batch58':0,'new_parent_events':0})
 sb=json.loads((out/'solver_binding.json').read_text());sb['hydraulic_end_to_end_qualified']=bool(aud['all_checks_passed']and all(c['pass']for c in checks));dump(out/'solver_binding.json',sb)
 (report/'solver_binding.json').write_bytes((out/'solver_binding.json').read_bytes());(report/'source_binding.json').write_bytes((out/'source_binding.json').read_bytes())
 print(json.dumps({'controls':display,'rain_checks_passed':all(c['pass']for c in checks)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
