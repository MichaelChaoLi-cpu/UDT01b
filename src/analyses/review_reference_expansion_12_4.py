"""Portable forcing-coverage and qualified-response inventory for the finite 12/4 expansion."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,csv,html,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/preprocessing'));sys.path.insert(0,str(ROOT/'src/analyses'))
from prepare_reference_expansion_12_4 import OUT,REPORT,PRIOR,PRIOR_REPORT,LIB,FEATURES,SHAPES,features,csvwrite
from prepare_sfincs_cpu_build import dump
from load_three_ward_minimum_scenarios import find_case
from audit_three_ward_negative_volumes import readnp
from compare_adaptive_positivity import wquantile

def table(rows):
 if not rows:return ''
 keys=list(rows[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>'for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>'for k in keys)+'</tr>'for row in rows)+'</table></div>'

def main(partial=False):
 o=ROOT/OUT;r=ROOT/REPORT;s=json.loads((o/'selection.json').read_text());m=json.loads((o/'expansion_manifest.json').read_text());new={x['case_id']:x for x in m['runs']};old={x['case_id']:x for x in m['reused']};rows=[];metrics=[];coverage=[];variables=[]
 support=readnp(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz');valid=support['Eligible Ground Mask'];area=support['Ground Area'];pause=json.loads((r/'pause_receipt.json').read_text())if(r/'pause_receipt.json').exists()else{};paused_cases={n.removeprefix('udt01b-expand124-')for n in pause.get('paused_containers',[])}
 for k in s['selected_case_ids']:
  c=find_case(k);v=features(c);ap=ROOT/old[k]['audit_path']if k in old else r/'expansion'/k/'audit.json';lp=ROOT/old[k]['label_path']if k in old else o/'expansion'/(k+'_candidate_reference.npz');row={'case_id':k,'parent_id':c['parent_id'],'split':c['split'],'shape':c['parameters']['shape_class'],'reused':k in old,**{n:float(x)if np.isfinite(x)else None for n,x in zip(FEATURES,v)},'response_status':'pending','original_admission_state':'not_audited','label_path':None}
  if ap.exists():
   a=json.loads(ap.read_text());row.update(response_status='qualified_candidate'if a['all_checks_passed']and a['admission']['technical_reference']else'hold_event',original_admission_state=a['admission']['state'],label_path=str(lp.relative_to(ROOT)));z=a['summary'];metrics.append({'case_id':k,'split':c['split'],'reused':k in old,'all_checks_passed':a['all_checks_passed'],'negative_cells':z['negative_cells'],'rain_error_m3':z['rain_difference_m3'],'max_budget_residual_m3':z['max_budget_residual_m3'],'actual_final_time_s':z['actual_final_time_s'],'map_records':z['map_records'],'closed_faces':z['closed_boundary_faces'],'original_admission_state':a['admission']['state'],'recorded_elapsed_minutes':z['runtime_s']/60,'interrupted_by_user_pause':k in paused_cases,'runtime_comparison_status':'excluded_user_pause'if k in paused_cases else'not_benchmarked','training_release':False})
   with ap.with_name('label_support.csv').open()as f:cov=list(csv.DictReader(f));coverage+=cov
   d=readnp(lp)
   for name,x in [('Peak Mean Depth Lower',d['Peak Mean Depth Lower']),('Peak Mean Depth Upper',d['Peak Mean Depth Upper']),('Peak Interval Width',d['Peak Mean Depth Upper']-d['Peak Mean Depth Lower'])]:
    ok=valid&np.isfinite(x);variables.append({'Case':k,'Readable Variable':name,'Eligible Count':int(valid.sum()),'Nonmissing Count':int(ok.sum()),'Minimum':float(x[ok].min()),'Area Weighted Median':wquantile(x[ok],area[ok],.5),'Area Weighted 95th Percentile':wquantile(x[ok],area[ok],.95),'Maximum':float(x[ok].max()),'Final Variable Status':'no'})
  rows.append(row)
 if not partial:assert len(metrics)==16 and all(x['all_checks_passed']for x in metrics)
 csvwrite(r/'response_inventory.csv',rows)
 if metrics:csvwrite(r/'response_numerical_summary.csv',metrics);csvwrite(r/'label_support.csv',coverage)
 brief=ROOT/'data/exp/data-briefing/reference_expansion_12_4';brief.mkdir(parents=True,exist_ok=True);csvwrite(brief/'response_inventory.csv',rows)
 if variables:csvwrite(brief/'variable_summary.csv',variables)
 csvwrite(brief/'skipped_files.csv',[{'scope':'Original raw inputs, original forcing catalog, prior frozen response packages','reason':'Unchanged protected sources; reuse existing inventories and verify checksums rather than rescan or alter them.'}])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 selected=set(s['selected_case_ids']);reused=set(s['reused_case_ids']);allcases=json.loads((ROOT/LIB/'catalog.json').read_text())['cases'];names=['Peak rainfall (mm/h)','Total rainfall (mm)','Rain duration (min)','Stage increment (m)','Meaningful stage-peak lag (min)'];fig,axes=plt.subplots(2,5,figsize=(16,6),layout='constrained')
 for iy,split in enumerate(['train','validation']):
  pool=[x for x in allcases if x['split']==split]
  for j in range(5):
   ax=axes[iy,j]
   for group,color,offset,label in [('pool','#c8d0d6',0,'Original partition'),('retained','#26728b',.13,'Retained'),('added','#e69242',-.13,'Added')]:
    cs=pool if group=='pool'else[x for x in pool if(x['case_id']in reused if group=='retained'else x['case_id']in selected and x['case_id']not in reused)];values=np.array([features(c)[j]for c in cs]);values=values[np.isfinite(values)];ax.scatter(values,np.zeros(len(values))+offset,s=22 if group=='pool'else 36,color=color,label=label,alpha=.85)
   ax.set_xlabel(names[j]);ax.set_yticks([]);ax.set_ylim(-.4,.4);ax.set_title(split if j==0 else'',loc='left');ax.grid(axis='x',alpha=.15)
 axes[0,0].legend(fontsize=7,loc='upper left');fig.suptitle('Input-only parent selection: fixed original splits; pulse-free lag omitted');fig.savefig(r/'forcing_coverage.png',dpi=160);fig.savefig(brief/'forcing_coverage.png',dpi=160);plt.close(fig)
 qualified=[x['case_id']for x in rows if x['response_status']=='qualified_candidate'];held=[x['case_id']for x in rows if x['response_status']=='hold_event'];pending=[x['case_id']for x in rows if x['response_status']=='pending'];summary={'target_training_parents':12,'target_validation_parents':4,'retained_parents':6,'new_response_limit':10,'selected':s['selected_case_ids'],'qualified':qualified,'held':held,'pending':pending,'all16_numerically_qualified':len(qualified)==16,'training':False,'formal_reference_release':False,'sealed_response_execution':False,'real_world_accuracy_verified':False,'limitations':s['coverage_limitations'],'pause_resumed_cases':sorted(paused_cases),'runtime_note':'Original monotonic-clock elapsed readings are preserved. Interrupted runs are affected by the pause and are excluded from runtime comparisons; these readings are not measured active-compute times or UTC wall durations. No performance benchmark is claimed.'};dump(r/'summary.json',summary)
 if(o/'execution_overlay.json').exists():
  idx=json.loads((o/'execution_overlay.json').read_text());(r/'execution_overlay.json').write_bytes((o/'execution_overlay.json').read_bytes());summary.update(cumulative_completed_subset_of58=idx['cumulative_completed_subset_of58'],remaining_unrun=idx['remaining_unrun']);dump(r/'summary.json',summary)
 display=[{'事件':x['case_id'],'谱系':x['parent_id'],'分区':x['split'],'雨型':x['shape'],'峰值雨强 mm/h':x['rain_peak_mm_h'],'总雨量 mm':round(x['rain_total_mm'],2),'历时 min':x['rain_duration_min'],'水位抬升 m':x['stage_amplitude_m'],'水位峰时差 min':x['meaningful_peak_lag_min']if x['meaningful_peak_lag_min']is not None else'N/A：无抬升','复用':x['reused'],'响应':x['response_status']}for x in rows]
 notes='累计12个训练、4个验证父事件候选；复用6个已合格父事件，只新增10次规则10 m水动力计算。四类雨型各3个训练、1个验证。父事件由原分区内输入参数距离和固定雨型配额选定，未使用新响应深度或模型得分。两份开放控制保留在原库，不计新增独立父事件。'
 limits='训练输入覆盖雨强23.14–99.78 mm/h、历时33–90分钟、水位抬升0–1.384 m；验证输入历时仅36–54分钟，正向水位峰时差仅到6分钟，尚未覆盖原验证池的全部长历时和时差组合。无抬升事件没有有意义的水位峰时差，不用计划时差冒充。距离指标只是输入空间覆盖，不能解释为概率或真实暴雨代表性。原合成情景、固定地貌和连动水位假设继续保留。第四批从用户暂停时保留的内存状态恢复，没有重启计算；其原始运行计时受暂停影响，仅作诊断保留，排除出性能比较。'
 status=f'目前合格 {len(qualified)}；待审隔离 {len(held)}；待完成 {len(pending)}。'+'原精确、窄区间、宽区间与隔离规则不变，质量信息不作预测输入，未训练、未拟合归一化、未打开封存测试或挑战响应。'
 (r/'README.md').write_text('# 三区父事件响应扩展：12训练＋4验证\n\n'+notes+'\n\n'+status+'\n\n'+limits+'\n\nReproducible execution: prepare_reference_expansion_12_4.py, then the initial preflight_reference_expansion_12_4.py before the first hydraulic launch, then bootstrap_reference_expansion_12_4.py to execute or resume. Reuse the initial preflight receipt when resuming completed executions. The bootstrap supplies the audit output directory and retains completed hydraulic runs. The first host-export failure and recovery are preserved in host_interface_recovery.json; no hydraulic rerun or numerical rule change occurred.\n\nFramelog: MILI-D-20261016-001.\n')
 (brief/'README.md').write_text('# Parent-response expansion inventory\n\nSixteen selected parent candidates retain the original train/validation partitions; six qualified response packages are reused and ten new regular-grid response packages are authorized. Static forcing-attribute distributions are shown; no time-series plot or training is generated. Peak-depth bounds, label support and numerical records remain candidate references under the unchanged policy. No variables are renamed or finalized.\n')
 links=[p.name for p in [r/'selection_inventory.csv',r/'forcing_coverage.csv',r/'selected_parent_pair_audit.csv',r/'response_inventory.csv',r/'response_numerical_summary.csv',r/'label_support.csv',r/'provenance.json',r/'execution_overlay.json',r/'preflight.json',r/'host_interface_recovery.json',r/'pause_receipt.json',r/'resume_receipt.json',r/'summary.json',r/'README.md']if p.exists()]
 page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区父事件扩样12＋4</title><style>body{font:16px/1.8 system-ui;background:#edf2f4;color:#203c48;margin:0}main{max-width:1350px;margin:auto;padding:28px}section{background:white;border-radius:12px;padding:22px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dfe6e9;text-align:left}img{width:100%}a{color:#146c81}</style><main><p>Mili · Whole-three-ward scenario library</p><h1>父事件响应扩展：12训练＋4验证</h1><p>'+notes+'</p><section><h2>输入覆盖与保留缺口</h2><img src="forcing_coverage.png" alt="原分区输入属性分布和保留、新增事件"><p>'+limits+'</p></section><section><h2>事件清单与响应状态</h2><p>'+status+'</p>'+table(display)+'</section><section><h2>原数值门槛与输出审查</h2>'+table(metrics)+'</section><section><h2>复核材料</h2><p>'+' · '.join('<a href="'+p+'">'+p+'</a>'for p in links)+'</p><p>Framelog MILI-D-20261016-001</p></section></main></html>'
 (r/'review.html').write_text(page);print(json.dumps({'qualified':len(qualified),'held':held,'pending':len(pending),'training':False}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');main(p.parse_args().partial)
