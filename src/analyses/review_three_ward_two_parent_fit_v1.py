"""Review the one completed fitting check; never trains or scores validation."""
from pathlib import Path
import sys,json,csv,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,Normalize
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,arrays,load_label
from run_three_ward_unet_controls_v2 import csv_write
from diagnose_three_ward_d_false_positives import auc
from three_ward_surrogate_protocol_v2 import macro_events
NAME='three_ward_two_parent_fit_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME;SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'
def fmt(x,scale=1,n=3):return 'NA' if x is None or x=='' else f'{float(x)*scale:.{n}f}'
def table(rr,cols):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(t)+'</th>' for k,t in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k,t in cols)+'</tr>' for r in rr)+'</tbody></table></div>'
def main():
 p=read(OUT/'protocol.json');done=read(REPORT/'completion.json');assert done['status']=='complete' and done['actual_fit_count']==1;ids=p['selected_case_ids'];frames=[read(REPORT/f'step_{s:04}.json') for s in p['evaluation_updates'] if s<=done['actual_optimizer_updates']];history=list(csv.DictReader((REPORT/'history.csv').open()));z=arrays(SOURCE/'prepared_inputs.npz');area=z['Ground Area'];eligible=z['Eligible Ground Mask'];ward=z['Ward Code'];labels=[load_label(cid,'train') for cid in ids];y=np.array([d['Peak Mean Depth Lower'].astype('f4') for d in labels]);masks=np.array([d['Exact Label Mask']&eligible for d in labels]);events=[];macros=[];spatial=[];fields={}
 sourceframe=read(ROOT/'data/exp/estimation/three_ward_unet_controls_v2/D_huber_lower_lr/epoch_040.jsonl');historical=[dict(r,point='historical_D40_context',optimizer_updates=480,parent_exposures=40) for r in sourceframe['event_metrics'] if r['case_id'] in ids and r['support']=='own_exact']
 for point,path,frame in [('initial','initial_predictions.npz',frames[0]),('endpoint','endpoint_predictions.npz',frames[-1]),('diagnostic_best','best_diagnostic_predictions.npz',read(REPORT/'best_diagnostic_metrics.json'))]:
  pred=arrays(OUT/path)['Peak Mean Depth'];fields[point]=pred
  for r in frame['event_metrics']:
   q={k:v for k,v in r.items() if not isinstance(v,dict)};q.update({'point':point,'optimizer_updates':frame['step'],'parent_exposures':frame['step']//2})
   if r['scope']=='three_wards':
    i=ids.index(r['case_id']);m=masks[i]&(ward>0);q['AUC']=auc(pred[i][m],y[i][m]>.1,area[m]);wet=y[i]>.1;pw=pred[i]>.1;spatial.append({'point':point,'case_id':r['case_id'],'AUC':q['AUC'],'reference_wet_area_km2':float(area[m&wet].sum()/1e6),'predicted_wet_area_km2':float(area[m&pw].sum()/1e6),'false_area_km2':float(area[m&~wet&pw].sum()/1e6),'missed_area_km2':float(area[m&wet&~pw].sum()/1e6)})
   events.append(q)
  for scope in ['domain','three_wards','Chiyoda','Chuo','Minato','cross_ward']:
   rr=[r for r in events if r['point']==point and r['scope']==scope];macros.append({'point':point,'scope':scope,**macro_events(rr)})
 csv_write(REPORT/'event_metrics.csv',events+[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in historical]);csv_write(REPORT/'spatial_fit_metrics.csv',spatial);write(REPORT/'parent_macro_metrics.json',macros)
 endpoint=frames[-1];best=read(REPORT/'best_diagnostic_metrics.json');failed=[{'case_id':r['case_id'],'scope':r['scope'],'failed_metrics':[k for k,v in r['checks'].items() if not v]} for r in endpoint['gates']['rows']];gate_count=sum(len(r['checks']) for r in endpoint['gates']['rows']);gate_pass=sum(sum(r['checks'].values()) for r in endpoint['gates']['rows']);steps=[json.loads(l) for l in (REPORT/'optimizer_steps.jsonl').read_text().splitlines()];initial_rows=[r for r in events if r['point']=='initial' and r['scope']=='three_wards'];endrows=[r for r in events if r['point']=='endpoint' and r['scope']=='three_wards'];summary={'status':'complete','decision':p['decision'],'fit_demonstrated':done['fit_demonstrated'],'stop_reason':done['stop_reason'],'actual_new_fits':1,'actual_updates':done['actual_optimizer_updates'],'parent_exposures':done['per_parent_exposures'],'gates_passed_at_endpoint':gate_pass,'applicable_gates':gate_count,'endpoint_gate_failures':failed,'endpoint_three_ward_events':endrows,'diagnostic_best_step':done['best_diagnostic_step'],'diagnostic_best_all_gates_pass':best['gates']['all_parent_ward_gates_pass'],'endpoint_all_gates_pass':endpoint['gates']['all_parent_ward_gates_pass'],'consecutive_success_evaluations':done['final_consecutive_passes'],'initial_three_ward_events':initial_rows,'gradient_finite_all_steps':all(r['gradient_pre_clip']['finite'] for r in steps),'gradient_max':max(r['gradient_pre_clip']['total_norm'] for r in steps),'clipping_count':sum(r['clipping_required'] for r in steps),'initial_fixed_objective_m2':float(history[0]['fixed_objective_m2']),'final_fixed_objective_m2':float(history[-1]['fixed_objective_m2']),'formal12plus4_split_unchanged':True,'quality_masks_unchanged':True,'validation_used_for_selection_or_scoring':False,'sealed_opened':False,'sample_expansion':False,'limits':p['interpretation_limits']};write(REPORT/'summary.json',summary)
 fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True);colors={ids[0]:'#0f766e',ids[1]:'#b45309'}
 charts=[('iou_0.1','Three-ward wet IoU',1,.85),('recall_0.1','Three-ward wet recall (%)',100,.90),('precision_0.1','Three-ward precision (%)',100,.90),('wet_mae_m','Three-ward wet MAE (cm)',100,.05),('deep_mae_m','Three-ward deep MAE (cm)',100,.20)]
 for ax,(key,title,scale,gate) in zip(axes.ravel(),charts):
  for cid in ids:
   yy=[next(r for r in f['event_metrics'] if r['case_id']==cid and r['scope']=='three_wards').get(key) for f in frames];ax.plot([f['step'] for f in frames],[v*scale if v is not None else np.nan for v in yy],color=colors[cid],label=cid)
  ax.axhline(gate*scale,color='#64748b',ls='--',lw=1,label='Predeclared gate');ax.set(title=title,xlabel='Actual optimizer updates');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);axes[1,2].plot([int(r['step']) for r in history],[float(r['fixed_objective_m2']) for r in history],color='#2563eb');axes[1,2].set(title='Fixed-model mean robust objective (m squared)',xlabel='Actual optimizer updates');axes[1,2].grid(alpha=.2);fig.savefig(REPORT/'fit_trajectories.png',dpi=170);plt.close(fig)
 # Whole-field location and depth views; no point-by-point geographic audit.
 from matplotlib.patches import Patch
 extent=[-11.36,-3.66,-42.44,-32.32];cm=ListedColormap(['white','#15803d','#2563eb','#f59e0b']);cm.set_bad('#e5e7eb');depthcm=plt.get_cmap('Blues').copy();depthcm.set_bad('#e5e7eb');fig=plt.figure(figsize=(11,8),layout='constrained');gs=fig.add_gridspec(3,4,height_ratios=[1,1,.09])
 for i,cid in enumerate(ids):
  m=masks[i]&(ward>0);wet=y[i]>.1
  for col in range(4):
   ax=fig.add_subplot(gs[i,col])
   if col in [0,1]:
    pp=fields['endpoint'][i]>.1 if col==1 else wet;v=np.zeros(m.shape,'f4');v[wet&pp]=1;v[wet&~pp]=2;v[~wet&pp]=3;v[~m]=np.nan;im=ax.imshow(v,origin='upper',extent=extent,cmap=cm,vmin=0,vmax=3,interpolation='nearest');title='Reference wet support' if col==0 else 'Endpoint extent errors'
   else:
    v=y[i] if col==2 else fields['endpoint'][i];im=ax.imshow(np.where(m,v,np.nan),origin='upper',extent=extent,cmap=depthcm,vmin=0,vmax=1,interpolation='nearest');title='Reference depth (m)' if col==2 else 'Endpoint depth (m)';fig.colorbar(im,ax=ax,fraction=.045,pad=.02,extend='max')
   ax.set_title(cid+' | '+title,fontsize=8);ax.tick_params(labelsize=7);ax.set_xlabel('Easting (km)',fontsize=8)
 legend=fig.add_subplot(gs[2,:]);legend.axis('off');legend.legend(handles=[Patch(color='#15803d',label='Reference / correctly wet'),Patch(color='#2563eb',label='Missed wet'),Patch(color='#f59e0b',label='False wet'),Patch(color='#e5e7eb',label='Excluded')],loc='center',ncol=4,fontsize=9);fig.savefig(REPORT/'whole_ward_fit.png',dpi=170);plt.close(fig)
 def show(r):return {'case':r['case_id'],'point':r['point'],'scope':r['scope'],'exposure':r.get('parent_exposures'),'iou':fmt(r.get('iou_0.1')),'precision':fmt(r.get('precision_0.1'),100,1),'recall':fmt(r.get('recall_0.1'),100,1),'wet':fmt(r.get('wet_mae_m'),100,2),'deep':fmt(r.get('deep_mae_m'),100,2),'FP':fmt(r.get('false_km2_0.1')),'FN':fmt(r.get('missed_km2_0.1')),'AUC':fmt(r.get('AUC'))}
 cols=[('case','父事件'),('point','检查点'),('scope','区域'),('exposure','每父事件曝光'),('iou','IoU'),('precision','精确率 %'),('recall','召回 %'),('wet','湿区 MAE cm'),('deep','深水 MAE cm'),('FP','误报 km²'),('FN','漏报 km²'),('AUC','排序 AUC')]
 title='达到预先规定的拟合标准' if done['fit_demonstrated'] else '用完预先预算，尚未展示充分拟合'
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Two-parent full-domain fitting diagnostic</title><style>body{font:15px/1.75 system-ui,sans-serif;max-width:1200px;margin:30px auto;padding:0 22px;color:#172033;background:#fafbfd}h1{font-size:27px}h2{font-size:21px;margin-top:32px}.box{padding:18px;background:#edf5f2;border-left:5px solid #0f766e}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{border:1px solid #d9e0e8;padding:8px;text-align:left;vertical-align:top}th{background:#edf2f7}img{width:100%;height:auto}a{color:#1762af}</style><h1>两父事件、三区全域：小样本拟合检查</h1>'''
 body+=f'<div class="box"><b>{title}。</b>只做一次联合拟合，实际 {done["actual_optimizer_updates"]} 次更新；train-01 和 train-04 各曝光 {done["actual_cycles"]} 次。终点通过 {gate_pass}/{gate_count} 项适用的父事件×区域×指标门槛，连续全通过次数为 {done["final_consecutive_passes"]}。<br>原 12＋4 划分、质量掩膜、共同未训练参数与原归一化保持不变。没有扩样、没有新水动力模拟，没有使用验证响应选模或评分，封存测试未打开。</div>'
 body+='<h2>先固定规则，再执行</h2><p>从原目录的训练事件中，按目录顺序选择第一个早峰 train-01／P0001 和第一个多峰 train-04／P0004，不按拟合结果挑选。两事件的雨量、历时和水位也不同，不能将差异归因于雨型。使用一个共享条件 U-Net，原 D 的四次池化与分层稳健损失，学习率 0.0003；没有额外误报惩罚、热启动、新输入或推理裁剪。</p><p>上限为 480 更新＝240 个双父事件循环，每 20 更新检查一次，另保留初始检查。预算、门槛及顺序在第一次神经网络前向与更新前登记。成功要求：两个父事件在三区整体和每个区分别满足 IoU≥0.85、precision/recall≥0.90、湿区 MAE≤5 cm、有深水支持处 MAE≤20 cm，并连续两次检查全通过。单次过关不算成功；预算耗尽不自动延长，数值异常即停并保留状态。这些是技术性拟合门槛，不是真实洪水精度或正式代理模型准入。</p>'
 body+='<h2>实际终点：两个父事件分别报告</h2>'+table([show(r) for r in endrows],cols)+'<p>所有指标基于原目标 own-exact 掩膜与可用地面，使用 Ground Area 加权；不是先前检索来源交集上的 paired-exact 支持。未知、区间、隔离单元仍被排除，没有缩小地理范围或只挑湿区训练。全域 1012×770 栅格仍包含必要跨区贡献区域；三区门槛只应用于原合格开放地面，非行政区全部土地面积。</p>'
 body+='<h2>各区是否同时达到门槛</h2>'+table([show(r) for r in events if r['point']=='endpoint' and r['scope'] in ['Chiyoda','Chuo','Minato']],cols)
 body+=table([{'case':r['case_id'],'scope':r['scope'],'failed':', '.join(r['failed_metrics']) or 'None'} for r in failed],[('case','父事件'),('scope','门槛区域'),('failed','未满足指标')])
 body+='<img src="fit_trajectories.png" alt="Training-only full-field fitting trajectories"><h2>初始、终点与诊断最佳</h2>'+table([show(r) for r in events if r['scope']=='three_wards'],cols)
 body+='<p>诊断最佳按最差父事件／区域 IoU 优先，再看最差湿区 MAE、最差总体 MAE和更早步数；它不决定是否成功，成功只按连续两次完整门槛确认。初始状态的 precision 若无预测湿区则为 NA，不能补成完美值；有参考湿区时这会阻止过关。</p><img src="whole_ward_fit.png" alt="Whole-three-ward reference and fitted location and depth"><p>深度图统一显示 0–1 m，超过 1 m 的颜色饱和仅用于显示；所有误差指标均保留原深水值，没有截断训练或标签。图上是各单元四小时峰值，不是某一时刻积水图或真实排水路径。</p>'
 body+='<h2>和原 12 父事件训练结果的关系</h2>'+table([show(r) for r in historical if r['scope']=='three_wards'],cols)
 body+='<p>历史 D 在同两父事件、同 own-exact 支持上的指标只作背景。原 D 各父事件曝光 40 次，此诊断最高 240 次，而且训练事件集合不同；即使总更新上限相同，也不能将差值作为新方法优于旧方法或泛化改善的证据。</p><h2>这一步能说明什么</h2>'
 if done['fit_demonstrated']:body+='<p>在固定预算内，当前输入与配方能够共同拟合这两个已知参考场；这支持继续检查多事件条件编码和样本覆盖的价值。但正式 12＋4 结果仍按原报告保留，不能把此诊断模型作为合格代理模型或独立验证结果。</p>'
 else:body+='<p>在相同网络与当前损失、优化配置下，即使把曝光集中到两个事件，固定预算内仍未达到预先的空间与水深拟合标准。不能把当前问题仅解释为训练事件太少或验证泛化差；当前输入表示、损失与优化仍需要定位。失败不证明网络天生不能拟合、地形必然错误或标签错位，也不授权自动增加步数或继续调参。</p>'
 body+=f'<p>所有 {len(steps)} 次更新的梯度均有限：{summary["gradient_finite_all_steps"]}；裁剪触发 {summary["clipping_count"]} 次。固定联合目标从 {summary["initial_fixed_objective_m2"]:.5f} 到 {summary["final_fixed_objective_m2"]:.5f} m²。目标下降与空间门槛通过分别报告，不能用前者代替后者。数值、掩膜和参数状态的独立核验另存。</p><ul>'
 for file,label in [('summary.json','结论与终点门槛'),('event_metrics.csv','两个父事件和各区完整指标'),('spatial_fit_metrics.csv','训练侧空间判别'),('history.csv','预定检查点轨迹'),('registration.json','更新前登记凭据'),('delivery_validation.json','独立核验')]:body+=f'<li><a href="{file}">{label}</a></li>'
 body+='</ul><p>来源完整性检查可对既有文件计算哈希，但本次优化、选模和新评分只使用两个训练响应。没有扩大训练／验证库，没有检查封存测试响应，Section 8 的独立验证及主动措施输出仍待完成。</p></html>'
 (REPORT/'review.html').write_text(body);print(json.dumps({'fit_demonstrated':done['fit_demonstrated'],'stop_reason':done['stop_reason'],'gates_passed':gate_pass,'gate_count':gate_count,'endpoint':[show(r) for r in endrows]},indent=2))
if __name__=='__main__':main()
