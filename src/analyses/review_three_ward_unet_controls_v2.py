"""Whole-ward development comparison of four completed paired fits."""
from pathlib import Path
import sys,csv,json,html,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_three_ward_unet_exact_pilot import read,write,arrays,load_label
NAME='three_ward_unet_controls_v2';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME

def csvrows(path):return list(csv.DictReader(Path(path).open()))
def writecsv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def f(v,scale=1,n=3):return 'NA' if v is None or v=='' else f'{float(v)*scale:.{n}f}'
def table(items,columns):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for k,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k,label in columns)+'</tr>' for r in items)+'</tbody></table></div>'
def main():
 done=read(REPORT/'completion.json');assert done['status']=='complete' and done['actual_fits']==4;arms=done['arms'];events=[];macros=[];summaryrows=[];optimization=[]
 for arm in arms:
  aid=arm['arm'];selected=read(REPORT/aid/'selected_metrics.json');events += [r|{'arm':aid,'selected_epoch':arm['best_epoch']} for r in selected['event_metrics']]
  macros += [r|{'arm':aid,'selected_epoch':arm['best_epoch']} for r in selected['parent_macro_metrics']]
  metric=arm['validation_three_ward_paired_macro']['values'];summaryrows.append({'arm':aid,'loss':arm['loss'],'learning_rate':arm['learning_rate'],'selected_epoch':arm['best_epoch'],'all_four_parents_pass_baseline_guard':arm['baseline_all_parents_pass'],**metric})
  h=csvrows(REPORT/aid/'history.csv');steps=[json.loads(line) for line in (REPORT/aid/'optimizer_steps.jsonl').read_text().splitlines() if line];fixed=np.array([float(r['fixed_train_loss_m2']) for r in h]);gradient=np.array([r['gradient_pre_clip']['total_norm'] for r in steps]);update=np.array([math.sqrt(sum(q['update_norm']**2 for q in r['parameter_change'].values())) for r in steps]);tail=fixed[-10:]
  optimization.append({'arm':aid,'loss':arm['loss'],'learning_rate':arm['learning_rate'],'optimizer_updates':len(steps),'clipping_count':sum(r['clipping_required'] for r in steps),'gradient_median':float(np.median(gradient)),'gradient_p95':float(np.quantile(gradient,.95)),'gradient_max':float(gradient.max()),'update_norm_median':float(np.median(update)),'update_norm_p95':float(np.quantile(update,.95)),'fixed_objective_epoch1':float(fixed[0]),'fixed_objective_epoch40':float(fixed[-1]),'fixed_objective_minimum':float(fixed.min()),'fixed_objective_final_over_epoch1':float(fixed[-1]/fixed[0]),'fixed_objective_final_over_minimum':float(fixed[-1]/fixed.min()),'fixed_objective_last10_cv':float(np.std(tail)/np.mean(tail)),'epochs_passing_all_parent_guard':sum(r['baseline_all_parents_pass']=='True' for r in h),'fixed_objective_increase_count':int(np.sum(np.diff(fixed)>0)),'finite_steps':all(r['gradient_pre_clip']['finite'] and r['gradient_post_clip']['finite'] for r in steps),'training_seconds_including_observer':arm['training_seconds_with_observer'],'evaluation_seconds':arm['evaluation_seconds'],'checkpoint_io_seconds':arm['checkpoint_io_seconds']})
 # Distinguish fixed mixture coefficients from actual realized loss shares.
 contribution=[]
 for arm in arms:
  aid=arm['arm'];sel=read(REPORT/aid/'selected_metrics.json');fin=json.loads((REPORT/aid/'epoch_040.jsonl').read_text().strip())
  for point,frame in [('selected_checkpoint',sel),('epoch40',fin)]:
   rr=[r for r in frame['event_metrics'] if r['split']=='train' and r['scope']=='domain' and r['support']=='own_exact'];total=float(np.mean([r['fixed_model_loss_m2'] for r in rr]))
   for stratum in ['nonwet','ordinary_wet','deep']:
    shares=[r['fixed_loss_strata'][stratum] for r in rr];mean_contribution=float(np.mean([r['contribution_m2'] for r in shares]));contribution.append({'arm':aid,'point':point,'stratum':stratum,'mean_area_fraction':float(np.mean([r['area_fraction'] for r in shares])),'mean_mixture_coefficient':float(np.mean([r['mixture_weight'] for r in shares])),'mean_loss_contribution_m2':mean_contribution,'fraction_of_mean_objective':mean_contribution/total,'mean_within_parent_fraction':float(np.mean([r['fixed_loss_strata'][stratum]['contribution_m2']/r['fixed_model_loss_m2'] for r in rr]))})
 writecsv(REPORT/'realized_loss_contributions.csv',contribution)
 # Flatten only scalar metrics; detailed fixed-loss strata remain in selected JSON.
 writecsv(REPORT/'selected_event_metrics.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in events])
 writecsv(REPORT/'selected_macro_metrics.csv',[{'arm':r['arm'],'selected_epoch':r['selected_epoch'],'split':r['split'],'scope':r['scope'],'support':r['support'],'parents':r['parents'],**r['values'],**{'defined_'+k:v for k,v in r['defined_parents'].items()}} for r in macros])
 writecsv(REPORT/'validation_summary.csv',summaryrows);writecsv(REPORT/'optimization_summary.csv',optimization)
 aids={a['arm'][0]:a['arm'] for a in arms};contrast_rows=[];cols=['balanced_extent_error','mae_m','wet_mae_m','deep_mae_m','iou_0.1','recall_0.1','fpr_0.1','missed_km2_0.1','false_km2_0.1']
 specs=[('loss_original_lr','B','A'),('loss_lower_lr','D','C'),('lr_mse','C','A'),('lr_stratified','D','B')]
 selected={r['arm']:r for r in summaryrows}
 for contrast,left,right in specs:
  aa=aids[left];bb=aids[right]
  for scope in ['three_wards','domain','Chiyoda','Chuo','Minato','cross_ward']:
   for support in ['paired_exact','own_exact']:
    leftrows={r['case_id']:r for r in events if r['arm']==aa and r['split']=='validation' and r['scope']==scope and r['support']==support};rightrows={r['case_id']:r for r in events if r['arm']==bb and r['split']=='validation' and r['scope']==scope and r['support']==support}
    for cid,lr in leftrows.items():
     rr=rightrows[cid];assert lr['support_sha256']==rr['support_sha256'];row={'contrast':contrast,'left_arm':aa,'right_arm':bb,'case_id':cid,'scope':scope,'support':support}
     for k in cols:row['delta_'+k]=lr.get(k)-rr.get(k) if lr.get(k) is not None and rr.get(k) is not None else None
     contrast_rows.append(row)
 writecsv(REPORT/'paired_event_contrasts.csv',contrast_rows)
 contrast_macro=[]
 for name,l,r in specs:
  for scope in ['three_wards','domain','Chiyoda','Chuo','Minato','cross_ward']:
   for support in ['paired_exact','own_exact']:
    rr=[x for x in contrast_rows if x['contrast']==name and x['scope']==scope and x['support']==support];d={}
    for k in cols:
     vv=[v['delta_'+k] for v in rr if v['delta_'+k] is not None];d['delta_'+k]=float(np.mean(vv)) if vv else None;d['defined_'+k]=len(vv)
    contrast_macro.append({'contrast':name,'left':l,'right':r,'scope':scope,'support':support,**d})
 writecsv(REPORT/'paired_contrast_macro.csv',contrast_macro)
 # Also compare the same final update budget, independent of checkpoint timing.
 final_metrics=[]
 for arm in arms:
  last=json.loads((REPORT/arm['arm']/'epoch_040.jsonl').read_text().strip())
  mm=next(r for r in last['parent_macro_metrics'] if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact')['values']
  final_metrics.append({'arm':arm['arm'],'epoch':40,**mm})
 writecsv(REPORT/'matched_epoch40_metrics.csv',final_metrics)
 final_by={r['arm'][0]:r for r in final_metrics};matched=[]
 for name,l,r in specs:
  matched.append({'contrast':name,'left':l,'right':r,'epoch':40,**{'delta_'+k:final_by[l][k]-final_by[r][k] if final_by[l].get(k) is not None and final_by[r].get(k) is not None else None for k in cols}})
 writecsv(REPORT/'matched_epoch40_contrasts.csv',matched)
 interaction={}
 for k in cols:
  vv=[selected[aids[a]].get(k) for a in ['A','B','C','D']];interaction[k]=(vv[3]-vv[2])-(vv[1]-vv[0]) if all(v is not None for v in vv) else None
 baselines=read(REPORT/'frozen_baseline_metrics.json');bmacro={name:{k:float(np.mean([r[k] for r in records.values() if r[k] is not None])) if any(r[k] is not None for r in records.values()) else None for k in cols} for name,records in baselines.items()}
 best=min(arms,key=lambda r:tuple(r['selection_key']));summary={'status':'complete','decision':'MILI-D-20261016-005','fits':4,'optimizer_updates':1920,'original_train_parents':12,'original_validation_parents':4,'shared_initialization_preserved':len({r['initialization_fingerprint'] for r in arms})==1,'selected_arm_under_frozen_development_key':best['arm'],'any_arm_passes_all_parent_baseline_guards':any(r['baseline_all_parents_pass'] for r in arms),'arms':summaryrows,'realized_training_loss_contributions':contribution,'same_final_epoch40_contrasts':matched,'baselines_on_same_primary_support':bmacro,'descriptive_interaction_lowerLR_loss_effect_minus_originalLR_loss_effect':interaction,'sealed_opened':False,'limitations':['Each checkpoint and the cross-arm best use the same already exposed four-event development validation; no independent test.','One seed; differences are conditional paired diagnostics without population significance.','Loss factor combines stratification and robust residuals; their individual effects are not isolated.','Fixed peak emulation does not verify dynamic water conservation, surface paths or observed-flood accuracy.','Runtime includes extensive observer and full-field scoring overhead; no hydraulic speedup claim.','Recorded timing segments exclude some selection-export and log overhead; loop residual is reported and initial data loading is excluded.']}
 summary['any_saved_epoch_passes_all_parent_guards']=any(r['epochs_passing_all_parent_guard']>0 for r in optimization)
 summary['recorded_timing']={'execution_loop_elapsed_seconds_after_data_loading':done['current_process_elapsed_seconds'],'instrumented_segments_seconds':sum(a['training_seconds_with_observer']+a['evaluation_seconds']+a['checkpoint_io_seconds'] for a in arms),'unassigned_loop_overhead_seconds':done['current_process_elapsed_seconds']-sum(a['training_seconds_with_observer']+a['evaluation_seconds']+a['checkpoint_io_seconds'] for a in arms)}
 write(REPORT/'summary.json',summary)
 # Full-budget trajectories. Axes use common evaluation metrics across both losses.
 colors={'A':'#64748b','B':'#2563eb','C':'#d97706','D':'#0f766e'};fig,axs=plt.subplots(2,3,figsize=(12,6.8),constrained_layout=True)
 charts=[('validation_balanced_extent_error','Validation balanced extent error',1),('validation_recall','Wet recall (%)',100),('validation_false_km2','False wet area (km squared)',1),('validation_MAE_m','All-exact MAE (cm)',100),('validation_wet_MAE_m','Reference-wet MAE (cm)',100),('validation_IoU','Wet IoU (>0.10m)',1)]
 for ax,(key,title,scale) in zip(axs.ravel(),charts):
  for arm in arms:
   aid=arm['arm'];h=csvrows(REPORT/aid/'history.csv');xx=[int(r['epoch']) for r in h];yy=[float(r[key])*scale for r in h];ax.plot(xx,yy,color=colors[aid[0]],label=aid[0]);idx=arm['best_epoch']-1;ax.scatter(xx[idx],yy[idx],color=colors[aid[0]],s=20)
  ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2)
 axs[0,0].legend(ncol=4);fig.savefig(REPORT/'validation_trajectories.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
 for arm in arms:
  aid=arm['arm'];h=csvrows(REPORT/aid/'history.csv');xx=np.arange(1,41);fixed=np.array([float(r['fixed_train_loss_m2']) for r in h]);axs[0].plot(xx,fixed/fixed[0],label=aid[0],color=colors[aid[0]]);axs[1].plot(xx,[float(r['gradient_max']) for r in h],label=aid[0],color=colors[aid[0]]);axs[2].plot(xx,np.cumsum([int(r['clipping_count']) for r in h]),label=aid[0],color=colors[aid[0]])
 for ax,title in zip(axs,['Fixed training objective / epoch1','Largest step gradient norm each epoch','Cumulative clipping count']):ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2)
 axs[0].legend(ncol=4);axs[1].set_yscale('log');fig.savefig(REPORT/'optimization_stability.png',dpi=170);plt.close(fig)
 # Whole-ward extent errors for every validation parent on exactly paired support.
 from matplotlib.colors import ListedColormap
 from matplotlib.patches import Patch
 source=ROOT/'data/processed/three_ward_unet_exact_pilot_v1';plan=read(ROOT/'data/processed/three_ward_unet_control_plan_v2/protocol.json');zz=arrays(source/'prepared_inputs.npz');record=plan['records'];index={r['case_id']:i for i,r in enumerate(record)};stored={a['arm']:arrays(OUT/a['arm']/'selected_predictions.npz')['Peak Mean Depth'] for a in arms}
 fig=plt.figure(figsize=(13,13),layout='constrained');gs=fig.add_gridspec(5,5,height_ratios=[1,1,1,1,.12]);error_cmap=ListedColormap(['#ffffff','#15803d','#2563eb','#f59e0b']);error_cmap.set_bad('#e5e7eb');ref_cmap=ListedColormap(['#ffffff','#2563eb']);ref_cmap.set_bad('#e5e7eb');extent=[-11.36,-3.66,-42.44,-32.32]
 for row,cid in enumerate(plan['validation_case_ids']):
  i=index[cid];r=record[i];j=index[r['retrieval_source_case_id']];lab=load_label(cid,'validation');other=load_label(record[j]['case_id'],'train');mask=lab['Exact Label Mask']&other['Exact Label Mask']&zz['Eligible Ground Mask']&(zz['Ward Code']>0);truth=lab['Peak Mean Depth Lower'].astype('f4')>.1
  for col in range(5):
   ax=fig.add_subplot(gs[row,col])
   if col==0:field=np.where(mask,truth.astype('f4'),np.nan);cmap=ref_cmap;vmax=1;title='Reference'
   else:
    a=arms[col-1];pred=stored[a['arm']][i]>.1;classes=np.zeros(mask.shape,'f4');classes[truth&pred]=1;classes[truth&~pred]=2;classes[~truth&pred]=3;field=np.where(mask,classes,np.nan);cmap=error_cmap;vmax=3;title=a['arm'][0]+': epoch '+str(a['best_epoch'])
   ax.imshow(field,origin='upper',extent=extent,cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest',rasterized=True);ax.set_title(cid+' | '+title,fontsize=8);ax.tick_params(labelsize=6);ax.set_xlabel('Easting (km)',fontsize=7)
   if col==0:ax.set_ylabel('Northing (km, EPSG:6677)',fontsize=7)
 legend=fig.add_subplot(gs[4,:]);legend.axis('off');legend.legend(handles=[Patch(facecolor='#15803d',label='True wet'),Patch(facecolor='#2563eb',label='Missed wet / reference wet'),Patch(facecolor='#f59e0b',label='False wet'),Patch(facecolor='#ffffff',edgecolor='#777',label='True nonwet'),Patch(facecolor='#e5e7eb',label='Excluded support')],loc='center',ncol=5,fontsize=9)
 fig.savefig(REPORT/'selected_validation_extent_maps.png',dpi=170);plt.close(fig)
 def display(r):return {'arm':r.get('arm',''),'epoch':r.get('selected_epoch',''),'B':f(r['balanced_extent_error']),'mae':f(r['mae_m'],100),'wet':f(r['wet_mae_m'],100),'deep':f(r['deep_mae_m'],100),'iou':f(r['iou_0.1']),'recall':f(r['recall_0.1'],100),'fp':f(r['false_km2_0.1']),'fn':f(r['missed_km2_0.1']),'guard':r.get('all_four_parents_pass_baseline_guard','')}
 baserows=[]
 for name,rr in bmacro.items():baserows.append({'arm':name,'epoch':'—','all_four_parents_pass_baseline_guard':'基线',**rr})
 guard='至少一组通过全部逐父事件基线比较' if summary['any_arm_passes_all_parent_baseline_guards'] else '四组均未通过全部逐父事件基线比较'
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Four-arm surrogate development comparison</title><style>body{font:15px/1.7 system-ui,sans-serif;max-width:1250px;margin:30px auto;padding:0 22px;color:#172033;background:#fafbfd}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.box{padding:18px;background:#edf5f2;border-left:5px solid #0f766e}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{border:1px solid #d9e0e8;padding:8px;text-align:left}th{background:#edf2f7}img{width:100%;height:auto}.note{color:#576476}a{color:#1762af}li{margin:8px 0}</style><h1>三区 U-Net 四组对照：分层稳健损失与学习率</h1>'''
 body+=f'<div class="box">四组各 40 轮、480 次更新，共完成 4 次拟合和 1,920 次更新。共同初始化、原 12 个训练与 4 个验证父事件、质量掩膜和训练集归一化均保留。<br>按冻结开发规则排序最前的是 {best["arm"]}；{guard}。封存测试未打开。此排序使用已暴露的验证集，不构成独立测试结论。</div>'
 loss40={r['contrast']:r for r in matched};op_by={r['arm'][0]:r for r in optimization}
 findings={'matched_epoch40_loss_package':{k:{'recall_change_percentage_points':loss40[k]['delta_recall_0.1']*100,'false_area_change_km2':loss40[k]['delta_false_km2_0.1'],'wet_MAE_change_m':loss40[k]['delta_wet_mae_m']} for k in ['loss_original_lr','loss_lower_lr']},'stratified_lower_lr_epoch40':loss40['lr_stratified'],'last10_fixed_objective_cv':{k:r['fixed_objective_last10_cv'] for k,r in op_by.items()},'clipping_counts':{k:r['clipping_count'] for k,r in op_by.items()},'saved_epochs_passing_all_guards':{k:r['epochs_passing_all_parent_guard'] for k,r in op_by.items()},'selection_time_caveat':'Selected B is epoch1 and selected D epoch36; selected-checkpoint differences include this timing, unlike matched epoch40 contrasts.','claims_conditional_on_one_seed_and_development_sample':True}
 write(REPORT/'findings.json',findings)
 body+='<h2>对照说明了什么</h2><ul><li>在相同第 40 轮预算下，分层损失相对 MSE 的湿区召回分别提高 '+f(loss40['loss_original_lr']['delta_recall_0.1'],100,2)+' 和 '+f(loss40['loss_lower_lr']['delta_recall_0.1'],100,2)+' 个百分点，同时平均误报面积增加 '+f(loss40['loss_original_lr']['delta_false_km2_0.1'])+' 和 '+f(loss40['loss_lower_lr']['delta_false_km2_0.1'])+' km²。漏报改善伴随明显误报代价。</li><li>较低学习率对分层损失的后期训练目标下降和波动有改善；MSE 组的末十轮目标变异反而更大，因此不能说降低学习率普遍稳定。全部 1,920 步均未触发裁剪。</li><li>B 的选中轮是第 1 轮，D 是第 36 轮；所选检查点的误报差异还包含选模时间的影响。同为第 40 轮时，D 比 B 的误报面积仅减少 '+f(-loss40['lr_stratified']['delta_false_km2_0.1'])+' km²，需与整条轨迹一起解释。</li><li>所有保留轮次通过完整基线比较的次数为 '+str(sum(r['epochs_passing_all_parent_guard'] for r in optimization))+'；本轮不能把任何一组直接升级为可靠代理模型。</li></ul>'
 body+='<h2>共同支持上的主要结果</h2><p>A：MSE /0.001；B：分层稳健损失 /0.001；C：MSE /0.0003；D：分层稳健损失 /0.0003。每组按 B、湿区 MAE、总体 MAE、较早轮的固定顺序选择检查点。主表是三区原目标精确掩膜与冻结训练检索来源精确掩膜交集上的四父事件宏平均；湿区阈值严格 >0.10 m。B=0.5×(漏报率+误报率)，越小越好。深水 MAE 对参考 >1m 支持计算，空层不补零。</p>'
 body+=table([display(r) for r in summaryrows+baserows],[('arm','方法'),('epoch','选中轮'),('B','平衡错误 B↓'),('mae','MAE cm↓'),('wet','湿区 MAE cm↓'),('deep','深水 MAE cm↓'),('iou','IoU↑'),('recall','召回 %↑'),('fp','误报 km²↓'),('fn','漏报 km²↓'),('guard','全部父事件过关')])
 body+='<p>“诊断最佳”与“通过基线比较”分别报告。每个验证父事件都需在 B、总体 MAE、召回和误报面积上满足冻结的检索/无积水比较条件；不因平均分改善而默认为可用。完整目标精确支持、各区和跨区贡献区域的结果另存，不能混用支持来宣称改善。</p><img src="validation_trajectories.png" alt="Four-arm validation trajectories; dots mark selected epochs">'
 primary=[r for r in contrast_macro if r['scope']=='three_wards' and r['support']=='paired_exact'];show=[]
 for r in primary:show.append({'contrast':r['contrast'],'pair':r['left']+'−'+r['right'],'B':f(r['delta_balanced_extent_error']),'mae':f(r['delta_mae_m'],100),'wet':f(r['delta_wet_mae_m'],100),'deep':f(r['delta_deep_mae_m'],100),'iou':f(r['delta_iou_0.1']),'recall':f(r['delta_recall_0.1'],100),'fp':f(r['delta_false_km2_0.1']),'fn':f(r['delta_missed_km2_0.1'])})
 body+='<h2>配对差值：损失与学习率</h2><p>所有差值均为左组减右组，各组使用自己的规则选中轮。误差和误报/漏报为负更好，IoU 与召回为正更好；召回差值单位为百分点。四个父事件逐一配对后平均；CSV 保留每个事件，未作像素级显著性检验。损失因子是分层+Huber 的组合，不能拆成两个单独效果。</p>'+table(show,[('contrast','比较'),('pair','差值'),('B','ΔB'),('mae','ΔMAE cm'),('wet','Δ湿区 cm'),('deep','Δ深水 cm'),('iou','ΔIoU'),('recall','Δ召回 pp'),('fp','Δ误报 km²'),('fn','Δ漏报 km²')])
 op=[]
 for r in optimization:op.append({'arm':r['arm'][0],'clip':r['clipping_count'],'g95':f(r['gradient_p95']),'gmax':f(r['gradient_max']),'ratio':f(r['fixed_objective_final_over_epoch1']),'cv':f(r['fixed_objective_last10_cv']),'rise':r['fixed_objective_increase_count'],'update':f(r['update_norm_p95'],1,5),'train':f(r['training_seconds_including_observer']),'eval':f(r['evaluation_seconds'])})
 body+='<h2>同为第 40 轮的对照</h2><p>各组选中轮可能不同。为区分检查点时间与最终预算结果，另报全部组第 40 轮的主指标和差值；不据此重新选模。</p>'+table([display(r) for r in final_metrics],[('arm','组'),('B','B'),('mae','MAE cm'),('wet','湿区 cm'),('iou','IoU'),('recall','召回 %'),('fp','误报 km²')])
 body+='<h2>优化稳定性与实际日志</h2><p>固定模型损失与逐步更新前损失分别记录。两种损失的分层权重和残差形状不同，原始损失值、梯度绝对尺度不能直接当作性能高低。这里看各自固定训练目标的相对变化、末十轮变异系数、上升次数、真实裁剪数和参数更新量；不把 40 轮结束宣称为收敛；低变异也可能只是停留在高误差平台。</p>'+table(op,[('arm','组'),('clip','裁剪次数/480'),('g95','梯度 p95'),('gmax','梯度最大'),('ratio','末轮/首轮固定目标'),('cv','末10轮目标 CV'),('rise','目标上升次数/39'),('update','更新范数 p95'),('train','训练含观察器 s'),('eval','固定评分 s')])+'<img src="optimization_stability.png" alt="Normalized fixed objective, gradient norms and actual clipping histories">'
 body+='<h2>全三区漏报与误报分布</h2><p>每行一个验证父事件；所有组使用相同的固定配对精确支持。模型列绿色为正确湿区，蓝色为漏报，橙色为误报，白色为正确非湿区，灰色为支持外。参考列蓝色表示参考湿区。这里只看全区空间分布，不作逐点核查或动态路径推断。</p><img src="selected_validation_extent_maps.png" alt="Full three-ward validation extent errors">'
 body+='<h2>设定权重与实际损失贡献</h2><p>0.50/0.40/0.10 是层内平均误差的混合系数，各层误差幅度不同，实际损失占比不会自动等于这些系数。下表是训练集各层平均贡献除以平均总目标；它是损失数值分解，不是梯度归因。原 MSE 的混合系数自然等于该层面积占比。</p>'+table([{'arm':r['arm'][0],'point':r['point'],'stratum':r['stratum'],'area':f(r['mean_area_fraction'],100),'coefficient':f(r['mean_mixture_coefficient'],100),'actual':f(r['fraction_of_mean_objective'],100)} for r in contribution],[('arm','组'),('point','检查点'),('stratum','层'),('area','面积 %'),('coefficient','混合系数 %'),('actual','实际目标贡献 %')])
 body+='<h2>逐父事件与各区</h2>'
 ve=[r for r in events if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact'];body+=table([display(r)|{'case':r['case_id']} for r in ve],[('arm','组'),('case','验证父事件'),('B','B'),('mae','MAE cm'),('wet','湿区 cm'),('iou','IoU'),('recall','召回 %'),('fp','误报 km²'),('fn','漏报 km²')])
 body+='<h2>解释边界与保留文件</h2><ul><li>真实优化器步数、共享初始化与顺序、掩膜和各版本绑定可审计；原训练、准备与诊断交付均保持冻结。</li><li>单初始化和四个已暴露验证父事件只能支持这次受限开发比较；不证明总体泛化、事件数量充分性或真实洪水精度。</li><li>只预测四小时峰值地图；没有训练动态路径、守恒代理、图模型、区间监督或主动措施。主动措施仍是后续第五阶段。</li><li>仪器化训练和逐轮全域评分消耗计入单独时钟；部分选中结果压缩导出、日志和设置开销不在三项分段时钟内，另报执行循环余项；初始数据装载不计入执行循环，此处不宣称端到端或水动力加速比。每轮模型与最新优化器/RNG 状态、真实逐步日志和完整 192 行评分帧均保留。</li></ul>'
 files=['summary.json','findings.json','validation_summary.csv','selected_event_metrics.csv','selected_macro_metrics.csv','paired_event_contrasts.csv','paired_contrast_macro.csv','optimization_summary.csv','realized_loss_contributions.csv','matched_epoch40_metrics.csv','matched_epoch40_contrasts.csv','frozen_baseline_metrics.json','completion.json','delivery_validation.json','anasop_update.json']
 body+='<p>'+ ' · '.join('<a href="'+q+'">'+q+'</a>' for q in files)+'</p></html>'; (REPORT/'review.html').write_text(body)
 (REPORT/'README.md').write_text('# Four-arm surrogate comparison\n\nCompleted the fixed two-by-two development experiment: original MSE versus stratified robust loss, each at0.001 or0.0003. Exactly four arms x40 epochs x12 parent updates =1920 steps.\n\nAll arms load the same frozen initialization and epoch order. Original12/4 partitions, exact masks and training-only normalization remain unchanged. No sealed test. Selection follows the prepared paired three-ward balanced-error/depth lexicographic rule; baseline readiness remains separate.\n\nRun the repository execution script once; it refuses a completed experiment in place. Review and delivery-validation scripts operate on retained outputs without fitting. Original preparation and source artifacts remain frozen. All files ending epoch_NNN.jsonl contain real fixed-checkpoint scores and update summaries; optimizer_steps.jsonl contains actual updates, unlike the separate preparation interface fixtures.\n')
 print(json.dumps({'summary':summary['selected_arm_under_frozen_development_key'],'any_pass':summary['any_arm_passes_all_parent_baseline_guards'],'arms':[{k:r[k] for k in ['arm','selected_epoch','balanced_extent_error','mae_m','iou_0.1','recall_0.1','false_km2_0.1']} for r in summaryrows]},indent=2))
if __name__=='__main__':main()
