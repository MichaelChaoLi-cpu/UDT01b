"""Exploratory failure-diagnosis report from frozen checkpoints; no fitting."""
from pathlib import Path
import csv,json,html,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_three_ward_unet_exact_pilot import read,write
NAME='three_ward_unet_shallow_diagnosis_v1';REPORT=ROOT/'data/exp/estimation'/NAME
SOURCE_REPORT=ROOT/'data/exp/estimation/three_ward_unet_exact_pilot_v1'

def rows(name):return list(csv.DictReader((REPORT/name).open()))
def f(v,scale=1,n=3):return 'NA' if v is None or v=='' else f'{float(v)*scale:.{n}f}'
def table(items,columns):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(b)+'</th>' for a,b in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(a,'')))+'</td>' for a,b in columns)+'</tr>' for r in items)+'</tbody></table></div>'
def main():
 s=read(REPORT/'summary.json');metrics=rows('macro_metrics.csv');st=rows('loss_strata_macro.csv');gr=rows('gradient_probes.csv');bias=rows('bias_gradient_probes.csv');op=read(REPORT/'optimizer_state.json');validation=read(REPORT/'validation.json');assert validation['all_checks_passed']
 cn=['seed_reconstructed_initial','best_epoch15','last_epoch35'];names=['Reconstructed initial','Saved epoch 15','Saved epoch 35'];colors=['#94a3b8','#2563eb','#f97316'];label=['<=0.01','0.01-0.10','0.10-0.30','0.30-1.00','>1.00']
 rr=[r for r in st if r['checkpoint']=='best_epoch15' and r['split']=='train'];tot=sum(float(r['mse_contribution_m2']) for r in rr)
 fig,axs=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
 for ax,vals,title in [(axs[0],[float(r['area_fraction'])*100 for r in rr],'Training exact-support area'),(axs[1],[float(r['mse_contribution_m2'])/tot*100 for r in rr],'Contribution to mean training MSE')]:
  bars=ax.bar(label,vals,color=['#94a3b8','#cbd5e1','#60a5fa','#2563eb','#1e3a8a']);ax.bar_label(bars,fmt='%.2f%%',fontsize=9,padding=3);ax.set(xlabel='Reference peak mean depth (m)',ylabel='Percent',title=title,ylim=(0,max(vals)*1.2));ax.grid(axis='y',alpha=.2)
 fig.savefig(REPORT/'training_loss_support.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(11,6.4),constrained_layout=True)
 for row,split in enumerate(['train','validation']):
  data=[next(r for r in metrics if r['checkpoint']==c and r['split']==split and r['scope']=='three_wards') for c in cn]
  for col,(key,title,scale) in enumerate([('mae_m','All-exact MAE (cm)',100),('wet_mae_m','Reference-wet MAE (cm)',100),('iou_0.1','Wet IoU (>0.10m)',1)]):
   ax=axs[row,col];vals=[float(r[key])*scale for r in data];b=ax.bar(range(3),vals,color=colors);ax.bar_label(b,fmt='%.3f',fontsize=8,padding=3);ax.set_xticks(range(3),['Initial*','Epoch15','Epoch35']);ax.set(title=split.title()+': '+title,ylim=(0,max(vals)*1.25));ax.grid(axis='y',alpha=.2)
 fig.savefig(REPORT/'saved_checkpoint_comparison.png',dpi=170);plt.close(fig)
 history=list(csv.DictReader((SOURCE_REPORT/'training_history.csv').open()));fig,axs=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
 axs[0].plot([int(r['epoch']) for r in history],[float(r['train_mse_m2']) for r in history],label='Mean of losses during 12 sequential updates',color='#475569')
 fixed=[next(r for r in metrics if r['checkpoint']==c and r['split']=='train' and r['scope']=='domain') for c in cn]
 axs[0].scatter([0,15,35],[float(r['mse_m2']) for r in fixed],c=colors,s=50,label='Frozen-model rescore (initial reconstructed)',zorder=5);axs[0].set(xlabel='Epoch',ylabel='Training MSE (m squared)',title='Online loss is not checkpoint fit');axs[0].legend(fontsize=7)
 checks=[r for r in history if r['validation_mae_m']];axs[1].plot([int(r['epoch']) for r in checks],[float(r['validation_mae_m'])*100 for r in checks],marker='o',color='#0f766e');axs[1].axvline(15,color='#2563eb',ls='--');axs[1].set(xlabel='Epoch',ylabel='Validation all-exact MAE (cm)',title='Historical checkpoint criterion')
 for ax in axs:ax.grid(alpha=.2)
 fig.savefig(REPORT/'optimization_clock.png',dpi=170);plt.close(fig)
 # Compact evidence tables, all development data; never relabel a selected model.
 comp=[]
 for split in ['train','validation']:
  for c in cn:
   r=next(r for r in metrics if r['checkpoint']==c and r['split']==split and r['scope']=='three_wards')
   comp.append({'split':split,'checkpoint':c,'mae':f(r['mae_m'],100),'mse':f(r['mse_m2'],1,5),'wet':f(r['wet_mae_m'],100),'iou':f(r['iou_0.1'],1,5),'recall':f(r['recall_0.1'],100),'false':f(r['false_km2_0.1']),'missed':f(r['missed_km2_0.1']),'depth':f(r['prediction_on_reference_wet_q50_m'],100)})
 strat=[{'stratum':label[j],'area':f(r['area_fraction'],100),'objective':f(float(r['mse_contribution_m2'])/tot,100),'parent':f(r['fraction_of_mse'],100),'mae':f(r['within_stratum_mae_m'],100)} for j,r in enumerate(rr)]
 grad=[]
 for cname in cn[1:]:
  for module in ['condition','enc.0','enc.4','dec.0','head']:
   q=[float(r['gradient_l2']) for r in gr if r['checkpoint']==cname and r['module']==module];grad.append({'checkpoint':cname,'module':module,'min':f(min(q),1,7),'median':f(np.median(q),1,7),'max':f(max(q),1,7)})
 bp=[r for r in bias if r['checkpoint']=='best_epoch15' and r['split']=='train'];derivative=np.mean([float(r['softplus_derivative_on_wet_mean']) for r in bp]);conflict=s['best_train_bias_objective_conflict_count'];best=s['best_metrics']['validation'];last=s['last_metrics']['validation'];train=s['best_metrics']['train'];lasttrain=s['last_metrics']['train']
 findings={'scope':'diagnosis only, no optimizer or parameter updates','objective_loss_shares':{'wet_above_0_1m':s['best_train_objective_MSE_share_above0_1m'],'deep_above_1m':s['best_train_objective_MSE_share_above1m']},'mean_within_parent_loss_shares':{'wet':s['best_train_MSE_share_above0_1m'],'deep':s['best_train_MSE_share_above1m']},'MSE_MAE_local_bias_conflict_training_events':conflict,'softplus_wet_derivative_mean_best':derivative,'training_online_loss_reduction_fraction':1-s['original_online_train_MSE_last']/s['original_online_train_MSE_first'],'checkpoint_15_to_35_validation':{k:{'best15':best[k],'last35':last[k]} for k in ['mae_m','wet_mae_m','iou_0.1','recall_0.1','false_km2_0.1']},'known_findings':['Reference wet support is sparse, but its squared-error contribution is dominant; this is not dry-pixel domination of the MSE value.','Very deep tail occupies little area and dominates MSE; squared depth loss and whole-area MAE do not directly optimize wet extent.','Saved epoch35 improves wet recall and IoU but worsens overall MAE and false wet area; swapping checkpoints alone does not solve emulation.','Frozen training rescoring is poor; the observed failure is not confined to held-out parents.','Parameter changes, finite Adam moments and nonzero current gradients argue against an entirely disconnected or never-updated model.','Per-step loss dynamics, clipping history and convergence are not recoverable from the saved telemetry.'],'unproven_explanations':['Specific causal contributions of rare deep cells, learning rate, conditioning instability, model capacity, Softplus initialization, sample diversity or fit duration.'],'proposals_not_applied':['Consider balanced evaluation of ordinary wet, nonwet false positives and deep-tail error before selecting a new loss; do not blindly multiply all wet MSE weights.','Consider explicit wet-extent/false-positive checks alongside all-ground MAE for future checkpoint selection, fixed using development data.','Before another fit, export fixed-checkpoint train/validation scores, stratum losses, gradient/clipping/update norms and periodic checkpoints; isolate optimization changes in a separately authorized bounded comparison.','Do not enlarge the hydraulic library or open the sealed test solely to compensate for failure to fit the current training events.']}
 write(REPORT/'findings.json',findings)
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>U-Net shallow prediction diagnosis</title><style>body{font:15px/1.7 system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 22px;background:#fafbfd;color:#172033}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.box{padding:18px;border-left:5px solid #b45309;background:#fff5e8}.note{color:#566273}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{border:1px solid #d9e0e8;padding:8px;text-align:left}th{background:#edf2f7}img{width:100%;height:auto}li{margin:9px 0}a{color:#1762af}</style><h1>首轮 U-Net 预测偏浅与湿区漏报：冻结检查点诊断</h1>'''
 body+=f'<div class="box">结论：损失目标、检查点指标与积水范围任务存在错位，同时这轮优化尚未稳定拟合训练事件。不能简单说是“干地淹没了 MSE”，也不能用更换末轮模型解决。<br>仅检查原 12 个训练与 4 个验证父事件；新增拟合 0 次、优化器更新 0 步。原质量掩膜和划分未改，封存测试未打开。研究决定：MILI-D-20261016-003。</div>'
 body+='<h2>1. 损失权重：湿区稀少，深尾支配平方误差</h2>'
 body+=f'<p>按原逐父事件归一化的 Ground Area 权重，训练标签中水深 >0.10 m 的面积占 {s["training_distribution"]["training_wet_area_fraction_event_macro"]*100:.2f}%，>1 m 仅占 {s["best_train_area_share_above1m"]*100:.3f}%。最佳检查点上，湿区贡献训练 MSE 约 {s["best_train_objective_MSE_share_above0_1m"]*100:.2f}%，其中 >1 m 的深尾贡献约 {s["best_train_objective_MSE_share_above1m"]*100:.2f}%。这两项是分层平均损失之和除以总平均损失；逐事件内部占比再平均分别为 {s["best_train_MSE_share_above0_1m"]*100:.2f}% 和 {s["best_train_MSE_share_above1m"]*100:.2f}%，两种聚合的分母不同。</p>'
 body+='<img src="training_loss_support.png" alt="Training support area versus MSE contribution">'+table(strat,[('stratum','参考峰值水深 (m)'),('area','有效面积 (%)'),('objective','训练目标 MSE 贡献 (%)'),('parent','事件内贡献再平均 (%)'),('mae','层内 MAE (cm)')])
 body+='<p>原训练只有开放地面面积权重和父事件等权更新，没有额外的湿区/干地区分类权重。平方误差对深尾敏感，却不直接奖励 >0.10 m 积水边界识别。继续无差别提高湿区 MSE 权重，未必能解决普通湿区漏报；这只是待比较的设计选择，当前没有修改损失。</p>'
 body+=f'<p>训练标签的事件平衡面积加权中位数约 {s["training_distribution"]["event_balanced_depth_quantiles_m"]["q50"]*100:.3f} cm，均值约 {s["training_distribution"]["event_balanced_depth_mean_m"]*100:.3f} cm，表明整体绝对误差可以被大量浅水支持影响。对最佳检查点作“所有输出 logit 同增一个微小偏置”的解析导数检查，{conflict}/12 个训练事件的 MSE 希望提高输出，而 MAE 希望降低输出。这证明了该局部方向的目标冲突，不能推广为所有网络参数方向的因果解释。</p>'
 body+='<h2>2. 优化过程：并非完全没有更新，也没有稳定拟合</h2>'
 body+=f'<p>原拟合完成 35 轮、420 次更新；Adam 的 {op["parameter_state_count"]} 个参数状态步数均为 420，动量有限，条件编码器和卷积权重相对种子重建初始化均发生变化。两个已保存检查点的训练梯度非零且有限；当前 24 次只读反向检查的总范数范围约 {s["gradient_probe_total_norm_range"][0]:.6f}–{s["gradient_probe_total_norm_range"][1]:.3f}，没有超过原裁剪阈值 10。这不能证明历史上没有裁剪。</p>'
 body+=f'<p>最佳检查点的 Softplus 在参考湿区的平均导数约 {derivative:.4f}，会衰减输出 logit 的梯度，但当前梯度仍存在；不能据此判定激活函数导致“死网络”。对训练集本身，最佳检查点的湿区召回率只有 {train["recall_0.1"]*100:.2f}%，故失败不能仅归结为验证父事件数量或泛化问题。</p>'
 body+='<img src="optimization_clock.png" alt="Sequential online training loss versus frozen checkpoint scores">'
 body+=f'<p>日志的每轮训练 MSE 是 12 次参数不断变化时的更新前损失平均，并非该轮末固定模型的训练分数。首末在线均值只下降约 {findings["training_online_loss_reduction_fraction"]*100:.1f}%。用原标签重算，三区训练 MSE 从第 15 轮的 {train["mse_m2"]:.5f} 升到第 35 轮的 {lasttrain["mse_m2"]:.5f} m²；末轮某些事件明显过预测。在线均值下降不能作为“训练集已拟合好”或“已收敛”的证据。具体是学习率、顺序更新、条件化或容量/预算造成，当前不能隔离。</p>'
 body+=table(grad,[('checkpoint','检查点'),('module','模块'),('min','梯度范数最小'),('median','中位'),('max','最大')])
 body+='<h2>3. 检查点选择：总体 MAE 与范围识别存在取舍</h2><p>原规则只用全域原精确支持的验证 MAE 选模，不检查是否超过无积水/检索，也没有湿区召回或误报约束。下面为三区原精确支持的父事件宏平均，保留原 >0.10 m 阈值。与上轮检索配对表的共同支持略有不同，不能把不同支持上的微小差异当作模型改善。</p>'
 body+=table(comp,[('split','划分'),('checkpoint','状态'),('mae','MAE (cm)'),('mse','MSE (m²)'),('wet','湿区 MAE (cm)'),('iou','IoU'),('recall','湿区召回 (%)'),('false','误报 (km²)'),('missed','漏报 (km²)'),('depth','湿区预测中位数 (cm)')])
 body+='<img src="saved_checkpoint_comparison.png" alt="Saved checkpoint performance on training and validation">'
 body+=f'<p>验证 IoU 从 {best["iou_0.1"]:.5f} 提升到 {last["iou_0.1"]:.5f}，召回从 {best["recall_0.1"]*100:.3f}% 提升到 {last["recall_0.1"]*100:.2f}%；但误报面积从 {best["false_km2_0.1"]:.6f} 增至 {last["false_km2_0.1"]:.3f} km²，整体 MAE 也由 {best["mae_m"]*100:.3f} 增至 {last["mae_m"]*100:.3f} cm。末轮不是可直接替换的合格模型，其 IoU 仍远低于原配对检索约 0.648。没有根据本次诊断另选模型、改阈值或校准输出。</p>'
 body+='<h2>已知与尚不能判断的部分</h2><ul><li>已核验：标签、质量掩膜、归一化、原划分、原曲线与检查点均与冻结交付一致；最佳检查点重现原预测；模型确实更新过，当前反向信号存在。</li><li>仅实际保留了第 15 轮与第 35 轮；图中的 Initial 是按种子和冻结构造函数重建的未训练参考，未经原始初始检查点核对，不能当作历史遥测。</li><li>没有历史逐步梯度/裁剪/更新范数、逐轮固定训练分数或湿区指标，因此无法认定其他轮次的最优 IoU、历史裁剪率或真正收敛时点。</li><li>四个验证父事件已用于选择和诊断；不能把事后指标或末轮比较表述为独立测试验证。没有对真实洪水、动态路径或主动措施作新结论。</li></ul>'
 body+='<h2>后续建议（均未执行）</h2><ol><li>先定义能同时揭示普通湿区、深尾误差和干地误报的开发指标；下一轮损失对照可考虑分层或稳健深度损失，避免直接把所有湿区平方误差放大。</li><li>未来检查点选择同时约束积水范围与误报，并与两种便宜基线对照；具体权重、阈值和选模规则须在下一轮拟合前固定，本次不改。</li><li>下一轮拟合前先补足固定模型训练/验证分数、分层损失、梯度与裁剪/更新统计及周期检查点，再用有限单因素对照区分优化问题。先确认现有训练事件能拟合，再判断是否需要扩充父事件。</li></ol>'
 body+='<h2>可复现文件</h2><p>'+ ' · '.join('<a href="'+x+'">'+x+'</a>' for x in ['findings.json','summary.json','event_metrics.csv','macro_metrics.csv','loss_strata.csv','loss_strata_macro.csv','gradient_probes.csv','bias_gradient_probes.csv','parameter_changes.csv','training_distribution.json','optimizer_state.json','validation.json','delivery_validation.json','anasop_update.json'])+'</p><p class="note">所有反向检查只计算梯度，不构造优化器、不执行参数更新。原训练交付保留冻结状态；本次诊断另存输出。物理模型与数值标签的既有局限仍然适用。</p></html>'
 (REPORT/'review.html').write_text(body)
 (REPORT/'README.md').write_text('# Read-only shallow-prediction diagnosis\n\nRun src/analyses/diagnose_three_ward_unet_shallow.py then src/analyses/review_three_ward_unet_shallow.py with the project Python. The analysis reads only the authorized original 12/4 labels and the two saved checkpoints; the initial model is seed-reconstructed. No optimizer is constructed or stepped. Existing successful diagnosis refuses silent overwrite.\n\nAll source pilot files remain frozen. The first diagnosis attempt stopped during a GPU float64 norm reduction before any parameter updates; the corrected code moves norm reduction to CPU first. This affected diagnostic bookkeeping only. The preserved attempt log is not an additional fit.\n\nScores use original target exact support. The original retrieval comparison uses paired source-and-target exact support; do not mix supports in a claimed improvement. Loss value shares are provided both as a mean of within-event fractions and as a ratio of event-mean contributions, with their denominators explicit.\n')
 print(json.dumps({'report':str(REPORT/'review.html'),'diagnostic_checks':len(validation['checks']),'objective_wet_share':s['best_train_objective_MSE_share_above0_1m'],'objective_deep_share':s['best_train_objective_MSE_share_above1m']}))
if __name__=='__main__':main()
