"""Render diagnostic evidence from existing frozen outputs; no model operations."""
from pathlib import Path
import sys,csv,json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/preprocessing'))
from prepare_three_ward_unet_exact_pilot import read,write
NAME='three_ward_d_false_positive_diagnosis_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME

def rows(name):return list(csv.DictReader((REPORT/name).open()))
def fmt(x,n=3):return 'NA' if x is None else f'{float(x):.{n}f}'
def table(data,cols):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(t)+'</th>' for k,t in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k,t in cols)+'</tr>' for r in data)+'</tbody></table></div>'
def main():
 s=read(REPORT/'summary.json');assert read(REPORT/'validation.json')['all_checks_passed'];t=rows('forty_epoch_trajectories.csv');st=rows('spatial_strata_macro.csv');p=s['primary_metrics'];a=p['train'];b=p['validation'];trade=s['retrospective_checkpoints_not_promoted'];old=read(ROOT/'data/exp/estimation/three_ward_unet_controls_v2/summary.json')
 fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
 charts=[('fixed_domain_objective_m2','Fixed full-domain robust objective',1),('false_km2_0.1','False wet area (km squared)',1),('recall_0.1','Wet recall (%)',100),('balanced_extent_error','Balanced extent error',1),('iou_0.1','Wet IoU',1),('deep_loss_value_share','Deep share of fixed objective (%)',100)]
 for ax,(key,title,scale) in zip(axes.ravel(),charts):
  for split,color in [('train','#0f766e'),('validation','#b45309')]:
   rr=[v for v in t if v['split']==split];x=[int(v['epoch']) for v in rr];y=[float(v[key])*scale for v in rr];ax.plot(x,y,label=split,color=color);ax.scatter([36],[y[35]],s=22,color=color)
  ax.axvline(36,color='#64748b',ls=':',lw=1);ax.set(title=title,xlabel='Existing epoch');ax.grid(alpha=.2)
 axes[0,0].legend();fig.savefig(REPORT/'fit_and_selection_trajectories.png',dpi=170);plt.close(fig)
 maps=np.load(OUT/'diagnostic_spatial_aggregates.npz');ward=maps['Ward Code'];extent=[-11.36,-3.66,-42.44,-32.32];fig,axes=plt.subplots(2,2,figsize=(9,10),constrained_layout=True)
 for row,split in enumerate(['train','validation']):
  for col,(key,title,cmap) in enumerate([('reference_wet_frequency','Reference wet frequency','Blues'),('false_frequency','False wet frequency','YlOrRd')]):
   ax=axes[row,col];cm=plt.get_cmap(cmap).copy();cm.set_bad('#e5e7eb');im=ax.imshow(maps[split+'_'+key],origin='upper',extent=extent,vmin=0,vmax=1,cmap=cm,interpolation='nearest');ax.set(title=split+' | '+title,xlabel='Easting (km)',ylabel='Northing (km, EPSG:6677)');fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
   for code,label in [(1,'Chiyoda'),(2,'Chuo'),(3,'Minato')]:
    rr,cc=np.where((ward==code)&maps[split+'_common_mask']);ax.text(-11.36+(cc.mean()+.5)*.01,-32.32-(rr.mean()+.5)*.01,label,fontsize=8,ha='center',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':2})
 fig.savefig(REPORT/'whole_ward_recurrence.png',dpi=170);plt.close(fig)
 # Fixed bins; charts do not choose or recalibrate a wet threshold.
 specs=[('prediction_depth_m','False prediction depth (m)',True),('reference_nonwet_depth_m','Reference depth at false cells (m)',True),('distance_to_reference_wet_m','Straight-line distance to reference wet (m)',True),('Road Ground Fraction','Road fraction / nonwet FPR (%)',False),('PLATEAU Ground Fraction','PLATEAU fraction / nonwet FPR (%)',False),('Subcell Connectivity Ambiguity','Ambiguity flag / nonwet FPR (%)',False)]
 labelsets={'prediction_depth_m':['(0.1,0.15]','(0.15,0.3]','(0.3,0.5]','(0.5,inf]'],'reference_nonwet_depth_m':['(-inf,0.01]','(0.01,0.05]','(0.05,0.1]'],'distance_to_reference_wet_m':['(0,20]','(20,50]','(50,100]','(100,inf]'],'Road Ground Fraction':['(-inf,0.1]','(0.1,0.5]','(0.5,inf]'],'PLATEAU Ground Fraction':['(-inf,0]','(0,0.5]','(0.5,inf]'],'Subcell Connectivity Ambiguity':['(-inf,0.5]','(0.5,inf]']}
 fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
 for ax,(family,title,share) in zip(axes.ravel(),specs):
  rr={r['stratum']:r for r in st if r['split']=='validation' and r['family']==family};labs=labelsets[family];vals=[float(rr[k]['fraction_of_event_false_area'] if share else rr[k]['fpr'])*100 for k in labs];ax.bar(range(len(labs)),vals,color='#b45309' if share else '#0f766e');ax.set_xticks(range(len(labs)),labs,rotation=25,ha='right',fontsize=8);ax.set(title=title,ylabel='Mean within-parent FP share (%)' if share else 'Mean within-parent nonwet FPR (%)');ax.grid(axis='y',alpha=.2)
 fig.savefig(REPORT/'false_area_strata.png',dpi=170);plt.close(fig)
 def ss(fam,name):return next(r for r in s['validation_spatial_strata'][fam] if r['stratum']==name)
 near=sum(ss('distance_to_reference_wet_m',k)['fraction_of_event_false_area'] for k in ['(0,20]','(20,50]']);dry=ss('reference_nonwet_depth_m','(-inf,0.01]')['fraction_of_event_false_area'];never=ss('training_reference_wet_frequency','never_wet_in_12')['fraction_of_event_false_area'];shallow=ss('prediction_depth_m','(0.1,0.15]')['fraction_of_event_false_area'];con=s['training_loss_strata'];tot=sum(r['contribution_m2'] for r in con);deep=next(r for r in con if r['stratum']=='deep');nonwet=next(r for r in con if r['stratum']=='nonwet');g=s['training_head_bias_directions'];net=sum(r['head_bias_derivative'] for r in g);fpg=next(r for r in g if r['category']=='false_wet');grow=rows('head_bias_derivative_by_event.csv');trainids={r['case_id'] for r in grow if r['split']=='train'};wantdown=sum(sum(float(r['head_bias_derivative']) for r in grow if r['case_id']==cid)>0 for cid in trainids)
 findings={'primary_inference':'Current-recipe spatial fitting/discrimination failure is directly observed even on training events. Loss allocation and selection tradeoffs contribute, but intrinsic representation capacity and eventual convergence are not isolated.','validation_false_share_within50m_of_reference_wet':near,'validation_false_share_reference_at_most1cm':dry,'validation_false_share_never_wet_in_all12_training':never,'validation_false_share_predicted_10to15cm':shallow,'training_deep_loss_value_share':deep['contribution_m2']/tot,'training_nonwet_loss_value_share':nonwet['contribution_m2']/tot,'training_false_wet_loss_value_share':fpg['loss_contribution_m2']/tot,'analytic_macro_head_bias_derivative':net,'training_parents_favoring_uniform_logit_lowering':wantdown,'training_parent_count':12,'current_selected_checkpoint_remains':36,'retrospective_epoch40_false_area_reduction_fraction':1-trade[1]['false_km2_0.1']/trade[0]['false_km2_0.1'],'retrospective_epoch40_recall_change':trade[1]['recall_0.1']-trade[0]['recall_0.1'],'any_saved_D_epoch_passes_original_guard':s['any_saved_D_epoch_passes_original_guard'],'no_training_or_test_access':True}
 write(REPORT/'findings.json',findings)
 metricrows=[]
 for split,label in [('train','训练 12 父事件'),('validation','验证 4 父事件')]:
  r=p[split];metricrows.append({'split':label,'fp':fmt(r['false_km2_0.1']),'recall':fmt(r['recall_0.1']*100,2),'precision':fmt(r['precision_0.1']*100,2),'iou':fmt(r['iou_0.1']),'auc':fmt(r['wet_nonwet_rank_auc']),'mae':fmt(r['mae_m']*100,2)})
 diagnosis=[{'item':'训练拟合','evidence':f'训练与验证 AUC 均约 0.75；训练误报 {a["false_km2_0.1"]:.2f} km²、IoU {a["iou_0.1"]:.3f}。训练集同样不能准确分开湿区和周边背景。','verdict':'直接支持当前配方下拟合不足；不能解释成单纯的验证泛化问题。'}, {'item':'空间表征','evidence':f'{near*100:.1f}% 误报距参考湿区不超过 50 m；{never*100:.1f}% 位于 12 个训练事件均未湿的共同支持。','verdict':'支持湿区周边出现重复宽泛误报；尚未隔离输入编码、网络容量、损失与优化各自责任。'}, {'item':'损失分配','evidence':f'非湿地占训练面积 {nonwet["area_fraction"]*100:.1f}%，但仅贡献目标值 {nonwet["contribution_m2"]/tot*100:.1f}%；深水占 {deep["area_fraction"]*100:.2f}% 面积、贡献 {deep["contribution_m2"]/tot*100:.1f}%。','verdict':'目标值预算与足迹误报不直接一致；误报仍受到下压梯度，不是被损失完全忽略。'}, {'item':'选模规则','evidence':'BER 首先平衡类内漏报率和误报率；未按误报面积设置候选排序门槛，完整基线准入在排序后独立检查。','verdict':'可以接受一定误报来换召回，但准入门槛正确拒绝了全部 D 检查点；换轮次不能使现有候选通过。'}]
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D-arm false inundation diagnosis</title><style>body{font:15px/1.75 system-ui,sans-serif;max-width:1150px;margin:30px auto;padding:0 22px;background:#fafbfd;color:#172033}h1{font-size:27px}h2{font-size:21px;margin-top:32px}.box{padding:18px;background:#edf5f2;border-left:5px solid #0f766e}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{border:1px solid #d9e0e8;padding:8px;text-align:left;vertical-align:top}th{background:#edf2f7}img{width:100%;height:auto}a{color:#1762af}.note{color:#576476}</style><h1>D 组为何仍大面积误报</h1>'''
 body+='<div class="box"><b>证据最强的是：现有配方还没有把“哪里湿、哪里不湿”拟合好，训练集也有同样问题。</b>损失分配和选模规则放大了召回与误报的取舍；目前不能把原因单独归为网络空间容量不足，更没有证据把它归咎于 PLATEAU 高程。<br>仅使用已有 12 训练、4 验证父事件的冻结预测、40 轮日志；0 新拟合、0 更新、0 新推理、0 新水动力运行。封存测试未读取，D 第 36 轮仍是原选中模型。</div>'
 body+='<h2>三类解释，各自能确认到哪一步</h2>'+table(diagnosis,[('item','问题'),('evidence','已有证据'),('verdict','判断与边界')])
 body+='<h2>训练集没有表现出足够好的空间判别</h2>'+table(metricrows,[('split','支持'),('fp','误报 km²'),('recall','召回 %'),('precision','精确率 %'),('iou','IoU'),('auc','湿／非湿排序 AUC'),('mae','总体 MAE cm')])
 body+='<p>所有值均先按父事件计算，再宏平均；三区主比较保持目标与原检索来源精确掩膜的交集、原可用地面面积和严格 >0.10 m 阈值。AUC 是水深对湿／非湿的面积加权排序描述，不拟合新分类器、不选择阈值。两组事件强度不同，不能把训练误报面积更大直接解释为泛化更好；两组同样偏低的空间精确率与 IoU 才是关键。</p>'
 body+=f'<p>验证误报中，{shallow*100:.1f}% 的预测深度位于 10–15 cm，但 {dry*100:.1f}% 的参考深度不超过 1 cm。因此既有预测跨过 10 cm 的敏感性，也有把本来很浅的地面抬成湿区的实质错误。不能通过更改阈值直接宣称修复。</p><img src="false_area_strata.png" alt="Whole-ward false-area strata">'
 wardrows=[]
 for r in s['validation_spatial_strata']['ward']:wardrows.append({'ward':r['stratum'],'fp':fmt(r['false_km2']),'fpr':fmt(r['fpr']*100,2),'share':fmt(r['fraction_of_event_false_area']*100,2)})
 body+='<h2>空间上是三区共同问题，主要表现为邻近湿区的宽泛误报</h2>'+table(wardrows,[('ward','区'),('fp','平均误报 km²'),('fpr','非湿地误报率 %'),('share','事件内误报占比均值 %')])
 body+=f'<p>{near*100:.1f}% 误报距同事件已认证参考湿区不超过 50 m；{never*100:.1f}% 位于全部 12 个训练事件共同精确且从未超过 10 cm 的区域。这两项并不矛盾：真实积水周边有大量始终不湿的地面。直线距离只用于定位误报分布，不代表水力连通路径。</p>'
 rec=s['spatial_recurrence'];body+=f'<p>在每组事件都合格的共同支持上，训练集有 {rec[0]["never_reference_wet_but_false_in_at_least75pct_km2"]:.3f} km² 在全部训练事件都非湿，却在至少 9/12 个事件误报；验证集对应 {rec[1]["never_reference_wet_but_false_in_at_least75pct_km2"]:.3f} km²，在全部 4 个验证事件非湿、至少 3/4 个误报。这是重复的空间偏差，不能只用某一个事件异常解释。</p><img src="whole_ward_recurrence.png" alt="Whole-ward wet and false-wet recurrence maps">'
 road=s['validation_spatial_strata']['Road Ground Fraction'];fallback=ss('PLATEAU Ground Fraction','(0.5,inf]');amb=ss('Subcell Connectivity Ambiguity','(0.5,inf]')
 body+=f'<p>道路占地 >50% 单元的非湿地误报率约 {road[2]["fpr"]*100:.1f}%；道路占地 ≤10% 的约 {road[0]["fpr"]*100:.1f}%。但显式子格连通歧义区仅占误报 {amb["fraction_of_event_false_area"]*100:.1f}%，PLATEAU 补缺比例 >50% 的区仅占 {fallback["fraction_of_event_false_area"]*100:.1f}%。这些是带基数的描述性关联，不能把误报主要归因于道路本身、补缺地形或某个掩膜标志。所有详细分层均保留未知值和各自非湿地分母。</p>'
 body+='<h2>优化在改善目标，但还不能证明已经收敛</h2><p>历史 480 步梯度均有限，0 次触发裁剪。固定训练目标从第 1 轮 0.04443 降至第 40 轮 0.02208 m²；第 36 轮为 0.02519 m²，目标最小在第 38 轮。末期仍有轮间起伏。因此没有“梯度爆炸／一直被裁剪”的证据，也不能把结果视为已充分收敛。原始逐步日志还保留每个模块的真实参数变化；本次没有重建或伪造新的优化轨迹。</p><img src="fit_and_selection_trajectories.png" alt="Existing forty-epoch fit and selection trajectories">'
 body+='<h2>损失确实惩罚误报，但面积与梯度预算并不相同</h2>'
 lossdisplay=[{'stratum':r['stratum'],'area':fmt(r['area_fraction']*100,3),'coeff':fmt(r['mixture_weight'],2),'share':fmt(r['contribution_m2']/tot*100,2),'gradient':fmt(r['head_bias_derivative'],6)} for r in con]
 body+=table(lossdisplay,[('stratum','训练层：全域原精确支持'),('area','面积占比 %'),('coeff','固定混合系数'),('share','目标值贡献 %'),('gradient','共享输出偏置方向导数')])
 body+=f'<p>全部误报贡献平均训练目标值的 {fpg["loss_contribution_m2"]/tot*100:.1f}%。在冻结预测上解析计算现有 Softplus 输出头共享偏置的导数，并用独立中心差分核验：误报项为 +{fpg["head_bias_derivative"]:.6f}，总导数 +{net:.6f}，{wantdown}/12 个训练父事件倾向整体下调输出。这说明损失没有忽略误报，也不能简单解释为深水项必然把所有地面一同抬高。下调共享偏置又可能增加漏报；真正需要的是区分位置的拟合。这里仅检查一个共享参数方向，不是全部参数的梯度归因，也没有实际修改模型。</p>'
 body+='<h2>选模放大了取舍，但不能解释全部问题</h2><p>BER=0.5×(FN/参考湿面积 + FP/参考非湿面积)。每个父事件的参考非湿面积远大于湿面积；同一事件内，挽回 1 m² 漏报允许增加“非湿面积/湿面积”倍的误报而使 BER 不变。四个验证父事件的这个比值如下；不能用面积宏平均之比替代逐事件比值。</p>'
 body+=table([{'case':k,'ratio':fmt(v,2)} for k,v in s['selection_area_exchange_illustration']['per_parent_nonwet_to_wet_area_ratio'].items()],[('case','父事件'),('ratio','同事件 BER 的误报／漏报面积交换比')])
 reason={'selected_original':'原冻结规则选择','final_budget':'同预算末轮诊断','minimum_false_area_only':'仅误报最小：回看','minimum_overall_MAE_only':'仅 MAE 最小：回看','minimum_training_objective_only':'训练目标最小轮：回看'}
 body+=table([{'why':reason[r['reason']],'epoch':r['epoch'],'ber':fmt(r['balanced_extent_error'],6),'fp':fmt(r['false_km2_0.1']),'recall':fmt(r['recall_0.1']*100,2),'iou':fmt(r['iou_0.1']),'mae':fmt(r['mae_m']*100,2)} for r in trade],[('why','已有候选诊断'),('epoch','轮'),('ber','BER'),('fp','误报 km²'),('recall','召回 %'),('iou','IoU'),('mae','MAE cm')])
 body+=f'<p>第 40 轮误报比第 36 轮少 {findings["retrospective_epoch40_false_area_reduction_fraction"]*100:.1f}%，但召回下降 {-findings["retrospective_epoch40_recall_change"]*100:.2f} 个百分点，BER 略差；原规则选择第 36 轮符合既定定义。仅误报最小轮的召回只有 15.2%。全部 40 轮均未通过原逐父事件准入，因此“换一个现有轮次”还不能获得合格模型。原检索基线误报约 0.280 km²、IoU 0.648，D 仍相差明显。所有回看行只用于解释取舍，不构成重新选模。</p>'
 body+='<h2>下一轮应检验什么（尚未执行）</h2><p>优先用同一划分、掩膜、初始化与有限预算，分别检验空间判别是否可学好，以及对误报更直接的监督／候选准入能否改善精确率。一次只改变一类因素，避免同时增加网络容量、样本、时长和损失后无法归因。现有结果不支持直接扩张事件库或只延长训练就会解决误报，也不足以直接决定换图模型。本轮仅提出验证方向，未设新阈值、权重或训练预算。</p>'
 body+='<h2>核验与可复现材料</h2><p>逐父事件原掩膜、来源检索交集、原精度阈值、分层面积加和、冻结指标与选模键、解析导数中心差分和前后源文件哈希均通过核验。原 382 文件四组输出及输入、曲线、标签、归一化保留。没有把像素或轮次当成独立事件，也没有真实洪水观测验证或主测试结论。</p><ul>'
 for name,label in [('selected_event_diagnostics.csv','16 父事件结果'),('spatial_strata_by_event.csv','逐事件空间分层及分母'),('head_bias_derivative_by_event.csv','零更新解析导数'),('forty_epoch_trajectories.csv','40 轮原日志汇总'),('retrospective_tradeoffs.csv','候选取舍回看'),('findings.json','关键判断与数值'),('validation.json','核验记录')]:body+=f'<li><a href="{name}">{label}</a></li>'
 body+='</ul><p class="note">本次为冻结开发数据上的整体诊断。Section 8 的独立封存评估、主动措施与论文正式图表仍未完成。</p></html>'
 (REPORT/'review.html').write_text(body)
 print(json.dumps(findings,indent=2))
if __name__=='__main__':main()
