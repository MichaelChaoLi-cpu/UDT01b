"""Review actual completed comparisons; no training, response expansion or tests."""
from pathlib import Path
import sys,json,csv,html
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,arrays,load_label
from run_three_ward_unet_controls_v2 import csv_write
from diagnose_three_ward_d_false_positives import auc
from three_ward_surrogate_protocol_v2 import macro_events,baseline_readiness
NAME='three_ward_spatial_fp_controls_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME;OLDREPORT=ROOT/'data/exp/estimation/three_ward_unet_controls_v2';SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'
ARMS=['S0F0','S1F0','S0F1','S1F1']
def getframes(a):return [read((OLDREPORT/'D_huber_lower_lr'/f'epoch_{e:03}.jsonl') if a=='S0F0' else REPORT/a/f'epoch_{e:03}.json') for e in range(1,41)]
def metrics(frame,split='validation',scope='three_wards',support='paired_exact'):return next(r['values'] for r in frame['parent_macro_metrics'] if r['split']==split and r['scope']==scope and r['support']==support)
def fmt(v,scale=1,n=3):return 'NA' if v is None else f'{v*scale:.{n}f}'
def table(rr,cols):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(t)+'</th>' for k,t in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k,t in cols)+'</tr>' for r in rr)+'</tbody></table></div>'
def main():
 done=read(REPORT/'completion.json');assert done['new_fits']==3 and done['new_optimizer_updates']==1440 and not done['sealed_opened'];p=read(OUT/'protocol.json');records=p['records'];by={r['case_id']:i for i,r in enumerate(records)};z=arrays(SOURCE/'prepared_inputs.npz');area=z['Ground Area'];ward=z['Ward Code'];eligible=z['Eligible Ground Mask'];labels=[load_label(r['case_id'],r['split']) for r in records];y=np.array([l['Peak Mean Depth Lower'].astype('f4') for l in labels]);masks=np.array([l['Exact Label Mask']&eligible for l in labels]);paired=np.array([masks[i]&masks[by[r['retrieval_source_case_id']]]&(ward>0) for i,r in enumerate(records)])
 bases=read(REPORT/'frozen_baseline_metrics.json');selections={'S0F0':done['reference_selection'],**{r['arm']:r['selection'] for r in done['arms']}};frames={a:getframes(a) for a in ARMS};events=[];macros=[];primary=[];policy=[];trajectory=[];optimization=[];aucrows=[];strata=[];all_predictions={};recurrences=[]
 raw=arrays(ROOT/'data/processed/three_ward_surrogate_inputs/static_tensor_10m.npz');feature_names=list(raw['Static Channel Names']);train_common=np.logical_and.reduce(paired[[i for i,r in enumerate(records) if r['split']=='train']]);train_wet_count=np.sum(y[[i for i,r in enumerate(records) if r['split']=='train']]>.1,axis=0)
 for a in ARMS:
  selection=selections[a];selected=selection['selected_epoch'];shadow=selection['shadow_BER_epoch'];fr=frames[a]
  for epoch,f in enumerate(fr,1):
   vm=metrics(f);tm=metrics(f,'train','domain','own_exact');trajectory.append({'arm':a,'epoch':epoch,'fixed_train_base_loss_m2':tm.get('fixed_base_loss_m2',tm['fixed_model_loss_m2']),**{k:vm[k] for k in ['iou_0.1','false_km2_0.1','recall_0.1','precision_0.1','mae_m','wet_mae_m','balanced_extent_error']}})
  for key,ep in [('IoU_guard_policy',selected),('same_pool_BER_policy',shadow),('fixed_epoch40',40)]:policy.append({'arm':a,'policy':key,'epoch':ep,'status':selection['status'],'pool_kind':selection['pool_kind'],**metrics(fr[ep-1])})
  for point,ep in [('epoch40',40),('common_selected',selected)]:
   f=fr[ep-1];pred=arrays(OUT/a/('endpoint_predictions.npz' if point=='epoch40' else 'selected_predictions.npz'))['Peak Mean Depth'];all_predictions[(a,point)]=pred
   for r in f['event_metrics']:events.append({'arm':a,'point':point,'epoch':ep,**{k:v for k,v in r.items() if not isinstance(v,dict)}})
   for r in f['parent_macro_metrics']:macros.append({'arm':a,'point':point,'epoch':ep,'split':r['split'],'scope':r['scope'],'support':r['support'],**r['values']})
   for i,r in enumerate(records):
    m=paired[i];wet=y[i]>.1;pw=pred[i]>.1;fp=m&~wet&pw;rate=auc(pred[i][m],wet[m],area[m]);aucrows.append({'arm':a,'point':point,'epoch':ep,'case_id':r['case_id'],'split':r['split'],'AUC':rate});fp_total=area[fp].sum();dist=distance_transform_edt(~(m&wet),sampling=10)
    groups=[('FP_prediction_depth','10_to15cm',(pred[i]>.1)&(pred[i]<=.15)),('FP_prediction_depth','15_to30cm',(pred[i]>.15)&(pred[i]<=.3)),('FP_prediction_depth','30_to50cm',(pred[i]>.3)&(pred[i]<=.5)),('FP_prediction_depth','above50cm',pred[i]>.5),('reference_nonwet_depth','at_most1cm',y[i]<=.01),('reference_nonwet_depth','1_to5cm',(y[i]>.01)&(y[i]<=.05)),('reference_nonwet_depth','5_to10cm',(y[i]>.05)&(y[i]<=.1)),('proximity','within50m',dist<=50),('proximity','above50m',dist>50),('training_wet_frequency','never_wet_in12',train_common&(train_wet_count==0)),('training_wet_frequency','wet_in_at_least1',train_common&(train_wet_count>0)),('training_wet_frequency','incomplete_train_support',~train_common)]
    for feature,bounds in [('Road Ground Fraction',[-np.inf,.1,.5,np.inf]),('Building Occupancy Fraction',[-np.inf,.25,.5,.75,np.inf]),('Subcell Connectivity Ambiguity',[-np.inf,.5,np.inf]),('PLATEAU Ground Fraction',[-np.inf,0,.5,np.inf])]:
     j=feature_names.index(feature);known=raw['Static Feature Valid'][j]&np.isfinite(raw['Static Tensor'][j]);v=raw['Static Tensor'][j]
     for lo,hi in zip(bounds[:-1],bounds[1:]):groups.append((feature,f'({lo},{hi}]',known&(v>lo)&(v<=hi)))
     groups.append((feature,'unknown',~known))
    for family,name,g in groups:
     mass=area[fp&g].sum();nmass=area[m&~wet&g].sum();strata.append({'arm':a,'point':point,'case_id':r['case_id'],'split':r['split'],'family':family,'stratum':name,'FP_km2':float(mass/1e6),'nonwet_km2':float(nmass/1e6),'FPR':float(mass/nmass) if nmass else None,'FP_share':float(mass/fp_total) if fp_total else None})
   point_val={q['case_id']:q for q in f['event_metrics'] if q['split']=='validation' and q['scope']=='three_wards' and q['support']=='paired_exact'};point_ready=baseline_readiness(point_val,bases['dry'],bases['retrieval'])
   for split in ['train','validation']:
    ar=[r for r in aucrows if r['arm']==a and r['point']==point and r['split']==split];pm=metrics(f,split);primary.append({'arm':a,'point':point,'epoch':ep,'split':split,'status':'baseline_guard_passed' if point_ready['all_parents_pass'] else 'not_ready','arm_selected_status':selection['status'],**pm,'AUC':float(np.mean([r['AUC'] for r in ar]))})
    ii=[i for i,r in enumerate(records) if r['split']==split];common=np.logical_and.reduce(paired[ii]);fp=paired[ii]&(y[ii]<=.1)&(pred[ii]>.1);cnt=fp.sum(0);never=common&~np.any(y[ii]>.1,axis=0);recurrences.append({'arm':a,'point':point,'split':split,'common_support_km2':float(area[common].sum()/1e6),'false_in_at_least75pct_km2':float(area[common&(cnt>=int(np.ceil(.75*len(ii))))].sum()/1e6),'never_wet_and_false_at_least75pct_km2':float(area[never&(cnt>=int(np.ceil(.75*len(ii))))].sum()/1e6)})
  if a!='S0F0':
   steps=[json.loads(line) for line in (REPORT/a/'optimizer_steps.jsonl').read_text().splitlines()];base=np.array([r['fixed_train_base_loss_m2'] for r in trajectory if r['arm']==a]);opt=next(r for r in done['arms'] if r['arm']==a);optimization.append({'arm':a,'optimizer_updates':len(steps),'finite_gradients':all(r['gradient_pre_clip']['finite'] for r in steps),'clipping_count':sum(r['clipping_required'] for r in steps),'gradient_max':max(r['gradient_pre_clip']['total_norm'] for r in steps),'first_base_objective':float(base[0]),'final_base_objective':float(base[-1]),'final_over_first_base':float(base[-1]/base[0]),'last10_base_CV':float(base[-10:].std()/base[-10:].mean()),'training_seconds':opt['training_seconds'],'evaluation_seconds':opt['evaluation_seconds'],'maximum_sampled_mps_driver_allocated_bytes':opt['maximum_sampled_mps_driver_allocated_bytes']})
 columns=['mae_m','wet_mae_m','deep_mae_m','nonwet_mae_m','iou_0.1','precision_0.1','recall_0.1','false_km2_0.1','missed_km2_0.1','balanced_extent_error'];contrasts=[]
 for point in ['epoch40','common_selected']:
  for name,left,right in [('spatial_F0','S1F0','S0F0'),('spatial_F1','S1F1','S0F1'),('FP_S0','S0F1','S0F0'),('FP_S1','S1F1','S1F0')]:
   for scope in ['three_wards','Chiyoda','Chuo','Minato','domain','cross_ward']:
    ll={r['case_id']:r for r in events if r['arm']==left and r['point']==point and r['split']=='validation' and r['scope']==scope and r['support']=='paired_exact'};rr={r['case_id']:r for r in events if r['arm']==right and r['point']==point and r['split']=='validation' and r['scope']==scope and r['support']=='paired_exact'}
    for cid,x in ll.items():
     z0=rr[cid];assert x['support_sha256']==z0['support_sha256'];row={'point':point,'contrast':name,'left':left,'right':right,'scope':scope,'case_id':cid}
     for k in columns:row['delta_'+k]=x[k]-z0[k] if x.get(k) is not None and z0.get(k) is not None else None
     la=next(r['AUC'] for r in aucrows if r['arm']==left and r['point']==point and r['case_id']==cid);ra=next(r['AUC'] for r in aucrows if r['arm']==right and r['point']==point and r['case_id']==cid)
     if scope=='three_wards':row['delta_AUC']=la-ra
     contrasts.append(row)
 contrast_macro=[]
 for point in ['epoch40','common_selected']:
  for name in ['spatial_F0','spatial_F1','FP_S0','FP_S1']:
   rr=[r for r in contrasts if r['point']==point and r['contrast']==name and r['scope']=='three_wards'];v={}
   for k in ['delta_'+c for c in columns]+['delta_AUC']:
    vv=[r[k] for r in rr if r.get(k) is not None];v[k]=float(np.mean(vv)) if vv else None
   contrast_macro.append({'point':point,'contrast':name,**v})
 interaction=[]
 for point in ['epoch40','common_selected']:
  rr={r['contrast']:r for r in contrast_macro if r['point']==point};interaction.append({'point':point,**{k:rr['FP_S1'][k]-rr['FP_S0'][k] if rr['FP_S1'].get(k) is not None and rr['FP_S0'].get(k) is not None else None for k in rr['FP_S1'] if k.startswith('delta_')}})
 for name,rows in [('event_metrics.csv',events),('macro_metrics.csv',macros),('primary_comparison.csv',primary),('checkpoint_policy_comparison.csv',policy),('trajectory.csv',trajectory),('optimization.csv',optimization),('AUC_by_event.csv',aucrows),('FP_strata_by_event.csv',strata),('spatial_recurrence.csv',recurrences),('paired_contrasts.csv',contrasts),('paired_contrast_macro.csv',contrast_macro),('interactions.csv',interaction)]:csv_write(REPORT/name,rows)
 fixed={r['arm']:r for r in primary if r['point']=='epoch40' and r['split']=='validation'};selected={r['arm']:r for r in primary if r['point']=='common_selected' and r['split']=='validation'}
 summary={'status':'complete','decision':'MILI-D-20261016-008','new_fits':3,'new_optimizer_updates':1440,'original12train4validation':True,'shared_initialization_preserved':True,'primary_fixed_epoch40':fixed,'common_selected':selected,'selection':selections,'matched_epoch40_contrasts':[r for r in contrast_macro if r['point']=='epoch40'],'selected_checkpoint_contrasts':[r for r in contrast_macro if r['point']=='common_selected'],'interaction':interaction,'optimization':optimization,'any_new_arm_passes_original_guard':any(r['selection']['status']=='baseline_guard_passed' for r in done['arms']),'sealed_opened':False,'sample_expansion':False,'limitations':['One stored initialization, fixed synthetic geometry and four repeatedly exposed validation parents; no independent test or event observation.','Spatial treatment changes one pooling operation, receptive context and computation, not only information resolution.','Peak depth/footprint agreement does not prove road-length accuracy, dynamic paths or water conservation.','Same-budget endpoint contrasts and selected-checkpoint contrasts are separate; selection can choose different epochs.','MPS memory is sampled allocator usage, not a certified peak; historical D runtime is not contemporaneously rerun.']};write(REPORT/'summary.json',summary)
 colors={'S0F0':'#64748b','S1F0':'#2563eb','S0F1':'#d97706','S1F1':'#0f766e'};fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
 for ax,(key,title,scale) in zip(axes.ravel(),[('iou_0.1','Validation wet IoU',1),('false_km2_0.1','False wet area (km squared)',1),('recall_0.1','Wet recall (%)',100),('precision_0.1','Wet precision (%)',100),('fixed_train_base_loss_m2','Common fixed training base objective',1),('mae_m','Validation overall MAE (cm)',100)]):
  for a in ARMS:
   rr=[r for r in trajectory if r['arm']==a];xx=np.arange(1,41);yy=[r[key]*scale for r in rr];ax.plot(xx,yy,label=a,color=colors[a]);e=selections[a]['selected_epoch'];ax.scatter(e,yy[e-1],s=20,color=colors[a])
  ax.set(title=title,xlabel='Existing epoch');ax.grid(alpha=.2)
 axes[0,0].legend(ncol=2);fig.savefig(REPORT/'controlled_trajectories.png',dpi=170);plt.close(fig)
 # All four validation parents on matched epoch40, original comparison support.
 from matplotlib.patches import Patch
 fig=plt.figure(figsize=(12,12),layout='constrained');gs=fig.add_gridspec(5,5,height_ratios=[1,1,1,1,.12]);cm=ListedColormap(['white','#15803d','#2563eb','#f59e0b']);cm.set_bad('#e5e7eb');refcm=ListedColormap(['white','#2563eb']);refcm.set_bad('#e5e7eb');extent=[-11.36,-3.66,-42.44,-32.32]
 for row,cid in enumerate(p['validation_case_ids']):
  i=by[cid];m=paired[i];wet=y[i]>.1
  for col in range(5):
   ax=fig.add_subplot(gs[row,col])
   if col==0:field=np.where(m,wet.astype('f4'),np.nan);cmap=refcm;vmax=1;title='Reference'
   else:
    a=ARMS[col-1];pw=all_predictions[(a,'epoch40')][i]>.1;field=np.zeros(m.shape,'f4');field[wet&pw]=1;field[wet&~pw]=2;field[~wet&pw]=3;field[~m]=np.nan;cmap=cm;vmax=3;title=a
   ax.imshow(field,origin='upper',extent=extent,cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest');ax.set_title(cid+' | '+title,fontsize=8);ax.tick_params(labelsize=6)
 legend=fig.add_subplot(gs[4,:]);legend.axis('off');legend.legend(handles=[Patch(color='#15803d',label='True wet'),Patch(color='#2563eb',label='Missed/reference wet'),Patch(color='#f59e0b',label='False wet'),Patch(color='#e5e7eb',label='Excluded')],loc='center',ncol=4)
 fig.savefig(REPORT/'epoch40_whole_ward_extent.png',dpi=170);plt.close(fig)
 def display(r):return {'arm':r['arm'],'epoch':r['epoch'],'IoU':fmt(r['iou_0.1']),'AUC':fmt(r['AUC']),'precision':fmt(r['precision_0.1'],100,2),'recall':fmt(r['recall_0.1'],100,2),'FP':fmt(r['false_km2_0.1']),'FN':fmt(r['missed_km2_0.1']),'MAE':fmt(r['mae_m'],100,2),'wet':fmt(r['wet_mae_m'],100,2),'status':r['status']}
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spatial pooling and FP supervision controls</title><style>body{font:15px/1.75 system-ui,sans-serif;max-width:1200px;margin:30px auto;padding:0 22px;color:#172033;background:#fafbfd}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.box{padding:18px;background:#edf5f2;border-left:5px solid #0f766e}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{border:1px solid #d9e0e8;padding:8px;text-align:left;vertical-align:top}th{background:#edf2f7}img{width:100%;height:auto}a{color:#1762af}</style><h1>空间池化 × 显式误报监督：三组真实对照与 D 参照</h1>'''
 body+='<div class="box">已完成三次拟合，每组 40 轮、480 次更新，共新增 1,440 次更新。S0F0 复用原 D，不重训。原 12＋4 父事件、共同未训练参数、原归一化和质量掩膜均保留；未扩样，未读取封存测试。<br>接口核验和新规则不改写历史 D 第 36 轮；按本轮 IoU 规则，D 参照选中第 40 轮，仍须独立检查原准入门槛。</div>'
 body+='<p>S0：原四次池化；S1：只取消进入最深层前的最后一次池化。F0：原 D 分层稳健损失；F1：仅增加系数 1 的原非湿精确支持超 0.10 m 惩罚。所有参数均从原共同未训练状态严格载入。两项因素的配对对照和交互分别报告，不把组合组直接对参照的差异当作单项效果。</p>'
 reference=fixed['S0F0'];combined=fixed['S1F1'];selected_combined=selected['S1F1']
 findings={'fixed_epoch40_combo_vs_reference':{'FP_reduction_fraction':1-combined['false_km2_0.1']/reference['false_km2_0.1'],'recall_change_pp':100*(combined['recall_0.1']-reference['recall_0.1']),'IoU_change':combined['iou_0.1']-reference['iou_0.1'],'AUC_change':combined['AUC']-reference['AUC']},'selected_combo_vs_selected_reference':{'FP_reduction_fraction':1-selected_combined['false_km2_0.1']/selected['S0F0']['false_km2_0.1'],'recall_change_pp':100*(selected_combined['recall_0.1']-selected['S0F0']['recall_0.1']),'IoU_change':selected_combined['iou_0.1']-selected['S0F0']['iou_0.1']},'all_new_arms_failed_original_guard':not summary['any_new_arm_passes_original_guard'],'interpretation':'Under this one-seed fixed-data recipe the penalty reduces false area with substantial missed-wet cost. Pooling effects depend on supervision; the combined endpoint ranking remains close to reference. No new arm meets complete admission.'}
 write(REPORT/'findings.json',findings)
 body+='<h2>这轮最明确的结论</h2><p>固定末轮下，单独取消最后一次池化的 IoU 低于 D；只加误报惩罚也未提高 IoU。组合组误报比 D 少 '+fmt(findings['fixed_epoch40_combo_vs_reference']['FP_reduction_fraction'],100,1)+'%，但召回下降 '+fmt(-findings['fixed_epoch40_combo_vs_reference']['recall_change_pp'],1,2)+' 个百分点，IoU 仍较低；排序 AUC 为 '+fmt(combined['AUC'])+'，与 D 的 '+fmt(reference['AUC'])+' 接近。误报减少不能等同于积水识别更可靠。</p><p>两个因素存在条件性差异：在有误报惩罚时，保留更细瓶颈相对同样惩罚的原池化组改善了排序；在原 D 监督下则没有这种优势。因此不能把效果简单归给“更细一定更好”。统一选模后的组合组 IoU 为 '+fmt(selected_combined['iou_0.1'])+'，略高于 D 的 '+fmt(selected['S0F0']['iou_0.1'])+'，但召回仍降至 '+fmt(selected_combined['recall_0.1'],100,1)+'%。全部新组仍未通过原准入门槛；单种子、四个已暴露验证父事件的差异不构成独立确认。</p>'
 columns=[('arm','组'),('epoch','轮'),('IoU','IoU'),('AUC','排序 AUC'),('precision','精确率 %'),('recall','召回 %'),('FP','误报 km²'),('FN','漏报 km²'),('MAE','总体 MAE cm'),('wet','湿区 MAE cm'),('status','原准入')]
 body+='<h2>主要比较：同为第 40 轮／480 更新</h2><p>原 paired exact 三区支持，严格 >0.10 m，逐父事件面积加权后宏平均。AUC 只描述深度排序，不拟合分类器或选择新阈值。完整域、各区、跨区及 own-exact 的结果另存。</p>'+table([display(fixed[a]) for a in ARMS],columns)
 crow=[]
 for r in summary['matched_epoch40_contrasts']:crow.append({'contrast':r['contrast'],'IoU':fmt(r['delta_iou_0.1']),'AUC':fmt(r['delta_AUC']),'FP':fmt(r['delta_false_km2_0.1']),'recall':fmt(r['delta_recall_0.1'],100,2),'wet':fmt(r['delta_wet_mae_m'],100,2)})
 body+='<h2>单因素差值：左减右，保持另一因素不变</h2>'+table(crow,[('contrast','对照'),('IoU','Δ IoU'),('AUC','Δ AUC'),('FP','Δ 误报 km²'),('recall','Δ 召回百分点'),('wet','Δ 湿区 MAE cm')])
 body+='<p>spatial_F0=S1F0−S0F0；spatial_F1=S1F1−S0F1；FP_S0=S0F1−S0F0；FP_S1=S1F1−S1F0。IoU／AUC／召回增加通常有利，误报和 MAE 减少通常有利；要看逐事件结果而不能只看宏平均。误报下降若伴召回下降、AUC 没有改善，应解释为压低预测的取舍，不能直接称为空间判别改善。</p><img src="controlled_trajectories.png" alt="Forty epoch matched control trajectories">'
 body+='<h2>统一检查点选择与门槛</h2>'+table([display(selected[a]) for a in ARMS],columns)
 policyrows=[]
 for a in ARMS:
  sel=selections[a];policyrows.append({'arm':a,'iou':sel['selected_epoch'],'BER':sel['shadow_BER_epoch'],'passing':str(sel['passing_epochs']),'pool':sel['pool_kind']})
 body+=table(policyrows,[('arm','组'),('iou','IoU 规则轮次'),('BER','同池 BER 影子轮次'),('passing','通过全部父事件的轮次'),('pool','共同候选池')])
 body+='<p>任何通过原逐父事件检索／无积水比较门槛的轮次，先进入准入候选池；池内按 IoU、湿区 MAE、总体 MAE、较早轮选择。全部失败时只保留诊断最佳并标记 not_ready。BER 影子排序使用完全相同的池，单独分析排序键影响。不同组所选轮数不同，所选模型对照含选模时机差异，须与固定第 40 轮结果区分。不得因为未准入而自动改权重、预算、阈值或打开封存测试。</p><img src="epoch40_whole_ward_extent.png" alt="All validation parents whole ward extent errors at epoch40">'
 body+='<h2>训练与验证、优化和空间分层</h2>'+table([dict(display(r),arm=r['arm']+' / '+r['split']) for r in primary if r['point']=='epoch40'],columns)
 body+='<p>完整逐事件资料含道路／建筑比例、连通歧义和地形补缺分层及分母、参考很浅但预测湿的面积、邻近参考湿区及重复误报。它们用于描述错误，不作为新输入、标签排除规则或真实排水路径。所有未知支持保持未知。</p>'
 body+=table([{'arm':r['arm'],'clip':r['clipping_count'],'grad':fmt(r['gradient_max']),'ratio':fmt(r['final_over_first_base']),'cv':fmt(r['last10_base_CV']),'time':fmt(r['training_seconds'],1,1),'memory':fmt(r['maximum_sampled_mps_driver_allocated_bytes']/2**30,1,2)} for r in optimization],[('arm','组'),('clip','480 步裁剪次数'),('grad','最大梯度'),('ratio','末／首共同基础损失'),('cv','末十轮变异系数'),('time','训练秒（含观察器）'),('memory','采样 MPS 分配 GiB')])
 body+='<p>优化统计为真实步记录，固定模型目标与顺序更新时的在线目标分开；共同基础损失可横比，不同监督总损失不可直接横比。内存是采样到的 MPS 分配最大值，不是精确峰值；时间含观察器与传输，不能宣称端到端水动力加速。更细瓶颈改变算量和感受范围，单次单种子对照不能证明全体网络容量的因果作用。</p><h2>文件与证据边界</h2><ul>'
 for name,label in [('primary_comparison.csv','固定末轮与统一选模指标'),('paired_contrasts.csv','所有逐父事件、分区对照'),('checkpoint_policy_comparison.csv','同池检查点策略比较'),('AUC_by_event.csv','训练与验证空间排序'),('FP_strata_by_event.csv','误报分层和分母'),('spatial_recurrence.csv','整体重复误报'),('optimization.csv','真实优化摘要'),('summary.json','完整结果摘要'),('delivery_validation.json','独立交付核验')]:body+=f'<li><a href="{name}">{label}</a></li>'
 body+='</ul><p>本轮不扩大训练／验证样本，不生成新的水动力标签，不查看封存主测试或挑战响应。四个验证父事件已多次参与开发，当前结果不构成独立测试、真实洪水精度、路径／守恒验证或主动措施效果证明。AnaSOP 的正式 Section 8 输出仍待后续证据。</p></html>'
 (REPORT/'review.html').write_text(body)
 print(json.dumps({'fixed_epoch40':{a:{k:fixed[a][k] for k in ['iou_0.1','AUC','false_km2_0.1','recall_0.1']} for a in ARMS},'selections':{a:selections[a]['selected_epoch'] for a in ARMS},'any_guard_pass':summary['any_new_arm_passes_original_guard']},indent=2))
if __name__=='__main__':main()
