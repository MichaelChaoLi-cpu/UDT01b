"""Review completed single-parent fits against immutable exposure-matched joint fit.
No optimizer, model inference, validation responses or sealed tests.
"""
from pathlib import Path
import sys,json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_three_ward_unet_exact_pilot import read,write,arrays,load_label,sha
from run_three_ward_unet_controls_v2 import csv_write
from diagnose_three_ward_d_false_positives import auc
from three_ward_surrogate_protocol_v2 import macro_events
NAME='three_ward_single_joint_exposure_fit_v1';OUT=ROOT/'data/processed'/NAME;REPORT=ROOT/'data/exp/estimation'/NAME
JOINT=ROOT/'data/processed/three_ward_two_parent_fit_v1';JR=ROOT/'data/exp/estimation/three_ward_two_parent_fit_v1';SOURCE=ROOT/'data/processed/three_ward_unet_exact_pilot_v1'

def fmt(x,n=3):return 'NA' if x is None else f'{x:.{n}f}'
def table(rows,cols):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(t)+'</th>' for k,t in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k,t in cols)+'</tr>' for r in rows)+'</table></div>'
def main():
 done=read(REPORT/'completion.json');assert done['status']=='complete' and done['actual_new_optimizer_updates']==480 and done['actual_new_fits']==2
 p=read(OUT/'protocol.json');z=arrays(SOURCE/'prepared_inputs.npz');area=z['Ground Area'];ward=z['Ward Code'];eligible=z['Eligible Ground Mask'];jp=arrays(JOINT/'endpoint_predictions.npz');trajectory=[];end=[];delta=[];strata=[];fields={};gates=[];telemetry=[];sources=[]
 for arm,cid in p['arms'].items():
  label=load_label(cid,'train');y=label['Peak Mean Depth Lower'].astype('f4');mask=label['Exact Label Mask']&eligible;sp=arrays(OUT/arm/'endpoint_predictions.npz')['Peak Mean Depth'][0];j=list(jp['Case ID']).index(cid);ref=jp['Peak Mean Depth'][j];fields[cid]={'target':y,'single':sp,'joint':ref,'mask':mask}
  sf=[read(REPORT/arm/f'step_{e:04}.json') for e in p['evaluation_exposures']];jf=[read(JR/f'step_{e*2:04}.json') for e in p['evaluation_exposures']]
  for e,a,b in zip(p['evaluation_exposures'],sf,jf):
   for method,frame in [('single',a),('joint_reused',b)]:
    for r in frame['event_metrics']:
     if r['case_id']!=cid:continue
     row={k:v for k,v in r.items() if not isinstance(v,dict)};row.update(method=method,arm=arm if method=='single' else 'J',exposure=e,optimizer_update=e if method=='single' else 2*e);trajectory.append(row)
     if e==240:end.append(row)
  for scope,region in [('domain',np.ones_like(eligible)),('three_wards',ward>0),('Chiyoda',ward==1),('Chuo',ward==2),('Minato',ward==3),('cross_ward',ward==0)]:
   m=mask&region
   for method,pred in [('single',sp),('joint_reused',ref)]:
    r=next(r for r in end if r['case_id']==cid and r['scope']==scope and r['method']==method);r['AUC']=auc(pred[m],y[m]>.1,area[m])
    t=y[m];v=pred[m];a=area[m];err=v.astype('f8')-t.astype('f8')
    for name,g in [('all_exact',np.ones(t.shape,bool)),('nonwet',t<=.1),('ordinary_wet',(t>.1)&(t<=1)),('wet',t>.1),('deep',t>1)]:
     den=a[g].sum();strata.append({'case_id':cid,'scope':scope,'method':method,'stratum':name,'support_km2':float(den/1e6),'MAE_m':float(np.sum(a[g]*abs(err[g]))/den) if den else None,'RMSE_m':float(np.sqrt(np.sum(a[g]*err[g]**2)/den)) if den else None,'bias_m':float(np.sum(a[g]*err[g])/den) if den else None})
   a=next(r for r in end if r['case_id']==cid and r['scope']==scope and r['method']=='single');b=next(r for r in end if r['case_id']==cid and r['scope']==scope and r['method']=='joint_reused');assert a['support_sha256']==b['support_sha256'] and a['support_km2']==b['support_km2']
   d={'case_id':cid,'scope':scope,'single_exposure':240,'joint_exposure':240,'single_update':240,'joint_update':480,'support_sha256':a['support_sha256'],'support_km2':a['support_km2']}
   for k in ['iou_0.1','recall_0.1','precision_0.1','false_km2_0.1','missed_km2_0.1','mae_m','wet_mae_m','deep_mae_m','bias_m','AUC']:d['delta_'+k]=a[k]-b[k] if a.get(k) is not None and b.get(k) is not None else None
   delta.append(d)
  c=read(REPORT/arm/'completion.json');gates.append({'arm':arm,'case_id':cid,'first_confirmed_exposure':c['first_two_consecutive_pass_exposure'],'endpoint_all_gates_pass':c['endpoint_all_gates_pass'],'endpoint_consecutive_passes':c['endpoint_consecutive_passes'],'known_field_fit_demonstrated_at_endpoint':c['endpoint_all_gates_pass'] and c['endpoint_consecutive_passes']>=2,'failed_endpoint_checks':[{'scope':r['scope'],'failed':[k for k,v in r['checks'].items() if not v]} for r in sf[-1]['gates']['rows']]})
  steps=[json.loads(l) for l in (REPORT/arm/'optimizer_steps.jsonl').read_text().splitlines()];telemetry.append({'arm':arm,'actual_updates':len(steps),'finite_all':all(r['gradient_pre_clip']['finite'] and r['gradient_post_clip']['finite'] for r in steps),'clipping_count':sum(r['clipping_required'] for r in steps),'maximum_gradient_norm':max(r['gradient_pre_clip']['total_norm'] for r in steps),'initial_objective_m2':sf[0]['event_metrics'][0]['fixed_model_loss_m2'],'endpoint_objective_m2':sf[-1]['event_metrics'][0]['fixed_model_loss_m2']})
  sources.append({'arm':arm,'case_id':cid,'original_label':next(r['label_path'] for r in p['selected_records'] if r['case_id']==cid),'original_mask_hash':next(r['original_exact_eligible_mask_sha256'] for r in p['selected_records'] if r['case_id']==cid),'scope':'Only this training response in this fit; joint metrics are reused.'})
 csv_write(REPORT/'exposure_aligned_metrics.csv',trajectory);csv_write(REPORT/'endpoint_metrics.csv',end);csv_write(REPORT/'single_minus_joint.csv',delta);csv_write(REPORT/'endpoint_depth_strata.csv',strata);write(REPORT/'fitting_gates.json',gates);write(REPORT/'optimizer_summary.json',telemetry);write(REPORT/'response_access.json',sources)
 macro=[]
 for method in ['single','joint_reused']:
  for scope in ['domain','three_wards','Chiyoda','Chuo','Minato','cross_ward']:macro.append({'method':method,'scope':scope,**macro_events([r for r in end if r['method']==method and r['scope']==scope])})
 write(REPORT/'endpoint_macro.json',macro)
 passn=sum(g['known_field_fit_demonstrated_at_endpoint'] for g in gates)
 outcome='both_single_fields_fit' if passn==2 else 'one_single_field_fits' if passn==1 else 'neither_single_field_fits_within_budget'
 summary={'status':'complete','decision':p['decision'],'actual_new_fits':2,'actual_new_updates':480,'outcome':outcome,'singles_with_demonstrated_endpoint_fit':passn,'gates':gates,'whole_three_ward_endpoints':[r for r in end if r['scope']=='three_wards'],'whole_three_ward_deltas':[r for r in delta if r['scope']=='three_wards'],'optimizer_summary':telemetry,'joint_reference_reused_not_retrained':True,'formal12plus4_unchanged':True,'quality_masks_unchanged':True,'normalization_unchanged':True,'sample_expansion':False,'validation_response_access':False,'sealed_opened':False,'new_hydraulic_runs':0,'limits':p['interpretation_limits']};write(REPORT/'summary.json',summary)
 # Same exposure x axis; avoid comparing equal optimizer indices.
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 charts=[('iou_0.1','Wet IoU',1),('recall_0.1','Wet recall (%)',100),('precision_0.1','Precision (%)',100),('false_km2_0.1','False wet area (km squared)',1),('missed_km2_0.1','Missed wet area (km squared)',1),('wet_mae_m','Wet MAE (cm)',100)]
 colors={'train-01':'#0f766e','train-04':'#b45309'}
 for ax,(key,title,scale) in zip(axes.ravel(),charts):
  for cid in fields:
   for method,style in [('single','-'),('joint_reused','--')]:
    rr=[r for r in trajectory if r['case_id']==cid and r['method']==method and r['scope']=='three_wards'];ax.plot([r['exposure'] for r in rr],[r[key]*scale if r.get(key) is not None else np.nan for r in rr],ls=style,color=colors[cid],label=cid+' '+method)
  ax.set(title=title,xlabel='Own-parent exposures');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.savefig(REPORT/'exposure_aligned_comparison.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,5,figsize=(14,7),layout='constrained');extent=[-11.36,-3.66,-42.44,-32.32];cm=ListedColormap(['white','#15803d','#2563eb','#f59e0b']);cm.set_bad('#e5e7eb')
 for row,(cid,f) in enumerate(fields.items()):
  m=f['mask']&(ward>0);t=f['target'];
  for col,key in enumerate(['target','joint','single','joint','single']):
   ax=axes[row,col]
   if col<3:
    im=ax.imshow(np.where(m,f[key],np.nan),origin='upper',extent=extent,cmap='Blues',vmin=0,vmax=1,interpolation='nearest');fig.colorbar(im,ax=ax,fraction=.045,pad=.01,extend='max');title=key+' depth (m)'
   else:
    wet=t>.1;pw=f[key]>.1;v=np.zeros(t.shape,'f4');v[wet&pw]=1;v[wet&~pw]=2;v[~wet&pw]=3;v[~m]=np.nan;ax.imshow(v,origin='upper',extent=extent,cmap=cm,vmin=0,vmax=3,interpolation='nearest');title=key+' errors'
   ax.set_title(cid+' | '+title,fontsize=8);ax.tick_params(labelsize=6);ax.set_xlabel('Easting (km)',fontsize=7)
 fig.savefig(REPORT/'whole_three_ward_endpoints.png',dpi=165);plt.close(fig)
 def show(r):return {'case':r['case_id'],'method':r['method'],'scope':r['scope'],'IoU':fmt(r.get('iou_0.1')),'recall':fmt(r.get('recall_0.1')*100,1),'precision':fmt(r.get('precision_0.1')*100,1) if r.get('precision_0.1') is not None else 'NA','wet':fmt(r.get('wet_mae_m')*100,2),'deep':fmt(r.get('deep_mae_m')*100,2) if r.get('deep_mae_m') is not None else 'NA','FP':fmt(r.get('false_km2_0.1')),'FN':fmt(r.get('missed_km2_0.1')),'AUC':fmt(r.get('AUC'))}
 cols=[('case','父事件'),('method','方法'),('scope','区域'),('IoU','IoU'),('recall','召回 %'),('precision','精确率 %'),('wet','湿区 MAE cm'),('deep','深水 MAE cm'),('FP','误报 km²'),('FN','漏报 km²'),('AUC','AUC')]
 body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exposure-matched single versus joint fitting results</title><style>body{font:15px/1.85 system-ui;max-width:1250px;margin:30px auto;padding:0 22px;background:#fafbfd;color:#172033}h1{font-size:26px}h2{font-size:21px}.box{padding:18px;background:#edf5f2;border-left:5px solid #0f766e}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #cbd5e1}img{width:100%}a{color:#1762af}</style><h1>两个单父事件拟合与已有联合组：曝光匹配结果</h1>'''
 body+=f'<div class="box">两个单组各完成 240 次真实更新，共新增 480 次；已有联合组 480 次更新直接复用，每父事件均曝光 240 次。{passn}/2 个单组在终点达到连续确认的完整拟合门槛。原 12＋4、质量掩膜、训练集归一化和共同初始化保持不变；无扩样、无验证响应使用、封存测试未打开。</div>'
 body+='<h2>三区整体结果：固定终点</h2>'+table([show(r) for r in end if r['scope']=='three_wards'],cols)+'<p>所有方法使用同一目标自己的原 own-exact 支持和 Ground Area 权重。单组第 240 步对应联合组第 480 步；没有按最佳轮数重选结果。完整 1012×770 网格参与训练，图示和主要汇总保留三区整体视角。</p><img src="exposure_aligned_comparison.png" alt="Exposure-matched metrics over the three wards"><h2>湿区位置与水深</h2><img src="whole_three_ward_endpoints.png" alt="Reference, joint and individual peak-depth fields across the three wards"><p>误差图：绿色为正确湿区，蓝色为漏报，橙色为误报，灰色为排除支持。深度图显示 0–1 m，深水超过图上限仍完整计入训练和指标。四小时单元峰值图不代表同时刻水量或排水路径。</p>'
 body+='<h2>各区结果</h2>'+table([show(r) for r in end if r['scope'] in ['Chiyoda','Chuo','Minato']],cols)+'<h2>原门槛是否满足</h2>'+table([{'case':g['case_id'],'first':g['first_confirmed_exposure'],'endpass':g['endpoint_all_gates_pass'],'streak':g['endpoint_consecutive_passes'],'fail':'; '.join(r['scope']+': '+', '.join(r['failed']) for r in g['failed_endpoint_checks'])} for g in gates],[('case','事件'),('first','首次连续确认曝光'),('endpass','终点全部通过'),('streak','终点连续通过次数'),('fail','终点未过项')])
 body+='<p>门槛沿用原三区及各区 IoU≥0.85、precision/recall≥0.90、湿区 MAE≤5 cm、存在原深水支持时 MAE≤20 cm。首次连续两次通过单独记录，仍执行固定预算，晚期退步保留；没有自动扩展、切换损失或调阈值。门槛是已知参考场拟合筛查，不是物理精度或正式模型准入。</p><h2>本次能说明什么</h2>'
 if passn==0:body+='<p>两个单组在原预算内均未展示充分拟合，因此不能仅把联合组失败归因于两个事件互相干扰或样本太少。单组是否在某些指标上更好，必须和误报、漏报、水深共同解释。当前配方的逐场空间拟合本身仍是瓶颈；这不证明网络天生不能拟合或加大预算也无效。</p><p>按预定出口，应停止自动细调这套配方，下一次先在表示、监督或优化路线之间作明确选择，再决定更大情景库投入。本报告没有自动启动下一组实验。</p>'
 elif passn==1:body+='<p>仅一个单组达到完整终点拟合门槛，说明困难与具体父事件有关。应先分析未通过事件在当前监督与表示下的空间拟合；不能笼统断言条件链路失效或联合干扰解释了所有失败。</p>'
 else:body+='<p>两个已知场可分别拟好，而原联合组未过门槛，说明共享／交错更新使当前配方更困难。下一步可优先设计共享条件表示和优化的对照；本结果不能单独识别纯梯度冲突。</p>'
 body+='<p>曝光匹配仍不匹配 AdamW 全局步数、矩状态、衰减历史和中间更新。两个单模型总参数量为一个联合模型的两倍，且每个单组条件恒定；本次不是通用代理的部署对比，不证明泛化、条件辨别或现实积水准确度。所有观察来自一个初始化和两个既有合成父事件。</p><h2>与三维模型的关系和模拟原理</h2><p>PLATEAU 建筑、地面道路、水域和补缺地形参与原水动力几何约束；地形优先使用已有裸地 DEM。计算将城市几何转为地面单元储水关系、建筑阻水、道路过流和水域边界，再按降雨输入及水位差计算地表流动与积水。它是利用三维城市几何的二维地表水动力模拟，并未求解建筑周围完整三维流场；当前也没有完整管网或河床三维动力学。本轮只训练代理已有峰值水深结果，没有重跑物理模型。</p><ul>'
 for f in ['summary.json','endpoint_metrics.csv','single_minus_joint.csv','endpoint_depth_strata.csv','exposure_aligned_metrics.csv','fitting_gates.json','optimizer_summary.json','delivery_validation.json']:body+='<li><a href="'+f+'">'+f+'</a></li>'
 body+='</ul><p>AnaSOP5–7记录本次实测结果与界限。数据／变量未增加，第8节的独立验证、观测与主动措施输出仍待补充。</p></html>';(REPORT/'review.html').write_text(body);print(json.dumps({'outcome':outcome,'endpoints':[show(r) for r in end if r['scope']=='three_wards']},indent=2))
if __name__=='__main__':main()
