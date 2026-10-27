"""Reproducible exploratory report; not a sealed-test or publication result."""
from pathlib import Path
import csv,html,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing')]
from prepare_three_ward_unet_exact_pilot import OUT,REPORT,read,write,arrays,load_label

def rows(n):return list(csv.DictReader((REPORT/n).open()))
def num(v):return float(v) if v not in ('',None) else None

def table(items,columns):
 out='<table><thead><tr>'+''.join('<th>'+html.escape(title)+'</th>' for key,title in columns)+'</tr></thead><tbody>'
 for r in items:
  out+='<tr>'
  for key,title in columns:
   v=r.get(key,'');out+='<td>'+html.escape(str(v))+'</td>'
  out+='</tr>'
 return out+'</tbody></table>'
def main():
 d=read(REPORT/'completion.json');p=read(OUT/'protocol.json');env=read(REPORT/'environment.json');z=arrays(OUT/'prepared_inputs.npz');pred=arrays(OUT/'predictions.npz')['Conditional UNet Peak Mean Depth'];records=p['records'];indices={r['case_id']:i for i,r in enumerate(records)};retrieval={r['case_id']:r for r in rows('retrieval_choices.csv')}
 metric=rows('macro_metrics.csv');events=rows('event_metrics.csv');history=rows('training_history.csv');val=[i for i,r in enumerate(records) if r['split']=='validation']
 fig,axs=plt.subplots(1,2,figsize=(10,3.3),constrained_layout=True)
 axs[0].plot([int(r['epoch']) for r in history],[float(r['train_mse_m2']) for r in history],color='#2563eb');axs[0].set(xlabel='Epoch',ylabel='Training MSE (m squared)',title='Training exact-label loss')
 hh=[r for r in history if r['validation_mae_m']];axs[1].plot([int(r['epoch']) for r in hh],[float(r['validation_mae_m']) for r in hh],marker='o',color='#0f766e');axs[1].axvline(d['best_epoch'],color='#ef4444',ls='--',label='Selected checkpoint');axs[1].set(xlabel='Epoch',ylabel='Validation MAE (m)',title='Validation checkpoint selection');axs[1].legend()
 for ax in axs:ax.grid(alpha=.2)
 fig.savefig(REPORT/'learning_curve.png',dpi=160);plt.close(fig)
 extent=[-11.36,-3.66,-42.44,-32.32];fig,axs=plt.subplots(4,4,figsize=(13,13),constrained_layout=True)
 cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#eeeeee');errc=plt.get_cmap('RdBu_r').copy();errc.set_bad('#eeeeee')
 for row,i in enumerate(val):
  cid=records[i]['case_id'];lab=load_label(cid,'validation');mask=lab['Exact Label Mask']&z['Eligible Ground Mask'];truth=np.where(mask,lab['Peak Mean Depth Lower'],np.nan)
  j=indices[retrieval[cid]['source_case_id']];source=load_label(records[j]['case_id'],'train');lookup=np.where(source['Exact Label Mask']&mask,source['Peak Mean Depth Lower'],np.nan)
  unet=np.where(mask,pred[i],np.nan);fields=[truth,unet,lookup,unet-truth]
  for col,(field,title) in enumerate(zip(fields,['Reference exact','Conditional U-Net','Training-event retrieval','U-Net minus reference'])):
   ax=axs[row,col];im=ax.imshow(field,extent=extent,origin='upper',cmap=errc if col==3 else cmap,vmin=-.1 if col==3 else 0,vmax=.1 if col==3 else .5,interpolation='nearest',rasterized=True)
   ax.set_title(cid+' | '+title,fontsize=9);ax.set(xlabel='Easting (km, EPSG:6677)',ylabel='Northing (km)');ax.tick_params(labelsize=7)
 fig.colorbar(axs[0,0].images[0],ax=axs[:,:3].ravel().tolist(),location='bottom',fraction=.025,pad=.025,label='Peak mean depth (m; display clipped at 0.5)')
 fig.colorbar(axs[0,3].images[0],ax=axs[:,3].ravel().tolist(),location='bottom',fraction=.025,pad=.025,label='Difference (m; clipped)')
 fig.savefig(REPORT/'validation_whole_domain.png',dpi=160);plt.close(fig)
 def fmt(v,scale=1,digits=3):
  x=num(v);return 'NA' if x is None else f'{x*scale:.{digits}f}'
 def display(r):
  return {'method':r['method'],'scope':r['scope'],'case':r.get('case_id','4-parent macro'),'mae_cm':fmt(r['mae_m'],100),'wet_cm':fmt(r['wet_mae_m'],100),'rmse_cm':fmt(r['rmse_m'],100),'iou':fmt(r['iou_0.1']),'recall':fmt(r['recall_0.1']),'missed':fmt(r['missed_km2_0.1']),'false':fmt(r['false_km2_0.1']),'coverage':fmt(r['coverage_fraction'],100,2),'support':fmt(r['support_km2'])}
 columns=[('method','方法'),('mae_cm','MAE (cm)'),('wet_cm','湿区 MAE (cm)'),('rmse_cm','RMSE (cm)'),('iou','IoU >0.10m'),('recall','召回率'),('missed','漏报 (km²)'),('false','误报 (km²)'),('coverage','精确标签覆盖 (%)')]
 primary=[r for r in metric if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact']
 per_event=[r for r in events if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='paired_exact']
 own=[r for r in metric if r['split']=='validation' and r['scope']=='three_wards' and r['support']=='own_exact']
 ward=[r for r in metric if r['split']=='validation' and r['scope'].startswith('ward_') and r['support']=='paired_exact']
 by={r['method']:r for r in primary};u=by['conditional_unet'];n=by['retrieval'];dry=by['dry']
 summary={'selected_epoch':d['best_epoch'],'epochs':d['epochs_completed'],'parameters':env['parameters'],'validation_three_ward_paired_macro':{r['method']:{k:num(r[k]) for k in ['mae_m','wet_mae_m','rmse_m','iou_0.1','recall_0.1','missed_km2_0.1','false_km2_0.1','coverage_fraction','support_km2']} for r in primary},'unet_beats_retrieval_mae':float(u['mae_m'])<float(n['mae_m']),'unet_beats_retrieval_iou':float(u['iou_0.1'])>float(n['iou_0.1']),'unet_beats_dry_mae':float(u['mae_m'])<float(dry['mae_m']),'sealed_opened':False,'reference_accuracy_observed':False}
 write(REPORT/'summary.json',summary)
 exactstats=[]
 for i,r in enumerate(records):
  lab=load_label(r['case_id'],r['split']);area=z['Ground Area'];eligible=z['Eligible Ground Mask'];den=area[eligible].sum()
  rec={'case_id':r['case_id'],'split':r['split'],'eligible_km2':den/1e6}
  for key in ['Exact Label Mask','Interval Label Mask','Wide Interval Mask','Quarantine Mask','Online Ever Negative']:
   rec[key+' km2']=float(area[lab[key]&eligible].sum())/1e6
  exactstats.append(rec)
 with (REPORT/'label_support.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(exactstats[0]));w.writeheader();w.writerows(exactstats)
 support_table=[{'case':q['case_id'],'source':q['source_case_id'],'paired':fmt(q['paired_km2']),'fraction':fmt(q['paired_fraction_of_exact'],100,2),'abstained':fmt(q['abstained_km2'])} for q in retrieval.values() if q['split']=='validation']
 bettermae='低于' if summary['unet_beats_retrieval_mae'] else '未低于';betteriou='高于' if summary['unet_beats_retrieval_iou'] else '未高于'
 lead=f"首轮单次试训练已完成：12 个训练父事件、4 个验证父事件；共 {d['epochs_completed']} 轮，验证集选择第 {d['best_epoch']} 轮。三区共同精确支持上的 U-Net MAE 为 {fmt(u['mae_m'],100)} cm，{bettermae}事件检索；IoU 为 {fmt(u['iou_0.1'])}，{betteriou}事件检索。"
 htmltext='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Three-ward first exact-label pilot</title><style>body{font:15px/1.65 system-ui,sans-serif;max-width:1250px;margin:35px auto;padding:0 24px;color:#172033;background:#fafbfd}h1{font-size:28px}h2{margin-top:32px;font-size:21px}.box{padding:18px 22px;border-left:5px solid #0f766e;background:#e9f5f2}table{border-collapse:collapse;width:100%;font-size:13px;background:white}td,th{padding:9px;border:1px solid #dce2ea;text-align:left}th{background:#edf2f7}img{width:100%;height:auto;background:white}a{color:#1762af}code{font-size:13px}.muted{color:#556274}li{margin:6px 0}</style><h1>三区条件 U-Net：首轮精确标签试训练</h1>'''
 htmltext+=f'<div class="box">{lead}<br>湿区召回率仅约 0.027%，本次 U-Net 未保留主要积水范围。<br>这是模型条件下的验证集开发结果；验证集参与检查点选择，封存测试与挑战响应均未打开。</div>'
 htmltext+='<h2>比较规则</h2><p>主表覆盖都心三区，按每个父事件先计算开放地面面积加权指标，再等权平均 4 个事件。积水定义为峰值平均水深严格大于 0.10 m；湿区 MAE 只在真实参考湿区计算。共同支持是目标事件原精确掩膜与检索来源事件原精确掩膜的交集，三种方法使用完全相同的单元。被排除区域不是无积水。地图展示更广的各自可用支持，表格只在声明的支持上打分。</p>'
 htmltext+=table([display(r) for r in primary],columns)
 htmltext+='<h2>每个验证父事件</h2>'+table([display(r) for r in per_event],[('case','父事件')]+columns)
 htmltext+='<h2>各自可用精确支持与检索拒答</h2><p>U-Net 和无积水基线覆盖所有目标精确标签；检索在来源未知/区间/隔离单元拒答。以下各自支持的分数不能直接作为配对胜负。</p>'+table([display(r) for r in own],columns)+table(support_table,[('case','目标'),('source','仅训练来源'),('paired','全域共同支持 (km²)'),('fraction','目标精确覆盖 (%)'),('abstained','全域拒答 (km²)')])
 htmltext+='<h2>分区结果</h2><p>ward_1 = Chiyoda；ward_2 = Chuo；ward_3 = Minato。完整全域及三个区的逐事件指标见 CSV；训练分数仅为拟合诊断，训练事件检索排除自身谱系。</p>'+table([display(r) for r in ward],[('scope','区')]+columns)
 htmltext+='<h2>训练与全域图</h2><img src="learning_curve.png" alt="Training loss and validation checkpoint selection"><img src="validation_whole_domain.png" alt="Four validation events, full-domain reference, surrogate, retrieval and error maps"><p class="muted">灰色表示未知或不在该图支持内。深度色标截断于 0.5 m，误差色标截断于 ±0.1 m，仅为显示；指标采用完整数值。峰值图不能表示同时积水状态或动态地表排水路径。</p>'
 htmltext+=f'<h2>可复现设置和限制</h2><ul><li>一次拟合、固定种子 {p["seed"]}、{env["parameters"]:,} 个参数，设备 {env["device"]}，PyTorch {env["torch"]}。最多 40 轮；每个训练事件每轮一次全域更新；第 1 轮和每 5 轮验证，连续 4 次无改善可停止。</li><li>58 个静态输入通道：29 个已知值标准化通道（含处方低水位）及其 29 个有效性标记。降雨及 7 组水位增量使用 240 个一分钟精确步积分均值；保留总量，压缩分钟内形状。原秒级曲线和屋面分配不改动。</li><li>归一化只拟合 12 个训练事件；共享固定静态地理只在训练实例中计一次。缺测数值归零并附有效性标记。标签不标准化，原区间/隔离/负值掩膜不进入精确监督。</li><li>AdamW 学习率 0.001、权重衰减 0.00001、面积加权 MSE、梯度范数上限 10；以验证父事件宏平均 MAE 选择一次检查点。GroupNorm、FiLM 条件化、双线性上采样、Softplus 非负输出；不翻转方向地理数据，不做参数搜索。</li><li>检索距离对训练归一化雨型和七组水位时序各赋 1/2 权重；七组重复相干脉冲不会获得七倍权重。来源仅限训练父事件；选择不接触验证标签。</li><li>拟合耗时 {d["fit_elapsed_seconds"]:.1f} s；特征已在设备上、预热后的全域前向中位数 {d["forward_median_seconds_prepared_features"]:.3f} s。这不是端到端流程耗时，也不与受暂停影响的水动力时间计算加速比。</li><li>四个验证事件用于选择模型，不能充当独立测试；单种子不能衡量优化方差。所有情景为固定地形、干初始状态、均匀降雨和相干水位脉冲。没有实测洪水精度、时间演变、守恒路径或主动措施效果的证据。</li></ul>'
 htmltext+='<h2>文件</h2><p>'+ ' · '.join(f'<a href="{name}">{name}</a>' for name in ['summary.json','event_metrics.csv','macro_metrics.csv','retrieval_choices.csv','label_support.csv','training_history.csv','inference_runtime.csv','environment.json','completion.json','delivery_validation.json','preflight_validation.json','preparation_qc.json','anasop_update.json'])+'</p><p>协议、训练集归一化、受限准入列表、最佳/末轮检查点和 16 个事件预测保存在对应 processed 目录。历史标签、原注册表及封存响应状态保持不变。</p></html>'
 (REPORT/'review.html').write_text(htmltext)
 (REPORT/'README.md').write_text('# First exact-label surrogate pilot\n\nSingle full-domain conditional U-Net fit on the authorized 12/4 original partitions. Development validation only; no sealed test.\n\nReproduce in order: prepare_three_ward_unet_exact_pilot.py, validate_three_ward_unet_exact_pilot.py (preflight), train_three_ward_unet_exact_pilot.py, validate_three_ward_unet_exact_pilot.py (delivery), review_three_ward_unet_exact_pilot.py. Existing prepared protocols and completed fits refuse silent overwrite; use a new version for a new experiment. Scripts live in src/preprocessing and src/analyses.\n\nSee review.html for event-level performance, shared support, abstention and interpretation limits.\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
