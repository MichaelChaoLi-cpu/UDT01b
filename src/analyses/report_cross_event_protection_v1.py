"""Truthful live campaign review and, after qualification, fixed-plan transfer results."""
import os
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
from pathlib import Path
import sys,json,datetime as dt,collections,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src/preprocessing'),str(ROOT/'src/analyses')]
from prepare_cross_event_protection_v1 import P,R,D,OLD,MEASURE,MOTOR,load,read,write,dump,sha
from compare_urban_geometry_pair_controls_v1 import table
LABEL={'none':'不干预','corridor':'关键通道保障','distributed':'分散道路保障'}

def main():
 protocol=json.loads((D/'protocol.json').read_text());events=protocol['events'];state=json.loads((R/'campaign_state.json').read_text())if(R/'campaign_state.json').exists()else dict(status='inputs_prepared_not_started',completed_new_cases=[],queue=protocol['execution_order'],hydraulic_cases_started=0);status=state['status'];completed=len(state['completed_new_cases']);rows=[]
 for e in events:
  rows.append([e['event_id'],e['window_start_jst'].replace('T',' ')[:16]+' — '+e['window_end_jst'].replace('T',' ')[11:16],f'{e["reconstructed_rain_mm"]:.2f}',f'{e["peak_rolling60min_reconstructed_mm"]:.2f}',f'{e["peak60_fraction_of9h"]*100:.1f}%',e['wet_minutes_above1mm_h'],f'{e["rain_before_window_reconstructed_mm"]:.2f}'])
 # One static rainfall-only comparison; no model outcomes used for event selection.
 if not(R/'rainfall_profiles.png').exists():
  fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained');profiles=[]
  for event in events:
   rad=load(P/'rainfall'/event['event_id']/'radar_intervals.npz');from scipy.sparse import load_npz
   area=np.asarray(load_npz(OLD/'terrain_source_area_matrix.npz').sum(axis=0)).ravel();used=area>0;rate=rad['representative_intensity_mm_h'].reshape(108,-1)[:,used].astype('f8')@area[used]/area.sum();profiles.append(rate);x=np.arange(108)*5/60;axs[0].step(x,rate,where='post',label=event['event_id']);axs[1].plot(np.arange(109)*5/60,np.r_[0,np.cumsum(rate/12)]/np.sum(rate/12),label=event['event_id'])
  original=load(OLD/'radar_intervals.npz');rate=original['representative_intensity_mm_h'].reshape(108,-1)[:,used].astype('f8')@area[used]/area.sum();axs[0].step(x,rate,where='post',label='Original20240821',color='#777777',alpha=.65);axs[1].plot(np.arange(109)*5/60,np.r_[0,np.cumsum(rate/12)]/np.sum(rate/12),label='Original20240821',color='#777777',alpha=.65)
  axs[0].set(xlabel='Hours since selected window start',ylabel='Catchment-weighted representative rate (mm/h)',xlim=[0,9]);axs[1].set(xlabel='Hours since selected window start',ylabel='Fraction of reconstructed9h rainfall',xlim=[0,9],ylim=[0,1]);[ax.legend(fontsize=8)for ax in axs];fig.suptitle('Rainfall-only window selection | unchanged5min reconstruction');fig.savefig(R/'rainfall_profiles.png',dpi=150);plt.close(fig)
 body='<h1>两场新增历史降雨：固定道路保障方案验证</h1><p class="lead">保持约4.77 km的两套选路方案不变，检验模型内道路联系改善能否跨降雨事件保留。</p>'
 body+=f'<aside><b>当前状态：{html.escape(status)}；数值合格 {completed}/8 组。</b><p>本轮新增8组水动力算例，全部合格后接入3种道路保障情景，共1248组网络状态。未完成的算例不作为结果，也不提前判断措施是否有效。</p></aside>'
 if state.get('current_case'):body+='<p>当前算例：<b>'+html.escape(state['current_case'])+'</b>；已计算模型时间 '+str(state.get('latest_native_time_s'))+' /32400秒；该例活动计算时间 '+f'{state.get("case_active_wall_seconds",0)/60:.1f}'+' 分钟。模型时间来自实际诊断时钟。</p>'
 if state.get('error'):body+='<p class="error">已停止队列：'+html.escape(state['error'])+'。失败尝试保留，未自动重试或放宽阈值。</p>'
 body+='<h2>输入降雨与事件窗口</h2><p>两场使用与原事件相同的JMA参数203、5分钟、约1 km全国合成雷达代表强度，经RISH公开数值档案获取。先在完整调查时段内，仅按汇水面积加权雨量选出最大9小时窗口；不依据积水或道路保障效果选择时间。10 m仅为水动力接收网格，未把雷达分辨率提高到10 m。</p>'+table(['事件','日本时间窗口','9h重建雨量mm','最强60min重建雨量mm','最强60min占比','平均强度>1mm/h的分钟数','窗口前调查雨量mm'],rows)+'<img src="rainfall_profiles.png" alt="Radar rainfall profiles selected without access outcomes">'
 body+='<p><b>历史再现边界：</b>同一干初始地表、9小时窗口、无管网排水与入渗。2024年窗口前已有约43 mm面积加权重建降雨未施加，本轮不能复原其前期蓄水。2025年窗口覆盖约99%当日调查雨量，但仍无实际路面封闭验证。数值雨场是代表强度积分重建，不能称为精确观测累积雨量。</p>'
 body+='<h2>8组算例与计算预算</h2>'+table(['批次','新增算例','比较用途'],[[b['batch_id'],' / '.join(b['new_cases']),['第一场城市约束：低／抬升水位','第一场仅地形：低／抬升水位','第二场城市约束：低／抬升水位','第二场仅地形：低／抬升水位'][i]]for i,b in enumerate(protocol['batches'])])
 body+='<p>T=仅地形；U=加入建筑阻挡、道路约束及屋面分配的城市信息组合。均为同一二维求解器和10 m网格。H沿用相对窗口起点+3h开始、+3.5h达到1.5 m抬升、+4h开始回落、+5h回到基线的7组联合水位假设，L不抬升；没有假称其为当日实测河道水位，也没有逐事件重新调整水位以制造有利效果。</p><p>一次运行1例，6 CPU、7 GiB；每例最多2小时，8例活动计算上限16小时，新增输出上限120 GiB，至少保留100 GiB可用空间。按既有运行记录估计约6–12小时，非完成保证。任何原有数值准入、守恒、正性、关闭边界零交换、实际时钟或输入版本检查失败即停止队列，不自动重试。PAUSE／STOP标志分别支持暂停和停止。</p>'
 body+='<h2>固定措施及共同支持</h2><p>不干预、关键通道保障、分散道路保障三方案完全沿用已冻结道路清单，长度约4.77 km。保障假设只解除选中且有可靠地面对应道路的积水关闭门，不改水位、不增道路、不改权限／单行／转向，不把桥隧未知道路变成已知可用。措施尚未包含工程失效、费用、积水转移或容量影响。</p><p>先交集原12组与新增8组数值合格地表掩膜。失去支持的道路转为未知，重新计算匹配的正常未知分支；原93个出向／89个双向分析点作为支持上限，若新增缺口导致样本减少会单列。固定道路清单不重选，但缺乏支持的所选道路不会被强行恢复。保留原道路阈值0.1／0.3 m和两档未知分支。</p>'
 if(R/'motor_summary.json').exists():
  summary=json.loads((R/'motor_summary.json').read_text());val=json.loads((R/'motor_validation.json').read_text());assert val['all_passed'];body+=f'<h2>已完成的跨事件道路结果</h2><p>8组水动力及1248组道路网络状态已通过核验。共同支持出向点{summary["eligible_outward"]}个、双向{summary["eligible_two_direction"]}个；相较原出向93点，新增支持损失{summary["additional_support_attrition"]}个，单列证据不足。</p>'
  rows=summary['fixed_elapsed4h_outward'];body+='<p>以下固定为各窗口起点后4小时，时间含义保持一致，实际钟点因事件而异。稳健断连减少可以只转为条件敏感，不能等同全面恢复。</p>'+table(['算例','方案','稳健出向断连','条件敏感','四分支均有替代','证据不足'],[[r['case_id'],LABEL[r['arm']],r['robust_modeled_loss'],r['conditional_sensitivity'],r['retained_access'],r['insufficient_evidence']]for r in rows])
  inv={a:{r['case_id']:r for r in rows if r['arm']==a}for a in LABEL};cmp=[]
  for case in inv['none']:
   for arm in ['corridor','distributed']:cmp.append(dict(case_id=case,arm=arm,baseline_robust_loss=int(inv['none'][case]['robust_modeled_loss']),protected_robust_loss=int(inv[arm][case]['robust_modeled_loss']),robust_loss_reduction=int(inv['none'][case]['robust_modeled_loss'])-int(inv[arm][case]['robust_modeled_loss']),fully_retained_gain=int(inv[arm][case]['retained_access'])-int(inv['none'][case]['retained_access'])))
  write(R/'fixed_elapsed4h_transfer_comparison.csv',cmp)
  # Same-time transfer improvements over all12slices, without asserting duration from sparse slices.
  cls=read(R/'motor_classification_summary.csv');ix={(r['case_id'],r['frame_id'],r['direction'],r['arm']):r for r in cls};temporal=[]
  for case in inv['none']:
   for arm in ['corridor','distributed']:
    frames=[str(t)for t in protocol['grid']['times_s']];delta=[int(ix[case,f,'outward','none']['robust_modeled_loss'])-int(ix[case,f,'outward',arm]['robust_modeled_loss'])for f in frames];temporal.append(dict(case_id=case,arm=arm,matched_synchronous_slices=len(frames),slices_with_fewer_robust_losses=sum(d>0 for d in delta),largest_same_time_robust_loss_reduction=max(delta),not_duration=True))
  write(R/'all_synchronous_slice_transfer.csv',temporal)
  points=read(R/'motor_probe_classification.csv');probes=read(MOTOR/'probe_inventory.csv');pointix={(r['arm'],r['case_id'],r['frame_id'],r['probe_id']):r for r in points};colors={'insufficient_evidence':'#969fa7','robust_modeled_loss':'#b94a48','conditional_sensitivity':'#db9d35','retained_access':'#378a79'};grid=load(ROOT/'data/processed/three_ward_common_depth_labels/common_support_10m.npz')['Ward Code'];extent=[-11.36,-3.66,-42.44,-32.32];fig,axs=plt.subplots(2,3,figsize=(12,10),layout='constrained');persistent=[]
  for ri,event in enumerate(events):
   case=event['event_id']+'__H__U'
   for ci,arm in enumerate(LABEL):
    ax=axs[ri,ci];ax.imshow(np.where(grid>0,grid,np.nan),origin='upper',extent=extent,cmap='Greys',vmin=0,vmax=15,alpha=.35);rr=[pointix[arm,case,'14400',p['source_display_cell_id']]for p in probes]
    for k,color in colors.items():
     pp=[p for p,r in zip(probes,rr)if r['classification']==k];ax.scatter([float(p['x_m']or p['cell_center_x_m'])/1000 for p in pp],[float(p['y_m']or p['cell_center_y_m'])/1000 for p in pp],s=23,c=color,edgecolor='white',linewidth=.25)
    ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',title=event['event_id']+' | '+arm,xlabel='Easting(km)')
  from matplotlib.lines import Line2D
  fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,label=l)for c,l in zip(colors.values(),['Insufficient evidence','Robust modeled loss','Assumption-sensitive','Retained alternative'])],loc='outside lower center',ncol=2);fig.suptitle('Fixed4h after each window start | raised-stage assumption | urban geometry');fig.savefig(R/'cross_event_access_4h.png',dpi=150);plt.close(fig)
  for arm in LABEL:
   for pr in probes:
    both=[pointix[arm,e['event_id']+'__H__U','14400',pr['source_display_cell_id']]['classification']for e in events]
    if all(v=='robust_modeled_loss'for v in both):persistent.append(dict(arm=arm,probe_id=pr['source_display_cell_id'],x_m=pr['x_m'],y_m=pr['y_m'],criterion='Both additional events at elapsed4h,U,raised-stage,all4road uncertainty branches',not_entire_town_or_duration=True))
  if persistent:write(R/'persistent_4h_probe_candidates.csv',persistent)
  else:
   import csv
   with (R/'persistent_4h_probe_candidates.csv').open('w',newline='',encoding='utf-8-sig')as f:csv.DictWriter(f,fieldnames=['arm','probe_id','x_m','y_m','criterion','not_entire_town_or_duration']).writeheader()
  body+='<img src="cross_event_access_4h.png" alt="Regional fixed4h motor access comparison across new events"><p>两场新事件在相同相对时刻、U组和抬升水位下均稳健断连的候选点，按不干预／关键通道／分散方案分别为 '+str([sum(r['arm']==a for r in persistent)for a in LABEL])+'。这是同一分析点在两场条件情景中重复出现，不代表整个街区或全时段持续受困，位置清单另列。</p>'
  body+='<p>全时刻配对差异另见下载表；不把非同时峰值当作断连持续时间。方向恢复与距离节省见motor_directional_benefits.csv，距离节省只比较不干预时仍可达的同一批方向对，新恢复方向单列。未使用人口权重或车流需求。</p>'+table(['方案','冻结总长度m','共同支持下可核验保障长度m'],[[LABEL[a],f'{summary["fixed_length_m"][a]:.2f}',f'{summary["effective_known_length_m"][a]:.2f}']for a in LABEL])
 else:body+='<h2>尚无跨事件措施效果结论</h2><p>当前页面只交付雨场、输入合同及执行状态。已完成的单场结果不会冒充这两场新事件的结果。只有全部原数值检查通过后，才会生成措施效果表。</p>'
 body+='<h2>解释范围与复核文件</h2><p>不同日期构成新增历史父事件，同一事件的T/U和L/H均是其后代，不能重复计为独立事件。两场新增暴雨也不能证明普遍最优；道路方案同时改变选路依据和空间分布，不是单一集中度效应。保持现代城市／道路快照、人口不确定性和原12＋4机器学习划分，不训练、不打开封存测试、不计算受困人数、拥堵或实际工程防灾效果。</p><ul>'
 files=[D/'protocol.json',D/'rainfall_selection_protocol.json',P/'event_selection.json',P/'run_manifest.json',P/'audit_metadata_adapter.json',P/'audit_metadata_only.diff',R/'input_validation.json',R/'plan_validation.json',R/'forcing_volume_preflight.csv',R/'campaign_state.json',R/'motor_summary.json',R/'motor_validation.json',R/'motor_classification_summary.csv',R/'motor_directional_benefits.csv',R/'fixed_elapsed4h_transfer_comparison.csv',R/'all_synchronous_slice_transfer.csv',R/'persistent_4h_probe_candidates.csv']
 for f in files:
  if f.exists():body+='<li><a href="'+os.path.relpath(f,R)+'">'+f.name+'</a></li>'
 for e in events:body+='<li><a href="http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/jma-radar/synthetic/original/'+e['survey_start_jst'][:10].replace('-','/')+'/">RISH数值雷达档案 '+e['event_id']+'</a>；<a href="'+html.escape(e['gauge_context'],quote=True)+'">JMA日值数据：仅作事件背景</a></li>'
 body+='</ul><p>研究决定：MILI-D-20261017-011。更新时间 '+dt.datetime.now(dt.timezone.utc).isoformat()+'。<button onclick="location.reload()">刷新状态</button></p>'
 (R/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>跨事件固定道路保障验证</title><style>body{font:16px/1.85 system-ui;max-width:1250px;margin:auto;padding:28px;color:#263e4b;background:#f4f7f8}h1,h2{line-height:1.4}h2{margin-top:32px}.lead{font-size:20px}aside{background:#e1ece8;border-left:5px solid #397f6d;padding:18px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{border:1px solid #cddae2;padding:9px;text-align:left}th{background:#e7eef2}.scroll{overflow:auto}img{max-width:100%}a{color:#17648c}.error{background:#fae7e1;padding:15px}button{padding:7px;font:inherit}</style>'+body+'</html>')
 print('CROSS-EVENT REPORT',status,completed,'/8',flush=True)
if __name__=='__main__':main()
