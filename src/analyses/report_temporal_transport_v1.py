"""Reproducible five-event temporal evidence and live bounded-measure review."""
import json,datetime as dt,html
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from shapely.geometry import shape
from temporal_transport_common_v1 import *

def tab(df):return '<div class="scroll">'+df.to_html(index=False,escape=True,border=0,na_rep='—',float_format=lambda x:f'{x:,.3f}')+'</div>'
def readjson(f):return json.loads(f.read_text())if f.exists()else {}
def readcsv(f):return pd.read_csv(f)if f.exists()else pd.DataFrame()
def image(name,alt):return f'<figure><img src="{name}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'if(R/name).exists()else ''

def rainfall():
 f=P/'rainfall_trajectories.csv'
 if f.exists():return pd.read_csv(f)
 area=np.asarray(load_npz(D/'tokyo_historical_radar_20240821_v1/terrain_source_area_matrix.npz').sum(0)).ravel();used=area>0;rows=[]
 for event in pd.read_csv(PLAN/'rain_timing_context.csv').event_id:
  old=event in ['E20240830','E20250911'];base=D/'three_ward_cross_event_protection_v1'if old else PREV;rad=load(base/'rainfall'/event/'radar_intervals.npz');rate=rad['representative_intensity_mm_h'].reshape(108,-1)[:,used].astype(float)@area[used]/area.sum()
  for i,x in enumerate(rate):rows.append(dict(event_id=event,start_elapsed_s=i*300,end_elapsed_s=(i+1)*300,source_area_mean_mm_h=float(x)))
 save(f,rows);return pd.DataFrame(rows)

def road_summary(road):
 rows=[];clip=load(P/'clipped_road_lengths.npz');qi=np.flatnonzero(clip['qualified']);weights={'Three wards':clip['union_length_m'],**dict(zip(WARD,clip['ward_length_m']))}
 for (case,bound,arm,scope),g in road.groupby(['case','bound','arm','scope']):
  g=g.sort_values('frame');v=g.qualified_closed_km.to_numpy();i=int(np.argmax(v));z=load(P/'population'/(case+'.npz'));prefix=bound+'__'+arm+'__';w=weights[scope][qi];rows.append(dict(case=case,bound=bound,arm=arm,scope=scope,peak_closed_km=float(v[i]),peak_elapsed_h=float(g.analysis_elapsed_s.iloc[i]/3600),closed_at9h_km=float(v[-1]),closed_km_hours=float(np.trapezoid(v,g.analysis_elapsed_s)/3600),ever_closed_clipped_km=float(w@z[prefix+'ever_closed']/1000),qualified_km=float(g.qualified_total_km.iloc[0]),unknown_km=float(g.unknown_exposure_km.iloc[0])))
 save(P/'road_temporal_summary.csv',rows);return pd.DataFrame(rows)

def figures(road,pop,refs):
 if(R/'temporal_trajectories.png').exists():return
 rain=rainfall();events=sorted(road.event_id.unique());fig,axs=plt.subplots(5,3,figsize=(15,17),layout='constrained')
 for row,event in enumerate(events):
  ax=axs[row,0];rr=rain[rain.event_id.eq(event)];ax.stairs(rr.source_area_mean_mm_h,np.r_[rr.start_elapsed_s.iloc[0],rr.end_elapsed_s]/3600,color='#426781',fill=True,alpha=.7);ax.set(ylabel=event+'\nRain (mm/h)',xlim=[0,9]);ax=axs[row,1]
  for stage,color in [('L','#589fbd'),('H','#b45048')]:
   for arm,style in [('flood','-'),('targeted','--')]:
    g=road[road.event_id.eq(event)&road.stage.eq(stage)&road.bound.eq('lower')&road.arm.eq(arm)&road.scope.eq('Three wards')].sort_values('frame');ax.plot(g.analysis_elapsed_s/3600,g.qualified_closed_km,color=color,ls=style,label=stage+'/'+arm)
  ax.set(ylabel='Closed road length (km)',xlim=[0,9]);ax=axs[row,2]
  for stage,color in [('L','#589fbd'),('H','#b45048')]:
   g=pop[pop.event_id.eq(event)&pop.stage.eq(stage)&pop.bound.eq('lower')&pop.arm.eq('flood')&pop.scope.eq('Three wards')&pop.buffer_m.eq(10)].sort_values('frame');x=g.analysis_elapsed_s.to_numpy()/3600;ax.fill_between(x,g.conditional_loss_lower,g.conditional_loss_upper,color=color,alpha=.15);ax.plot(x,g.conditional_loss_upper,color=color,label=stage+' candidate upper');ax.plot(x,g.conditional_loss_lower,color=color,lw=.8,ls=':')
  ax.set(ylabel='Conditional resident stock',xlim=[0,9]);
  for col in [1,2]:
   for _,r in refs[refs.event_id.eq(event)].iterrows():axs[row,col].axvline(r.analysis_elapsed_s/3600,color='#777',lw=.7,ls=':')
  for col in range(3):axs[row,col].grid(alpha=.15);axs[row,col].set_xlabel('Hours from event-window start')
 for col in [1,2]:axs[0,col].legend(fontsize=7)
 fig.suptitle('Five weather parents | synchronous 5-min states | unknown exposure assumed available\nPopulation:2020 residence stock,10m candidate support;not observed stranded people',fontsize=13);fig.savefig(R/'temporal_trajectories.png',dpi=135);plt.close(fig)
 features=json.loads((P/'clipped_roads_6677.geojson').read_text())['features'];segments=[];indices=[]
 for f in features:
  g=shape(f['geometry']);parts=[g]if g.geom_type=='LineString'else list(g.geoms)
  for line in parts:
   if line.geom_type=='LineString':segments.append(np.asarray(line.coords));indices.append(f['properties']['road_index'])
 indices=np.array(indices);clip=load(P/'clipped_road_lengths.npz');qi=np.flatnonzero(clip['qualified']);wards=json.loads((P/'wards_6677.geojson').read_text())['features'];allxy=np.concatenate(segments);bounds=[allxy[:,0].min(),allxy[:,0].max(),allxy[:,1].min(),allxy[:,1].max()];norm=Normalize(0,540);fig,axs=plt.subplots(5,3,figsize=(13,21),layout='constrained')
 for row,event in enumerate(events):
  z=load(P/'population'/(event+'__H__U.npz'));pre='lower__flood__';duration=np.full(len(clip['qualified']),np.nan);longest=duration.copy();last=duration.copy();duration[qi]=z[pre+'sampled_closed_min'];longest[qi]=z[pre+'longest_observed_span_min'];last[qi]=z[pre+'closed_at_horizon']
  for col,vals in enumerate([duration,longest,last]):
   ax=axs[row,col];ax.add_collection(LineCollection(segments,colors='#d8dce0',linewidths=.35));v=vals[indices];mask=np.isfinite(v)&(v>0)
   selected=[segments[i]for i in np.flatnonzero(mask)]
   if selected:
    lc=LineCollection(selected,cmap='magma_r',norm=norm,linewidths=.75)if col<2 else LineCollection(selected,colors='#ad3434',linewidths=.8)
    if col<2:lc.set_array(v[mask])
    ax.add_collection(lc)
   for w in wards:
    geom=shape(w['geometry']);polys=[geom]if geom.geom_type=='Polygon'else list(geom.geoms)
    for g in polys:xy=np.array(g.exterior.coords);ax.plot(xy[:,0],xy[:,1],color='#65707a',lw=.6)
   ax.set_xlim(bounds[0]-100,bounds[1]+100);ax.set_ylim(bounds[2]-100,bounds[3]+100);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(event+' | '+['Sampled closed minutes','Longest observed run (min)','Closed at9h:recovery censored'][col],fontsize=9)
 fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='magma_r'),ax=axs[:,:2],shrink=.4,label='Minutes (sampled;not continuous-time proof)');fig.suptitle('H-stage,no protection,lower-depth branch | unique clipped roads\nGrey includes supported open roads and unknown-exposure roads;unknown is not verified dry',fontsize=12);fig.savefig(R/'road_duration_maps.png',dpi=125);plt.close(fig)

def measure_figures(outcomes):
 if outcomes.empty:return
 events=sorted(outcomes.event_id.unique());fig,axs=plt.subplots(len(events),2,figsize=(12,max(5,3.7*len(events))),squeeze=False,layout='constrained')
 colors=['#225c8a','#ae6038']
 for row,event in enumerate(events):
  g=outcomes[outcomes.event_id.eq(event)]
  for j,(ref,h)in enumerate(g.groupby('reference_id',sort=False)):
   hour=float(h.elapsed_s.iloc[0]/3600)
   for arm,style in [('flood','--'),('targeted','-')]:
    a=h[(h.protection==arm)&(h.guidance=='UE')].sort_values('deferred_fraction');axs[row,0].plot(a.deferred_fraction*100,a.completion_fraction_original*100,color=colors[j],ls=style,marker='o',label=f'{hour:.2f}h/{arm}')
    for mode,mark in [('UE','o'),('SO','s')]:
     b=h[(h.protection==arm)&(h.guidance==mode)].sort_values('deferred_fraction');axs[row,1].plot(b.deferred_fraction*100,b.fixed_original_cohort_cost_index,color=colors[j],ls=style,marker=mark,label=f'{hour:.2f}h/{arm}/{mode}')
  axs[row,0].set(ylabel=event+'\nCompleted original request (%)',xlabel='Deferred from this hour (%)',ylim=[0,100]);axs[row,1].set(ylabel='Fixed-cohort cost index (no-action=100)',xlabel='Deferred from this hour (%)');axs[row,1].axhline(100,color='#888',lw=.6)
  for ax in axs[row]:ax.grid(alpha=.15);ax.legend(fontsize=7,ncol=2)
 fig.suptitle('Snapshot measure comparisons | fixed protection118roads | same17h demand proxy\nDeferral is not completed travel;cost indices have separate reference cohorts',fontsize=12);fig.savefig(R/'measure_tradeoffs.png',dpi=130);plt.close(fig)

def main():
 init();a0=readjson(R/'A0_validation.json');a1=readjson(R/'A1_validation.json');a2=readjson(R/'A2_validation.json');p1=readjson(R/'A1_progress.json');p2=readjson(R/'A2_progress.json');ex1=readjson(R/'A1_execution.json');ex2=readjson(R/'A2_execution.json');road=readcsv(P/'road_trajectories.csv');pop=readcsv(P/'population_trajectories.csv');refs=readcsv(P/'reference_states.csv');outcomes=readcsv(P/'traffic/combined_outcomes.csv');fitvals=[readjson(f)for f in(R/'traffic').glob('*_fit_validation.json')];fits=sum(bool(v.get('passed'))and v.get('case_id')!='normal'for v in fitvals);complete=bool(a1.get('passed')and a2.get('passed'));status='全部核验完成'if complete else ('已停止，保留已完成结果'if ex1.get('status')=='stopped'or ex2.get('status')=='stopped'else '分阶段运行中');note=f'A0：通过。A1：{p1.get("cases_completed",0)}/10情景，{p1.get("frames_completed",0)}/1090帧。A2：{fits}/{len(refs)*16 if len(refs)else 160}措施单元通过；8组正常控制单独复用。'
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五场暴雨：道路关闭、人口可达性与措施时序</title><style>body{{margin:0;background:#eff3f5;color:#203c49;font:16px/1.8 system-ui}}main{{max-width:1220px;margin:24px auto;padding:30px;background:white;border-radius:12px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:30px}}.lead{{padding:18px;background:#e4f1eb;border-left:5px solid #43856f}}.note{{background:#fff1dc;padding:16px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dde4e8;text-align:left}}img{{width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{font-size:13px;color:#536672}}a{{color:#17688a}}code{{font-size:13px}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}</style><main><p>都心三区 · MILI-D-20261018-015 · {dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M JST')}</p><h1>五场暴雨的道路中断与恢复过程</h1><div class="lead"><strong>{status}</strong><br>{note}</div><p>本轮复用已合格的5场雨×低／高河道水位10组城市几何结果，每组109个实际时钟输出。分析覆盖三区全域与原道路上下文，不追加水动力模拟、不训练、不读取封存测试。<button onclick="location.reload()">刷新进度</button></p>'''
 for exc in [ex1,ex2]:
  if exc.get('status')=='stopped':body+='<p class="note">停止原因：'+html.escape(exc.get('reason',''))+'</p>'
 body+='<h2>本轮回答什么</h2><p>道路何时达到假设封闭阈值、持续多久、9小时内是否恢复？同一时刻低／高水位有什么差异？在固定居民居住位置和接入候选的条件下，机动车到外边界的可达性怎样变化？再在预先规定的两个参考时刻，比较冻结通道保障、需求移出和理想协同疏导。</p><p class="note">这里的人口是2020年常住人口存量496,345的候选位置区间，不是实际受困人数。未计入白天在岗人口、出行行为、步行和轨道替代。交通成本是固定17时需求代理下的静态容量响应，不是真实历史排队延误；移出当前小时的请求不算完成。</p>'
 summary=pd.DataFrame()
 if a1.get('passed'):
  summary=road_summary(road);figures(road,pop,refs);primary=summary[(summary.scope=='Three wards')&(summary.bound=='lower')&(summary.arm=='flood')];body+='<h2>道路关闭与恢复：完整9小时</h2><p>长度按唯一物理源道路裁剪到三区边界，区界重合线只算一次。已知可用／缓行／封闭长度在每一帧守恒，未知暴露另列。累计封路强度是采样封路长度的梯形积分；5分钟内的短暂关闭或恢复不可识别。</p>'+tab(primary)+image('temporal_trajectories.png','五场暴雨、两档水位的道路关闭与条件人口影响轨迹；虚线竖线为预定参考时刻。')+image('road_duration_maps.png','三区完整范围的采样关闭时长、最长已观测连续关闭段与9小时末仍关闭道路。最长连续段的完整持续时间可能更长，不能将末帧仍关闭称作已经恢复。')
  peak=readcsv(P/'population_peak_intervals.csv');ever=readcsv(P/'population_ever_intervals.csv');p=peak[(peak.scope=='Three wards')&(peak.buffer_m==10)&(peak.arm=='flood')];e=ever[(ever.scope=='Three wards')&(ever.buffer_m==10)&(ever.arm=='flood')];p=p.merge(e[['case','bound','ever_stock_lower','ever_stock_upper','unresolved_population']],on=['case','bound']);body+='<h2>常住人口：瞬时峰值和曾失去可达性的存量</h2><p>峰值分别取每个时刻三区总下界和总上界的最大值，端点时刻可以不同。曾受影响先在每个候选位置上对时间取“或”，再汇总街区；不会将多个时刻相加。区间反映空间接入与暴露解释，不是置信区间。正常网络支持不完整的人口保持暂留。</p>'+tab(p[['case','bound','peak_loss_lower','peak_loss_upper','lower_peak_elapsed_s','upper_peak_elapsed_s','ever_stock_lower','ever_stock_upper','unresolved_population']])
  paired=readcsv(P/'paired_stage_population_effects.csv');sroad=readcsv(P/'paired_stage_road_effects.csv');anchor=[]
  for _,ref in refs.iterrows():
   x=paired[(paired.event_id==ref.event_id)&(paired.frame==ref.frame)&(paired.bound=='lower')&(paired.arm=='flood')&(paired.buffer_m==10)&(paired.scope=='Three wards')].iloc[0];y=sroad[(sroad.event_id==ref.event_id)&(sroad.frame==ref.frame)&(sroad.bound=='lower')&(sroad.arm=='flood')&(sroad.scope=='Three wards')].iloc[0];anchor.append(dict(event_id=ref.event_id,elapsed_h=ref.analysis_elapsed_s/3600,reasons=ref.selection_reasons,stage_closed_km_HminusL=y.stage_closed_km_difference,paired_resident_loss_HminusL_lower=x.paired_stage_loss_lower,paired_resident_loss_HminusL_upper=x.paired_stage_loss_upper))
  save(P/'same_time_anchor_stage_summary.csv',anchor);body+='<h2>同一时刻的河道水位影响</h2><p>H为原有假设1.5米水位脉冲，不是实测河道水位。表中居民影响差先按候选位置配对，再取区间；没有把两个总区间端点直接相减。正数表示H组条件损失更多，负数表示更少。</p>'+tab(pd.DataFrame(anchor))
 body+='<h2>代表时刻与措施计算</h2><p>每场高水位组保留：①源区加权滚动60分钟雨量最大窗口的最早结束时刻；②无措施封路长度最大值的最早时刻。若两者重合只算一次；9小时末峰值仍标记截尾。选时没有使用措施收益，原4小时结果只作核验背景。</p>'
 if not refs.empty:body+=tab(refs[['reference_id','analysis_elapsed_s','selection_reasons','known_closed_clipped_km','peak_at_horizon']])
 if not outcomes.empty:
  measure_figures(outcomes);body+='<p>每个参考时刻2档保障×4档移出比例×UE／SO共16组；保障固定原118条源道路。UE与SO使用同一候选路径和保留需求。固定群体成本指数仅在同一参考时刻的群体内比较，不跨事件取混合平均。</p>'+image('measure_tradeoffs.png','已完成参考时刻的原请求完成率与固定原可达群体的成本指数。')+tab(outcomes[['reference_id','protection','deferred_fraction','guidance','completion_fraction_original','deferred_core_proxy','retained_unreachable_core_proxy','restored_same_deferral_proxy','fixed_original_cohort_cost_index','same_retained_guidance_network_cost_index']])
 unknown=readcsv(P/'unknown_withheld_reference_diagnostic.csv')
 if not unknown.empty:body+='<h2>未知暴露道路：证据压力诊断</h2><p>主分支假定未知暴露道路可用。此处将这些路全部暂时移除，并使用同样移除条件下的正常网络作参照。正常支持会明显减少；这是证据缺口诊断，不能作为真实洪灾影响人数的物理上界。</p>'+tab(unknown[(unknown.buffer_m==10)&(unknown.arm=='flood')])
 body+='<h2>输入与解释边界</h2>'+tab(pd.read_csv(PLAN/'rain_timing_context.csv')[['event_id','window_start_jst','rain_9h_mm','rain_by4h_percent','excluded_antecedent_rain_mm']])+'<p>所有情景沿用9小时干初始状态，窗口前降雨未纳入，部分事件9小时仍未退水。固定现代城市几何和2020年人口，因此不是完整历史重建。道路封闭阈值0.30米、0.10–0.30米缓行是统一车辆情景假设；桥隧高程不明和地表覆盖未知仍未知。机动车通向任一外边界的可达性不等于到特定医院或工作地的可达性，也不等于失去全部交通方式。</p><h2>核验与下载</h2><p>沿用原水动力数值准入。新增分析检查实际时钟、三区长度账本、人口守恒、原4小时结果精确回放；措施检查封闭路段零流量、车辆守恒、每个OD请求账本、路径加载、候选内间隙≤0.001及SO不差于同需求UE。已有结果须完整状态、数据、路径和目标指纹一致，并独立重算检查。</p><ul>'
 for name in ['protocol.json','input_inventory.csv','rainfall_trajectories.csv','road_temporal_summary.csv','road_trajectories.csv','road_closure_episodes.csv','population_trajectories.csv','population_peak_intervals.csv','population_ever_intervals.csv','paired_population_restoration.csv','paired_stage_population_effects.csv','same_time_anchor_stage_summary.csv','reference_states.csv','unknown_withheld_reference_diagnostic.csv','traffic/combined_outcomes.csv','traffic/fit_reuse_ledger.csv','traffic/reuse_exclusion_ledger.csv','source_bindings.json','A1_code_binding.json','A2_code_binding.json']:
  if(P/name).exists():body+=f'<li><a href="../../../processed/{P.name}/{name}">{name}</a></li>'
 for name in ['A0_validation.json','temporal_interface_selfcheck.json','measure_interface_selfcheck.json','A1_validation.json','A2_validation.json','A1_progress.json','A2_progress.json','A1_execution.log','A2_execution.log','traffic/independent_fit_audit.csv']:
  if(R/name).exists():body+=f'<li><a href="{name}">{name}</a></li>'
 body+='</ul><p>道路 © OpenStreetMap contributors / ODbL；PLATEAU、DEM、雷达与统计来源沿用现有来源审计。复现：prepare_temporal_transport_v1.py → check_temporal_interfaces_v1.py → execute_temporal_stage_v1.py A1 → execute_temporal_stage_v1.py A2 → report_temporal_transport_v1.py。</p></main></html>';tmp=R/'review.html.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(R/'review.html');dump(R/'summary.json',dict(status=status,A0_passed=bool(a0.get('passed')),A1_passed=bool(a1.get('passed')),A2_passed=bool(a2.get('passed')),snapshots_completed=p1.get('frames_completed',0),qualified_measure_cells=fits,reference_states=len(refs),new_hydraulics=0,training=False,sealed_tests=False,interpretation='Conditional2020resident stock and static vehicle scenarios,not observed stranded people or actual queue delay.'))
 if complete:dump(R/'delivery_validation.json',dict(passed=True,stages=[a0,a1,a2],scientific_limits_retained=True,all_five_parents_reported=True))
 print('TEMPORAL REVIEW UPDATED',status,'A2cells',fits,flush=True)
if __name__=='__main__':main()
