"""Add paired congestion-cost tradeoffs and regional pressure-transfer evidence.

Read completed scenario runs only; preserve the initial review. No rerouting or fit.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-mpl')
for k in ['PROJ_LIB','PROJ_DATA','GDAL_DATA']:os.environ.pop(k,None)
from pathlib import Path
import sys,json,gzip,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
import demand_protection_feedback_v1 as m
P=m.P;R=m.R
def main():
 m.check_sources();assert json.loads((R/'validation.json').read_text())['passed']
 protocol=json.loads((P/'protocol.json').read_text());dem=m.load(m.REP/'multi_routing_demands.npz');pr=m.load(m.M/'road_projection.npz');inv=pd.read_csv(m.M/'road_inventory.csv');core=m.load(P/'protection_masks.npz')['core_endpoint_od'];ridx=pr['arc_to_road'];known=m.load(m.F/'motor_common_support.npz')['qualified_roads'];rows=[];pressure=[];central={};availability={}
 # Exact clipped source-road lengths for pressure transitions;one road counted once.
 boundary=unary_union(gpd.read_file(ROOT/'data/processed/plateau_water/central_three_ward_boundaries.geojson').to_crs(6677).geometry.to_numpy());lookup={str(k):i for i,k in enumerate(inv.source_edge_id)};geoms={};length=np.zeros(len(inv))
 with gzip.open(m.N/'surface_physical_edges.jsonl.gz','rt')as h:
  for line in h:
   e=json.loads(line);j=lookup.get(e['edge_id'])
   if j is not None and inv.inside_core.iloc[j]:
    g=shape(e['geometry']).intersection(boundary)
    if not g.is_empty:geoms[j]=g;length[j]=g.length
 for spec in protocol['schedule']:
  name=spec['name'];f=m.load(P/f'flood__{name}.npz');requested=f['requested'];mask=core&(dem['base']if spec['shape']=='base'else dem['through_plus']);paired=mask&f['reachable'];vf=f['arc_flow']/f['capacity'];closed_f=m.road_state('flood')[0][ridx]
  for arm in ['old','targeted']:
   z=m.load(P/f'{arm}__{name}.npz');assert (z['reachable'][paired]).all();difference=z['od_mean_cost_seconds'][paired]-f['od_mean_cost_seconds'][paired];w=requested[paired]
   rows.append(dict(scenario=name,arm=arm,paired_originally_reachable_proxy=float(w.sum()),mean_model_cost_saving_min=float(-w@difference/w.sum()/60),cost_increase_over1min_proxy=float(w[difference>60].sum()),cost_increase_over1min_share=float(w[difference>60].sum()/w.sum()),cost_decrease_over1min_proxy=float(w[difference < -60].sum()),threshold_seconds=60,not_observed_delay=True))
   vp=z['arc_flow']/z['capacity'];closed_p=m.road_state(arm)[0][ridx];common=~closed_f&~closed_p;new=common&(vf<=1)&(vp>1);relief=common&(vf>1)&(vp<=1);reopened=closed_f&~closed_p
   flags={}
   for key,sel in [('new_overload_known',new&known[ridx]),('new_overload_unknown',new&~known[ridx]),('relieved_overload_known',relief&known[ridx]),('relieved_overload_unknown',relief&~known[ridx]),('reopened_any',reopened),('reopened_overload',reopened&(vp>1))]:flags[key]=np.bincount(ridx,weights=sel.astype(int),minlength=len(inv))>0
   pressure.append(dict(scenario=name,arm=arm,**{key+'_clipped_source_km':float(length[flag].sum()/1000)for key,flag in flags.items()},unit='ward-clipped source-road km,one road counted once if any permitted direction crosses threshold;different categories may overlap'))
   if name=='central':central[arm]=flags
  if name=='central':
   for arm in m.ARMS:
    z=m.load(P/f'{arm}__{name}.npz');closed=m.road_state(arm)[0];availability[arm]=np.where(closed[ridx],0.,z['capacity'])
 m.save(P/'paired_cost_tradeoffs.csv',rows);m.save(P/'paired_pressure_transfer.csv',pressure);np.savez_compressed(P/'central_available_capacity.npz',**availability)
 # Pressure on roads that were already open differs from traffic on restored roads.
 fig,axs=plt.subplots(1,2,figsize=(10,8),layout='constrained');bounds=np.array(boundary.bounds)/1000
 def parts(g):
  if g.geom_type=='LineString':return[np.asarray(g.coords)/1000]
  return[np.asarray(x.coords)/1000 for x in getattr(g,'geoms',[])if x.geom_type=='LineString']
 for ax,arm in zip(axs,['old','targeted']):
  ax.add_collection(LineCollection([x for g in geoms.values()for x in parts(g)],colors='#dce3e5',linewidths=.35))
  for k,c in [('reopened_any','#2679aa'),('relieved_overload_known','#3e9369'),('relieved_overload_unknown','#3e9369'),('new_overload_known','#c14742'),('new_overload_unknown','#9e4aa3')]:
   ax.add_collection(LineCollection([x for j,g in geoms.items()if central[arm][k][j]for x in parts(g)],colors=c,linewidths=1.6))
  ax.set(xlim=[bounds[0]-.1,bounds[2]+.1],ylim=[bounds[1]-.1,bounds[3]+.1],aspect='equal',title=arm,xlabel='Easting(km)',ylabel='Northing(km)')
 fig.legend(handles=[Line2D([0],[0],color=c,lw=3,label=l)for c,l in [('#2679aa','Restored road'),('#3e9369','Pressure relieved on open road'),('#c14742','New pressure:known surface'),('#9e4aa3','New pressure:unknown surface')]],loc='outside lower center',ncol=2);fig.suptitle('Central assumed capacity | pressure shifts relative to flooded no-action');fig.savefig(R/'pressure_transfer.png',dpi=150);plt.close(fig)
 metrics=pd.read_csv(P/'paired_capacity_response_metrics.csv');benefits=pd.read_csv(P/'protection_benefits.csv');transfer=pd.read_csv(P/'transfer_access_check.csv');trade=pd.DataFrame(rows);press=pd.DataFrame(pressure)
 primary=metrics[metrics.scenario.eq('central')].set_index('arm');b=benefits[benefits.scenario.eq('central')].set_index('arm');tt=trade[trade.scenario.eq('central')].set_index('arm');transferidx=transfer.set_index('arm');lost=float(transferidx.loc['flood','unreachable_proxy']);transfer_gain={a:(lost-float(transferidx.loc[a,'unreachable_proxy']))/lost for a in ['old','targeted']}
 high=trade[trade.scenario.eq('high_demand')&trade.arm.eq('targeted')].iloc[0]
 result=dict(primary_new_lost_fraction={a:float(primary.loc[a,'new_unreachable_share'])for a in m.ARMS},primary_restored_fraction={a:float(b.loc[a,'restored_fraction_of_flood_lost'])for a in ['old','targeted']},primary_paired_cost_tradeoff=trade[trade.scenario.eq('central')].to_dict('records'),demand_capacity_sensitivity_tradeoff=trade[trade.arm.eq('targeted')].to_dict('records'),transfer_elapsed18000_restored_fraction=transfer_gain,primary_pressure_transfer=press[press.scenario.eq('central')].to_dict('records'),interpretation='In-sample ideal accessibility design,restricted-path assumed capacity response,one adjacent-time transfer;not full-network equilibrium,observed delays or engineering effectiveness.')
 m.dump(R/'interpretation.json',result)
 initial=R/'review_initial.html'
 if not initial.exists():shutil.copyfile(R/'review.html',initial)
 text=initial.read_text();lead=f'''<div class="note"><b>首要结果：</b>在同一选定时刻和中央需求下，新方案将新增不可达出行代理比例从{primary.loc['flood','new_unreachable_share']*100:.1f}%降至{primary.loc['targeted','new_unreachable_share']*100:.1f}%，恢复{b.loc['targeted','restored_fraction_of_flood_lost']*100:.1f}%的原新增不可达需求（原方案恢复{b.loc['old','restored_fraction_of_flood_lost']*100:.1f}%）。在另一已有时刻，冻结新方案恢复{transfer_gain['targeted']*100:.1f}%。这是情景与样内设计结果，不是实测恢复率。</div>'''
 text=text.replace('<h2>选路目标与公平比较</h2>',lead+'<h2>选路目标与公平比较</h2>')
 table=lambda d:'<div class="scroll">'+d.to_html(index=False,border=0,float_format=lambda x:f'{x:,.3f}')+'</div>'
 added=f'''<h2>恢复可达与局部成本上升须同时报告</h2><p>在无措施下已可达的同一批出行中，新方案平均模型成本节省{tt.loc['targeted','mean_model_cost_saving_min']:.2f}分钟；但其中{tt.loc['targeted','cost_increase_over1min_share']*100:.1f}%的出行代理，模型成本增加超过1分钟。下表将1分钟作为实质变化的展示阈值，原结果中任何微小数值变化不当作实质变慢。</p>{table(trade[trade.scenario.eq('central')])}<p>受淹后有些需求根本无法完成，静态分配不把它们留在道路上排队，因此平均成本或过载减少不能解释为暴雨缓解交通。新方案恢复车辆后可能使其他出行变慢。这里是容量函数下的成本响应，仍不等同真实排队时间。</p><h2>同一批原本开放道路上的压力转移</h2>{table(press[press.scenario.eq('central')])}<p>上述长度按三区边界裁剪，每条源道路最多计一次，若任一允许方向跨过v/c=1就标记。新出现压力只在无措施与措施都开放的道路中比较；恢复道路另列。不同类别可能在同一条道路不同方向出现，不能直接相加。未知地表对应与可靠地表对应分开。</p><img src="pressure_transfer.png" alt="Restored roads and hypothetical pressure shifts,with unknown surface correspondence separated"><p>已有主表中的有向弧压力长度按与三区相交道路整段计、含方向；不能与本节裁剪后的源道路长度混用。求解器内部capacity字段是计算分母，封闭通过路径掩膜强制零流量；另导出central_available_capacity.npz将实际关闭方向的可用容量明确置零。</p><p><a href="../../../processed/{m.NAME}/paired_cost_tradeoffs.csv">配对成本权衡</a> · <a href="../../../processed/{m.NAME}/paired_pressure_transfer.csv">同路段压力转移</a> · <a href="../../../processed/{m.NAME}/central_available_capacity.npz">中央可用容量</a></p>'''
 sensitivity=f'''<h2>高需求／低容量下出现反转</h2><p class="note">高需求（原先验×2）下，新方案恢复可达性，但对无措施下原本可达的同一批出行，平均模型成本反而增加{-high.mean_model_cost_saving_min:.2f}分钟，{high.cost_increase_over1min_share*100:.1f}%的出行代理增加超过1分钟。恢复通行与缓解拥堵压力不是同一个目标。低容量（容量×0.5）具有相同的需求/容量比例，结果由齐次关系得到，不能当作第二次独立证据。</p>{table(trade[trade.arm.eq('targeted')])}<p>这组敏感性结果支持后续比较通道保障与主动需求削减、交通疏导的组合，避免只优化恢复出行比例。外部穿越增量也会改变收益和成本转移。不同需求下保持不变的连通恢复率是固定封路拓扑的结果，不是对真实交通不确定性的经验稳健性证明。</p>'''
 text=text.replace('<h2>冻结方案后，另一个时刻是否有效</h2>',added+sensitivity+'<h2>冻结方案后，另一个时刻是否有效</h2>');(R/'review.html').write_text(text)
 m.dump(R/'interpretation_validation.json',dict(passed=True,cost_tradeoff_same_original_survivors=True,pressure_common_open_arcs_only=True,source_lengths_clipped_once=True,closed_available_capacity_zero=True,original_review_retained=True))
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
