"""Gate-triggered read-only interpretation of the frozen physical-measure campaign.
No new hydraulic cases, fitted parameters, population allocation or quality masks.
"""
import argparse, datetime as dt, hashlib, html, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/processed/three_ward_barrier_drainage_v1'
R=ROOT/'data/exp/estimation'/P.name
O=P/'effect_summary'
V=R/'effect_summary'
ARMS=['barrier','drainage','combined']

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));tmp.replace(p)
def gate():
 if (P/'STOP').exists():raise RuntimeError('User STOP')
 while (P/'PAUSE').exists():
  if (P/'STOP').exists():raise RuntimeError('User STOP')
  time.sleep(5)
def percentage(avoided,baseline):return np.where(np.asarray(baseline)!=0,100*np.asarray(avoided)/np.where(np.asarray(baseline)!=0,baseline,1),np.nan)
def avoided_interval(added_lower,added_upper):
 assert np.all(np.asarray(added_lower)<=np.asarray(added_upper))
 return -np.asarray(added_upper),-np.asarray(added_lower)

def compare(frame,keys,metrics):
 assert not frame.duplicated(keys+['measure']).any()
 base=frame[frame.measure=='none'][keys+metrics]
 joined=frame[frame.measure.isin(ARMS)].merge(base,on=keys,suffixes=('_measure','_none'),validate='many_to_one')
 assert len(joined)==len(frame[frame.measure.isin(ARMS)])
 for m in metrics:
  joined[m+'_avoided']=joined[m+'_none']-joined[m+'_measure']
  joined[m+'_avoided_pct']=percentage(joined[m+'_avoided'],joined[m+'_none'])
 return joined

def tables():
 roads=pd.read_csv(P/'road_summary.csv')
 road=compare(roads,['event_id','stage','bound','scope','stratum'],['peak_closed_km','closed_km_hours','closed_at9h_km'])
 road.to_csv(O/'road_reductions.csv',index=False)
 traffic=pd.read_csv(P/'traffic/physical_traffic_outcomes.csv')
 tr=compare(traffic,['event_id','stage','frame'],['unreachable_proxy','common_survivor_cost_index'])
 # Index is relative to the identical no-measure common-OD cost (=100).
 assert np.allclose(tr.common_survivor_cost_index_none,100)
 tr['common_od_cost_reduction_pct']=100-tr.common_survivor_cost_index_measure
 tr.to_csv(O/'traffic_reductions.csv',index=False)
 paired=pd.read_csv(P/'paired_population_effects.csv')
 anchors=traffic[traffic.measure=='none'][['event_id','stage','frame']]
 pop=paired.merge(anchors,on=['event_id','stage','frame'],validate='many_to_one')
 assert len(pop)==2*2*3*2*3*4
 pop['net_avoided_loss_lower'],pop['net_avoided_loss_upper']=avoided_interval(pop.net_added_loss_lower,pop.net_added_loss_upper)
 assert (pop.net_avoided_loss_lower<=pop.net_avoided_loss_upper).all()
 pop.to_csv(O/'paired_population_at_reference_time.csv',index=False)
 transfer=pd.read_csv(P/'neighbor_transfer_screen.csv')
 inc='outside_corridor_cell_area_head_increase_ge0_1m_km2'
 dec='outside_corridor_cell_area_head_decrease_ge0_1m_km2'
 risk=transfer.groupby(['event_id','stage','measure'],as_index=False).agg(peak_outside_increase_km2=(inc,'max'),peak_outside_decrease_km2=(dec,'max'))
 risk.to_csv(O/'risk_transfer_peaks.csv',index=False)
 return road,tr,pop,risk

def transfer_maps(risk):
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.frame-control/runtimes/physical-measures-mpl'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from shapely.geometry import shape
 from shapely import contains_xy
 with np.load(P/'measures/spatial_contract.npz') as z:
  idx=z['native_grid_north_index'];xy=z['native_xy'];corridor=z['corridor_native']
 with np.load(P/'motor_common_support.npz') as z:eligible=z['common_admitted'].ravel()[idx]&~corridor
 wards=read(ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1/wards_6677.geojson')['features']
 sels={'Three wards':np.zeros(len(idx),bool)}
 for f in wards:
  sel=contains_xy(shape(f['geometry']),xy[:,0],xy[:,1]);sels[str(f['properties']['ward'])]=sel;sels['Three wards']|=sel
 rows=[];images=[]
 for event in ['E20250911','E20240618']:
  for level in ['H','L']:
   with np.load(P/'exposure'/f'{event}__{level}__U__none.npz') as z:base=z['native_depth'][:,idx]
   assert base.shape==(109,len(idx))
   for arm in ARMS:
    gate();case=f'{event}__{level}__U__{arm}'
    with np.load(P/'exposure'/(case+'.npz')) as z:treated=z['native_depth'][:,idx]
    up=np.zeros(len(idx),dtype='f4');down=up.copy();max_frame_area=0.
    for i in range(109):
     # Exactly the existing screened 0..0.01m censor convention;no new imputation.
     u=np.where(np.isfinite(treated[i]),treated[i],0)-np.where(np.isfinite(base[i]),base[i],.01)
     d=np.where(np.isfinite(base[i]),base[i],0)-np.where(np.isfinite(treated[i]),treated[i],.01)
     up=np.maximum(up,u);down=np.maximum(down,d)
     max_frame_area=max(max_frame_area,np.count_nonzero(eligible&(u>=.1))*.0001)
    ref=risk[(risk.event_id==event)&(risk.stage==level)&(risk.measure==arm)].iloc[0]
    assert np.isclose(max_frame_area,ref.peak_outside_increase_km2,atol=1e-10),'Spatial map must reproduce original transfer screen'
    inc=eligible&(up>=.1);dec=eligible&(down>=.1)
    np.savez_compressed(O/(case+'_transfer_locations.npz'),native_xy=xy,native_grid_north_index=idx,eligible_outside=eligible,max_increase_lower_m=up,max_decrease_lower_m=down)
    for scope,sel in sels.items():rows.append(dict(event_id=event,stage=level,measure=arm,scope=scope,ever_increased_cell_area_km2=np.count_nonzero(inc&sel)*.0001,ever_decreased_cell_area_km2=np.count_nonzero(dec&sel)*.0001,center_assigned_10m_cell_area=True))
    fig,axs=plt.subplots(1,2,figsize=(10,7),layout='constrained')
    for ax,mask,color,title in [(axs[0],inc,'#c83f36','Increase >=0.10 m'),(axs[1],dec,'#16869b','Decrease >=0.10 m')]:
     for f in wards:
      geom=shape(f['geometry']);geoms=list(geom.geoms) if hasattr(geom,'geoms') else [geom]
      for g in geoms:
       x,y=g.exterior.xy;ax.plot(x,y,lw=.6,color='#8699a2')
     ax.scatter(xy[mask,0],xy[mask,1],s=2,c=color,rasterized=True)
     ax.set(aspect='equal',title=title,xlabel='EPSG:6677 Easting (m)',ylabel='Northing (m)')
    fig.suptitle(f'{event} / {level} / {arm}\nOutside frozen corridor;any sampled time;screened depth-change lower bound')
    filename=case+'_transfer.png';fig.savefig(V/filename,dpi=140);plt.close(fig);images.append(filename)
    del treated
   del base
 pd.DataFrame(rows).to_csv(O/'risk_transfer_ward_locations.csv',index=False)
 return images

def report(road,tr,pop,risk,images):
 def table(f):return '<div class="scroll">'+f.to_html(index=False,border=0,float_format=lambda v:f'{v:,.3f}',na_rep='—')+'</div>'
 main=road[(road.scope=='Three wards')&(road.stratum=='all')]
 columns=['event_id','stage','measure','bound','peak_closed_km_none','peak_closed_km_measure','peak_closed_km_avoided','peak_closed_km_avoided_pct','closed_km_hours_avoided']
 cp=pop[(pop.scope=='Three wards')&(pop.buffer_m==10)]
 parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>挡排措施效果汇总</title><style>body{font:16px/1.7 system-ui;color:#203c49;background:#eef3f5}main{max-width:1180px;margin:auto;padding:26px;background:white}h1{font-size:28px}h2{font-size:22px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{padding:7px;border-bottom:1px solid #ddd}img{width:100%;height:auto}.note{padding:15px;background:#fff0d9}</style><main>',
 '<h1>挡水、排水与挡排结合：相对无措施的变化</h1><p>全部12组新增水动力情景、4组复用基线及道路／人口／交通检查通过后生成。两场降雨×两档水位分别比较，不合并成单一收益。正的 avoided 表示改善，负值表示恶化；零基线的百分比留空。</p>',
 '<p class="note">假设措施效果，尚无实测工程验证。交通成本来自固定需求、固定容量的静态UE模型，不是实测排队时间。人口是2020年常住人口存量的车行接入损失区间，不是实际受困人数或昼间人数。相同原质量掩膜和未知分支继续保留。</p>',
 '<h2>封路减少多少</h2><p>单位km及km·h。peak为各自窗口峰值之差，峰值可能出现在不同时刻；closed_km_hours为相同9小时窗口的封路长度积分。三区及各区、保护走廊和其他道路分层结果见CSV。</p>',table(main[columns]),
 '<h2>交通成本与不可达需求</h2><p>使用预先冻结的无措施封路峰值时刻，同一事件两档水位和所有措施使用同一时刻。成本只比较四组都仍可达的同一批OD；新增恢复出行和被排除的请求不能解释为成本下降。需求为车辆请求代理量。</p>',table(tr[['event_id','stage','measure','frame','unreachable_proxy_none','unreachable_proxy_measure','unreachable_proxy_avoided','common_od_cost_reduction_pct','common_survivor_original_request_proxy','baseline_reachable_excluded_from_cost_proxy']]),
 '<h2>人口影响的配对区间</h2><p>以下采用原10m接入主分支，在上述固定时刻比较；lower／upper保留道路水深分支。net_avoided_loss直接来自同一候选点配对结果，未通过边际区间相减生成。restored和new_loss分别保留，不能用净收益掩盖新增损失；0／25m分支和各区见CSV。</p>',table(cp[['event_id','stage','measure','bound','frame','net_avoided_loss_lower','net_avoided_loss_upper','restored_lower','restored_upper','new_loss_lower','new_loss_upper']]),
 '<h2>是否转移到其他片区</h2><p>只检查原保护走廊外、共同准入单元。瞬时峰值面积与任一时刻曾增加的单元面积分开，不能相互替代；面积为10m模型单元筛查，不是实测淹没范围。以下水深差与水面差等价，因为同一单元地形未改变。</p>',table(risk),
 '<p>地图左右分别显示曾增加、曾减少至少0.10m的位置，同一位置可以在不同时刻分别出现两种变化。各区面积按单元中心归属，跨界单元存在约10m定位尺度。</p>']
 for image in images:parts.append(f'<img loading="lazy" src="{html.escape(image)}" alt="{html.escape(image)}">')
 parts.append('<h2>可复核文件</h2><ul>')
 for p in sorted(O.glob('*.csv')):parts.append(f'<li><a href="../../../../processed/{P.name}/effect_summary/{p.name}">{p.name}</a></li>')
 parts.append('</ul><p><a href="../review.html">原实验进度与完整输出</a></p></main></html>')
 tmp=V/'review.html.tmp';tmp.write_text(''.join(parts));tmp.replace(V/'review.html')

def selftest():
 f=pd.DataFrame([dict(event='x',measure='none',loss=10),dict(event='x',measure='barrier',loss=12),dict(event='x',measure='drainage',loss=4),dict(event='y',measure='none',loss=0),dict(event='y',measure='combined',loss=1)])
 r=compare(f,['event'],['loss']).set_index('measure')
 assert r.loc['barrier','loss_avoided']==-2 and r.loc['drainage','loss_avoided_pct']==60
 assert np.isnan(r.loc['combined','loss_avoided_pct']) and r.loc['combined','loss_avoided']==-1
 lo,hi=avoided_interval(np.array([-3,-10,4]),np.array([7,-5,9]))
 assert np.array_equal(lo,[-7,5,-9]) and np.array_equal(hi,[3,10,-4])
 print('SELFTEST PASSED: negative effects retained;zero denominator unknown;paired interval direction')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
 if args.self_test:selftest();return
 V.mkdir(exist_ok=True);O.mkdir(exist_ok=True)
 binding=sha(Path(__file__))
 if args.wait:
  if not (V/'validation.json').exists():
   (V/'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>等待挡排措施汇总</title><main style="max-width:850px;margin:50px auto;font:18px/1.8 system-ui"><h1>自动效果汇总等待中</h1><p>原队列继续运行。全部12组新增水动力情景及道路、人口、交通检查通过后，本页将自动显示挡水、排水、挡排结合的封路、交通成本、人口区间及风险转移地图。此页每60秒刷新。</p><p><a href="../review.html">查看实时实验进度</a></p></main></html>')
  begun=time.monotonic()
  while not (R/'delivery_validation.json').exists():
   gate();assert sha(Path(__file__))==binding,'Summary code changed';state=read(R/'campaign_state.json')
   if state['status']=='held':raise RuntimeError('Campaign held;no automatic numeric failure bypass')
   assert time.monotonic()-begun<36*3600,'Summary wait limit'
   dump(V/'status.json',dict(status='waiting_for_all_gates',pid=os.getpid(),qualified=len(state.get('completed_new_cases',[])),required=12,script_sha256=binding));time.sleep(30)
 gate()
 for n in ['delivery_validation.json','postprocessing_validation.json','traffic_validation.json']:assert read(R/n)['passed'],n
 for p in (R/'execution').glob('*/audit.json'):assert read(p)['all_checks_passed']
 assert len(read(R/'campaign_state.json')['completed_new_cases'])==12
 names=['road_summary.csv','paired_population_effects.csv','neighbor_transfer_screen.csv','traffic/physical_traffic_outcomes.csv','motor_common_support.npz','measures/spatial_contract.npz']
 sources={n:sha(P/n) for n in names};dump(V/'status.json',dict(status='summarizing',pid=os.getpid()))
 road,tr,pop,risk=tables();assert len(road)==288 and len(tr)==12 and len(risk)==12
 images=transfer_maps(risk);report(road,tr,pop,risk,images)
 assert sha(Path(__file__))==binding and all(sha(P/n)==h for n,h in sources.items())
 dump(V/'validation.json',dict(passed=True,all_campaign_gates_required=True,original_sources_unchanged=True,source_sha256=sources,script_sha256=binding,road_rows=len(road),traffic_rows=len(tr),population_rows=len(pop),spatial_maps=len(images),negative_effects_retained=True,no_new_simulations=True,no_training=True,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
 dump(V/'status.json',dict(status='qualified_complete',review=str(V/'review.html')))
 print('ALL PHYSICAL EFFECT SUMMARIES QUALIFIED',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if V.exists():dump(V/'status.json',dict(status='held',reason=str(exc)));(V/'error.txt').write_text(traceback.format_exc())
  raise
