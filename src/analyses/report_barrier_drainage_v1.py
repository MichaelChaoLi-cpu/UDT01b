"""Local experiment review:assumptions,live status,positive and adverse effects."""
from pathlib import Path
import json,datetime as dt,html,sys
import numpy as np,pandas as pd
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/udt01b-measures-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src/analyses'))
from observed_load_common_v1 import dump
P=ROOT/'data/processed/three_ward_barrier_drainage_v1';R=ROOT/'data/exp/estimation'/P.name;TEMP=ROOT/'data/processed/three_ward_five_event_temporal_analysis_v1'
def read(p):return json.loads(p.read_text())if p.exists()else{}
def csv(p):return pd.read_csv(p)if p.exists()and p.stat().st_size>2 else pd.DataFrame()
def table(d):return'<div class="scroll">'+d.to_html(index=False,border=0,escape=True,na_rep='—',float_format=lambda x:f'{x:,.3f}')+'</div>'
def main():
 p=read(P/'protocol.json');s=read(R/'campaign_state.json');pipeline=read(R/'pipeline_status.json');q=read(R/'solver_qualification.json');progress=read(R/'solver_qualification_progress.json');road=csv(P/'road_trajectories.csv');out=csv(P/'traffic/physical_traffic_outcomes.csv');status=s.get('status',pipeline.get('status','资格核验中'));status={'waiting_for_qualification':'资格核验中','running_physical_campaign':'正式批次运行中','running':'正式批次运行中','auditing':'数值审计中','held':'已停止并保留结果','qualified_complete':'全部通过','road_population_postprocessing':'道路与人口后处理','traffic_postprocessing':'交通对照计算中'}.get(status,status);done=len(s.get('completed_new_cases',[]));matrix=read(P/'run_manifest.json');R.mkdir(parents=True,exist_ok=True)
 if not(R/'deployment.png').exists():
  g=read(P/'measures/deployment_6677.geojson');fig,ax=plt.subplots(figsize=(8,9),layout='constrained');wards=read(TEMP/'wards_6677.geojson')
  for f in wards.get('features',[]):
   c=f['geometry']['coordinates'];c=c if f['geometry']['type']=='MultiPolygon'else[c]
   for ring in c:a=np.array(ring[0]);ax.plot(a[:,0],a[:,1],color='#81919b',lw=.8)
  segments=[f['geometry']['coordinates']for f in g['features']if f['properties']['kind']=='barrier'];ax.add_collection(LineCollection(segments,colors='#2d729a',lw=1.1,label='Overtoppable barriers'))
  pumps=[f for f in g['features']if f['properties']['kind']=='pump'];links=[f['geometry']['coordinates']for f in pumps];ax.add_collection(LineCollection(links,colors='#d48136',lw=.7,ls='--',label='Hypothetical transfer link'))
  a=np.array([f['geometry']['coordinates'][0]for f in pumps]);b=np.array([f['geometry']['coordinates'][1]for f in pumps]);ax.scatter(a[:,0],a[:,1],s=18,c='#b75b24',label='Pump intake');ax.scatter(b[:,0],b[:,1],s=17,c='#348b88',marker='x',label='Fixed-stage receiver');ax.set_aspect('equal');ax.autoscale();ax.set(xlabel='EPSG:6677 Easting (m)',ylabel='Northing (m)',title='Hypothetical deployment on frozen road corridors\n0.5m barriers;19 aggregated0.5m3/s pumping units');ax.legend(fontsize=8);fig.savefig(R/'deployment.png',dpi=145);plt.close(fig)
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三区挡水与排水措施实验</title><style>body{{margin:0;background:#eef3f5;color:#203c49;font:16px/1.8 system-ui}}main{{max-width:1180px;margin:24px auto;padding:30px;background:white;border-radius:12px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:28px}}.lead{{padding:18px;background:#e3f0e9;border-left:5px solid #42846c}}.note{{background:#fff0d9;padding:16px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #dde4e8}}.scroll{{overflow:auto}}a{{color:#176487}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}</style><main><p>MILI-D-20261018-016 · {dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M JST')}</p><h1>挡水、排水能否保住道路通行？</h1><div class="lead">状态：{html.escape(status)}。新增物理情景通过 {done}/12，另有4组既有无措施基线。<br>当前：{html.escape(s.get('current_case')or'—')}；实际计算时钟 {s.get('latest_native_time_s',0):,.1f}/32,400秒。<button onclick="location.reload()">刷新</button></div><p>先检验物理措施的水量与地形作用，再接入相同道路和人口分析。2025-09-11短时强雨与2024-06-18持续降雨，两档既有河道水位，每组无措施／仅挡水／仅排水／挡排结合，共16组对照。模型采用全域10m网格与原PLATEAU建筑、道路约束。</p>'''
 reason=s.get('reason',pipeline.get('reason'))
 if reason:body+='<p class="note">队列停止：'+html.escape(reason)+'</p>'
 body+='<h2>固定参数与部署</h2><p class="note">以下全部是实验假设，尚无设施调查支撑。挡水线按静态地面高程加0.5m设置，允许漫顶，系数0.6；696个10m网格面共约6.96km，并保留映射道路交叉开口。19个聚合抽排点，每点0.5m³/s，总上限9.5m³/s；水深低于0.05m不抽排，假设扬程超过5m停泵。单个直线连接不超过500m，未证明现实管线／软管可铺设。</p><p>选址沿用此前118条通道；泵点按静态地形洼地、间距和受纳水体距离批量选择，没有根据本轮措施结果调参。2025事件是原通道选择事件的样内背景，2024事件作迁移检查；两场雨不代表全部暴雨。</p><img src="deployment.png" alt="全域假设挡水线、泵吸水点与指定受纳水体连接"><p>排出的水明确进入指定的固定水位水体，计为显式抽排出域；内部转移则同时记来源和去向。受纳水位参与扬程截止，但未模拟河道新增蓄水、排口容量、泵性能曲线、能耗或下水道。固定水位受纳边界不是无限容量已获验证。</p><h2>资格核验</h2>'
 body+=table(pd.DataFrame(progress.get('checks',[]))) if progress.get('checks')else'<p>正在构建并核验。</p>'
 body+='<p>核验包括多泵共享储量、与格面出流共同限流、干单元不抽水、启动阈值、扬程截止、内部／外部水量配对；无措施新旧镜像须精确回放。正式情景继续检查原雨量积分、负体积、实际时钟、完整边界交换、水量平衡和质量掩膜；没有放宽门槛。</p>'
 if matrix:
  rows=[]
  for r in matrix['runs']:rows.append(dict(event=r['event_id'],stage=r['stage'],measure=r['measure'],status='既有合格基线'if r['reused']else'已通过'if r['case_id']in s.get('completed_new_cases',[])else'计算／审计中'if s.get('current_case')==r['case_id']else'待运行'))
  body+='<h2>全部实验矩阵</h2>'+table(pd.DataFrame(rows))
 if not road.empty:
  fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=dict(none='#586975',barrier='#467eb1',drainage='#299b80',combined='#b16d38')
  for row,event in enumerate(['E20250911','E20240618']):
   for col,stage in enumerate(['H','L']):
    ax=axs[row,col]
    for arm,color in colors.items():g=road[(road.event_id==event)&(road.stage==stage)&(road.measure==arm)&(road.bound=='lower')&(road.scope=='Three wards')&(road.stratum=='all')].sort_values('frame');ax.plot(g.elapsed_s/3600,g.closed_km,color=color,label=arm)
    ax.set(title=event+'/'+stage,xlabel='Hours from window start',ylabel='Qualified closed roads (km)',xlim=[0,9]);ax.legend(fontsize=8);ax.grid(alpha=.15)
  fig.suptitle('Physical measures | same thresholds and common road support | no forced-dry overlay');fig.savefig(R/'road_response.png',dpi=140);plt.close(fig);body+='<h2>道路关闭与恢复</h2><img src="road_response.png" alt="相同降雨下四组措施道路关闭轨迹">';a=csv(P/'road_summary.csv');body+=table(a[(a.scope=='Three wards')&(a.bound=='lower')&(a.stratum=='all')])if not a.empty else''
 pop=csv(P/'population_peak_intervals.csv')
 if not pop.empty:body+='<h2>常住人口条件影响</h2><p>2020常住人口存量496,345；保持0／10／25m接入候选、用途和楼宇分配的不确定性。表为10m接入主分支的瞬时总量峰值端点，不是各街区峰值相加，也不是实际受困人数。</p>'+table(pop[(pop.scope=='Three wards')&(pop.buffer_m==10)&(pop.bound=='lower')])
 x=csv(P/'neighbor_transfer_screen.csv')
 if not x.empty:body+='<h2>邻近区域的风险转移</h2><p>分别记录走廊外水面抬高和降低，不用净改善抵消新增受损。这里是水面变化≥0.10m的10m单元面积筛查，不是实测淹没面积。道路与人口的新增受损亦单独保留。</p>'+table(x.groupby(['event_id','stage','measure'],as_index=False).agg(peak_worsened_cell_area_km2=('outside_corridor_cell_area_head_increase_ge0_1m_km2','max'),peak_improved_cell_area_km2=('outside_corridor_cell_area_head_decrease_ge0_1m_km2','max')))
 if not out.empty:body+='<h2>机动车可达性与容量成本</h2><p>在本轮措施开始前已确定的无措施封路峰值时刻比较，L／H与四组均用同一时刻。沿用固定17时需求代理和容量假设，仅比较UE静态分配。成本比较固定为四组全部仍可达的共同OD，原可达但被排除的请求另列；新增恢复出行不混入成本均值。没有把措施道路强行改为干燥，没有真实排队延误或工程收益认证。</p>'+table(out)
 body+='<h2>研究边界与文件</h2><p>保留原9小时干初始状态、窗口前降雨遗漏、假设河道水位、桥隧暴露未知及2020人口局限。挡水可能加重其他位置积水；净收益、零收益和负收益都保留，不因效果不好重选位置。无机器学习训练、无封存测试读取；此前交通措施批次独立继续。</p><ul>'
 for f in ['protocol.json','source_binding.json','solver_binding.json','execution_code_binding.json','run_manifest.json','measures/barrier_faces.csv','measures/barrier_exclusions.csv','measures/pump_sites.csv','measures/deployment_6677.geojson','road_summary.csv','road_trajectories.csv','population_peak_intervals.csv','paired_population_effects.csv','neighbor_transfer_screen.csv','traffic/physical_traffic_outcomes.csv','traffic/protocol.json']:
  if(P/f).exists():body+=f'<li><a href="../../../processed/{P.name}/{f}">{f}</a></li>'
 for f in ['campaign_state.json','solver_qualification.json','qualification_execution_v2.log','input_validation.json','postprocessing_validation.json','traffic_validation.json','delivery_validation.json']:
  if(R/f).exists():body+=f'<li><a href="{f}">{f}</a></li>'
 body+='</ul><p>道路 © OpenStreetMap contributors / ODbL；PLATEAU、DEM、雷达与统计来源沿用项目已有来源审计。</p></main></html>';tmp=R/'review.html.tmp';tmp.write_text(body,encoding='utf-8');tmp.replace(R/'review.html');dump(R/'summary.json',dict(status=status,qualified_new_cases=done,maximum_new_cases=12,reused_baselines=4,solver_qualification_passed=q.get('passed',False),training=False,sealed_tests=False,engineering_effectiveness_observed=False));print('PHYSICAL REVIEW UPDATED',status,done,'/12',flush=True)
if __name__=='__main__':main()
