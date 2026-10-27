#!/usr/bin/env python3
"""Live/offline pilot review;no physical changes or population calculations."""
from pathlib import Path
import json,html
import pandas as pd
from shared_receiver_pilot_common_v1 import ROOT,P,R,dump

def main():
 manifest=json.loads((P/'run_manifest.json').read_text());state=json.loads((R/'campaign_state.json').read_text()) if (R/'campaign_state.json').exists() else {'status':'preflight_passed','completed_cases':[]};status=state['status'];completed=state.get('completed_cases',[]);current=state.get('current_case');rows=[]
 for run in manifest['runs']:rows.append({'情景':run['baseline_case'],'措施':run['portfolio'],'状态':'已通过' if run['case_id'] in completed else (status if current==run['case_id'] else '等待前组通过'),'对照':'复用原无措施基线'})
 refresh='<meta http-equiv="refresh" content="30">' if status not in ['qualified_complete','held','stopped_by_user'] else ''
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>共享接收额度小批次水动力核验</title><style>body{{font:16px/1.8 system-ui;max-width:1320px;margin:30px auto;padding:0 24px;color:#263f4a}}aside{{padding:18px;background:#eaf3f5;border-left:4px solid #438094}}.warn{{padding:16px;background:#fff0d8}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #d2dee3;padding:8px;text-align:left}}</style><h1>共享接收额度：首批水动力核验</h1><aside>状态：<b>{html.escape(status)}</b>；已通过{len(completed)}/2组。仅运行7月31日高／低水位下的同一方案：62个模型取水位置、1000m直线连接假设、每具名接收池共享0.5m³/s、总预留2.5m³/s。<b>位置数为模型取水口，不是机器台数。</b></aside><div class="scroll">{pd.DataFrame(rows).to_html(index=False)}</div>'''
 if current:page+=f'<p>当前组：{html.escape(current)}；实际模型时钟约{state.get("latest_native_time_s",0)/3600:.3f}/9小时；本组活跃计算时间{state.get("case_active_wall_seconds",0):,.0f}s。</p>'
 if state.get('error'):page+='<div class="warn">任务已保留并停止扩跑：'+html.escape(state['error'])+'</div>'
 page+='<h2>实验及预算</h2><p>相同原10m网格、建筑道路约束、屋面分配降雨、河道水位曲线和9小时窗口，只加入已分配的固定流量抽排；未加入新挡水设施。采用原已核验抽排CPU镜像，逐组串行、6CPU／7GiB，每组活跃计算最多2小时，失败即停止队列，不自动重试或扩展更高额度。新输出上限40GiB、剩余磁盘至少100GiB。</p><p>30分钟清除本地超阈值储量仅用于此前离线分配；实际运行按原0.05m启动水深、5m扬程截断和既有可用储量限流，不能保证30分钟服务目标。源水位下降会改变来水、河道交换及可用抽排能力。接收端为规定水位水库式边界，排出计为显式出域，不模拟河道新增蓄水或实际排口许可。</p><h2>数值核验</h2><p>保留原实际时钟、秒级降雨积分、负体积、闭合格面零流量、边界通量、地图储量及标签准入门槛。另逐步独立核验62个位置的可变设定流量、端点、启动／扬程、储量限制、显式抽排账本和共享接收池总额。古川两模型组共用一份额度，未命名水域不分配。</p>'
 actual=[]
 for run in manifest['runs']:
  f=R/'execution'/run['case_id']/'pump_budget_summary.json'
  if f.exists():
   a=json.loads(f.read_text());actual.append({'情景':run['baseline_case'],'实际抽排(m3)':a['actual_transferred_m3'],'逐步记录':a['pump_records'],'最大独立账本差(m3)':a['max_independent_sink_ledger_difference_m3'],'核验通过':a['passed']})
 if actual:page+='<div class="scroll">'+pd.DataFrame(actual).to_html(index=False,float_format=lambda v:f'{v:,.6f}')+'</div>'
 f=P/'road_summary.csv'
 if f.exists():
  a=pd.read_csv(f);a=a[(a.Scope=='Three wards')&(a['Depth Branch']=='lower')];page+='<h2>三区道路结果</h2><p>所有基线和措施在同一共同质量掩膜上比较，0.30m关闭、0.10m减速规则不变；未知桥隧不记安全。恢复与新增关闭分开，不能用净改善抵消新增受损。单位为km和km·h。</p><div class="scroll">'+a.to_html(index=False,float_format=lambda v:f'{v:,.4f}')+'</div>'
 f=P/'off_target_head_changes.csv'
 if f.exists():
  a=pd.read_csv(f);a=a[a.Scope=='Three wards'].groupby(['Case','Stage'],as_index=False).agg({'Outside Target Cells Certain Head Rise Area':'max','Outside Target Cells Certain Head Fall Area':'max'});page+='<h2>控制道路单元以外的变化</h2><p>分别保留水面确定升高或降低至少0.10m的10m单元面积(km²)，考虑干单元0—0.01m输出删失。这是网格面积筛查，不是实测浸水面积。</p><div class="scroll">'+a.to_html(index=False,float_format=lambda v:f'{v:,.5f}')+'</div>'
 page+='<p>本轮不计算新的交通拥堵或人口影响，不训练、不打开封存测试。道路和周边变化只有在两组通过后自动汇总；未通过时不发布措施改善结论。同一历史父事件用于位置选择和本轮核验，不能称为独立现实验证。</p><h2>文件</h2><ul>'
 for name in ['protocol.json','run_manifest.json','pump_sites.csv','pool_allowances.csv','execution_code_binding.json','road_summary.csv','road_trajectories.csv','off_target_head_changes.csv']:
  if (P/name).exists():page+=f'<li><a href="../../../processed/{P.name}/{name}">{name}</a></li>'
 for name in ['preflight_validation.json','campaign_state.json','postprocessing_validation.json','delivery_validation.json']:
  if (R/name).exists():page+=f'<li><a href="{name}">{name}</a></li>'
 for run in manifest['runs']:
  for name in ['audit.json','pump_budget_summary.json']:
   path=R/'execution'/run['case_id']/name
   if path.exists():page+=f'<li><a href="execution/{run["case_id"]}/{name}">{run["baseline_case"]} {name}</a></li>'
 page+='</ul><p><a href="../../research-design/three_ward_shared_receiver_allocation_v1/review.html">离线分配依据</a> · <a href="README.md">复现及暂停</a></p></html>'
 tmp=R/'review.html.tmp';tmp.write_text(page);tmp.replace(R/'review.html');dump(R/'summary.json',{'status':status,'qualified_cases':len(completed),'new_runs':2,'new_population_counts':False,'training':False})
if __name__=='__main__':main()
