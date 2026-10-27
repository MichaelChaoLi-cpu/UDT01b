"""Truthful auto-refreshing progress, then qualified hydraulic/population/measure results."""
from pathlib import Path
import json,html,datetime as dt
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];TAG='three_ward_five_event_population_validation_v1';P=ROOT/'data/processed'/TAG;R=ROOT/'data/exp/estimation'/TAG

def main():
 p=json.loads((P/'protocol.json').read_text());s=json.loads((R/'campaign_state.json').read_text())if(R/'campaign_state.json').exists()else dict(status='prepared_not_started',completed_new_cases=[],queue=p['execution_order']);current=s.get('current_case');body='<h1>三场新增暴雨：水动力验证 → 人口 → 措施实验</h1><p>新增2024-07-31、2025-07-10、2024-06-18；各运行原城市几何下低水位／抬升水位，共6组。原网格、求解器、道路保护方案及数值规则保持不变。</p><aside><b>状态：'+html.escape(s['status'])+'</b>；水动力数值合格 '+str(len(s['completed_new_cases']))+'/6组。<br>更新时间：'+dt.datetime.now(dt.timezone.utc).isoformat()+'</aside>'
 if current:body+='<p>当前：'+html.escape(current)+'；实际模型时钟 '+str(s.get('latest_native_time_s'))+' / 32400秒；当前组活动计算时间 '+f'{s.get("case_active_wall_seconds",0)/60:.1f}'+' 分钟。</p>'
 if s.get('error'):body+='<p class="error">流程停止并保留尝试：'+html.escape(s['error'])+'。未自动重试或放宽规则。</p>'
 rows=[]
 for r in json.loads((P/'run_manifest.json').read_text())['runs']:
  a=R/'execution'/r['case_id']/'audit.json';v=json.loads(a.read_text())if a.exists()else{};rows.append(dict(case=r['case_id'],status='数值合格'if v.get('all_checks_passed')else'检查未通过'if v else'运行中'if r['case_id']==current and s['status']in['running','auditing']else'未完成',runtime_minutes=v.get('summary',{}).get('runtime_s',0)/60 if v else None,negative_cells=v.get('summary',{}).get('negative_cells'),rain_difference_m3=v.get('summary',{}).get('rain_difference_m3')))
 body+='<h2>水动力队列</h2>'+pd.DataFrame(rows).to_html(index=False,na_rep='—')+'<h2>自动执行顺序</h2><ol><li>每组检查实际时钟、秒级降雨、步积分水量、正性、关闭边界零交换和原标签准入。任一失败即停队列。</li><li>6组全部合格后，交集原新地表支持并导出道路积水；桥隧和支持丢失道路继续未知，不强行恢复。</li><li>接入官方2020年496,345名常住人口，复用原住地候选接入和0／10／25m位置敏感性；保留证据不足人口，不从车辆数推人数。</li><li>人口账本和路由检查通过后，对三场新增抬升水位情景，继续固定道路保护×需求移出×理想疏导的48组对照，复用8组正常控制。</li></ol><h2>验证含义和运行预算</h2><p>这是数值／接口检查和跨事件受控验证。尚无独立实测积水或交通封闭来验证历史准确率。所有事件均使用同一现代城市快照和干初态，河道水位是原假设，交通使用同一17时负荷代理。</p><p>6月18日窗口前约21.53mm面积加权降雨未施加，因此不能称为完整历史洪水再现。一次1组，6CPU、7GiB，每组活动上限2小时，水动力总上限12小时；基于既有每组61—72分钟，预计水动力约6—8小时，随后继续后处理。PAUSE／STOP文件分别支持暂停／停止。未训练机器学习模型，未打开封存测试。</p>'
 for sub,title,file in [('population','人口接入','population_effect_summary.csv'),('traffic','措施与交通对照','combined_outcomes.csv')]:
  f=P/sub/file
  if f.exists():
   body+='<h2>'+title+'</h2><p><a href="../../../processed/'+TAG+'/'+sub+'/'+file+'">完整CSV结果</a></p>'
   val=R/sub/'validation.json';valid=val.exists()and json.loads(val.read_text()).get('passed',False)
   if valid:
    data=pd.read_csv(f)
    if sub=='population':data=data[(data.buffer_m==10)&(data.bound=='lower')][['case','arm','conditional_loss_lower','conditional_loss_upper','baseline_unresolved_population']]
    else:data=data[data.deferred_fraction.isin([0.,.1])&data.guidance.eq('UE')][['case_id','protection','deferred_fraction','completion_fraction_original','fixed_original_cohort_cost_index']]
    body+='<p>已通过本阶段核验。</p>'+data.to_html(index=False)

   else:body+='<p>计算或核验尚未完成，已写出的文件仅是中间结果。</p>'
 body+='<p>人口与措施主对照统一取各窗口起点后4小时，该时刻不一定是各场最强积水。道路积水另导出12个同步时刻，非同步峰值单列；本轮不将峰值拼成同时失联，也不从零散切片推算持续受困时间。</p>';body+='<p><a href="../../../processed/'+TAG+'/protocol.json">冻结实验合同</a> · <a href="campaign_state.json">实时状态</a> · <a href="plan_validation.json">运行前核验</a></p>'
 style='body{font:16px/1.8 system-ui,sans-serif;color:#243647;max-width:1120px;margin:28px auto;padding:0 18px}aside{padding:16px;background:#e9f3ed}table{display:block;overflow-x:auto;max-width:100%;border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}.error{background:#ffe3df;padding:14px}h1{font-size:27px}'
 (R/'review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="60"><title>三场新增暴雨与人口自动验证</title><style>'+style+'</style>'+body+'</html>')
if __name__=='__main__':main()
